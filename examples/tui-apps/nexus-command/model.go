package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/sahilm/fuzzy"
)

const (
	tabOverview = iota
	tabAgents
	tabLogs
	tabMetrics
)

var tabNames = []string{"Overview", "Agents", "Logs", "Metrics"}

const (
	focusAgents = iota
	focusLogs
	focusMetrics
)

type (
	animMsg        time.Time
	clockMsg       time.Time
	streamMsg      struct{}
	toastGoneMsg   struct{ id int }
	stepMsg        struct{ idx int }
	pipelineDoneMs struct{}
)

type command struct {
	name, desc string
	run        func(m model) (model, tea.Cmd)
}

type rect struct{ x, y, w, h int }

func (r rect) contains(x, y int) bool {
	return x >= r.x && x < r.x+r.w && y >= r.y && y < r.y+r.h
}

type model struct {
	width, height int
	tab, focus    int

	agents   []agent
	pos, vel []float64
	selected int
	spring   harmonica.Spring

	logs       []logEntry
	logOffset  int
	follow     bool
	errorsOnly bool

	task        string
	running     bool
	paused      bool
	script      []step
	stepIdx     int
	heldStep    int // step waiting while paused (-1 none)
	failPending bool
	failUsed    bool

	quality, qPos, qVel float64
	qualityHist         []float64
	tokHist             []float64
	tokens, lastTokens  int
	cost                float64
	iter                int
	runStart            time.Time
	runEnd              time.Time
	now                 time.Time
	animating           bool

	paletteOpen bool
	palette     textinput.Model
	cmds        []command
	matches     []int
	palCursor   int

	confirmOpen bool
	confirmText string
	confirmYes  func(model) (model, tea.Cmd)

	helpOpen bool

	toast      string
	toastOK    bool
	toastID    int
	toastUntil time.Time

	agentsRect, logsRect rect
}

func newModel(task string) model {
	ti := textinput.New()
	ti.Prompt = "❯ "
	ti.PromptStyle = keyStyle
	ti.TextStyle = textStyle
	ti.Placeholder = "type a command…"
	ti.CharLimit = 48
	agents := newAgents()
	m := model{
		agents:   agents,
		pos:      make([]float64, len(agents)),
		vel:      make([]float64, len(agents)),
		spring:   harmonica.NewSpring(harmonica.FPS(30), 7.0, 0.55),
		follow:   true,
		task:     task,
		heldStep: -1,
		palette:  ti,
		now:      time.Now(),
	}
	m.cmds = commands()
	m.addLog(lvInfo, "nexus", "NEXUS command center online — press r to run the pipeline, ctrl+k for commands", false)
	return m
}

func (m model) Init() tea.Cmd {
	return tea.Batch(clockTick(), textinput.Blink)
}

func clockTick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return clockMsg(t) })
}
func animTick() tea.Cmd {
	return tea.Tick(33*time.Millisecond, func(t time.Time) tea.Msg { return animMsg(t) })
}
func streamTick() tea.Cmd {
	return tea.Tick(28*time.Millisecond, func(time.Time) tea.Msg { return streamMsg{} })
}
func scheduleStep(idx int, after time.Duration) tea.Cmd {
	return tea.Tick(after, func(time.Time) tea.Msg { return stepMsg{idx} })
}

func (m *model) addLog(lvl level, ag, text string, stream bool) {
	e := logEntry{at: time.Now(), lvl: lvl, agent: ag, text: text, stream: stream}
	if !stream {
		e.revealed = len([]rune(text))
	}
	m.logs = append(m.logs, e)
	if len(m.logs) > 2000 {
		m.logs = m.logs[len(m.logs)-2000:]
	}
}

func (m *model) setToast(text string, ok bool) tea.Cmd {
	m.toastID++
	m.toast, m.toastOK = text, ok
	m.toastUntil = time.Now().Add(2500 * time.Millisecond)
	id := m.toastID
	return tea.Tick(2500*time.Millisecond, func(time.Time) tea.Msg { return toastGoneMsg{id} })
}

func (m *model) agentIndex(name string) int {
	for i, a := range m.agents {
		if a.name == name {
			return i
		}
	}
	return -1
}

func (m *model) resetRun() {
	m.agents = newAgents()
	for i := range m.pos {
		m.pos[i], m.vel[i] = 0, 0
	}
	m.quality, m.qPos, m.qVel = 0, 0, 0
	m.qualityHist = nil
	m.tokHist = nil
	m.tokens, m.lastTokens = 0, 0
	m.cost = 0
	m.iter = 0
	m.stepIdx = 0
	m.heldStep = -1
	m.failUsed = false
	m.runEnd = time.Time{}
}

func (m model) startRun() (model, tea.Cmd) {
	if m.running {
		return m, m.setToast("pipeline already running", false)
	}
	m.resetRun()
	m.running, m.paused = true, false
	m.runStart = time.Now()
	m.script = pipelineScript(m.task)
	m.addLog(lvInfo, "nexus", "▶ pipeline started — "+m.task, false)
	return m, tea.Batch(scheduleStep(0, 200*time.Millisecond), animTick())
}

// applyStep executes script[idx] and schedules the next one.
func (m model) applyStep(idx int) (model, tea.Cmd) {
	if !m.running || idx >= len(m.script) {
		return m, nil
	}
	if m.paused {
		m.heldStep = idx
		return m, nil
	}
	s := m.script[idx]
	m.stepIdx = idx
	ai := m.agentIndex(s.agent)
	if ai < 0 {
		return m, nil
	}
	a := &m.agents[ai]
	if a.state == stKilled {
		m.running = false
		m.addLog(lvErr, "nexus", "pipeline halted — "+a.name+" was killed. Use ctrl+k → Resume to continue", false)
		return m, nil
	}
	if s.start {
		a.state = stRunning
		a.started = time.Now()
		a.target = 0.05
		m.selected = ai
	}
	if s.iter > 0 {
		m.iter = s.iter
	}
	m.addLog(s.lvl, s.agent, s.text, s.stream)
	a.lastMsg = s.text
	if s.tokens > 0 {
		a.tokens += s.tokens
		m.tokens += s.tokens
		c := costFor(a.model, s.tokens)
		a.costUSD += c
		m.cost += c
	}
	if s.progress > 0 {
		a.target = s.progress
	}
	if s.quality > 0 {
		m.quality = s.quality
		m.qualityHist = append(m.qualityHist, s.quality)
	}
	if s.finish {
		a.state = stDone
		a.elapsed = time.Since(a.started)
		a.target = 1
	}
	cmds := []tea.Cmd{animTick()}
	if s.stream {
		cmds = append(cmds, streamTick())
	}
	if s.failable && m.failPending && !m.failUsed {
		m.failPending, m.failUsed = false, true
		rest := append([]step{}, m.script[idx+1:]...)
		m.script = append(append(m.script[:idx+1], failureSteps...), rest...)
	}
	if idx+1 < len(m.script) {
		cmds = append(cmds, scheduleStep(idx+1, jitter(s.delayMs)))
	} else {
		m.running = false
		m.runEnd = time.Now()
		m.addLog(lvOK, "nexus", fmt.Sprintf("■ pipeline complete in %s — %d iterations, quality %.2f, %d tokens, $%.2f",
			fmtDuration(m.runEnd.Sub(m.runStart)), m.iter, m.quality, m.tokens, m.cost), false)
		cmds = append(cmds, m.setToast("pipeline complete ✓", true))
	}
	return m, tea.Batch(cmds...)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.palette.Width = 40
		return m, nil

	case clockMsg:
		m.now = time.Time(msg)
		if m.running && !m.paused {
			m.tokHist = append(m.tokHist, float64(m.tokens-m.lastTokens))
			if len(m.tokHist) > 120 {
				m.tokHist = m.tokHist[len(m.tokHist)-120:]
			}
			m.lastTokens = m.tokens
		}
		return m, clockTick()

	case animMsg:
		settled := true
		for i := range m.agents {
			m.pos[i], m.vel[i] = m.spring.Update(m.pos[i], m.vel[i], m.agents[i].target)
			if abs(m.pos[i]-m.agents[i].target) > 0.001 || abs(m.vel[i]) > 0.001 {
				settled = false
			}
		}
		m.qPos, m.qVel = m.spring.Update(m.qPos, m.qVel, m.quality)
		if abs(m.qPos-m.quality) > 0.001 || abs(m.qVel) > 0.001 {
			settled = false
		}
		m.animating = !settled || (m.running && !m.paused)
		if m.animating {
			return m, animTick()
		}
		return m, nil

	case streamMsg:
		more := false
		for i := range m.logs {
			e := &m.logs[i]
			if e.stream && e.revealed < len([]rune(e.text)) {
				e.revealed = min(len([]rune(e.text)), e.revealed+3)
				more = true
			}
		}
		if more {
			return m, streamTick()
		}
		return m, nil

	case stepMsg:
		return m.applyStep(msg.idx)

	case toastGoneMsg:
		if msg.id == m.toastID {
			m.toast = ""
		}
		return m, nil

	case tea.MouseMsg:
		return m.handleMouse(msg)

	case tea.KeyMsg:
		return m.handleKey(msg)
	}
	return m, nil
}

func (m model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.paletteOpen || m.confirmOpen || m.helpOpen {
		return m, nil
	}
	m.agentsRect, m.logsRect, _ = m.layoutRects()
	switch msg.Button {
	case tea.MouseButtonWheelUp:
		if m.logsRect.contains(msg.X, msg.Y) {
			m.scrollLogs(-3)
		} else if m.agentsRect.contains(msg.X, msg.Y) {
			m.selected = max(0, m.selected-1)
		}
	case tea.MouseButtonWheelDown:
		if m.logsRect.contains(msg.X, msg.Y) {
			m.scrollLogs(3)
		} else if m.agentsRect.contains(msg.X, msg.Y) {
			m.selected = min(len(m.agents)-1, m.selected+1)
		}
	case tea.MouseButtonLeft:
		if msg.Action != tea.MouseActionPress {
			break
		}
		if msg.Y == 1 {
			x := 0
			for i, n := range tabNames {
				w := len(n) + 4
				if msg.X >= x && msg.X < x+w {
					m.tab = i
				}
				x += w
			}
		} else if m.agentsRect.contains(msg.X, msg.Y) {
			m.focus = focusAgents
			row := (msg.Y - m.agentsRect.y - 1) / 2
			if row >= 0 && row < len(m.agents) {
				m.selected = row
			}
		} else if m.logsRect.contains(msg.X, msg.Y) {
			m.focus = focusLogs
		}
	}
	return m, nil
}

func (m *model) visibleLogs() []logEntry {
	if !m.errorsOnly {
		return m.logs
	}
	out := make([]logEntry, 0, 16)
	for _, e := range m.logs {
		if e.lvl == lvErr || e.lvl == lvWarn {
			out = append(out, e)
		}
	}
	return out
}

func (m *model) scrollLogs(delta int) {
	n := len(m.visibleLogs())
	_, m.logsRect, _ = m.layoutRects()
	h := max(1, m.logsRect.h-2)
	if m.follow {
		m.logOffset = max(0, n-h)
	}
	m.logOffset = max(0, min(n-1, m.logOffset+delta))
	m.follow = m.logOffset >= n-h
}

func (m model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := msg.String()

	if m.confirmOpen {
		switch key {
		case "y", "Y", "enter":
			m.confirmOpen = false
			return m.confirmYes(m)
		case "n", "N", "esc", "q":
			m.confirmOpen = false
		}
		return m, nil
	}
	if m.helpOpen {
		if key == "?" || key == "esc" || key == "q" {
			m.helpOpen = false
		}
		return m, nil
	}
	if m.paletteOpen {
		switch key {
		case "esc", "ctrl+k":
			m.paletteOpen = false
			m.palette.Blur()
			return m, nil
		case "up", "ctrl+p":
			m.palCursor = max(0, m.palCursor-1)
			return m, nil
		case "down", "ctrl+n", "tab":
			m.palCursor = min(max(0, len(m.matches)-1), m.palCursor+1)
			return m, nil
		case "enter":
			if len(m.matches) == 0 {
				return m, nil
			}
			c := m.cmds[m.matches[m.palCursor]]
			m.paletteOpen = false
			m.palette.Blur()
			return c.run(m)
		}
		var cmd tea.Cmd
		m.palette, cmd = m.palette.Update(msg)
		m.filterCommands()
		return m, cmd
	}

	switch key {
	case "ctrl+c", "q":
		return m, tea.Quit
	case "ctrl+k", ":":
		m.paletteOpen = true
		m.palette.SetValue("")
		m.palCursor = 0
		m.filterCommands()
		return m, m.palette.Focus()
	case "?":
		m.helpOpen = true
	case "1", "2", "3", "4":
		m.tab = int(key[0] - '1')
	case "tab":
		m.focus = (m.focus + 1) % 3
	case "shift+tab":
		m.focus = (m.focus + 2) % 3
	case "r":
		return m.startRun()
	case "p", " ":
		return m.togglePause()
	case "f":
		m.follow = !m.follow
	case "e":
		m.errorsOnly = !m.errorsOnly
		m.follow = true
	case "i":
		return m.injectFailure()
	case "k", "x":
		return m.confirmKill()
	case "j", "down":
		if m.focus == focusLogs {
			m.scrollLogs(1)
		} else {
			m.selected = min(len(m.agents)-1, m.selected+1)
		}
	case "up":
		if m.focus == focusLogs {
			m.scrollLogs(-1)
		} else {
			m.selected = max(0, m.selected-1)
		}
	case "ctrl+d", "pgdown":
		m.scrollLogs(m.logsRect.h / 2)
	case "ctrl+u", "pgup":
		m.scrollLogs(-m.logsRect.h / 2)
	case "g", "home":
		if m.focus == focusLogs {
			m.follow = false
			m.logOffset = 0
		} else {
			m.selected = 0
		}
	case "G", "end":
		if m.focus == focusLogs {
			m.follow = true
		} else {
			m.selected = len(m.agents) - 1
		}
	}
	if key == "k" && m.focus == focusLogs {
		m.scrollLogs(-1)
	}
	return m, nil
}

func (m model) togglePause() (model, tea.Cmd) {
	if !m.running {
		return m, m.setToast("nothing running — press r", false)
	}
	m.paused = !m.paused
	if m.paused {
		m.addLog(lvWarn, "nexus", "⏸ pipeline paused", false)
		return m, m.setToast("paused", true)
	}
	m.addLog(lvInfo, "nexus", "▶ pipeline resumed", false)
	var cmd tea.Cmd
	if m.heldStep >= 0 {
		idx := m.heldStep
		m.heldStep = -1
		cmd = scheduleStep(idx, 150*time.Millisecond)
	}
	return m, tea.Batch(cmd, animTick(), m.setToast("resumed", true))
}

func (m model) injectFailure() (model, tea.Cmd) {
	if !m.running {
		return m, m.setToast("start the pipeline first (r)", false)
	}
	if m.failUsed || m.failPending {
		return m, m.setToast("failure already injected", false)
	}
	m.failPending = true
	m.addLog(lvWarn, "chaos", "☠ test failure armed — will fire when Tester runs", false)
	return m, m.setToast("chaos armed", true)
}

func (m model) confirmKill() (model, tea.Cmd) {
	a := m.agents[m.selected]
	if a.state != stRunning {
		return m, m.setToast(a.name+" is not running", false)
	}
	m.confirmOpen = true
	m.confirmText = "Kill " + a.name + " mid-task?"
	m.confirmYes = func(m model) (model, tea.Cmd) {
		a := &m.agents[m.selected]
		a.state = stKilled
		a.target = m.pos[m.selected]
		m.addLog(lvErr, a.name, "killed by operator (SIGTERM)", false)
		return m, tea.Batch(animTick(), m.setToast(a.name+" killed", false))
	}
	return m, nil
}

func (m model) resumeAfterKill() (model, tea.Cmd) {
	revived := false
	for i := range m.agents {
		if m.agents[i].state == stKilled {
			m.agents[i].state = stRunning
			revived = true
		}
	}
	if !revived {
		return m, m.setToast("no killed agents", false)
	}
	m.running, m.paused = true, false
	m.addLog(lvInfo, "nexus", "agent restarted — resuming from step "+fmt.Sprint(m.stepIdx+1), false)
	return m, tea.Batch(scheduleStep(m.stepIdx, 200*time.Millisecond), animTick(), m.setToast("resumed", true))
}

func (m *model) filterCommands() {
	q := strings.TrimSpace(m.palette.Value())
	m.matches = m.matches[:0]
	if q == "" {
		for i := range m.cmds {
			m.matches = append(m.matches, i)
		}
	} else {
		names := make([]string, len(m.cmds))
		for i, c := range m.cmds {
			names[i] = c.name
		}
		for _, r := range fuzzy.Find(q, names) {
			m.matches = append(m.matches, r.Index)
		}
	}
	m.palCursor = min(m.palCursor, max(0, len(m.matches)-1))
}

func commands() []command {
	return []command{
		{"Run pipeline", "start a fresh generation run", func(m model) (model, tea.Cmd) { return m.startRun() }},
		{"Pause / resume", "toggle the scheduler", func(m model) (model, tea.Cmd) { return m.togglePause() }},
		{"Inject test failure", "chaos: make the Tester fail once, watch the Coder self-heal", func(m model) (model, tea.Cmd) { return m.injectFailure() }},
		{"Kill selected agent", "terminate the highlighted agent", func(m model) (model, tea.Cmd) { return m.confirmKill() }},
		{"Resume after kill", "restart killed agents and continue", func(m model) (model, tea.Cmd) { return m.resumeAfterKill() }},
		{"Toggle follow logs", "auto-scroll to newest", func(m model) (model, tea.Cmd) { m.follow = !m.follow; return m, nil }},
		{"Filter: errors & warnings", "show only problems", func(m model) (model, tea.Cmd) { m.errorsOnly = !m.errorsOnly; return m, nil }},
		{"Clear logs", "empty the log buffer", func(m model) (model, tea.Cmd) {
			m.logs = nil
			m.logOffset = 0
			return m, m.setToast("logs cleared", true)
		}},
		{"Go to Overview", "tab 1", func(m model) (model, tea.Cmd) { m.tab = tabOverview; return m, nil }},
		{"Go to Agents", "tab 2", func(m model) (model, tea.Cmd) { m.tab = tabAgents; return m, nil }},
		{"Go to Logs", "tab 3", func(m model) (model, tea.Cmd) { m.tab = tabLogs; return m, nil }},
		{"Go to Metrics", "tab 4", func(m model) (model, tea.Cmd) { m.tab = tabMetrics; return m, nil }},
		{"Help", "show key bindings", func(m model) (model, tea.Cmd) { m.helpOpen = true; return m, nil }},
		{"Quit", "exit nexus-command", func(m model) (model, tea.Cmd) { return m, tea.Quit }},
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
