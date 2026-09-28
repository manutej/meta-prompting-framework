package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/aymanbagabas/go-osc52/v2"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"

	"alembic/harness"
	"alembic/jev"
)

type tab int

const (
	tabTasks tab = iota
	tabWorktrees
	tabJev
	tabAgents
)

var tabNames = []string{"Tasks", "Worktrees", "Jev", "Agents"}

type config struct {
	Feed     string
	Outbox   string
	Packs    string
	Receipts string
	Repo     string
	RepoOK   bool
	Demo     bool
	Live     bool // with Demo: use the real Jev client instead of the mock
	Poll     time.Duration
}

// Messages produced by tea.Cmds. Tests inject these directly.
type (
	pollTickMsg struct{}
	demoTickMsg struct{}
	animTickMsg struct{}
	wtTickMsg   struct{}
	feedMsg     struct {
		recs    []harness.FeedRecord
		missing bool
		err     error
	}
	demoStepMsg  struct{ err error }
	toastGoneMsg struct{ id int }
	wtListMsg    struct {
		wts []harness.Worktree
		err error
	}
	wtAddedMsg struct {
		path string
		err  error
	}
	wtRemovedMsg struct {
		path string
		err  error
	}
	packsMsg struct {
		packs []*jev.Pack
		errs  map[string]error
	}
	jevStateMsg struct {
		key   string
		state []byte
		err   error
	}
	jevDoneMsg struct {
		runID    int
		taskID   string
		pack     *jev.Pack
		stateRef string
		state    []byte
		resp     *jev.Response
		err      error
	}
	execDoneMsg struct{ err error }
	openedMsg   struct {
		what string
		err  error
	}
)

type toast struct {
	text  string
	style lipgloss.Style
	id    int
}

type confirmState struct {
	open  bool
	text  string
	hint  string
	onYes func(m model) (model, tea.Cmd)
}

type composerKind int

const (
	composerPing composerKind = iota
	composerJev
)

type composerState struct {
	open    bool
	kind    composerKind
	agentID string
	agent   string
	taskID  string
	input   textinput.Model
	canned  int
	receipt *jev.Receipt
}

var cannedPings = []string{"status?", "stop after current step", "add tests first", "ship it"}

// picker is a small list overlay used for agents, tasks, worktrees and diff kinds.
type pickerState struct {
	open   bool
	title  string
	items  []string
	labels []string
	cursor int
	query  textinput.Model
	fuzzy  bool
	onPick func(m model, idx int) (model, tea.Cmd)
}

type taskJev struct {
	running bool
	receipt *jev.Receipt
	path    string
	at      time.Time
}

type model struct {
	cfg    config
	width  int
	height int
	tab    tab
	now    time.Time

	snap   *harness.Snapshot
	feed   *harness.Feed
	outbox *harness.Outbox
	client *jev.Client
	demo   *harness.Demo

	feedMissing  bool
	lastRecordAt time.Time
	feedErr      string

	packs    []*jev.Pack
	packErrs []packError

	toast       toast
	toastSeq    int
	animPending bool
	frame       int

	tasks  tasksState
	wt     wtState
	jv     jevState
	agents agentsState

	palette  paletteState
	help     bool
	helpOff  int
	confirm  confirmState
	composer composerState
	picker   pickerState
	form     *huh.Form
	formKind string
	formVals *formValues

	pingsSent map[string]int
	taskJev   map[string]*taskJev
	copier    func(string)
	runSeq    int
}

// formValues is heap-allocated so huh's bound pointers survive model copies.
type formValues struct {
	branch, path string
}

// Timings that tests shorten.
var (
	toastTTL     = 2500 * time.Millisecond
	demoInterval = 2500 * time.Millisecond
	wtInterval   = 5 * time.Second
)

type packError struct {
	name string
	err  error
}

func newModel(cfg config) model {
	snap, feed, err := harness.LoadAll(cfg.Feed)
	m := model{
		cfg:       cfg,
		snap:      snap,
		feed:      feed,
		outbox:    harness.NewOutbox(cfg.Outbox),
		client:    jev.NewFromEnv(),
		now:       time.Now(),
		pingsSent: map[string]int{},
		taskJev:   map[string]*taskJev{},
		copier:    copyOSC52,
	}
	if err != nil {
		m.feedErr = err.Error()
	}
	if _, statErr := os.Stat(cfg.Feed); statErr != nil {
		m.feedMissing = true
	} else if snap.Records > 0 {
		m.lastRecordAt = time.Now()
	}
	if cfg.Demo {
		m.demo = harness.NewDemo(cfg.Feed, cfg.Outbox)
		if !cfg.Live {
			// a demo is repeatable and free: deterministic MOCK unless --live
			m.client.APIKey = ""
		}
	}
	m.tasks = newTasksState()
	m.wt = newWtState()
	m.jv = newJevState()
	m.palette = newPaletteState()
	m.loadPacks()
	m.loadTaskReceipts()
	if cfg.Demo {
		// the demo opens on the pack whose state (the selected task) is always there
		for i, p := range m.packs {
			if p.ID == "task-readiness" {
				m.jv.packCursor = i
			}
		}
	}
	ti := textinput.New()
	ti.Prompt = ""
	ti.CharLimit = 200
	m.composer.input = ti
	q := textinput.New()
	q.Prompt = "❯ "
	q.PromptStyle = keyStyle
	m.picker.query = q
	m.rebuildRows()
	return m
}

func (m *model) loadPacks() {
	packs, errs := jev.LoadDir(m.cfg.Packs)
	m.applyPacks(packs, errs)
}

// applyPacks installs a (re)loaded pack set, keeping the cursor in range.
func (m *model) applyPacks(packs []*jev.Pack, errs map[string]error) {
	m.packs = packs
	m.packErrs = nil
	for name, err := range errs {
		m.packErrs = append(m.packErrs, packError{name: name, err: err})
	}
	sort.Slice(m.packErrs, func(i, j int) bool { return m.packErrs[i].name < m.packErrs[j].name })
	if m.jv.packCursor >= len(m.packs) {
		m.jv.packCursor = max(0, len(m.packs)-1)
	}
}

// loadTaskReceipts indexes the newest receipt per task so the detail pane
// can show "last jev" without touching disk on every render.
func (m *model) loadTaskReceipts() {
	for _, rc := range jev.LoadReceipts(m.cfg.Receipts, "") {
		if rc.StateRef == "" || (rc.StateSource != "task" && rc.StateSource != "events") {
			continue
		}
		if _, ok := m.taskJev[rc.StateRef]; ok {
			continue
		}
		r := rc
		m.taskJev[rc.StateRef] = &taskJev{receipt: &r, at: rc.At}
	}
}

func copyOSC52(s string) {
	_, _ = os.Stderr.WriteString(osc52.New(s).String())
}

func (m model) Init() tea.Cmd {
	cmds := []tea.Cmd{pollTick(m.cfg.Poll), m.refreshWorktreesCmd()}
	if m.demo != nil {
		cmds = append(cmds, demoTick())
	}
	if m.needsAnim() {
		cmds = append(cmds, animTick())
	}
	return tea.Batch(cmds...)
}

func pollTick(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(time.Time) tea.Msg { return pollTickMsg{} })
}

func demoTick() tea.Cmd {
	return tea.Tick(demoInterval, func(time.Time) tea.Msg { return demoTickMsg{} })
}

func animTick() tea.Cmd {
	return tea.Tick(90*time.Millisecond, func(time.Time) tea.Msg { return animTickMsg{} })
}

func wtTick() tea.Cmd {
	return tea.Tick(wtInterval, func(time.Time) tea.Msg { return wtTickMsg{} })
}

func (m model) pollCmd() tea.Cmd {
	feed, path := m.feed, m.cfg.Feed
	return func() tea.Msg {
		_, statErr := os.Stat(path)
		recs, err := feed.Poll()
		return feedMsg{recs: recs, missing: statErr != nil, err: err}
	}
}

// demoStepCmd runs the simulated harness against a copy of the snapshot so
// the Cmd never races with Update.
func (m model) demoStepCmd() tea.Cmd {
	demo := m.demo
	clone := harness.NewSnapshot()
	for id, t := range m.snap.Tasks {
		c := *t
		clone.Tasks[id] = &c
	}
	return func() tea.Msg { return demoStepMsg{err: demo.Step(clone)} }
}

// needsAnim reports whether anything on screen is animating.
func (m model) needsAnim() bool {
	if m.jv.running || m.wt.loading {
		return true
	}
	for _, tj := range m.taskJev {
		if tj.running {
			return true
		}
	}
	if m.tab == tabTasks {
		for _, t := range m.snap.Tasks {
			if t.State == harness.StateRunning {
				return true
			}
		}
	}
	return false
}

// kickAnim starts the spinner loop if needed and not already running.
func (m *model) kickAnim() tea.Cmd {
	if m.animPending || !m.needsAnim() {
		return nil
	}
	m.animPending = true
	return animTick()
}

func (m *model) setToast(text string, st lipgloss.Style) tea.Cmd {
	m.toastSeq++
	m.toast = toast{text: text, style: st, id: m.toastSeq}
	id := m.toastSeq
	return tea.Tick(toastTTL, func(time.Time) tea.Msg { return toastGoneMsg{id: id} })
}

func (m *model) ok(text string) tea.Cmd   { return m.setToast(text, toastOK) }
func (m *model) warn(text string) tea.Cmd { return m.setToast(text, toastWarn) }
func (m *model) fail(text string) tea.Cmd { return m.setToast(text, toastErr) }

func (m model) anyOverlay() bool {
	return m.palette.open || m.help || m.confirm.open || m.picker.open || m.form != nil || m.jv.history.open
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.resize()
		return m, nil

	case pollTickMsg:
		m.now = time.Now()
		return m, tea.Batch(m.pollCmd(), pollTick(m.cfg.Poll))

	case feedMsg:
		m.feedMissing = msg.missing
		if msg.err != nil {
			m.feedErr = msg.err.Error()
		} else {
			m.feedErr = ""
		}
		cmd := m.applyRecords(msg.recs)
		return m, tea.Batch(cmd, m.kickAnim())

	case demoTickMsg:
		if m.demo == nil {
			return m, nil
		}
		return m, tea.Batch(m.demoStepCmd(), demoTick())

	case demoStepMsg:
		if msg.err != nil {
			return m, m.fail("demo: " + msg.err.Error())
		}
		return m, nil

	case animTickMsg:
		m.animPending = false
		if !m.needsAnim() {
			return m, nil
		}
		m.frame++
		m.animPending = true
		return m, animTick()

	case toastGoneMsg:
		if m.toast.id == msg.id {
			m.toast = toast{}
		}
		return m, nil

	case wtTickMsg:
		if m.tab != tabWorktrees {
			m.wt.ticking = false
			return m, nil
		}
		return m, tea.Batch(m.refreshWorktreesCmd(), wtTick())

	case wtListMsg:
		m.wt.loading = false
		m.wt.err = msg.err
		if msg.err == nil {
			m.wt.list = msg.wts
			harness.LinkTasks(m.wt.list, m.snap)
			m.wt.cursor = min(m.wt.cursor, max(0, len(m.wt.list)-1))
		}
		return m, nil

	case wtAddedMsg:
		if msg.err != nil {
			return m, m.fail(msg.err.Error())
		}
		return m, tea.Batch(m.ok("worktree created "+shortPath(msg.path)), m.refreshWorktreesCmd())

	case wtRemovedMsg:
		if msg.err != nil {
			return m, m.fail(msg.err.Error())
		}
		return m, tea.Batch(m.ok("worktree removed "+shortPath(msg.path)), m.refreshWorktreesCmd())

	case packsMsg:
		m.applyPacks(msg.packs, msg.errs)
		toast := m.ok(fmt.Sprintf("packs reloaded · %d valid", len(m.packs)))
		if len(m.packErrs) > 0 {
			toast = m.warn(fmt.Sprintf("packs reloaded · %d valid · %d invalid", len(m.packs), len(m.packErrs)))
		}
		return m, tea.Batch(toast, m.jevStateCmd())

	case jevStateMsg:
		return m.applyJevState(msg)

	case jevDoneMsg:
		return m.applyJevDone(msg)

	case execDoneMsg:
		if msg.err != nil {
			return m, m.fail("editor: " + msg.err.Error())
		}
		return m, nil

	case openedMsg:
		if msg.err != nil {
			return m, m.warn(msg.what + " (opener unavailable)")
		}
		return m, m.ok(msg.what)

	case tea.MouseMsg:
		return m.handleMouse(msg)

	case tea.KeyMsg:
		return m.handleKey(msg)
	}

	if m.form != nil {
		return m.updateForm(msg)
	}
	return m, nil
}

// applyRecords folds feed records into the snapshot, keeping the selection.
func (m *model) applyRecords(recs []harness.FeedRecord) tea.Cmd {
	if len(recs) == 0 {
		return nil
	}
	var cmds []tea.Cmd
	acksBefore := len(m.snap.Acks)
	for _, r := range recs {
		m.snap.Apply(r)
		if !r.TS.IsZero() && r.TS.After(m.lastRecordAt) {
			m.lastRecordAt = r.TS
		}
	}
	if m.lastRecordAt.IsZero() {
		m.lastRecordAt = time.Now()
	}
	for _, a := range m.snap.Acks[acksBefore:] {
		name := m.agentName(a.Agent)
		cmds = append(cmds, m.ok(name+": "+a.Text))
	}
	m.rebuildRows()
	harness.LinkTasks(m.wt.list, m.snap)
	if src, _ := m.jevSource(); m.tab == tabJev && (src == "task" || src == "events") {
		cmds = append(cmds, m.jevStateCmd())
	}
	return tea.Batch(cmds...)
}

func (m model) agentName(id string) string {
	if a, ok := m.snap.Agents[id]; ok && a.Name != "" {
		return a.Name
	}
	if id == "" {
		return "—"
	}
	return id
}

func (m *model) resize() {
	m.composer.input.Width = max(10, m.width-30)
	m.picker.query.Width = max(10, min(50, m.width-12))
	m.palette.input.Width = max(10, min(50, m.width-12))
	m.tasks.filter.Width = max(10, m.tasksListWidth()-14)
	m.jv.text.SetWidth(max(10, m.jevStateWidth()-4))
	m.jv.text.SetHeight(max(3, min(8, m.bodyHeight()-8)))
	m.jv.file.Width = max(10, m.jevStateWidth()-8)
	if m.form != nil {
		m.form = m.form.WithWidth(boxWidth(60, m.width) - 2)
	}
}

func (m model) bodyHeight() int {
	h := m.height - 3
	if m.composer.open {
		h--
	}
	return max(0, h)
}

func (m model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	switch {
	case m.form != nil:
		if k == "esc" {
			m.form = nil
			return m, nil
		}
		return m.updateForm(msg)
	case m.palette.open:
		return m.updatePalette(msg)
	case m.help:
		return m.updateHelp(msg)
	case m.confirm.open:
		switch k {
		case "y", "Y":
			m.confirm.open = false
			if m.confirm.onYes != nil {
				return m.confirm.onYes(m)
			}
			return m, nil
		case "n", "N", "esc", "q":
			m.confirm.open = false
		}
		return m, nil
	case m.picker.open:
		return m.updatePicker(msg)
	case m.jv.history.open:
		return m.updateHistory(msg)
	case m.composer.open:
		return m.updateComposer(msg)
	case m.tasks.filtering:
		return m.updateFilter(msg)
	case m.tab == tabJev && m.jv.editing:
		return m.updateJevEditor(msg)
	}

	switch k {
	case "ctrl+c", "q":
		return m, tea.Quit
	case "?":
		m.help, m.helpOff = true, 0
		return m, nil
	case "ctrl+k", ":":
		return m.openPalette()
	case "1", "2", "3", "4":
		return m.switchTab(tab(int(k[0] - '1')))
	}
	switch m.tab {
	case tabTasks:
		return m.updateTasks(msg)
	case tabWorktrees:
		return m.updateWorktrees(msg)
	case tabJev:
		return m.updateJev(msg)
	case tabAgents:
		return m.updateAgents(msg)
	}
	return m, nil
}

func (m model) switchTab(t tab) (model, tea.Cmd) {
	if t < tabTasks || t > tabAgents {
		return m, nil
	}
	m.tab = t
	var cmds []tea.Cmd
	switch t {
	case tabWorktrees:
		cmds = append(cmds, m.refreshWorktreesCmd())
		if !m.wt.ticking {
			m.wt.ticking = true
			cmds = append(cmds, wtTick())
		}
	case tabJev:
		cmds = append(cmds, m.jevStateCmd())
	}
	cmds = append(cmds, m.kickAnim())
	return m, tea.Batch(cmds...)
}

func (m model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.anyOverlay() || m.composer.open {
		return m, nil
	}
	switch msg.Button {
	case tea.MouseButtonWheelUp:
		return m.scrollBy(-3)
	case tea.MouseButtonWheelDown:
		return m.scrollBy(3)
	case tea.MouseButtonLeft:
		if msg.Action != tea.MouseActionPress {
			return m, nil
		}
		if msg.Y == 1 {
			x := 0
			for i, name := range tabNames {
				w := lipgloss.Width(tabInactive.Render(name))
				if msg.X >= x && msg.X < x+w {
					return m.switchTab(tab(i))
				}
				x += w
			}
			return m, nil
		}
		if msg.Y >= 2 && msg.Y < 2+m.bodyHeight() {
			return m.clickBody(msg.X, msg.Y-2)
		}
	}
	return m, nil
}

func (m model) scrollBy(n int) (model, tea.Cmd) {
	switch m.tab {
	case tabTasks:
		if m.tasks.focus == focusDetail {
			m.tasks.evOffset = max(0, m.tasks.evOffset+n)
			return m, nil
		}
		return m.moveTask(n), nil
	case tabWorktrees:
		m.wt.cursor = clampInt(m.wt.cursor+n, 0, max(0, len(m.wt.list)-1))
	case tabJev:
		if m.jv.focus == jevFocusResult {
			m.jv.resultOffset = max(0, m.jv.resultOffset+n)
		} else {
			m.jv.packCursor = clampInt(m.jv.packCursor+n, 0, max(0, len(m.packs)-1))
			return m, m.jevStateCmd()
		}
	case tabAgents:
		m.agents.cursor = clampInt(m.agents.cursor+n, 0, max(0, len(m.agentList())-1))
	}
	return m, nil
}

func (m model) clickBody(x, y int) (model, tea.Cmd) {
	switch m.tab {
	case tabTasks:
		lw := m.tasksListWidth()
		if x < lw {
			m.tasks.focus = focusList
			if id := m.taskAtLine(y - 1); id != "" {
				m.tasks.selID = id
				m.tasks.elemCursor, m.tasks.evOffset = 0, 0
			}
		} else {
			m.tasks.focus = focusDetail
		}
	case tabWorktrees:
		if idx := y - 3; idx >= 0 && idx < len(m.wt.list) {
			m.wt.cursor = idx
		}
	case tabJev:
		switch {
		case x < packsPaneWidth:
			m.jv.focus = jevFocusPacks
			if idx := (y-1)/2 + m.jv.packOffset; y >= 1 && idx >= 0 && idx < len(m.packs) {
				m.jv.packCursor = idx
				return m, m.jevStateCmd()
			}
		case x < packsPaneWidth+m.jevStateWidth():
			m.jv.focus = jevFocusState
		default:
			m.jv.focus = jevFocusResult
		}
	case tabAgents:
		if idx := y - 3; idx >= 0 && idx < len(m.agentList()) {
			m.agents.cursor = idx
		}
	}
	return m, nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ---------- composer ----------

func (m model) openComposer(kind composerKind, agentID, taskID string, rc *jev.Receipt) (model, tea.Cmd) {
	m.composer.open = true
	m.composer.kind = kind
	m.composer.agentID = agentID
	m.composer.agent = m.agentName(agentID)
	m.composer.taskID = taskID
	m.composer.receipt = rc
	m.composer.canned = -1
	m.composer.input.SetValue("")
	m.resize()
	return m, m.composer.input.Focus()
}

func (m model) updateComposer(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.composer.open = false
		m.composer.input.Blur()
		return m, nil
	case "tab":
		if m.composer.kind == composerPing {
			m.composer.canned = (m.composer.canned + 1) % len(cannedPings)
			m.composer.input.SetValue(cannedPings[m.composer.canned])
			m.composer.input.CursorEnd()
		}
		return m, nil
	case "enter":
		text := strings.TrimSpace(m.composer.input.Value())
		m.composer.open = false
		m.composer.input.Blur()
		switch m.composer.kind {
		case composerPing:
			if text == "" {
				return m, m.warn("empty ping not sent")
			}
			if _, err := m.outbox.Ping(m.composer.agentID, m.composer.taskID, text); err != nil {
				return m, m.fail("outbox: " + err.Error())
			}
			m.pingsSent[m.composer.agentID]++
			return m, m.ok("ping sent → " + m.composer.agent)
		case composerJev:
			if m.composer.receipt == nil {
				return m, m.warn("no receipt to send")
			}
			_, err := m.outbox.Send(harness.Command{Type: "jev.receipt", TaskID: m.composer.taskID, Text: text, Data: m.composer.receipt})
			if err != nil {
				return m, m.fail("outbox: " + err.Error())
			}
			return m, m.ok("receipt sent → harness")
		}
		return m, nil
	}
	var cmd tea.Cmd
	m.composer.input, cmd = m.composer.input.Update(msg)
	return m, cmd
}

func (m model) viewComposer(w int) string {
	var prompt string
	switch m.composer.kind {
	case composerPing:
		prompt = keyStyle.Render("ping → " + m.composer.agent)
		if m.composer.taskID != "" {
			prompt += mutedStyle.Render(" (" + m.composer.taskID + ")")
		}
	case composerJev:
		prompt = keyStyle.Render("jev.receipt → harness")
		if m.composer.taskID != "" {
			prompt += mutedStyle.Render(" (" + m.composer.taskID + ")")
		}
	}
	prompt += keyStyle.Render(" ❯ ")
	hint := mutedStyle.Render("tab canned · enter send · esc")
	if m.composer.kind == composerJev {
		hint = mutedStyle.Render("enter send · esc")
	}
	line := prompt + m.composer.input.View()
	return splitRow(line, hint, w)
}

// ---------- picker ----------

func (m model) openPicker(title string, labels, items []string, fuzzy bool, onPick func(m model, idx int) (model, tea.Cmd)) (model, tea.Cmd) {
	m.picker = pickerState{open: true, title: title, items: items, labels: labels, fuzzy: fuzzy, onPick: onPick, query: m.picker.query}
	m.picker.query.SetValue("")
	if fuzzy {
		return m, m.picker.query.Focus()
	}
	return m, nil
}

func (m model) pickerMatches() []int {
	q := strings.TrimSpace(m.picker.query.Value())
	if !m.picker.fuzzy || q == "" {
		out := make([]int, len(m.picker.labels))
		for i := range out {
			out[i] = i
		}
		return out
	}
	var out []int
	for _, r := range fuzzyFind(q, m.picker.labels) {
		out = append(out, r.Index)
	}
	return out
}

func (m model) updatePicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	matches := m.pickerMatches()
	switch msg.String() {
	case "esc":
		m.picker.open = false
		m.picker.query.Blur()
		return m, nil
	case "up", "ctrl+p":
		m.picker.cursor = max(0, m.picker.cursor-1)
		return m, nil
	case "down", "ctrl+n":
		m.picker.cursor = min(max(0, len(matches)-1), m.picker.cursor+1)
		return m, nil
	case "enter":
		if len(matches) == 0 {
			return m, nil
		}
		idx := matches[clampInt(m.picker.cursor, 0, len(matches)-1)]
		m.picker.open = false
		m.picker.query.Blur()
		if m.picker.onPick != nil {
			return m.picker.onPick(m, idx)
		}
		return m, nil
	}
	if !m.picker.fuzzy {
		switch msg.String() {
		case "j":
			m.picker.cursor = min(max(0, len(matches)-1), m.picker.cursor+1)
		case "k":
			m.picker.cursor = max(0, m.picker.cursor-1)
		default:
			for i, it := range m.picker.items {
				if len(it) > 0 && msg.String() == strings.ToLower(it[:1]) {
					m.picker.open = false
					if m.picker.onPick != nil {
						return m.picker.onPick(m, i)
					}
				}
			}
		}
		return m, nil
	}
	var cmd tea.Cmd
	m.picker.query, cmd = m.picker.query.Update(msg)
	m.picker.cursor = min(m.picker.cursor, max(0, len(m.pickerMatches())-1))
	return m, cmd
}

func (m model) viewPicker() string {
	w := boxWidth(60, m.width)
	lines := []string{titleStyle.Render(fit(m.picker.title, w))}
	if m.picker.fuzzy {
		lines = append(lines, fit(m.picker.query.View(), w))
	}
	lines = append(lines, lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("─", w)))
	matches := m.pickerMatches()
	maxRows := max(3, min(12, m.height-8))
	for i, idx := range matches {
		if i >= maxRows {
			lines = append(lines, mutedStyle.Render(fmt.Sprintf("  … %d more", len(matches)-i)))
			break
		}
		row := "  " + textStyle.Render(m.picker.labels[idx])
		if i == m.picker.cursor {
			row = selStyle.Render(fit("▸ "+m.picker.labels[idx], w))
		}
		lines = append(lines, fit(row, w))
	}
	if len(matches) == 0 {
		lines = append(lines, mutedStyle.Render("  nothing matches"))
	}
	lines = append(lines, "", mutedStyle.Render(fit("  ↑↓ move   enter pick   esc close", w)))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

// ---------- confirm ----------

func (m model) askConfirm(text, hint string, onYes func(m model) (model, tea.Cmd)) (model, tea.Cmd) {
	m.confirm = confirmState{open: true, text: text, hint: hint, onYes: onYes}
	return m, nil
}

func (m model) viewConfirm() string {
	w := boxWidth(56, m.width)
	lines := []string{errorStyle.Render(fit("⚠ "+m.confirm.text, w))}
	if m.confirm.hint != "" {
		lines = append(lines, warnStyle.Render(fit("  "+m.confirm.hint, w)))
	}
	lines = append(lines, "", fit(keyStyle.Render("y")+mutedStyle.Render(" yes   ")+keyStyle.Render("n")+mutedStyle.Render(" no"), w))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

// ---------- huh form ----------

func (m model) updateForm(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.form == nil {
		return m, nil
	}
	f, cmd := m.form.Update(msg)
	if ff, ok := f.(*huh.Form); ok {
		m.form = ff
	}
	switch m.form.State {
	case huh.StateAborted:
		m.form = nil
		return m, nil
	case huh.StateCompleted:
		m.form = nil
		return m.formCompleted(cmd)
	}
	return m, cmd
}

func (m model) formCompleted(cmd tea.Cmd) (tea.Model, tea.Cmd) {
	switch m.formKind {
	case "worktree":
		if m.formVals == nil {
			return m, cmd
		}
		branch := strings.TrimSpace(m.formVals.branch)
		path := strings.TrimSpace(m.formVals.path)
		if branch == "" {
			return m, m.warn("branch name required")
		}
		repo := m.cfg.Repo
		return m, tea.Batch(cmd, func() tea.Msg {
			p, err := harness.AddWorktree(repo, branch, path)
			return wtAddedMsg{path: p, err: err}
		})
	}
	return m, cmd
}

func (m model) viewForm() string {
	w := boxWidth(60, m.width)
	title := titleStyle.Render(fit("New worktree", w))
	body := m.form.View()
	lines := []string{title, ""}
	for _, l := range strings.Split(body, "\n") {
		lines = append(lines, fit(l, w))
	}
	lines = append(lines, mutedStyle.Render(fit("  enter next/submit   esc cancel", w)))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

func alembicTheme() *huh.Theme {
	t := huh.ThemeBase()
	t.Focused.Title = t.Focused.Title.Foreground(gold).Bold(true)
	t.Focused.Description = t.Focused.Description.Foreground(muted)
	t.Focused.Base = t.Focused.Base.BorderForeground(gold)
	t.Focused.TextInput.Prompt = t.Focused.TextInput.Prompt.Foreground(gold)
	t.Focused.TextInput.Cursor = t.Focused.TextInput.Cursor.Foreground(gold)
	t.Focused.FocusedButton = t.Focused.FocusedButton.Background(gold).Foreground(darkNavy)
	t.Focused.ErrorMessage = t.Focused.ErrorMessage.Foreground(red)
	t.Blurred.Title = t.Blurred.Title.Foreground(muted)
	t.Blurred.Base = t.Blurred.Base.BorderForeground(navy)
	return t
}

// ---------- view ----------

func (m model) View() string {
	if m.width < 20 || m.height < 6 {
		return "alembic: terminal too small"
	}
	w, h := m.width, m.height
	var rows []string
	rows = append(rows, m.viewHeader(w), m.viewTabs(w))
	body := m.bodyHeight()
	var content string
	switch m.tab {
	case tabTasks:
		content = m.viewTasksTab(w, body)
	case tabWorktrees:
		content = m.viewWorktreesTab(w, body)
	case tabJev:
		content = m.viewJevTab(w, body)
	case tabAgents:
		content = m.viewAgentsTab(w, body)
	}
	rows = append(rows, content)
	if m.composer.open {
		rows = append(rows, m.viewComposer(w))
	}
	rows = append(rows, m.viewStatus(w))
	base := strings.Join(rows, "\n")
	lines := strings.Split(base, "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	for len(lines) < h {
		lines = append(lines, strings.Repeat(" ", w))
	}
	base = strings.Join(lines, "\n")

	switch {
	case m.form != nil:
		return overlay(base, m.viewForm(), w, h)
	case m.palette.open:
		return overlay(base, m.viewPalette(), w, h)
	case m.help:
		return overlay(base, m.viewHelp(), w, h)
	case m.confirm.open:
		return overlay(base, m.viewConfirm(), w, h)
	case m.picker.open:
		return overlay(base, m.viewPicker(), w, h)
	case m.jv.history.open:
		return overlay(base, m.viewHistory(), w, h)
	}
	return base
}

func (m model) viewHeader(w int) string {
	badge := badgeStyle.Render("ALEMBIC")
	label := headerDim.Render(" harness ")
	var fresh string
	switch {
	case m.feedMissing:
		fresh = errorStyle.Render("no feed")
	case m.lastRecordAt.IsZero():
		fresh = mutedStyle.Render("waiting")
	case m.now.Sub(m.lastRecordAt) > time.Minute:
		fresh = warnStyle.Render("stale " + relTime(m.now, m.lastRecordAt))
	default:
		fresh = successStyle.Render("live " + relTime(m.now, m.lastRecordAt) + " ago")
	}
	counts := mutedStyle.Render(fmt.Sprintf("%s · %s · %s",
		plural(len(m.snap.Workflows), "workflow"), plural(len(m.snap.Tasks), "task"), plural(len(m.snap.Agents), "agent")))
	jv := successStyle.Render("jev: live")
	if m.client.IsMock() {
		jv = warnStyle.Render("jev: MOCK")
	}
	sep := mutedStyle.Render("  ")
	left := badge + label + " " + fresh + sep + counts + sep + jv
	if m.demo != nil {
		left += sep + cyanStyle.Render("demo")
	}
	clock := headerDim.Render(" " + m.now.Format("15:04:05") + " ")
	return splitRow(left, clock, w)
}

func (m model) viewTabs(w int) string {
	var b strings.Builder
	for i, name := range tabNames {
		if tab(i) == m.tab {
			b.WriteString(tabActive.Render(name))
		} else {
			b.WriteString(tabInactive.Render(name))
		}
	}
	return fit(b.String(), w)
}

func (m model) viewStatus(w int) string {
	hints := m.hints()
	var right string
	if m.toast.text != "" {
		right = m.toast.style.Render(m.toast.text)
	}
	left := statusStyle.Render(" " + hints)
	return statusStyle.Render(splitRow(left, right, w))
}
