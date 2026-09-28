package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/wordwrap"

	"alembic/harness"
	"alembic/jev"
)

type jevFocus int

const (
	jevFocusPacks jevFocus = iota
	jevFocusState
	jevFocusResult
)

const packsPaneWidth = 30

// jevResult is what the Result column shows: a receipt (fresh or loaded from
// history) and where it lives on disk.
type jevResult struct {
	packID      string
	receipt     jev.Receipt
	path        string
	fromHistory bool
}

type jevHistory struct {
	open   bool
	packID string
	list   []jev.Receipt
	cursor int
	mark   int // -1 when nothing is marked for compare
}

type jevState struct {
	focus        jevFocus
	packCursor   int
	packOffset   int
	editing      bool
	text         textarea.Model
	file         textinput.Model
	taskID       string // explicit task override (T); "" follows the Tasks tab selection
	wtPath       string // explicit worktree override (W); "" follows the Worktrees tab selection
	state        []byte
	stateFor     string // jevStateKey the state bytes belong to
	stateErr     error
	running      bool
	runID        int
	result       *jevResult
	resultOffset int
	history      jevHistory
}

func newJevState() jevState {
	ta := textarea.New()
	ta.Prompt = ""
	ta.ShowLineNumbers = false
	ta.Placeholder = "type or paste the state here…"
	ta.CharLimit = 8000
	ta.SetWidth(30)
	ta.SetHeight(4)
	ti := textinput.New()
	ti.Prompt = "▸ "
	ti.PromptStyle = keyStyle
	ti.Placeholder = "path/to/state.json"
	ti.CharLimit = 400
	return jevState{text: ta, file: ti, history: jevHistory{mark: -1}}
}

func contextBG() context.Context { return context.Background() }

func expandHome(p string) string {
	if strings.HasPrefix(p, "~/") || p == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}

func reloadPacksCmd(dir string) tea.Cmd {
	return func() tea.Msg {
		packs, errs := jev.LoadDir(dir)
		return packsMsg{packs: packs, errs: errs}
	}
}

// ---------- state source ----------

func (m model) selectedPack() *jev.Pack {
	if len(m.packs) == 0 {
		return nil
	}
	return m.packs[clampInt(m.jv.packCursor, 0, len(m.packs)-1)]
}

func (m model) jevTaskID() string {
	if m.jv.taskID != "" {
		if _, ok := m.snap.Tasks[m.jv.taskID]; ok {
			return m.jv.taskID
		}
	}
	return m.tasks.selID
}

func (m model) jevWorktreePath() string {
	if m.jv.wtPath != "" {
		return m.jv.wtPath
	}
	if wt := m.selectedWorktree(); wt != nil {
		return wt.Path
	}
	if m.cfg.RepoOK {
		return m.cfg.Repo
	}
	return ""
}

// jevSource returns the selected pack's state source and the reference the
// state is built from (task id, worktree path, file path).
func (m model) jevSource() (src, ref string) {
	p := m.selectedPack()
	if p == nil {
		return "", ""
	}
	switch p.StateSource {
	case "task", "events":
		return p.StateSource, m.jevTaskID()
	case "staged_diff", "working_diff":
		return p.StateSource, m.jevWorktreePath()
	case "file":
		return "file", strings.TrimSpace(m.jv.file.Value())
	case "text":
		return "text", ""
	}
	return p.StateSource, ""
}

func (m model) jevStateKey() string {
	src, ref := m.jevSource()
	return src + ":" + ref
}

// jevFetcher returns a function that produces the state bytes for a source.
// Snapshot-backed sources are captured now so the Cmd never touches the model.
func (m model) jevFetcher(src, ref string) func() ([]byte, error) {
	switch src {
	case "task", "events":
		t := m.snap.Tasks[ref]
		if t == nil {
			return func() ([]byte, error) { return nil, nil }
		}
		state := m.taskState(t, src == "events")
		return func() ([]byte, error) { return state, nil }
	case "staged_diff", "working_diff":
		if ref == "" {
			return func() ([]byte, error) { return nil, nil }
		}
		return func() ([]byte, error) {
			var out string
			var err error
			if src == "staged_diff" {
				out, err = harness.StagedDiff(ref)
			} else {
				out, err = harness.WorkingDiff(ref)
			}
			return []byte(out), err
		}
	case "file":
		if ref == "" {
			return func() ([]byte, error) { return nil, nil }
		}
		return func() ([]byte, error) { return os.ReadFile(expandHome(ref)) }
	case "text":
		state := []byte(m.jv.text.Value())
		return func() ([]byte, error) { return state, nil }
	}
	return nil
}

// jevStateCmd loads the state for the selected pack in a Cmd.
func (m model) jevStateCmd() tea.Cmd {
	src, ref := m.jevSource()
	fetch := m.jevFetcher(src, ref)
	if fetch == nil {
		return nil
	}
	key := src + ":" + ref
	return func() tea.Msg {
		state, err := fetch()
		return jevStateMsg{key: key, state: state, err: err}
	}
}

func (m model) applyJevState(msg jevStateMsg) (tea.Model, tea.Cmd) {
	if msg.key != m.jevStateKey() {
		return m, nil // stale: the pack or its source changed meanwhile
	}
	m.jv.state = msg.state
	m.jv.stateFor = msg.key
	m.jv.stateErr = msg.err
	return m, nil
}

func (m model) jevStateLoaded() bool { return m.jv.stateFor == m.jevStateKey() }

// ---------- running ----------

// startJevRun evaluates the selected pack against its state, loading the
// state inside the Cmd when it has not arrived yet.
func (m model) startJevRun() (model, tea.Cmd) {
	p := m.selectedPack()
	if p == nil {
		return m, m.warn("no pack selected")
	}
	if m.jv.running {
		return m, m.warn("a run is already in progress")
	}
	src, ref := m.jevSource()
	loaded := m.jevStateLoaded()
	if loaded && len(bytes.TrimSpace(m.jv.state)) == 0 {
		return m, m.warn("state is empty · nothing to evaluate")
	}
	if (src == "task" || src == "events") && ref == "" {
		return m, m.warn("no task selected · T picks one")
	}
	taskID := ""
	if src == "task" || src == "events" {
		taskID = ref
	}
	m.runSeq++
	m.jv.running = true
	m.jv.runID = m.runSeq
	m.jv.focus = jevFocusResult
	m.jv.resultOffset = 0
	var cmd tea.Cmd
	if loaded {
		cmd = m.askJevCmd(m.runSeq, taskID, p, ref, m.jv.state)
	} else {
		cmd = m.fetchAndAskCmd(m.runSeq, taskID, p, src, ref)
	}
	return m, tea.Batch(cmd, m.kickAnim())
}

func (m model) fetchAndAskCmd(runID int, taskID string, p *jev.Pack, src, ref string) tea.Cmd {
	client := m.client
	fetch := m.jevFetcher(src, ref)
	return func() tea.Msg {
		var state []byte
		var err error
		if fetch != nil {
			state, err = fetch()
		}
		if err != nil {
			return jevDoneMsg{runID: runID, taskID: taskID, pack: p, stateRef: ref, err: err}
		}
		if len(bytes.TrimSpace(state)) == 0 {
			return jevDoneMsg{runID: runID, taskID: taskID, pack: p, stateRef: ref, err: errors.New("state is empty · nothing to evaluate")}
		}
		var st any
		if json.Unmarshal(state, &st) != nil {
			st = string(state)
		}
		resp, err := client.Ask(contextBG(), st, p.Questions)
		return jevDoneMsg{runID: runID, taskID: taskID, pack: p, stateRef: ref, state: state, resp: resp, err: err}
	}
}

func (m model) applyJevDone(msg jevDoneMsg) (tea.Model, tea.Cmd) {
	isTab := m.jv.running && msg.runID == m.jv.runID
	tj := m.taskJev[msg.taskID]
	isTask := !isTab && msg.taskID != "" && tj != nil && tj.running
	if !isTab && !isTask {
		return m, nil // a run that was superseded
	}
	if isTab {
		m.jv.running = false
	}
	if isTask {
		tj.running = false
	}
	if msg.err != nil {
		return m, m.fail("jev: " + msg.err.Error())
	}
	if msg.pack == nil || msg.resp == nil {
		return m, m.fail("jev: empty response")
	}
	verdict := msg.pack.Evaluate(msg.resp)
	rc := jev.NewReceipt(msg.pack, msg.stateRef, msg.state, msg.resp, verdict)
	path, err := jev.SaveReceipt(m.cfg.Receipts, rc)
	var cmds []tea.Cmd
	if err != nil {
		cmds = append(cmds, m.fail("receipt not saved: "+err.Error()))
	}
	if msg.taskID != "" && (msg.pack.StateSource == "task" || msg.pack.StateSource == "events") {
		if tj == nil {
			tj = &taskJev{}
			m.taskJev[msg.taskID] = tj
		}
		r := rc
		tj.receipt, tj.path, tj.at = &r, path, rc.At
	}
	if isTab {
		m.jv.result = &jevResult{packID: msg.pack.ID, receipt: rc, path: path}
		m.jv.resultOffset = 0
		if key := msg.pack.StateSource + ":" + msg.stateRef; key == m.jevStateKey() {
			m.jv.state, m.jv.stateFor, m.jv.stateErr = msg.state, key, nil
		}
	}
	label := strings.ToUpper(string(rc.Decision))
	if msg.pack.Gate == nil {
		label = "done"
	}
	st := toastOK
	switch rc.Decision {
	case jev.DecisionEscalate:
		st = toastWarn
	case jev.DecisionRefuse:
		st = toastErr
	}
	if msg.pack.Gate == nil {
		st = toastOK
	}
	cmds = append(cmds, m.setToast(fmt.Sprintf("%s: %s · %d ms", msg.pack.Name, label, rc.LatencyMs), st))
	return m, tea.Batch(cmds...)
}

// ---------- keys ----------

func (m model) updateJev(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	src, _ := m.jevSource()
	switch k {
	case "tab":
		m.jv.focus = (m.jv.focus + 1) % 3
		return m, nil
	case "shift+tab":
		m.jv.focus = (m.jv.focus + 2) % 3
		return m, nil
	case "enter":
		return m.startJevRun()
	case "r":
		return m, reloadPacksCmd(m.cfg.Packs)
	case "T":
		return m.pickTaskForJev()
	case "W":
		return m.pickWorktreeForJev()
	case "h":
		return m.openHistory()
	case "y":
		if m.jv.result == nil {
			return m, m.warn("no receipt yet · enter runs the pack")
		}
		m.copier(m.jv.result.path)
		return m, m.ok("copied " + shortPath(m.jv.result.path))
	case "s":
		return m.sendReceipt("")
	case "S":
		if m.jv.result == nil {
			return m, m.warn("no receipt yet · enter runs the pack")
		}
		rc := m.jv.result.receipt
		return m.openComposer(composerJev, "", rc.StateRef, &rc)
	case "e", "i":
		switch src {
		case "text":
			m.jv.editing, m.jv.focus = true, jevFocusState
			return m, m.jv.text.Focus()
		case "file":
			m.jv.editing, m.jv.focus = true, jevFocusState
			return m, m.jv.file.Focus()
		}
		return m, m.warn("this pack's state comes from " + strings.ReplaceAll(src, "_", " ") + " · nothing to edit")
	}
	switch m.jv.focus {
	case jevFocusPacks:
		n := len(m.packs)
		switch k {
		case "j", "down":
			m.jv.packCursor = clampInt(m.jv.packCursor+1, 0, max(0, n-1))
		case "k", "up":
			m.jv.packCursor = clampInt(m.jv.packCursor-1, 0, max(0, n-1))
		case "g", "home":
			m.jv.packCursor = 0
		case "G", "end":
			m.jv.packCursor = max(0, n-1)
		default:
			return m, nil
		}
		m.jv.packOffset = packOffsetFor(m.jv.packCursor, m.jv.packOffset, m.bodyHeight()-2)
		m.jv.resultOffset = 0
		return m, m.jevStateCmd()
	case jevFocusState:
		if src == "text" || src == "file" {
			if k == "j" || k == "k" || k == "down" || k == "up" {
				return m, m.warn("e edits the state")
			}
		}
		return m, nil
	case jevFocusResult:
		iw := m.jevResultWidth() - 2
		lines := m.jevResultLines(iw)
		maxOff := max(0, len(lines)-(m.bodyHeight()-2))
		switch k {
		case "j", "down":
			m.jv.resultOffset = clampInt(m.jv.resultOffset+1, 0, maxOff)
		case "k", "up":
			m.jv.resultOffset = clampInt(m.jv.resultOffset-1, 0, maxOff)
		case "ctrl+d", "pgdown":
			m.jv.resultOffset = clampInt(m.jv.resultOffset+max(1, (m.bodyHeight()-2)/2), 0, maxOff)
		case "ctrl+u", "pgup":
			m.jv.resultOffset = clampInt(m.jv.resultOffset-max(1, (m.bodyHeight()-2)/2), 0, maxOff)
		case "g", "home":
			m.jv.resultOffset = 0
		case "G", "end":
			m.jv.resultOffset = maxOff
		}
	}
	return m, nil
}

func packOffsetFor(cursor, offset, inner int) int {
	if inner <= 0 {
		return 0
	}
	top := cursor * 2
	if top < offset {
		offset = top
	}
	if top+2 > offset+inner {
		offset = top + 2 - inner
	}
	return max(0, offset)
}

// updateJevEditor routes keys to the inline textarea / path input.
func (m model) updateJevEditor(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	src, _ := m.jevSource()
	k := msg.String()
	if k == "ctrl+c" {
		return m, tea.Quit
	}
	if k == "esc" {
		m.jv.editing = false
		m.jv.text.Blur()
		m.jv.file.Blur()
		return m, m.jevStateCmd()
	}
	var cmd tea.Cmd
	switch src {
	case "text":
		m.jv.text, cmd = m.jv.text.Update(msg)
		m.jv.state = []byte(m.jv.text.Value())
		m.jv.stateFor, m.jv.stateErr = "text:", nil
		return m, cmd
	case "file":
		if k == "enter" {
			m.jv.editing = false
			m.jv.file.Blur()
			return m, m.jevStateCmd()
		}
		m.jv.file, cmd = m.jv.file.Update(msg)
		return m, cmd
	}
	m.jv.editing = false
	return m, nil
}

func (m model) pickTaskForJev() (model, tea.Cmd) {
	if len(m.snap.Tasks) == 0 {
		return m, m.warn("no tasks in feed")
	}
	ids := make([]string, 0, len(m.snap.Tasks))
	for id := range m.snap.Tasks {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	labels := make([]string, len(ids))
	for i, id := range ids {
		t := m.snap.Tasks[id]
		labels[i] = fmt.Sprintf("%-8s %-8s %s", id, t.State, t.Title)
	}
	return m.openPicker("State: which task?", labels, ids, true, func(m model, idx int) (model, tea.Cmd) {
		m.jv.taskID = ids[idx]
		m.jv.resultOffset = 0
		return m, tea.Batch(m.jevStateCmd(), m.ok("state ← "+ids[idx]))
	})
}

func (m model) pickWorktreeForJev() (model, tea.Cmd) {
	if !m.cfg.RepoOK {
		return m, m.warn("not a git repo: " + shortPath(m.cfg.Repo))
	}
	if len(m.wt.list) == 0 {
		return m, tea.Batch(m.warn("no worktrees listed yet · refreshing"), m.refreshWorktreesCmd())
	}
	paths := make([]string, len(m.wt.list))
	labels := make([]string, len(m.wt.list))
	for i, wt := range m.wt.list {
		branch := wt.Branch
		if wt.Detached {
			branch = "detached"
		}
		labels[i] = fmt.Sprintf("%-28s %s", fit(shortPath(wt.Path), 28), branch)
		paths[i] = wt.Path
	}
	return m.openPicker("State: which worktree?", labels, paths, true, func(m model, idx int) (model, tea.Cmd) {
		m.jv.wtPath = paths[idx]
		m.jv.resultOffset = 0
		return m, tea.Batch(m.jevStateCmd(), m.ok("state ← "+shortPath(paths[idx])))
	})
}

func (m model) sendReceipt(note string) (model, tea.Cmd) {
	if m.jv.result == nil {
		return m, m.warn("no receipt yet · enter runs the pack")
	}
	rc := m.jv.result.receipt
	if _, err := m.outbox.Send(harness.Command{Type: "jev.receipt", TaskID: rc.StateRef, Text: note, Data: rc}); err != nil {
		return m, m.fail("outbox: " + err.Error())
	}
	return m, m.ok("receipt sent → harness")
}

// ---------- history ----------

func (m model) openHistory() (model, tea.Cmd) {
	if m.jv.history.open {
		m.jv.history.open = false
		return m, nil
	}
	p := m.selectedPack()
	if p == nil {
		return m, m.warn("no pack selected")
	}
	list := jev.LoadReceipts(m.cfg.Receipts, p.ID)
	if len(list) == 0 {
		return m, m.warn("no receipts for " + p.Name + " yet · enter runs one")
	}
	m.jv.history = jevHistory{open: true, packID: p.ID, list: list, mark: -1}
	return m, nil
}

func (m model) updateHistory(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	h := &m.jv.history
	n := len(h.list)
	switch msg.String() {
	case "esc", "h", "q":
		h.open = false
	case "ctrl+c":
		return m, tea.Quit
	case "j", "down":
		h.cursor = clampInt(h.cursor+1, 0, max(0, n-1))
	case "k", "up":
		h.cursor = clampInt(h.cursor-1, 0, max(0, n-1))
	case "g", "home":
		h.cursor = 0
	case "G", "end":
		h.cursor = max(0, n-1)
	case "c":
		if h.mark == h.cursor {
			h.mark = -1
		} else {
			h.mark = h.cursor
		}
	case "enter":
		if n == 0 {
			h.open = false
			return m, nil
		}
		rc := h.list[clampInt(h.cursor, 0, n-1)]
		m.jv.result = &jevResult{packID: rc.PackID, receipt: rc, path: filepath.Join(m.cfg.Receipts, rc.ID+".json"), fromHistory: true}
		m.jv.resultOffset = 0
		m.jv.focus = jevFocusResult
		h.open = false
		return m, m.ok("loaded receipt " + rc.At.Local().Format("15:04:05"))
	}
	return m, nil
}

func headlineProb(a jev.Answer) float64 {
	if a.Type == jev.Score && a.Score != nil {
		return *a.Score
	}
	return a.Probability()
}

func (m model) viewHistory() string {
	h := m.jv.history
	w := boxWidth(72, m.width)
	name := h.packID
	if p := m.findPack(h.packID); p != nil {
		name = p.Name
	}
	lines := []string{titleStyle.Render(fit(fmt.Sprintf("History · %s · %s", name, plural(len(h.list), "receipt")), w))}
	sep := lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("─", w))
	lines = append(lines, sep)

	var compare []string
	if h.mark >= 0 && h.mark < len(h.list) && h.mark != h.cursor && h.cursor < len(h.list) {
		a, b := h.list[h.mark], h.list[h.cursor]
		compare = append(compare, sep, fit(mutedStyle.Render("compare  ")+keyStyle.Render("◆ "+a.At.Local().Format("15:04:05"))+mutedStyle.Render("  →  ")+keyStyle.Render("▸ "+b.At.Local().Format("15:04:05"))+
			mutedStyle.Render("   "+strings.ToUpper(string(a.Decision))+" → ")+decisionStyle(b.Decision).Render(strings.ToUpper(string(b.Decision))), w))
		ids := map[string]bool{}
		for id := range a.Answers {
			ids[id] = true
		}
		for id := range b.Answers {
			ids[id] = true
		}
		order := make([]string, 0, len(ids))
		for id := range ids {
			order = append(order, id)
		}
		sort.Strings(order)
		if p := m.findPack(h.packID); p != nil {
			order = orderIDs(p.QuestionIDs(), order)
		}
		for _, id := range order {
			pa, pb := headlineProb(a.Answers[id]), headlineProb(b.Answers[id])
			d := pb - pa
			ds := mutedStyle.Render(fmt.Sprintf("%+.2f", d))
			switch {
			case d > 0.005:
				ds = successStyle.Render(fmt.Sprintf("%+.2f", d))
			case d < -0.005:
				ds = errorStyle.Render(fmt.Sprintf("%+.2f", d))
			}
			extra := ""
			if ab, bb := a.Answers[id], b.Answers[id]; ab.Type == jev.Choice || bb.Type == jev.Choice {
				extra = mutedStyle.Render(fmt.Sprintf("  %s → %s", ab.Choice, bb.Choice))
			} else if ab.Type == jev.Score {
				extra = mutedStyle.Render(fmt.Sprintf("  %s → %s", ab.Level(), bb.Level()))
			}
			lines := fmt.Sprintf("  %-18s %s → %s  ", fit(id, 18), textStyle.Render(fmt.Sprintf("%.2f", pa)), textStyle.Render(fmt.Sprintf("%.2f", pb)))
			compare = append(compare, fit(lines+ds+extra, w))
		}
	}
	avail := max(3, m.height-2-4)
	rows := max(1, avail-len(compare))
	start := 0
	if h.cursor >= rows {
		start = h.cursor - rows + 1
	}
	for i := start; i < len(h.list) && i < start+rows; i++ {
		rc := h.list[i]
		mark := " "
		if i == h.mark {
			mark = "◆"
		}
		mock := ""
		if rc.Mock {
			mock = "MOCK"
		}
		plain := fmt.Sprintf("%s %s  %-9s %5d ms  %-4s  %s", mark, rc.At.Local().Format("2006-01-02 15:04:05"), strings.ToUpper(string(rc.Decision)), rc.LatencyMs, mock, rc.StateRef)
		if i == h.cursor {
			lines = append(lines, selStyle.Render(fit("▸"+plain, w)))
			continue
		}
		row := " " + keyStyle.Render(mark) + " " + textStyle.Render(rc.At.Local().Format("2006-01-02 15:04:05")) + "  " +
			decisionStyle(rc.Decision).Render(fmt.Sprintf("%-9s", strings.ToUpper(string(rc.Decision)))) + " " +
			mutedStyle.Render(fmt.Sprintf("%5d ms", rc.LatencyMs)) + "  " + warnStyle.Render(fmt.Sprintf("%-4s", mock)) + "  " + mutedStyle.Render(rc.StateRef)
		lines = append(lines, fit(row, w))
	}
	if start+rows < len(h.list) {
		lines[len(lines)-1] = fit(mutedStyle.Render(fmt.Sprintf("  … %d more", len(h.list)-start-rows)), w)
	}
	lines = append(lines, compare...)
	lines = append(lines, "", mutedStyle.Render(fit("  j/k move   enter load   c mark → compare   esc close", w)))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

// orderIDs puts ids in pack order, appending any ids the pack does not know.
func orderIDs(pref, ids []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, id := range pref {
		for _, x := range ids {
			if x == id && !seen[id] {
				out = append(out, id)
				seen[id] = true
			}
		}
	}
	for _, id := range ids {
		if !seen[id] {
			out = append(out, id)
		}
	}
	return out
}

// ---------- view ----------

func (m model) jevStateWidth() int {
	rest := m.width - packsPaneWidth
	return clampInt(rest*35/100, 16, max(16, rest-12))
}

func (m model) jevResultWidth() int {
	return max(0, m.width-packsPaneWidth-m.jevStateWidth())
}

func (m model) viewJevTab(w, h int) string {
	pw := packsPaneWidth
	sw := m.jevStateWidth()
	rw := w - pw - sw
	packs := m.viewPacksPane(pw, h)
	if sw+rw < 12 {
		return packs
	}
	state := m.viewStatePane(sw, h)
	if rw < 6 {
		return lipgloss.JoinHorizontal(lipgloss.Top, packs, state)
	}
	result := m.viewResultPane(rw, h)
	return lipgloss.JoinHorizontal(lipgloss.Top, packs, state, result)
}

func (m model) viewPacksPane(w, h int) string {
	focused := m.jv.focus == jevFocusPacks
	iw := w - 2
	title := fmt.Sprintf("Packs · %d", len(m.packs))
	var all []string
	if len(m.packs) == 0 && len(m.packErrs) == 0 {
		all = append(all, "", mutedStyle.Render("  no packs found in"), mutedStyle.Render("  "+shortPath(m.cfg.Packs)), "", mutedStyle.Render("  drop *.json packs there"), mutedStyle.Render("  and press r"))
		return pane(title, all, w, h, focused)
	}
	for i, p := range m.packs {
		if i == m.jv.packCursor {
			all = append(all, selStyle.Render(fit("▸ "+p.Name, iw)), selStyle.Render(fit("  "+p.Description, iw)))
			continue
		}
		all = append(all, fit("  "+textStyle.Render(p.Name), iw), fit("  "+mutedStyle.Render(p.Description), iw))
	}
	for _, pe := range m.packErrs {
		all = append(all, fit("  "+errorStyle.Render("✖ "+pe.name), iw), fit("  "+lipgloss.NewStyle().Foreground(red).Render(pe.err.Error()), iw))
	}
	inner := h - 2
	off := clampInt(packOffsetFor(m.jv.packCursor, m.jv.packOffset, inner), 0, max(0, len(all)-inner))
	end := min(len(all), off+inner)
	lines := all[off:end]
	return pane(title, lines, w, h, focused)
}

func (m model) viewStatePane(w, h int) string {
	focused := m.jv.focus == jevFocusState
	iw := w - 2
	p := m.selectedPack()
	if p == nil {
		return pane("State", []string{"", mutedStyle.Render("  select a pack")}, w, h, focused)
	}
	src, ref := m.jevSource()
	title := "State · " + strings.ReplaceAll(src, "_", " ")
	var lines []string
	switch src {
	case "task", "events":
		if t := m.snap.Tasks[ref]; t != nil {
			lines = append(lines, fit(mutedStyle.Render("task ")+keyStyle.Render(t.ID), iw), fit(textStyle.Render(t.Title), iw))
		} else {
			lines = append(lines, warnStyle.Render(fit("no task selected", iw)))
		}
		lines = append(lines, mutedStyle.Render(fit("T picks another task", iw)))
	case "staged_diff", "working_diff":
		if ref == "" {
			lines = append(lines, warnStyle.Render(fit("no worktree", iw)))
		} else {
			lines = append(lines, fit(mutedStyle.Render("worktree"), iw), fit(keyStyle.Render(shortPath(ref)), iw))
		}
		lines = append(lines, mutedStyle.Render(fit("W picks another worktree", iw)))
	case "text":
		for _, l := range strings.Split(m.jv.text.View(), "\n") {
			lines = append(lines, fit(l, iw))
		}
		if m.jv.editing {
			lines = append(lines, mutedStyle.Render(fit("esc done editing", iw)))
		} else {
			lines = append(lines, mutedStyle.Render(fit("e edits the text", iw)))
		}
	case "file":
		lines = append(lines, fit(m.jv.file.View(), iw))
		if m.jv.editing {
			lines = append(lines, mutedStyle.Render(fit("enter loads · esc", iw)))
		} else {
			lines = append(lines, mutedStyle.Render(fit("e edits the path", iw)))
		}
	default:
		lines = append(lines, errorStyle.Render(fit("unknown source "+src, iw)))
	}
	lines = append(lines, "", fit(sectionStyle.Render("─ preview ")+sectionStyle.Render(strings.Repeat("─", max(0, iw-10))), iw))
	switch {
	case m.jv.stateErr != nil && m.jevStateLoaded():
		for _, l := range strings.Split(wordwrap.String(m.jv.stateErr.Error(), max(8, iw)), "\n") {
			lines = append(lines, errorStyle.Render(fit(l, iw)))
		}
	case !m.jevStateLoaded() && src != "text":
		lines = append(lines, mutedStyle.Render(fit("loading…", iw)))
	case len(bytes.TrimSpace(m.jv.state)) == 0:
		lines = append(lines, warnStyle.Render(fit("⚠ state is empty", iw)), mutedStyle.Render(fit("nothing to evaluate", iw)))
	default:
		inner := h - 2
		room := max(1, inner-len(lines)-1)
		preview := strings.Split(strings.TrimRight(string(m.jv.state), "\n"), "\n")
		for i, l := range preview {
			if i >= room {
				break
			}
			lines = append(lines, fit(mutedStyle.Render(l), iw))
		}
		lines = append(lines, fit(textStyle.Render(fmt.Sprintf("%s bytes", commas(len(m.jv.state))))+mutedStyle.Render(fmt.Sprintf(" · %d lines", len(preview))), iw))
	}
	return pane(title, lines, w, h, focused)
}

func commas(n int) string {
	s := strconv.Itoa(n)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
	}
	for i := pre; i < len(s); i += 3 {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(s[i : i+3])
	}
	return b.String()
}

func (m model) viewResultPane(w, h int) string {
	focused := m.jv.focus == jevFocusResult
	iw := w - 2
	inner := h - 2
	title := "Result"
	if p := m.selectedPack(); p != nil {
		title += " · " + p.ID
	}
	if m.jv.running {
		title += " " + spinnerAt(m.frame) + " running…"
	} else if r := m.jv.result; r != nil && r.fromHistory && m.selectedPack() != nil && r.packID == m.selectedPack().ID {
		title += " · " + r.receipt.At.Local().Format("15:04:05") + " (history)"
	}
	lines := m.jevResultLines(iw)
	off := clampInt(m.jv.resultOffset, 0, max(0, len(lines)-inner))
	end := min(len(lines), off+inner)
	shown := lines[off:end]
	if end < len(lines) && inner > 0 {
		shown = append([]string{}, shown...)
		shown[len(shown)-1] = fit(mutedStyle.Render(fmt.Sprintf("↓ %d more · j scrolls", len(lines)-end)), iw)
	}
	return pane(title, shown, w, h, focused)
}

// jevResultLines renders the Result column content: the pack's questions
// before a run, the answer visualizations and verdict after.
func (m model) jevResultLines(iw int) []string {
	p := m.selectedPack()
	if p == nil {
		return []string{"", mutedStyle.Render("  no pack selected")}
	}
	r := m.jv.result
	if r == nil || r.packID != p.ID {
		return m.jevQuestionLines(p, iw)
	}
	return m.jevReceiptLines(p, r, iw)
}

func (m model) jevQuestionLines(p *jev.Pack, iw int) []string {
	var lines []string
	lines = append(lines, fit(titleStyle.Render(p.Name), iw))
	if p.Description != "" {
		lines = append(lines, fit(mutedStyle.Render(p.Description), iw))
	}
	lines = append(lines, "")
	for _, id := range p.QuestionIDs() {
		q, ok := p.Questions[id]
		if !ok {
			continue
		}
		lines = append(lines, fit(keyStyle.Render(id)+" "+qtypeBadge(q.Type), iw))
		lines = append(lines, fit("  "+mutedStyle.Render(q.Instructions), iw))
		switch q.Type {
		case jev.Choice:
			lines = append(lines, fit("  "+dimStyle.Render(strings.Join(q.Options(), " · ")), iw))
		case jev.Score:
			lines = append(lines, fit("  "+dimStyle.Render(strings.Join(q.Levels(), " → ")), iw))
		}
	}
	lines = append(lines, "")
	if g := p.Gate; g != nil {
		lines = append(lines, fit(mutedStyle.Render("gate ")+textStyle.Render(g.Action)+mutedStyle.Render(fmt.Sprintf("  p≥%.2f · conf≥%.2f · refuse<%.2f", g.MinProbability, g.AutoConfidence, g.RefuseBelow)), iw))
	} else {
		lines = append(lines, fit(mutedStyle.Render("no gate · informational"), iw))
	}
	if m.jv.running {
		lines = append(lines, fit(keyStyle.Render(spinnerAt(m.frame))+textStyle.Render(" asking Jev…"), iw))
	} else {
		lines = append(lines, fit(keyStyle.Render("enter")+mutedStyle.Render(" runs the pack"), iw))
	}
	return lines
}

func (m model) jevReceiptLines(p *jev.Pack, r *jevResult, iw int) []string {
	rc := r.receipt
	var lines []string
	ids := make([]string, 0, len(rc.Answers))
	for id := range rc.Answers {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range orderIDs(p.QuestionIDs(), ids) {
		a, ok := rc.Answers[id]
		if !ok {
			continue
		}
		lines = append(lines, renderAnswer(id, a, iw)...)
		lines = append(lines, "")
	}
	if rc.Mock {
		lines = append(lines, warnStyle.Render(fit("MOCK — deterministic, not a model", iw)))
	}
	lines = append(lines, renderVerdict(p, rc, iw)...)
	lines = append(lines, "")
	lines = append(lines, fit(mutedStyle.Render(fmt.Sprintf("%s · %d ms · %d in / %d out tokens", rc.Model, rc.LatencyMs, rc.Usage.InputTokens, rc.Usage.OutputTokens)), iw))
	lines = append(lines, fit(mutedStyle.Render("receipt ")+textStyle.Render(shortPath(r.path)), iw))
	lines = append(lines, fit(mutedStyle.Render("y copy path · s send to harness · h history"), iw))
	return lines
}

func renderVerdict(p *jev.Pack, rc jev.Receipt, iw int) []string {
	if p.Gate == nil {
		return []string{
			fit(mutedStyle.Bold(true).Render("▌INFO")+"  "+textStyle.Render("informational"), iw),
			fit(mutedStyle.Render("no gate defined for this pack"), iw),
		}
	}
	g := p.Gate
	st := decisionStyle(rc.Decision)
	lines := []string{
		fit(st.Render("▌"+strings.ToUpper(string(rc.Decision)))+"  "+textStyle.Render(g.Action), iw),
		fit(st.Render("▌")+mutedStyle.Render(rc.Reason), iw),
		fit(st.Render("▌")+mutedStyle.Render(fmt.Sprintf("p≥%.2f · conf≥%.2f · refuse<%.2f", g.MinProbability, g.AutoConfidence, g.RefuseBelow)), iw),
	}
	return lines
}

func confStyle(c float64) lipgloss.Style {
	return lipgloss.NewStyle().Foreground(confColor(c)).Bold(true)
}

// renderAnswer draws one answer as a small visualization.
func renderAnswer(id string, a jev.Answer, iw int) []string {
	conf := a.Conf()
	c := confColor(conf)
	switch a.Type {
	case jev.Noul:
		p := a.Probability()
		half := clampInt((iw-12)/2, 3, 14)
		return []string{
			fit(keyStyle.Render(id)+" "+qtypeBadge(a.Type), iw),
			fit(mutedStyle.Render("no ◀ ")+bipolar(p, half, c)+mutedStyle.Render(" ▶ yes"), iw),
			fit(confStyle(conf).Render(fmt.Sprintf("p=%.2f", p))+mutedStyle.Render("  conf ")+confStyle(conf).Render(fmt.Sprintf("%.2f", conf)), iw),
		}
	case jev.Choice:
		opts := make([]string, 0, len(a.Probabilities))
		for o := range a.Probabilities {
			opts = append(opts, o)
		}
		sort.Slice(opts, func(i, j int) bool {
			pi, pj := a.Probabilities[opts[i]], a.Probabilities[opts[j]]
			if pi != pj {
				return pi > pj
			}
			return opts[i] < opts[j]
		})
		nameW := 4
		for _, o := range opts {
			nameW = max(nameW, lipgloss.Width(o))
		}
		nameW = min(nameW, max(4, iw/3))
		bw := clampInt(iw-nameW-9, 4, 20)
		lines := []string{fit(keyStyle.Render(id)+" "+qtypeBadge(a.Type)+"  "+keyStyle.Render("✓ "+a.Choice), iw)}
		for _, o := range opts {
			pr := a.Probabilities[o]
			name := fit(o, nameW)
			if o == a.Choice {
				lines = append(lines, fit(keyStyle.Render(name)+" "+gauge(pr, bw, gold)+keyStyle.Render(fmt.Sprintf("  %.2f", pr)), iw))
			} else {
				lines = append(lines, fit(mutedStyle.Render(name)+" "+gauge(pr, bw, muted)+mutedStyle.Render(fmt.Sprintf("  %.2f", pr)), iw))
			}
		}
		lines = append(lines, fit(mutedStyle.Render("conf ")+confStyle(conf).Render(fmt.Sprintf("%.2f", conf)), iw))
		return lines
	case jev.Score:
		keys := make([]int, 0, len(a.Legend))
		for k := range a.Legend {
			if i, err := strconv.Atoi(k); err == nil {
				keys = append(keys, i)
			}
		}
		sort.Ints(keys)
		score := 0.0
		if a.Score != nil {
			score = *a.Score
		}
		nearest := int(score + 0.5)
		nameW := 4
		for _, k := range keys {
			nameW = max(nameW, lipgloss.Width(a.Legend[strconv.Itoa(k)]))
		}
		nameW = min(nameW, max(4, iw/3))
		bw := clampInt(iw-nameW-11, 4, 20)
		lines := []string{fit(keyStyle.Render(id)+" "+qtypeBadge(a.Type)+"  "+keyStyle.Render(a.Level()), iw)}
		for _, k := range keys {
			ks := strconv.Itoa(k)
			pr := a.Probabilities[ks]
			name := fit(a.Legend[ks], nameW)
			if k == nearest {
				lines = append(lines, fit(keyStyle.Render(name)+" "+gauge(pr, bw, gold)+keyStyle.Render(fmt.Sprintf("  %.2f ◀", pr)), iw))
			} else {
				lines = append(lines, fit(mutedStyle.Render(name)+" "+gauge(pr, bw, muted)+mutedStyle.Render(fmt.Sprintf("  %.2f", pr)), iw))
			}
		}
		lines = append(lines, fit(textStyle.Render(fmt.Sprintf("score %.2f", score))+mutedStyle.Render(" · conf ")+confStyle(conf).Render(fmt.Sprintf("%.2f", conf)), iw))
		return lines
	}
	return []string{fit(keyStyle.Render(id)+" "+mutedStyle.Render("unknown answer type "+string(a.Type)), iw)}
}

// bipolar draws "░░░░████" centred: mass to the right for P(yes) > 0.5,
// to the left for P(no) > 0.5.
func bipolar(p float64, half int, c lipgloss.Color) string {
	p = clamp(p, 0, 1)
	fillL, fillR := 0, 0
	if p >= 0.5 {
		fillR = int((p-0.5)*2*float64(half) + 0.5)
	} else {
		fillL = int((0.5-p)*2*float64(half) + 0.5)
	}
	empty := lipgloss.NewStyle().Foreground(darkNavy)
	full := lipgloss.NewStyle().Foreground(c)
	return empty.Render(strings.Repeat("░", half-fillL)) + full.Render(strings.Repeat("█", fillL)) +
		full.Render(strings.Repeat("█", fillR)) + empty.Render(strings.Repeat("░", half-fillR))
}
