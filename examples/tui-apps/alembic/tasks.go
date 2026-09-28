package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/sahilm/fuzzy"

	"alembic/harness"
	"alembic/jev"
)

type tasksFocus int

const (
	focusList tasksFocus = iota
	focusDetail
)

type rowKind int

const (
	rowHeader rowKind = iota
	rowTask
)

type taskRow struct {
	kind rowKind
	wf   *harness.Workflow
	task *harness.Task
	line int // first screen line of this row in the list
}

type tasksState struct {
	focus      tasksFocus
	selID      string
	rows       []taskRow
	offset     int
	wfFilter   string
	hideDone   bool
	filter     textinput.Model
	filtering  bool
	elemCursor int
	evOffset   int
}

func newTasksState() tasksState {
	ti := textinput.New()
	ti.Prompt = "/"
	ti.PromptStyle = keyStyle
	ti.CharLimit = 80
	return tasksState{filter: ti}
}

func fuzzyFind(q string, data []string) fuzzy.Matches { return fuzzy.Find(q, data) }

// sortedWorkflows returns workflows by name.
func (m model) sortedWorkflows() []*harness.Workflow {
	out := make([]*harness.Workflow, 0, len(m.snap.Workflows))
	for _, w := range m.snap.Workflows {
		out = append(out, w)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func sortTasks(ts []*harness.Task) {
	sort.SliceStable(ts, func(i, j int) bool {
		a, b := ts[i], ts[j]
		if a.Priority != b.Priority {
			return a.Priority > b.Priority
		}
		if sa, sb := stateOrder(a.State), stateOrder(b.State); sa != sb {
			return sa < sb
		}
		if !a.Updated.Equal(b.Updated) {
			return a.Updated.After(b.Updated)
		}
		return a.ID < b.ID
	})
}

// rebuildRows recomputes the grouped, sorted, filtered list and keeps the
// selection on the same task id when it is still visible.
func (m *model) rebuildRows() {
	var visible map[string]bool
	if q := strings.TrimSpace(m.tasks.filter.Value()); q != "" {
		ids := make([]string, 0, len(m.snap.Tasks))
		hay := make([]string, 0, len(m.snap.Tasks))
		for id, t := range m.snap.Tasks {
			ids = append(ids, id)
			hay = append(hay, t.ID+" "+t.Title+" "+t.StatusLine+" "+m.agentName(t.Agent))
		}
		visible = map[string]bool{}
		for _, r := range fuzzyFind(q, hay) {
			visible[ids[r.Index]] = true
		}
	}
	m.tasks.rows = m.tasks.rows[:0]
	smart := m.triage.sort == sortSmart
	type group struct {
		wf   *harness.Workflow
		ts   []*harness.Task
		best int
	}
	var groups []group
	for _, wf := range m.sortedWorkflows() {
		if m.tasks.wfFilter != "" && wf.ID != m.tasks.wfFilter {
			continue
		}
		var ts []*harness.Task
		for _, t := range m.snap.Tasks {
			if t.Workflow != wf.ID {
				continue
			}
			if m.tasks.hideDone && t.State == harness.StateDone {
				continue
			}
			if visible != nil && !visible[t.ID] {
				continue
			}
			ts = append(ts, t)
		}
		if len(ts) == 0 && (visible != nil || m.tasks.hideDone) {
			continue
		}
		sortTasks(ts)
		g := group{wf: wf, ts: ts, best: 1 << 30}
		if smart {
			m.orderSmart(ts)
			if len(ts) > 0 {
				g.best = m.triageRank(ts[0].ID)
			}
		}
		groups = append(groups, g)
	}
	if smart {
		// workflows by their best-ranked task; name order breaks ties
		sort.SliceStable(groups, func(i, j int) bool { return groups[i].best < groups[j].best })
	}
	line := 0
	for _, g := range groups {
		m.tasks.rows = append(m.tasks.rows, taskRow{kind: rowHeader, wf: g.wf, line: line})
		line++
		for _, t := range g.ts {
			m.tasks.rows = append(m.tasks.rows, taskRow{kind: rowTask, wf: g.wf, task: t, line: line})
			line += 2
		}
	}
	if m.selectedRow() < 0 {
		m.tasks.selID = ""
		for _, r := range m.tasks.rows {
			if r.kind == rowTask {
				m.tasks.selID = r.task.ID
				break
			}
		}
	}
}

func (m model) selectedRow() int {
	for i, r := range m.tasks.rows {
		if r.kind == rowTask && r.task.ID == m.tasks.selID {
			return i
		}
	}
	return -1
}

func (m model) selectedTask() *harness.Task {
	if m.tasks.selID == "" {
		return nil
	}
	return m.snap.Tasks[m.tasks.selID]
}

func (m model) taskRows() []int {
	var idx []int
	for i, r := range m.tasks.rows {
		if r.kind == rowTask {
			idx = append(idx, i)
		}
	}
	return idx
}

// moveTask moves the selection by n tasks (skipping headers).
func (m model) moveTask(n int) model {
	idx := m.taskRows()
	if len(idx) == 0 {
		return m
	}
	pos := 0
	cur := m.selectedRow()
	for i, ri := range idx {
		if ri == cur {
			pos = i
		}
	}
	pos = clampInt(pos+n, 0, len(idx)-1)
	if id := m.tasks.rows[idx[pos]].task.ID; id != m.tasks.selID {
		m.tasks.selID = id
		m.tasks.elemCursor, m.tasks.evOffset = 0, 0
	}
	m.ensureVisible(m.bodyHeight() - 2)
	return m
}

func (m model) taskAtLine(line int) string {
	line += m.tasks.offset
	for _, r := range m.tasks.rows {
		if r.kind == rowTask && (line == r.line || line == r.line+1) {
			return r.task.ID
		}
	}
	return ""
}

func (m model) tasksListWidth() int {
	return clampInt(m.width*45/100, min(40, m.width), m.width)
}

func (m model) totalListLines() int {
	if len(m.tasks.rows) == 0 {
		return 0
	}
	last := m.tasks.rows[len(m.tasks.rows)-1]
	if last.kind == rowTask {
		return last.line + 2
	}
	return last.line + 1
}

// ensureVisible scrolls the list so the selected task is on screen.
func (m *model) ensureVisible(inner int) {
	sel := m.selectedRow()
	if sel < 0 || inner <= 0 {
		m.tasks.offset = 0
		return
	}
	r := m.tasks.rows[sel]
	if r.line < m.tasks.offset {
		m.tasks.offset = r.line
		if sel > 0 && m.tasks.rows[sel-1].kind == rowHeader {
			m.tasks.offset = m.tasks.rows[sel-1].line
		}
	}
	if r.line+2 > m.tasks.offset+inner {
		m.tasks.offset = r.line + 2 - inner
	}
	m.tasks.offset = clampInt(m.tasks.offset, 0, max(0, m.totalListLines()-inner))
}

func (m model) updateTasks(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	k := msg.String()
	switch k {
	case "tab":
		if m.tasks.focus == focusList {
			m.tasks.focus = focusDetail
		} else {
			m.tasks.focus = focusList
		}
		return m, nil
	case "shift+tab":
		if m.tasks.focus == focusList {
			m.tasks.focus = focusDetail
		} else {
			m.tasks.focus = focusList
		}
		return m, nil
	case "/":
		m.tasks.filtering = true
		return m, m.tasks.filter.Focus()
	case "w":
		return m.cycleWorkflow(), nil
	case "e":
		m.tasks.hideDone = !m.tasks.hideDone
		m.rebuildRows()
		if m.tasks.hideDone {
			return m, m.ok("hiding done tasks")
		}
		return m, m.ok("showing done tasks")
	case "p":
		return m.pingSelected()
	case "P":
		return m.pickAgentForPing()
	case "x":
		return m.cancelSelected()
	case "R":
		return m.retrySelected()
	case "J":
		return m.runTaskReadiness()
	case "s":
		return m.toggleSort()
	case "N":
		return m.whatNext()
	}
	if m.tasks.focus == focusDetail {
		return m.updateDetail(msg)
	}
	switch k {
	case "t":
		return m.runTriageNow()
	case "j", "down":
		m = m.moveTask(1)
	case "k", "up":
		m = m.moveTask(-1)
	case "g", "home":
		m = m.moveTask(-len(m.tasks.rows))
	case "G", "end":
		m = m.moveTask(len(m.tasks.rows))
	case "ctrl+d", "pgdown":
		m = m.moveTask(max(1, (m.bodyHeight()-2)/4))
	case "ctrl+u", "pgup":
		m = m.moveTask(-max(1, (m.bodyHeight()-2)/4))
	case "o", "enter":
		t := m.selectedTask()
		if t == nil {
			return m, m.warn("no task selected")
		}
		if len(t.Elements) == 0 {
			return m, m.warn("task has no elements")
		}
		return m.openElement(t, t.Elements[0])
	}
	return m, nil
}

func (m model) updateFilter(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.tasks.filtering = false
		m.tasks.filter.SetValue("")
		m.tasks.filter.Blur()
		m.rebuildRows()
		return m, nil
	case "enter":
		m.tasks.filtering = false
		m.tasks.filter.Blur()
		return m, nil
	case "down", "ctrl+n":
		m = m.moveTask(1)
		return m, nil
	case "up", "ctrl+p":
		m = m.moveTask(-1)
		return m, nil
	}
	var cmd tea.Cmd
	m.tasks.filter, cmd = m.tasks.filter.Update(msg)
	m.rebuildRows()
	return m, cmd
}

func (m model) cycleWorkflow() model {
	wfs := m.sortedWorkflows()
	if len(wfs) == 0 {
		return m
	}
	next := ""
	if m.tasks.wfFilter == "" {
		next = wfs[0].ID
	} else {
		for i, w := range wfs {
			if w.ID == m.tasks.wfFilter && i+1 < len(wfs) {
				next = wfs[i+1].ID
			}
		}
	}
	m.tasks.wfFilter = next
	m.rebuildRows()
	return m
}

func (m model) pingSelected() (model, tea.Cmd) {
	t := m.selectedTask()
	if t == nil {
		return m, m.warn("no task selected")
	}
	if t.Agent == "" {
		return m, m.warn("task has no agent · P picks one")
	}
	return m.openComposer(composerPing, t.Agent, t.ID, nil)
}

func (m model) pickAgentForPing() (model, tea.Cmd) {
	agents := m.agentList()
	if len(agents) == 0 {
		return m, m.warn("no agents in feed")
	}
	labels := make([]string, len(agents))
	ids := make([]string, len(agents))
	for i, a := range agents {
		labels[i] = fmt.Sprintf("%-12s %s", a.Name, a.State)
		ids[i] = a.ID
	}
	taskID := ""
	if t := m.selectedTask(); t != nil {
		taskID = t.ID
	}
	return m.openPicker("Ping which agent?", labels, ids, true, func(m model, idx int) (model, tea.Cmd) {
		return m.openComposer(composerPing, ids[idx], taskID, nil)
	})
}

func (m model) cancelSelected() (model, tea.Cmd) {
	t := m.selectedTask()
	if t == nil {
		return m, m.warn("no task selected")
	}
	if t.State == harness.StateDone || t.State == harness.StateFailed {
		return m, m.warn("task is already " + string(t.State))
	}
	id := t.ID
	return m.askConfirm("Cancel "+id+" · "+t.Title+"?", "the agent is told to stop; this cannot be undone", func(m model) (model, tea.Cmd) {
		if _, err := m.outbox.Cancel(id); err != nil {
			return m, m.fail("outbox: " + err.Error())
		}
		return m, m.warn("cancel sent → " + id)
	})
}

func (m model) retrySelected() (model, tea.Cmd) {
	t := m.selectedTask()
	if t == nil {
		return m, m.warn("no task selected")
	}
	if t.State != harness.StateFailed && t.State != harness.StateBlocked {
		return m, m.warn("retry only applies to failed or blocked tasks")
	}
	if _, err := m.outbox.Retry(t.ID); err != nil {
		return m, m.fail("outbox: " + err.Error())
	}
	return m, m.ok("retry sent → " + t.ID)
}

// taskState is the JSON state sent to Jev for task-sourced packs.
func (m model) taskState(t *harness.Task, withEvents bool) []byte {
	payload := map[string]any{"task": t}
	if withEvents {
		evs := m.snap.Events[t.ID]
		if len(evs) > 30 {
			evs = evs[len(evs)-30:]
		}
		payload["events"] = evs
	}
	b, _ := json.MarshalIndent(payload, "", "  ")
	return b
}

func (m model) findPack(id string) *jev.Pack {
	for _, p := range m.packs {
		if p.ID == id {
			return p
		}
	}
	return nil
}

func (m model) runTaskReadiness() (model, tea.Cmd) {
	t := m.selectedTask()
	if t == nil {
		return m, m.warn("no task selected")
	}
	p := m.findPack("task-readiness")
	if p == nil {
		return m, m.fail("pack task-readiness not loaded")
	}
	if tj := m.taskJev[t.ID]; tj != nil && tj.running {
		return m, m.warn("already running for " + t.ID)
	}
	tj := m.taskJev[t.ID]
	if tj == nil {
		tj = &taskJev{}
		m.taskJev[t.ID] = tj
	}
	tj.running = true
	m.runSeq++
	state := m.taskState(t, true)
	return m, tea.Batch(m.askJevCmd(m.runSeq, t.ID, p, t.ID, state), m.kickAnim())
}

func (m model) askJevCmd(runID int, taskID string, p *jev.Pack, ref string, state []byte) tea.Cmd {
	client := m.client
	return func() tea.Msg {
		var st any
		if err := json.Unmarshal(state, &st); err != nil {
			st = string(state)
		}
		resp, err := client.Ask(contextBG(), st, p.Questions)
		return jevDoneMsg{runID: runID, taskID: taskID, pack: p, stateRef: ref, state: state, resp: resp, err: err}
	}
}

// ---------- view ----------

func (m model) viewTasksTab(w, h int) string {
	lw := m.tasksListWidth()
	rw := w - lw
	left := m.viewTaskList(lw, h)
	right := m.viewDetail(rw, h)
	if rw < 6 {
		return left
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, left, right)
}

func (m model) viewTaskList(w, h int) string {
	inner := h - 2
	mm := m
	mm.ensureVisible(inner)
	m.tasks.offset = mm.tasks.offset
	iw := w - 2
	title := fmt.Sprintf("Tasks · %d · %s", len(m.taskRows()), m.triage.sort)
	if m.tasks.wfFilter != "" {
		if wf, ok := m.snap.Workflows[m.tasks.wfFilter]; ok {
			title += " · " + wf.Name
		}
	}
	if m.tasks.hideDone {
		title += " · -done"
	}
	if m.tasks.filtering || m.tasks.filter.Value() != "" {
		title += " · /" + m.tasks.filter.Value()
	}
	var lines []string
	if len(m.snap.Tasks) == 0 {
		lines = append(lines, "", mutedStyle.Render("  no tasks yet"))
		if m.feedMissing {
			lines = append(lines, mutedStyle.Render("  feed file missing:"), mutedStyle.Render("  "+shortPath(m.cfg.Feed)), "", mutedStyle.Render("  start alembic --demo to see a simulated harness"))
		} else {
			lines = append(lines, mutedStyle.Render("  waiting for task.upsert records"))
		}
		return pane(title, lines, w, h, m.tasks.focus == focusList)
	}
	if len(m.tasks.rows) == 0 {
		lines = append(lines, "", mutedStyle.Render("  nothing matches the current filter"), mutedStyle.Render("  esc clears / · w cycles workflow · e shows done"))
		return pane(title, lines, w, h, m.tasks.focus == focusList)
	}
	all := make([]string, 0, m.totalListLines())
	for _, r := range m.tasks.rows {
		switch r.kind {
		case rowHeader:
			all = append(all, m.renderWorkflowHeader(r.wf, iw))
		case rowTask:
			l1, l2 := m.renderTaskRow(r.task, iw, r.task.ID == m.tasks.selID)
			all = append(all, l1, l2)
		}
	}
	end := min(len(all), m.tasks.offset+inner)
	if m.tasks.offset < end {
		lines = all[m.tasks.offset:end]
	}
	if m.tasks.filtering {
		title = "Tasks · " + m.tasks.filter.View()
	}
	return pane(title, lines, w, h, m.tasks.focus == focusList)
}

func (m model) renderWorkflowHeader(wf *harness.Workflow, w int) string {
	name := lipgloss.NewStyle().Foreground(fgText).Bold(true).Render(wf.Name)
	env := ""
	if wf.Env != "" {
		env = "  " + mutedStyle.Render(wf.Env)
	}
	state := ""
	if wf.State != "" {
		state = "  " + workflowStateStyle(wf.State).Render("● "+wf.State)
	}
	return fit(name+env+state, w)
}

func (m model) renderTaskRow(t *harness.Task, w int, selected bool) (string, string) {
	glyph := stateGlyph(t.State, m.frame)
	prio := priorityMark(t.Priority)
	head := " " + glyph + " " + t.ID
	if prio != "" {
		head += " " + prio
	}
	head += " "
	// the triage chip is right-aligned; the title is truncated first so the
	// chip never pushes the row past the pane
	chip, chipPlain := m.triageChip(t.ID)
	tw := w
	if chip != "" {
		cw := lipgloss.Width(chipPlain)
		if w-cw-1 < 8 {
			chip, chipPlain = "", ""
		} else {
			tw = w - cw - 1
		}
	}
	if selected {
		plain := " " + stripANSI(glyph) + " " + t.ID
		if prio != "" {
			plain += " " + stripANSI(prio)
		}
		l1 := fit(plain+" "+t.Title, tw)
		if chip != "" {
			l1 += " " + chipPlain
		}
		l1 = selStyle.Render(l1)
		l2 := selStyle.Render(fit("     "+t.StatusLine, w))
		return l1, l2
	}
	l1 := fit(head+textStyle.Render(t.Title), tw)
	if chip != "" {
		l1 += " " + chip
	}
	l2 := fit(mutedStyle.Render("     "+t.StatusLine), w)
	return l1, l2
}
