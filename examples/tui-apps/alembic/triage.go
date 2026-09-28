package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"alembic/harness"
	"alembic/jev"
)

// The triage scheduler: one Triager per session (it keeps stall and budget
// state between ticks), a tea.Tick that runs it off the Update goroutine,
// and the ranking that drives the smart sort, the chips and "what next".

type triageSort int

const (
	sortStatus triageSort = iota
	sortSmart
)

func (s triageSort) String() string {
	if s == sortSmart {
		return "smart"
	}
	return "status"
}

type triageState struct {
	tr       *jev.Triager
	last     *jev.Triage
	lastAt   time.Time
	running  bool
	seq      int
	sort     triageSort
	sortSet  bool // the operator chose a sort; do not auto-switch to smart
	prevTop  string
	prevNext map[string]jev.Action
	hasPrev  bool
}

type (
	triageTickMsg struct{}
	triageDoneMsg struct {
		seq    int
		t      jev.Triage
		manual bool
		logErr error
	}
)

// triageFirstDelay is how long after start the first scheduled tick fires;
// tests shorten it.
var triageFirstDelay = 2 * time.Second

const triagePingText = "status? (triage: possibly stuck)"

func triageTick(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(time.Time) tea.Msg { return triageTickMsg{} })
}

func (m model) triageBudget() jev.Budget {
	b := jev.DefaultBudget()
	b.MaxTasksPerTick = m.cfg.TriageTasks
	b.MaxCallsPerHour = m.cfg.TriageCalls
	return b
}

// pendingPings counts the pings this session sent per task that have not
// been acknowledged yet.
func (m model) pendingPings() map[string]int {
	out := map[string]int{}
	for _, taskID := range m.pingsOpen {
		out[taskID]++
	}
	return out
}

// startTriage marks a run in flight and returns the Cmd that performs it
// against a copy of the snapshot, so the Triager never races with Update.
func (m *model) startTriage(manual bool) tea.Cmd {
	m.triage.running = true
	m.triage.seq++
	seq, tr, path := m.triage.seq, m.triage.tr, m.cfg.TriageLog
	clone := harness.NewSnapshot()
	for id, t := range m.snap.Tasks {
		c := *t
		clone.Tasks[id] = &c
	}
	for id, evs := range m.snap.Events {
		clone.Events[id] = evs
	}
	pending := m.pendingPings()
	return func() tea.Msg {
		t := tr.Run(contextBG(), clone, pending)
		return triageDoneMsg{seq: seq, t: t, manual: manual, logErr: jev.AppendTriage(path, t)}
	}
}

// runTriageNow is the manual tick (t key, palette).
func (m model) runTriageNow() (model, tea.Cmd) {
	if m.triage.running {
		return m, m.warn("triage already running")
	}
	if len(m.snap.Tasks) == 0 {
		return m, m.warn("no tasks to triage")
	}
	return m, m.startTriage(true)
}

// triageChanged reports whether the ranking's top task or any task's
// recommended action differs from the previous tick.
func triageChanged(prev *jev.Triage, cur jev.Triage) bool {
	if prev == nil {
		return len(cur.Tasks) > 0
	}
	prevTop, curTop := "", ""
	if n := prev.Next(); n != nil {
		prevTop = n.ID
	}
	if n := cur.Next(); n != nil {
		curTop = n.ID
	}
	if prevTop != curTop {
		return true
	}
	for _, tt := range cur.Tasks {
		p := prev.Get(tt.ID)
		if p == nil || p.Next != tt.Next {
			return true
		}
	}
	return false
}

func (m model) applyTriage(msg triageDoneMsg) (model, tea.Cmd) {
	if msg.seq != m.triage.seq {
		return m, nil // a superseded run
	}
	m.triage.running = false
	t := msg.t
	changed := triageChanged(m.triage.last, t)
	m.triage.last = &t
	m.triage.lastAt = time.Now()
	if !m.triage.sortSet {
		m.triage.sort = sortSmart
	}
	m.rebuildRows()
	var cmds []tea.Cmd
	if changed {
		if _, err := m.outbox.Send(harness.Command{Type: "triage", Data: t}); err != nil {
			cmds = append(cmds, m.fail("outbox: "+err.Error()))
		}
	}
	if msg.logErr != nil {
		cmds = append(cmds, m.warn("triage log: "+msg.logErr.Error()))
	} else if msg.manual {
		cmds = append(cmds, m.ok(triageSummary(t)))
	}
	return m, tea.Batch(cmds...)
}

// triageSummary is the one-line toast for a tick.
func triageSummary(t jev.Triage) string {
	if t.Deterministic {
		why := t.Skipped
		switch {
		case strings.HasPrefix(why, "budget"):
			why = "budget"
		case strings.HasPrefix(why, "jev error"):
			why = "jev error"
		case why == "":
			why = "no jev"
		}
		return fmt.Sprintf("triage: deterministic (%s) · %s", why, plural(len(t.Tasks), "task"))
	}
	s := fmt.Sprintf("triage: %s · jev %s · $%.4f", plural(len(t.Tasks), "task"), plural(t.JevCalls, "call"), t.CostUSD)
	if t.Mock {
		s += " · mock"
	}
	return s
}

func (m model) triageFor(id string) *jev.TaskTriage {
	if m.triage.last == nil {
		return nil
	}
	return m.triage.last.Get(id)
}

// triageRank orders tasks in smart mode; unknown tasks sort last.
func (m model) triageRank(id string) int {
	if tt := m.triageFor(id); tt != nil {
		return tt.Rank
	}
	return 1 << 30
}

func (m model) toggleSort() (model, tea.Cmd) {
	m.triage.sortSet = true
	if m.triage.sort == sortSmart {
		m.triage.sort = sortStatus
	} else {
		m.triage.sort = sortSmart
	}
	m.rebuildRows()
	if m.triage.sort == sortSmart && m.triage.last == nil {
		return m, m.warn("sort: smart · no triage yet, t runs one")
	}
	return m, m.ok("sort: " + m.triage.sort.String())
}

// whatNext selects the top-ranked task and pre-arms its recommended action.
func (m model) whatNext() (model, tea.Cmd) {
	if m.triage.last == nil {
		return m, m.warn("no triage yet · t runs it")
	}
	top := m.triage.last.Next()
	if top == nil {
		return m, m.ok("nothing needs you — every task is done")
	}
	t, ok := m.snap.Tasks[top.ID]
	if !ok {
		return m, m.warn(top.ID + " is no longer in the feed")
	}
	m = m.selectTaskClearingFilters(top.ID)
	mm, cmd := m.switchTab(tabTasks)
	m = mm
	reasons := top.ID + " → " + string(top.Next) + " · " + strings.Join(top.Reasons, " · ")
	switch top.Next {
	case jev.ActPing:
		if t.Agent == "" {
			mm, c := m.pickAgentForPing()
			return mm, tea.Batch(cmd, c, mm.warn(top.ID+" has no agent · pick one"))
		}
		mm, c := m.openComposer(composerPing, t.Agent, t.ID, nil)
		mm.composer.input.SetValue(triagePingText)
		mm.composer.input.CursorEnd()
		return mm, tea.Batch(cmd, c, mm.ok(reasons))
	case jev.ActReview:
		m.tasks.focus = focusDetail
		m.tasks.elemCursor, m.tasks.evOffset = 0, 0
		return m, tea.Batch(cmd, m.ok(reasons))
	case jev.ActRetry:
		mm, c := m.confirmRetry()
		return mm, tea.Batch(cmd, c, mm.ok(reasons))
	case jev.ActCancel:
		mm, c := m.cancelSelected()
		return mm, tea.Batch(cmd, c, mm.ok(reasons))
	}
	return m, tea.Batch(cmd, m.ok("nothing needs you — top task is progressing"))
}

// confirmRetry is the confirmed form of R, used when triage recommends it.
func (m model) confirmRetry() (model, tea.Cmd) {
	t := m.selectedTask()
	if t == nil {
		return m, m.warn("no task selected")
	}
	if t.State != harness.StateFailed && t.State != harness.StateBlocked {
		return m, m.warn("retry only applies to failed or blocked tasks")
	}
	id := t.ID
	return m.askConfirm("Retry "+id+" · "+t.Title+"?", "the task is re-queued from its last good state", func(m model) (model, tea.Cmd) {
		if _, err := m.outbox.Retry(id); err != nil {
			return m, m.fail("outbox: " + err.Error())
		}
		return m, m.ok("retry sent → " + id)
	})
}

// selectTaskClearingFilters lands the selection on id even when a filter
// hides it.
func (m model) selectTaskClearingFilters(id string) model {
	m.tasks.selID = id
	m.tasks.elemCursor, m.tasks.evOffset = 0, 0
	m.tasks.focus = focusList
	m.rebuildRows()
	if m.tasks.selID != id {
		m.tasks.wfFilter = ""
		m.tasks.hideDone = false
		m.tasks.filter.SetValue("")
		m.tasks.selID = id
		m.rebuildRows()
	}
	return m.moveTask(0)
}

// ---------- view ----------

func scoreColor(s float64) lipgloss.Color {
	switch {
	case s >= 0.6:
		return gold
	case s >= 0.3:
		return amber
	}
	return navy
}

func scoreBar(s float64) string {
	filled := clampInt(int(clamp(s, 0, 1)*4+0.5), 0, 4)
	return lipgloss.NewStyle().Foreground(scoreColor(s)).Render(strings.Repeat("▰", filled) + strings.Repeat("▱", 4-filled))
}

func nextGlyph(a jev.Action) (string, lipgloss.Style) {
	switch a {
	case jev.ActPing:
		return "✉", warnStyle
	case jev.ActReview:
		return "◉", cyanStyle
	case jev.ActCancel:
		return "✖", errorStyle
	case jev.ActRetry:
		return "↻", keyStyle
	}
	return "·", mutedStyle
}

// triageChip is the right-aligned score bar + next-action glyph for a task
// row; plain is the unstyled text (same width) for selected rows.
func (m model) triageChip(id string) (styled, plain string) {
	tt := m.triageFor(id)
	if tt == nil {
		return "", ""
	}
	g, st := nextGlyph(tt.Next)
	plain = scoreBarPlain(tt.Score) + " " + g
	styled = scoreBar(tt.Score) + " " + st.Render(g)
	if tt.JevUsed {
		plain += "◆"
		styled += cyanStyle.Render("◆")
	}
	return styled, plain
}

func scoreBarPlain(s float64) string {
	filled := clampInt(int(clamp(s, 0, 1)*4+0.5), 0, 4)
	return strings.Repeat("▰", filled) + strings.Repeat("▱", 4-filled)
}

// renderTriageLines is the detail pane's "─ triage ─" section.
func (m model) renderTriageLines(id string, w int) []string {
	prefix := sectionStyle.Render("─ triage ─ ")
	tt := m.triageFor(id)
	if tt == nil {
		hint := "not run yet · t runs it"
		if m.triage.last != nil {
			hint = "not in the last tick · t runs it"
		}
		return []string{fit(prefix+mutedStyle.Render(hint), w)}
	}
	_, st := nextGlyph(tt.Next)
	next := st.Render(string(tt.Next))
	if tt.NextConf > 0 {
		next += mutedStyle.Render(fmt.Sprintf(" (%.2f)", tt.NextConf))
	}
	head := prefix + keyStyle.Render(fmt.Sprintf("#%d", tt.Rank)) + mutedStyle.Render(" · ") +
		lipgloss.NewStyle().Foreground(scoreColor(tt.Score)).Bold(true).Render(fmt.Sprintf("%.2f", tt.Score)) +
		mutedStyle.Render(" · next: ") + next
	if tt.JevUsed {
		head += mutedStyle.Render(" · ") + cyanStyle.Render("◆ jev")
	}
	reasons := strings.Join(tt.Reasons, " · ")
	if reasons == "" {
		reasons = "no reasons"
	}
	return []string{fit(head, w), fit(mutedStyle.Render("           "+reasons), w)}
}

// viewTriageBadge is the header segment.
func (m model) viewTriageBadge() string {
	switch {
	case m.triage.running:
		return mutedStyle.Render("triage running")
	case m.triage.last == nil && m.cfg.TriageInterval <= 0:
		return mutedStyle.Render("triage: off")
	case m.triage.last == nil:
		return mutedStyle.Render("triage: pending")
	}
	s := mutedStyle.Render("triage " + relTime(m.now, m.triage.lastAt) + " ago")
	if t := m.triage.last; t.CostUSD > 0 && !t.Mock {
		s += " " + warnStyle.Render(fmt.Sprintf("$%.4f", t.CostUSD))
	}
	return s
}

// orderSmart sorts tasks by triage rank (stable over the status order).
func (m model) orderSmart(ts []*harness.Task) {
	sort.SliceStable(ts, func(i, j int) bool { return m.triageRank(ts[i].ID) < m.triageRank(ts[j].ID) })
}
