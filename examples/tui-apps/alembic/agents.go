package main

import (
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"alembic/harness"
)

type agentsState struct {
	cursor int
}

// agentList returns agents sorted by name, then id, so rows are stable.
func (m model) agentList() []*harness.Agent {
	out := make([]*harness.Agent, 0, len(m.snap.Agents))
	for _, a := range m.snap.Agents {
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func (m model) selectedAgent() *harness.Agent {
	list := m.agentList()
	if len(list) == 0 {
		return nil
	}
	return list[clampInt(m.agents.cursor, 0, len(list)-1)]
}

// agentTask is the task an agent is on: the harness's current_task when it
// says so, else the most active task assigned to the agent.
func (m model) agentTask(a *harness.Agent) *harness.Task {
	if a == nil {
		return nil
	}
	if t, ok := m.snap.Tasks[a.CurrentTask]; ok && a.CurrentTask != "" {
		return t
	}
	var best *harness.Task
	for _, t := range m.snap.Tasks {
		if t.Agent != a.ID || t.State == harness.StateDone || t.State == harness.StateFailed {
			continue
		}
		if best == nil || stateOrder(t.State) < stateOrder(best.State) ||
			(stateOrder(t.State) == stateOrder(best.State) && t.Updated.After(best.Updated)) {
			best = t
		}
	}
	return best
}

func (m model) acksFor(agentID string) int {
	n := 0
	for _, a := range m.snap.Acks {
		if a.Agent == agentID {
			n++
		}
	}
	return n
}

func (m model) updateAgents(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	list := m.agentList()
	switch msg.String() {
	case "j", "down":
		m.agents.cursor = clampInt(m.agents.cursor+1, 0, max(0, len(list)-1))
	case "k", "up":
		m.agents.cursor = clampInt(m.agents.cursor-1, 0, max(0, len(list)-1))
	case "g", "home":
		m.agents.cursor = 0
	case "G", "end":
		m.agents.cursor = max(0, len(list)-1)
	case "p":
		return m.pingSelectedAgent()
	case "enter", "o":
		return m.jumpToAgentTask()
	}
	return m, nil
}

func (m model) pingSelectedAgent() (model, tea.Cmd) {
	a := m.selectedAgent()
	if a == nil {
		return m, m.warn("no agents in feed")
	}
	return m.openComposer(composerPing, a.ID, "", nil)
}

func (m model) jumpToAgentTask() (model, tea.Cmd) {
	a := m.selectedAgent()
	if a == nil {
		return m, m.warn("no agents in feed")
	}
	t := m.agentTask(a)
	if t == nil {
		return m, m.warn(a.Name + " has no current task")
	}
	m.tasks.selID = t.ID
	m.tasks.elemCursor, m.tasks.evOffset = 0, 0
	m.tasks.focus = focusList
	m.rebuildRows()
	if m.tasks.selID != t.ID {
		// hidden by a filter: clear filters so the jump lands
		m.tasks.wfFilter = ""
		m.tasks.hideDone = false
		m.tasks.filter.SetValue("")
		m.tasks.selID = t.ID
		m.rebuildRows()
	}
	m = m.moveTask(0)
	mm, cmd := m.switchTab(tabTasks)
	return mm, tea.Batch(cmd, mm.ok("→ "+t.ID))
}

func (m model) viewAgentsTab(w, h int) string {
	list := m.agentList()
	title := "Agents"
	if len(list) > 0 {
		title += fmt.Sprintf(" · %d", len(list))
	}
	iw := w - 2
	if len(list) == 0 {
		lines := []string{"", mutedStyle.Render("  no agents in feed")}
		if m.feedMissing {
			lines = append(lines, mutedStyle.Render("  feed file missing: "+shortPath(m.cfg.Feed)), "", mutedStyle.Render("  start alembic --demo to see a simulated harness"))
		} else {
			lines = append(lines, mutedStyle.Render("  waiting for agent.upsert records"))
		}
		return pane(title, lines, w, h, true)
	}
	// columns: name | state | model | task | seen | ping/ack
	nameW, stateW, seenW, pingW := 10, 7, 7, 9
	for _, a := range list {
		nameW = max(nameW, lipgloss.Width(a.Name))
	}
	nameW = min(nameW, max(10, iw/5))
	modelW := clampInt((iw-nameW-stateW-seenW-pingW-12)*35/100, 8, 20)
	taskW := max(6, iw-nameW-stateW-modelW-seenW-pingW-12)
	hdr := fmt.Sprintf(" %-*s  %-*s  %-*s  %-*s  %-*s  %s", nameW, "agent", stateW, "state", modelW, "model", taskW, "current task", seenW, "seen", "ping/ack")
	lines := []string{fit(mutedStyle.Bold(true).Render(hdr), iw), fit(lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("─", iw)), iw)}
	for i, a := range list {
		t := m.agentTask(a)
		task := "—"
		if t != nil {
			task = t.ID + " " + t.Title
		}
		seen := relTime(m.now, a.Seen)
		pa := fmt.Sprintf("%d / %d", m.pingsSent[a.ID], m.acksFor(a.ID))
		state := a.State
		if state == "" {
			state = "unknown"
		}
		plain := fmt.Sprintf(" %s  %s  %s  %s  %s  %s", fit(a.Name, nameW), fit(state, stateW), fit(a.Model, modelW), fit(task, taskW), fit(seen, seenW), pa)
		if i == m.agents.cursor {
			lines = append(lines, selStyle.Render(fit(plain, iw)))
			continue
		}
		row := " " + textStyle.Render(fit(a.Name, nameW)) + "  " + agentStateStyle(state).Render(fit(state, stateW)) + "  " +
			mutedStyle.Render(fit(a.Model, modelW)) + "  "
		if t != nil {
			row += keyStyle.Render(t.ID) + " " + textStyle.Render(fit(t.Title, max(0, taskW-len(t.ID)-1)))
		} else {
			row += mutedStyle.Render(fit("—", taskW))
		}
		row += "  " + mutedStyle.Render(fit(seen, seenW)) + "  " + textStyle.Render(pa)
		lines = append(lines, fit(row, iw))
	}
	lines = append(lines, "", mutedStyle.Render(fit("  pings sent this session / acks received from the harness", iw)))
	return pane(title, lines, w, h, true)
}
