package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type command struct {
	name, desc string
	run        func(m model) (model, tea.Cmd)
}

type paletteState struct {
	open    bool
	input   textinput.Model
	cursor  int
	matches []int
	cmds    []command
}

func newPaletteState() paletteState {
	ti := textinput.New()
	ti.Prompt = "❯ "
	ti.PromptStyle = keyStyle
	ti.Placeholder = "type a command…"
	ti.CharLimit = 60
	ti.Width = 40
	return paletteState{input: ti, cmds: commands()}
}

func commands() []command {
	return []command{
		{"Ask Jev: is the selected task stuck?", "runs the task-readiness pack on the selected task (J)", func(m model) (model, tea.Cmd) { return m.runTaskReadiness() }},
		{"Run triage now", "score every task, Jev for the ambiguous ones (t)", func(m model) (model, tea.Cmd) { return m.runTriageNow() }},
		{"What next?", "select the top-ranked task and arm its action (N)", func(m model) (model, tea.Cmd) { return m.whatNext() }},
		{"Toggle sort smart / status", "order tasks by triage rank or by state (s)", func(m model) (model, tea.Cmd) { return m.toggleSort() }},
		{"Ping selected agent", "send a short note to the agent", func(m model) (model, tea.Cmd) { return m.paletteping() }},
		{"Cancel task", "tell the harness to stop the selected task", func(m model) (model, tea.Cmd) { return m.cancelSelected() }},
		{"Retry task", "re-queue a failed or blocked task", func(m model) (model, tea.Cmd) { return m.retrySelected() }},
		{"New worktree", "git worktree add on a new branch", func(m model) (model, tea.Cmd) {
			if !m.cfg.RepoOK {
				return m, m.warn("not a git repo: " + shortPath(m.cfg.Repo))
			}
			return m.openWorktreeForm()
		}},
		{"Refresh worktrees", "re-run git worktree list", func(m model) (model, tea.Cmd) {
			if !m.cfg.RepoOK {
				return m, m.warn("not a git repo: " + shortPath(m.cfg.Repo))
			}
			m.wt.loading = true
			return m, tea.Batch(m.refreshWorktreesCmd(), m.kickAnim())
		}},
		{"Reload packs", "re-read the question packs directory", func(m model) (model, tea.Cmd) { return m, reloadPacksCmd(m.cfg.Packs) }},
		{"Toggle hide done", "show or hide finished tasks", func(m model) (model, tea.Cmd) {
			m.tasks.hideDone = !m.tasks.hideDone
			m.rebuildRows()
			if m.tasks.hideDone {
				return m, m.ok("hiding done tasks")
			}
			return m, m.ok("showing done tasks")
		}},
		{"Cycle workflow", "filter the task list by the next workflow", func(m model) (model, tea.Cmd) {
			m = m.cycleWorkflow()
			if m.tasks.wfFilter == "" {
				return m, m.ok("all workflows")
			}
			return m, m.ok("workflow: " + m.snap.Workflows[m.tasks.wfFilter].Name)
		}},
		{"Go to Tasks", "tab 1", func(m model) (model, tea.Cmd) { return m.switchTab(tabTasks) }},
		{"Go to Worktrees", "tab 2", func(m model) (model, tea.Cmd) { return m.switchTab(tabWorktrees) }},
		{"Go to Jev", "tab 3", func(m model) (model, tea.Cmd) { return m.switchTab(tabJev) }},
		{"Go to Agents", "tab 4", func(m model) (model, tea.Cmd) { return m.switchTab(tabAgents) }},
		{"Help", "show key bindings", func(m model) (model, tea.Cmd) { m.help, m.helpOff = true, 0; return m, nil }},
		{"Quit", "exit alembic", func(m model) (model, tea.Cmd) { return m, tea.Quit }},
	}
}

// paletteping pings the agent that is "selected" in the current context:
// the Agents tab row, else the selected task's agent, else a picker.
func (m model) paletteping() (model, tea.Cmd) {
	if m.tab == tabAgents {
		return m.pingSelectedAgent()
	}
	if t := m.selectedTask(); t != nil && t.Agent != "" {
		return m.pingSelected()
	}
	return m.pickAgentForPing()
}

func (m model) openPalette() (tea.Model, tea.Cmd) {
	m.palette.open = true
	m.palette.cursor = 0
	m.palette.input.SetValue("")
	m.filterCommands()
	return m, m.palette.input.Focus()
}

func (m *model) filterCommands() {
	q := strings.TrimSpace(m.palette.input.Value())
	m.palette.matches = m.palette.matches[:0]
	if q == "" {
		for i := range m.palette.cmds {
			m.palette.matches = append(m.palette.matches, i)
		}
	} else {
		names := make([]string, len(m.palette.cmds))
		for i, c := range m.palette.cmds {
			names[i] = c.name
		}
		for _, r := range fuzzyFind(q, names) {
			m.palette.matches = append(m.palette.matches, r.Index)
		}
	}
	m.palette.cursor = min(m.palette.cursor, max(0, len(m.palette.matches)-1))
}

func (m model) updatePalette(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "ctrl+k", "ctrl+c":
		m.palette.open = false
		m.palette.input.Blur()
		return m, nil
	case "up", "ctrl+p":
		m.palette.cursor = max(0, m.palette.cursor-1)
		return m, nil
	case "down", "ctrl+n", "tab":
		m.palette.cursor = min(max(0, len(m.palette.matches)-1), m.palette.cursor+1)
		return m, nil
	case "enter":
		if len(m.palette.matches) == 0 {
			return m, nil
		}
		c := m.palette.cmds[m.palette.matches[clampInt(m.palette.cursor, 0, len(m.palette.matches)-1)]]
		m.palette.open = false
		m.palette.input.Blur()
		return c.run(m)
	}
	var cmd tea.Cmd
	m.palette.input, cmd = m.palette.input.Update(msg)
	m.filterCommands()
	return m, cmd
}

func (m model) viewPalette() string {
	w := boxWidth(64, m.width)
	var lines []string
	lines = append(lines, fit(m.palette.input.View(), w))
	lines = append(lines, lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("─", w)))
	names := make([]string, len(m.palette.cmds))
	for i, c := range m.palette.cmds {
		names[i] = c.name
	}
	q := strings.TrimSpace(m.palette.input.Value())
	var hits map[int][]int
	if q != "" {
		hits = map[int][]int{}
		for _, r := range fuzzyFind(q, names) {
			hits[r.Index] = r.MatchedIndexes
		}
	}
	maxRows := max(3, min(12, m.height-8))
	nameW := min(36, max(16, w*55/100))
	shown := 0
	for i, idx := range m.palette.matches {
		if shown >= maxRows {
			lines = append(lines, mutedStyle.Render(fmt.Sprintf("  … %d more", len(m.palette.matches)-shown)))
			break
		}
		c := m.palette.cmds[idx]
		if i == m.palette.cursor {
			lines = append(lines, selStyle.Render(fit("▸ "+c.name, nameW))+selStyle.Render(fit("  "+c.desc, w-nameW)))
		} else {
			lines = append(lines, fit("  "+highlight(c.name, hits[idx]), nameW)+fit("  "+mutedStyle.Render(c.desc), w-nameW))
		}
		shown++
	}
	if len(m.palette.matches) == 0 {
		lines = append(lines, mutedStyle.Render("  no matching commands"))
	}
	lines = append(lines, "", mutedStyle.Render(fit("  ↑↓ move   enter run   esc close", w)))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}
