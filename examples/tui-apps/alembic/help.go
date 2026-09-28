package main

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type helpEntry struct{ key, desc string }

type helpSection struct {
	title   string
	entries []helpEntry
}

func helpSections() []helpSection {
	return []helpSection{
		{"Everywhere", []helpEntry{
			{"1-4", "switch tab (or click a tab)"},
			{"ctrl+k / :", "command palette (fuzzy)"},
			{"?", "this help · j/k scroll it"},
			{"mouse", "click rows/tabs · wheel scrolls"},
			{"q / ctrl+c", "quit"},
		}},
		{"Tasks", []helpEntry{
			{"j / k", "move · g/G top/bottom · ctrl+d/u half page"},
			{"tab", "focus list ↔ detail"},
			{"/", "fuzzy filter · esc clears"},
			{"w", "cycle workflow filter"},
			{"e", "hide / show done tasks"},
			{"o / enter", "open element ($EDITOR, browser, pager)"},
			{"y", "copy element (detail focus)"},
			{"p / P", "ping task's agent / pick an agent"},
			{"x", "cancel task (confirms)"},
			{"R", "retry a failed or blocked task"},
			{"J", "ask Jev: task-readiness"},
		}},
		{"Worktrees", []helpEntry{
			{"j / k", "move"},
			{"n", "new worktree (form)"},
			{"d", "remove worktree (confirms; dirty arms --force)"},
			{"r", "refresh"},
			{"enter / o", "open in $EDITOR"},
			{"y", "copy path"},
			{"J", "ask Jev: staged or working diff"},
		}},
		{"Jev", []helpEntry{
			{"tab", "focus Packs → State → Result"},
			{"j / k", "move in the focused column"},
			{"enter", "run the pack"},
			{"T / W", "pick the task / worktree for the state"},
			{"e / i", "edit inline text or file path · esc done"},
			{"h", "history: enter loads · c marks → compare"},
			{"y", "copy receipt path"},
			{"s / S", "send receipt to harness / with a note"},
			{"r", "reload packs"},
		}},
		{"Agents", []helpEntry{
			{"j / k", "move"},
			{"p", "ping the agent"},
			{"enter", "go to the agent's current task"},
		}},
	}
}

func (m model) updateHelp(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "?", "esc", "q":
		m.help = false
		m.helpOff = 0
	case "ctrl+c":
		return m, tea.Quit
	case "j", "down":
		m.helpOff++
	case "k", "up":
		m.helpOff = max(0, m.helpOff-1)
	case "g", "home":
		m.helpOff = 0
	}
	return m, nil
}

func (m model) viewHelp() string {
	w := boxWidth(96, m.width)
	keyW := 11
	twoCol := w >= 84
	colW := w
	if twoCol {
		colW = (w - 2) / 2
	}
	render := func(secs []helpSection) []string {
		var out []string
		for i, s := range secs {
			if i > 0 {
				out = append(out, "")
			}
			out = append(out, fit(sectionStyle.Render("─ "+s.title+" ")+sectionStyle.Render(strings.Repeat("─", max(0, colW-len(s.title)-3))), colW))
			for _, e := range s.entries {
				out = append(out, fit("  "+keyStyle.Render(fmt.Sprintf("%-*s", keyW, e.key))+textStyle.Render(e.desc), colW))
			}
		}
		return out
	}
	secs := helpSections()
	var body []string
	if twoCol {
		left := render(secs[:2])
		right := render(secs[2:])
		n := max(len(left), len(right))
		for i := 0; i < n; i++ {
			l, r := strings.Repeat(" ", colW), strings.Repeat(" ", colW)
			if i < len(left) {
				l = left[i]
			}
			if i < len(right) {
				r = right[i]
			}
			body = append(body, l+"  "+r)
		}
	} else {
		body = render(secs)
	}
	avail := max(3, m.height-2-3)
	off := clampInt(m.helpOff, 0, max(0, len(body)-avail))
	end := min(len(body), off+avail)
	shown := body[off:end]
	lines := []string{fit(titleStyle.Render("alembic — keys")+mutedStyle.Italic(true).Render("   Liquid gold · empower, don't extract."), w), ""}
	for _, l := range shown {
		lines = append(lines, fit(l, w))
	}
	foot := "  ? or esc closes"
	if end < len(body) || off > 0 {
		foot = fmt.Sprintf("  j/k scroll (%d more below)   ? or esc closes", len(body)-end)
	}
	lines = append(lines, mutedStyle.Render(fit(foot, w)))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

// hints is the per-tab key strip in the status bar.
func (m model) hints() string {
	switch {
	case m.form != nil:
		return "enter next · esc cancel"
	case m.palette.open:
		return "↑↓ move · enter run · esc"
	case m.help:
		return "j/k scroll · ? close"
	case m.confirm.open:
		return "y confirm · n cancel"
	case m.picker.open:
		return "↑↓ move · enter pick · esc"
	case m.jv.history.open:
		return "j/k · enter load · c compare · esc"
	case m.composer.open:
		return "enter send · tab canned · esc"
	case m.tasks.filtering:
		return "type to filter · ↑↓ move · enter keep · esc clear"
	case m.tab == tabJev && m.jv.editing:
		return "type · esc done"
	}
	switch m.tab {
	case tabTasks:
		return "j/k · tab · / filter · w wf · e done · p ping · x cancel · R retry · J jev · o open · ? help"
	case tabWorktrees:
		return "j/k · n new · d remove · r refresh · enter open · y copy · J jev · ? help"
	case tabJev:
		return "tab focus · j/k · enter run · T task · W worktree · e edit · h history · y copy · s send · r reload · ? help"
	case tabAgents:
		return "j/k · p ping · enter → task · ctrl+k palette · ? help"
	}
	return "? help"
}

var _ = lipgloss.Width
