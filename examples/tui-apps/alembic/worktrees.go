package main

import (
	"errors"
	"fmt"
	"os/exec"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"

	"alembic/harness"
)

type wtState struct {
	list       []harness.Worktree
	err        error
	loading    bool
	ticking    bool
	cursor     int
	forceArmed string // path of a dirty worktree whose removal was confirmed once
}

func newWtState() wtState { return wtState{} }

func (m model) refreshWorktreesCmd() tea.Cmd {
	if !m.cfg.RepoOK {
		return nil
	}
	repo := m.cfg.Repo
	return func() tea.Msg {
		wts, err := harness.ListWorktrees(repo)
		return wtListMsg{wts: wts, err: err}
	}
}

func (m model) selectedWorktree() *harness.Worktree {
	if len(m.wt.list) == 0 {
		return nil
	}
	return &m.wt.list[clampInt(m.wt.cursor, 0, len(m.wt.list)-1)]
}

func (m model) updateWorktrees(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if !m.cfg.RepoOK {
		switch msg.String() {
		case "r":
			return m, m.warn("not a git repo: " + shortPath(m.cfg.Repo))
		}
		return m, nil
	}
	switch msg.String() {
	case "j", "down":
		m.wt.cursor = clampInt(m.wt.cursor+1, 0, max(0, len(m.wt.list)-1))
	case "k", "up":
		m.wt.cursor = clampInt(m.wt.cursor-1, 0, max(0, len(m.wt.list)-1))
	case "g", "home":
		m.wt.cursor = 0
	case "G", "end":
		m.wt.cursor = max(0, len(m.wt.list)-1)
	case "r":
		m.wt.loading = true
		return m, tea.Batch(m.refreshWorktreesCmd(), m.kickAnim())
	case "enter", "o":
		wt := m.selectedWorktree()
		if wt == nil {
			return m, m.warn("no worktree selected")
		}
		c := exec.Command(editorCmd(), wt.Path)
		c.Dir = wt.Path
		return m, tea.ExecProcess(c, func(err error) tea.Msg { return execDoneMsg{err: err} })
	case "n":
		return m.openWorktreeForm()
	case "d":
		return m.removeSelectedWorktree()
	case "J":
		return m.runDiffPack()
	case "y":
		if wt := m.selectedWorktree(); wt != nil {
			m.copier(wt.Path)
			return m, m.ok("copied " + shortPath(wt.Path))
		}
	}
	return m, nil
}

func (m model) openWorktreeForm() (model, tea.Cmd) {
	m.formVals = &formValues{}
	m.formKind = "worktree"
	m.form = huh.NewForm(huh.NewGroup(
		huh.NewInput().Key("branch").Title("Branch").Description("new or existing branch name").
			Placeholder("feat/…").Value(&m.formVals.branch).
			Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return errors.New("branch name is required")
				}
				if strings.ContainsAny(s, " ~^:?*[\\") {
					return errors.New("not a valid git ref")
				}
				return nil
			}),
		huh.NewInput().Key("path").Title("Path").Description("optional · default ../<repo>-<branch>").
			Placeholder(shortPath(m.cfg.Repo)+"-<branch>").Value(&m.formVals.path),
	)).WithTheme(alembicTheme()).WithShowHelp(false).WithWidth(boxWidth(60, m.width) - 2)
	return m, m.form.Init()
}

func (m model) removeSelectedWorktree() (model, tea.Cmd) {
	wt := m.selectedWorktree()
	if wt == nil {
		return m, m.warn("no worktree selected")
	}
	if wt.Main {
		return m, m.warn("the main worktree cannot be removed")
	}
	path := wt.Path
	repo := m.cfg.Repo
	if wt.Dirty > 0 {
		if m.wt.forceArmed != path {
			m.wt.forceArmed = path
			return m.askConfirm(fmt.Sprintf("Remove %s?", shortPath(path)),
				fmt.Sprintf("%d uncommitted change(s) will be lost · confirming arms --force; press d then y again", wt.Dirty),
				func(m model) (model, tea.Cmd) { return m, m.warn("armed: d + y again removes with --force") })
		}
		m.wt.forceArmed = ""
		return m.askConfirm(fmt.Sprintf("FORCE remove %s?", shortPath(path)), fmt.Sprintf("%d change(s) discarded", wt.Dirty), func(m model) (model, tea.Cmd) {
			return m, func() tea.Msg { return wtRemovedMsg{path: path, err: harness.RemoveWorktree(repo, path, true)} }
		})
	}
	return m.askConfirm(fmt.Sprintf("Remove worktree %s?", shortPath(path)), "clean tree · branch is kept", func(m model) (model, tea.Cmd) {
		return m, func() tea.Msg { return wtRemovedMsg{path: path, err: harness.RemoveWorktree(repo, path, false)} }
	})
}

// runDiffPack asks staged vs working, then runs the first pack with that
// state source against the selected worktree on the Jev tab.
func (m model) runDiffPack() (model, tea.Cmd) {
	wt := m.selectedWorktree()
	if wt == nil {
		return m, m.warn("no worktree selected")
	}
	path := wt.Path
	kinds := []string{"staged_diff", "working_diff"}
	labels := []string{"staged diff   (git diff --cached)", "working diff  (git diff)"}
	return m.openPicker("Run a pack on "+shortPath(path), labels, kinds, false, func(m model, idx int) (model, tea.Cmd) {
		var pack int = -1
		for i, p := range m.packs {
			if p.StateSource == kinds[idx] {
				pack = i
				break
			}
		}
		if pack < 0 {
			return m, m.warn("no pack with state_source " + kinds[idx])
		}
		m.jv.packCursor = pack
		m.jv.wtPath = path
		m.jv.focus = jevFocusResult
		m.tab = tabJev
		return m.startJevRun()
	})
}

func (m model) viewWorktreesTab(w, h int) string {
	title := "Worktrees"
	if m.wt.loading {
		title += " " + spinnerAt(m.frame)
	} else if len(m.wt.list) > 0 {
		title += fmt.Sprintf(" · %d", len(m.wt.list))
	}
	iw := w - 2
	if !m.cfg.RepoOK {
		lines := []string{"",
			warnStyle.Render("  not a git repo: ") + textStyle.Render(shortPath(m.cfg.Repo)),
			"",
			mutedStyle.Render("  start alembic inside a repository, or pass --repo DIR,"),
			mutedStyle.Render("  to list worktrees, link them to tasks and gate diffs with Jev."),
		}
		return pane(title, lines, w, h, true)
	}
	if m.wt.err != nil {
		lines := []string{"", errorStyle.Render("  git error: ") + textStyle.Render(m.wt.err.Error()), "", mutedStyle.Render("  r retries")}
		return pane(title, lines, w, h, true)
	}
	if len(m.wt.list) == 0 {
		msg := "  no worktrees yet · n creates one"
		if m.wt.loading {
			msg = "  listing worktrees…"
		}
		return pane(title, []string{"", mutedStyle.Render(msg)}, w, h, true)
	}
	// column widths: path | branch | ↑↓ | ✎ | HEAD | tasks
	fixed := 2 + 8 + 2 + 5 + 2 + 8 + 2 + 3
	pw := max(12, (iw-fixed)*45/100)
	bw := max(10, iw-fixed-pw-14)
	tw := max(6, iw-pw-bw-fixed)
	hdr := fmt.Sprintf(" %-*s  %-*s  %-8s %-5s  %-8s  %-*s", pw, "path", bw, "branch", "↑ ↓", "✎", "HEAD", tw, "tasks")
	lines := []string{fit(mutedStyle.Bold(true).Render(hdr), iw), fit(lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("─", iw)), iw)}
	for i, wt := range m.wt.list {
		path := fit(shortPath(wt.Path), pw)
		branch := wt.Branch
		if wt.Detached {
			branch = "detached @" + shortHead(wt.Head)
		}
		if wt.Main {
			branch = fit(branch, max(0, bw-5)) + " main"
		}
		branch = fit(branch, bw)
		ab := fmt.Sprintf("↑%d ↓%d", wt.Ahead, wt.Behind)
		dirty := "clean"
		if wt.Dirty > 0 {
			dirty = fmt.Sprintf("✎%d", wt.Dirty)
		}
		tasks := strings.Join(wt.TaskIDs, " ")
		if tasks == "" {
			tasks = "—"
		}
		plain := fmt.Sprintf(" %s  %s  %-8s %-5s  %-8s  %s", path, branch, ab, dirty, shortHead(wt.Head), fit(tasks, tw))
		if i == m.wt.cursor {
			lines = append(lines, selStyle.Render(fit(plain, iw)))
			continue
		}
		dirtyS := successStyle.Render(fmt.Sprintf("%-5s", dirty))
		if wt.Dirty > 0 {
			dirtyS = warnStyle.Render(fmt.Sprintf("%-5s", dirty))
		}
		branchS := cyanStyle.Render(branch)
		if wt.Main {
			branchS = cyanStyle.Render(fit(wt.Branch, max(0, bw-5))) + keyStyle.Render(" main")
		}
		row := " " + textStyle.Render(path) + "  " + branchS + "  " + mutedStyle.Render(fmt.Sprintf("%-8s", ab)) + " " + dirtyS + "  " + mutedStyle.Render(fmt.Sprintf("%-8s", shortHead(wt.Head))) + "  " + textStyle.Render(fit(tasks, tw))
		lines = append(lines, fit(row, iw))
	}
	return pane(title, lines, w, h, true)
}
