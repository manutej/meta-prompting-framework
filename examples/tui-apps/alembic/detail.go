package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"alembic/harness"
)

func (m model) updateDetail(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	t := m.selectedTask()
	if t == nil {
		return m, nil
	}
	switch msg.String() {
	case "j", "down":
		m.tasks.elemCursor = clampInt(m.tasks.elemCursor+1, 0, max(0, len(t.Elements)-1))
	case "k", "up":
		m.tasks.elemCursor = clampInt(m.tasks.elemCursor-1, 0, max(0, len(t.Elements)-1))
	case "g", "home":
		m.tasks.elemCursor = 0
	case "G", "end":
		m.tasks.elemCursor = max(0, len(t.Elements)-1)
	case "ctrl+d", "pgdown":
		m.tasks.evOffset = clampInt(m.tasks.evOffset+5, 0, max(0, len(m.snap.Events[t.ID])-1))
	case "ctrl+u", "pgup":
		m.tasks.evOffset = max(0, m.tasks.evOffset-5)
	case "o", "enter":
		if len(t.Elements) == 0 {
			return m, m.warn("task has no elements")
		}
		return m.openElement(t, t.Elements[clampInt(m.tasks.elemCursor, 0, len(t.Elements)-1)])
	case "y":
		if len(t.Elements) == 0 {
			return m, m.warn("task has no elements")
		}
		el := t.Elements[clampInt(m.tasks.elemCursor, 0, len(t.Elements)-1)]
		m.copier(elementTarget(el))
		return m, m.ok("copied " + elementTarget(el))
	}
	return m, nil
}

var prRe = regexp.MustCompile(`^([\w.-]+/[\w.-]+)#(\d+)$`)

// elementTarget is what an element resolves to when copied or opened.
func elementTarget(el harness.Element) string {
	if el.Kind == harness.ElemPR {
		if mm := prRe.FindStringSubmatch(el.Ref); mm != nil {
			return "https://github.com/" + mm[1] + "/pull/" + mm[2]
		}
	}
	return el.Ref
}

// resolvePath resolves a task-relative path against its worktree, else the repo.
func (m model) resolvePath(t *harness.Task, ref string) string {
	if filepath.IsAbs(ref) {
		return ref
	}
	if t.Worktree != "" {
		if st, err := os.Stat(t.Worktree); err == nil && st.IsDir() {
			return filepath.Join(t.Worktree, ref)
		}
	}
	return filepath.Join(m.cfg.Repo, ref)
}

func editorCmd() string {
	if e := os.Getenv("EDITOR"); e != "" {
		return e
	}
	return "vi"
}

func pagerCmd() string {
	if p := os.Getenv("PAGER"); p != "" {
		return p
	}
	return editorCmd()
}

func (m model) openElement(t *harness.Task, el harness.Element) (model, tea.Cmd) {
	switch el.Kind {
	case harness.ElemFile:
		path := m.resolvePath(t, el.Ref)
		args := []string{}
		if el.Line > 0 {
			args = append(args, "+"+strconv.Itoa(el.Line))
		}
		args = append(args, path)
		c := exec.Command(editorCmd(), args...)
		return m, tea.ExecProcess(c, func(err error) tea.Msg { return execDoneMsg{err: err} })
	case harness.ElemURL, harness.ElemPR:
		url := elementTarget(el)
		m.copier(url)
		return m, openURLCmd(url)
	case harness.ElemLog, harness.ElemDir:
		path := m.resolvePath(t, el.Ref)
		if _, err := os.Stat(path); err != nil {
			return m, m.warn("not found: " + shortPath(path))
		}
		prog := editorCmd()
		if el.Kind == harness.ElemLog {
			prog = pagerCmd()
		}
		c := exec.Command(prog, path)
		return m, tea.ExecProcess(c, func(err error) tea.Msg { return execDoneMsg{err: err} })
	}
	return m, m.warn("unknown element kind " + string(el.Kind))
}

// openURLCmd launches the platform opener detached; failure only degrades the toast.
func openURLCmd(url string) tea.Cmd {
	return func() tea.Msg {
		for _, opener := range []string{"xdg-open", "open"} {
			if _, err := exec.LookPath(opener); err != nil {
				continue
			}
			c := exec.Command(opener, url)
			c.Stdout, c.Stderr = nil, nil
			if err := c.Start(); err == nil {
				go func() { _ = c.Wait() }()
				return openedMsg{what: "opened + copied " + url}
			}
		}
		return openedMsg{what: "copied " + url, err: fmt.Errorf("no opener")}
	}
}

func (m model) worktreeFor(path string) *harness.Worktree {
	if path == "" {
		return nil
	}
	for i := range m.wt.list {
		if filepath.Clean(m.wt.list[i].Path) == filepath.Clean(path) {
			return &m.wt.list[i]
		}
	}
	return nil
}

func (m model) viewDetail(w, h int) string {
	focused := m.tasks.focus == focusDetail
	t := m.selectedTask()
	if t == nil {
		return pane("Detail", []string{"", mutedStyle.Render("  select a task to see its detail")}, w, h, focused)
	}
	iw := w - 2
	title := t.ID + " · " + t.Title
	label := func(s string) string { return mutedStyle.Render(fmt.Sprintf(" %-9s", s)) }

	var head []string
	agent := m.agentName(t.Agent)
	if t.Agent == "" {
		agent = "unassigned"
	}
	gw := max(6, min(20, iw-40))
	head = append(head, fit(label("state")+stateGlyph(t.State, m.frame)+" "+
		lipgloss.NewStyle().Foreground(stateColor(t.State)).Bold(true).Render(string(t.State))+
		"  "+gauge(t.Progress, gw, stateColor(t.State))+mutedStyle.Render(fmt.Sprintf(" %3.0f%%", t.Progress*100))+
		"  "+textStyle.Render(agent), iw))
	wfName, wfEnv := t.Workflow, ""
	if wf, ok := m.snap.Workflows[t.Workflow]; ok {
		wfName, wfEnv = wf.Name, wf.Env
	}
	head = append(head, fit(label("workflow")+textStyle.Render(wfName)+"  "+mutedStyle.Render(wfEnv), iw))
	wtLine := mutedStyle.Render("none")
	if t.Worktree != "" {
		wtLine = textStyle.Render(shortPath(t.Worktree))
		if t.Branch != "" {
			wtLine += "  " + cyanStyle.Render(t.Branch)
		}
		if wt := m.worktreeFor(t.Worktree); wt != nil {
			wtLine += "  " + mutedStyle.Render(fmt.Sprintf("↑%d ↓%d", wt.Ahead, wt.Behind))
			if wt.Dirty > 0 {
				wtLine += " " + warnStyle.Render(fmt.Sprintf("✎%d", wt.Dirty))
			}
		}
	}
	head = append(head, fit(label("worktree")+wtLine, iw))
	head = append(head, fit(label("updated")+textStyle.Render(relTime(m.now, t.Updated)+" ago")+mutedStyle.Render("  created "+relTime(m.now, t.Created)+" ago"), iw))

	section := func(name string) string {
		return fit(sectionStyle.Render("─ "+name+" ")+sectionStyle.Render(strings.Repeat("─", max(0, iw-len(name)-4))), iw)
	}

	var elems []string
	elems = append(elems, section("elements"))
	if len(t.Elements) == 0 {
		elems = append(elems, mutedStyle.Render("   none"))
	}
	for i, el := range t.Elements {
		ref := el.Ref
		if el.Line > 0 {
			ref += ":" + strconv.Itoa(el.Line)
		}
		lbl := ""
		if el.Label != "" && el.Label != el.Ref {
			lbl = "  " + mutedStyle.Render(el.Label)
		}
		row := "   " + cyanStyle.Render(kindIcon(el.Kind)) + " " + textStyle.Render(ref) + lbl
		if focused && i == m.tasks.elemCursor {
			plain := "▸  " + kindIcon(el.Kind) + " " + ref
			if el.Label != "" && el.Label != el.Ref {
				plain += "  " + el.Label
			}
			row = selStyle.Render(fit(plain, iw))
		}
		elems = append(elems, fit(row, iw))
	}

	jevLine := m.renderTaskJevLine(t.ID, iw)

	fixed := len(head) + len(elems) + 1 + 1 // sections + jev line
	inner := h - 2
	evRows := max(0, inner-fixed)
	evs := m.snap.Events[t.ID]
	var events []string
	events = append(events, section(fmt.Sprintf("events · %d", len(evs))))
	if len(evs) == 0 {
		events = append(events, mutedStyle.Render("   no events yet"))
	}
	off := clampInt(m.tasks.evOffset, 0, max(0, len(evs)-1))
	shown := 0
	for i := len(evs) - 1 - off; i >= 0 && shown < max(0, evRows-1); i-- {
		e := evs[i]
		row := " " + mutedStyle.Render(e.At.Local().Format("15:04:05")) + " " + levelGlyph(e.Level) + " " + textStyle.Render(e.Text)
		events = append(events, fit(row, iw))
		shown++
	}
	if off > 0 && evRows > 1 {
		events[len(events)-1] = fit(mutedStyle.Render(fmt.Sprintf("   ↓ %d newer", off)), iw)
	}

	lines := append([]string{}, head...)
	lines = append(lines, elems...)
	lines = append(lines, events...)
	for len(lines) < inner-1 {
		lines = append(lines, "")
	}
	if len(lines) > inner-1 {
		lines = lines[:inner-1]
	}
	lines = append(lines, jevLine)
	return pane(title, lines, w, h, focused)
}

// renderTaskJevLine shows the last Jev receipt for a task, or the running spinner.
func (m model) renderTaskJevLine(taskID string, w int) string {
	prefix := sectionStyle.Render("─ jev ─ ")
	tj := m.taskJev[taskID]
	switch {
	case tj == nil:
		return fit(prefix+mutedStyle.Render("no receipts · J runs task-readiness"), w)
	case tj.running:
		return fit(prefix+keyStyle.Render(spinnerAt(m.frame))+" "+textStyle.Render("task-readiness running…"), w)
	case tj.receipt != nil:
		rc := tj.receipt
		dec := decisionStyle(rc.Decision).Render(strings.ToUpper(string(rc.Decision)))
		reason := ""
		if len(rc.Unfavorable) > 0 {
			reason = " (" + strings.Join(rc.Unfavorable, ", ") + ")"
		}
		mock := ""
		if rc.Mock {
			mock = warnStyle.Render(" MOCK")
		}
		return fit(prefix+textStyle.Render(rc.PackID+": ")+dec+mutedStyle.Render(reason+" "+relTime(m.now, rc.At)+" ago")+mock, w)
	}
	return fit(prefix, w)
}
