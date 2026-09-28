package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/mattn/go-runewidth"
)

var sizes = [][2]int{{80, 24}, {120, 40}, {200, 60}}

func git(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=Test", "GIT_AUTHOR_EMAIL=t@example.com",
		"GIT_COMMITTER_NAME=Test", "GIT_COMMITTER_EMAIL=t@example.com",
		"HOME="+dir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return string(out)
}

func write(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// tempRepo builds a repository with two commits, one staged file, one
// unstaged modification and one untracked file.
func tempRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	git(t, dir, "init", "-q", "-b", "main")
	write(t, dir, "main.go", "package main\n\nfunc main() {}\n")
	write(t, dir, "README.md", "# demo\n")
	git(t, dir, "add", ".")
	git(t, dir, "commit", "-q", "-m", "feat: initial commit")
	write(t, dir, "README.md", "# demo\n\nmore\n")
	git(t, dir, "commit", "-q", "-am", "docs: expand readme")
	git(t, dir, "branch", "feature/x")
	write(t, dir, "staged.txt", "staged content\n")
	git(t, dir, "add", "staged.txt")
	write(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"hi\")\n}\n")
	write(t, dir, "untracked.txt", "new file\n")
	return dir
}

func run(t *testing.T, cmd tea.Cmd) tea.Msg {
	t.Helper()
	if cmd == nil {
		t.Fatal("expected a command")
	}
	return cmd()
}

func loaded(t *testing.T, w, h int) model {
	t.Helper()
	root := tempRepo(t)
	m := newModel(root, root)
	m = update(t, m, tea.WindowSizeMsg{Width: w, Height: h})
	m = update(t, m, run(t, loadStatus(root)))
	m = update(t, m, run(t, loadBranches(root)))
	m = update(t, m, run(t, loadCommits(root)))
	m = update(t, m, run(t, loadStashes(root)))
	return m
}

func update(t *testing.T, m model, msg tea.Msg) model {
	t.Helper()
	mm, _ := m.Update(msg)
	return mm.(model)
}

func updateCmd(t *testing.T, m model, msg tea.Msg) (model, tea.Cmd) {
	t.Helper()
	mm, cmd := m.Update(msg)
	return mm.(model), cmd
}

func key(s string) tea.KeyMsg {
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEscape}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	case "shift+tab":
		return tea.KeyMsg{Type: tea.KeyShiftTab}
	case "space":
		return tea.KeyMsg{Type: tea.KeySpace, Runes: []rune{' '}}
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case "backspace":
		return tea.KeyMsg{Type: tea.KeyBackspace}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
}

func checkFrame(t *testing.T, view string, w, h int) {
	t.Helper()
	if view == "" {
		t.Fatalf("empty view at %dx%d", w, h)
	}
	lines := strings.Split(view, "\n")
	if len(lines) != h {
		t.Fatalf("view has %d lines, want %d at %dx%d", len(lines), h, w, h)
	}
	for i, l := range lines {
		if lw := runewidth.StringWidth(stripANSI(l)); lw > w {
			t.Fatalf("line %d is %d wide (> %d): %q", i, lw, w, stripANSI(l))
		}
	}
}

func TestParseStatusPorcelain(t *testing.T) {
	out := "# branch.oid abc123\n# branch.head main\n# branch.upstream origin/main\n# branch.ab +2 -1\n" +
		"1 M. N... 100644 100644 100644 a b staged.go\n" +
		"1 .M N... 100644 100644 100644 a b dirty.go\n" +
		"1 MM N... 100644 100644 100644 a b both.go\n" +
		"2 R. N... 100644 100644 100644 a b R100 new.go\told.go\n" +
		"? untracked.txt\n"
	st := parseStatus(out)
	if st.Head != "main" || st.Upstream != "origin/main" || st.Ahead != 2 || st.Behind != 1 {
		t.Fatalf("header parsed wrong: %+v", st)
	}
	want := []statusEntry{
		{Path: "staged.go", Code: "M", Staged: true},
		{Path: "both.go", Code: "M", Staged: true},
		{Path: "new.go", OrigPath: "old.go", Code: "R", Staged: true},
		{Path: "dirty.go", Code: "M"},
		{Path: "both.go", Code: "M"},
		{Path: "untracked.txt", Code: "?"},
	}
	if len(st.Entries) != len(want) {
		t.Fatalf("got %d entries, want %d: %+v", len(st.Entries), len(want), st.Entries)
	}
	for i := range want {
		if st.Entries[i] != want[i] {
			t.Errorf("entry %d = %+v, want %+v", i, st.Entries[i], want[i])
		}
	}
}

func TestParseStatusDetachedAndUnborn(t *testing.T) {
	st := parseStatus("# branch.oid abcdef1234\n# branch.head (detached)\n")
	if !st.Detached {
		t.Fatal("expected detached")
	}
	st = parseStatus("# branch.oid (initial)\n# branch.head main\n")
	if !st.Unborn {
		t.Fatal("expected unborn")
	}
}

func TestRealRepoLoaders(t *testing.T) {
	root := tempRepo(t)
	st := run(t, loadStatus(root)).(statusMsg)
	if st.err != nil {
		t.Fatal(st.err)
	}
	var staged, unstaged, untracked int
	for _, e := range st.st.Entries {
		switch {
		case e.Code == "?":
			untracked++
		case e.Staged:
			staged++
		default:
			unstaged++
		}
	}
	if staged != 1 || unstaged != 1 || untracked != 1 {
		t.Fatalf("staged=%d unstaged=%d untracked=%d entries=%+v", staged, unstaged, untracked, st.st.Entries)
	}
	cs := run(t, loadCommits(root)).(commitsMsg)
	if cs.err != nil || len(cs.commits) != 2 || cs.commits[0].Subject != "docs: expand readme" {
		t.Fatalf("commits: %+v %v", cs.commits, cs.err)
	}
	bs := run(t, loadBranches(root)).(branchesMsg)
	if bs.err != nil || len(bs.branches) != 2 {
		t.Fatalf("branches: %+v %v", bs.branches, bs.err)
	}
	ss := run(t, loadStashes(root)).(stashesMsg)
	if ss.err != nil || len(ss.stashes) != 0 {
		t.Fatalf("stashes: %+v %v", ss.stashes, ss.err)
	}
}

func TestViewFitsAllSizes(t *testing.T) {
	for _, sz := range sizes {
		m := loaded(t, sz[0], sz[1])
		checkFrame(t, m.View(), sz[0], sz[1])
		for _, p := range []paneID{paneBranches, paneCommits, paneStash, paneDiff} {
			m, _ = m.focusPane(p)
			m = m.snapHeights()
			checkFrame(t, m.View(), sz[0], sz[1])
		}
	}
}

func TestViewWithDiffContent(t *testing.T) {
	for _, sz := range sizes {
		m := loaded(t, sz[0], sz[1])
		req, ok := m.currentRequest()
		if !ok {
			t.Fatal("expected a diff request for the first status entry")
		}
		msg := run(t, loadDiff(m.root, m.diffSeq, req)).(diffMsg)
		m = update(t, m, msg)
		view := m.View()
		checkFrame(t, view, sz[0], sz[1])
		if !strings.Contains(stripANSI(view), "staged content") {
			t.Fatalf("staged diff not rendered at %dx%d", sz[0], sz[1])
		}
	}
}

func TestLongLinesNeverOverflow(t *testing.T) {
	m := loaded(t, 80, 24)
	long := strings.Repeat("x", 500)
	m = update(t, m, diffMsg{seq: m.diffSeq, key: m.diffKey, kind: "diff", title: strings.Repeat("t", 300),
		text: "diff --git a/f b/f\n+++ b/f\n@@ -1 +1 @@\n+" + long + "\n-" + long + "\n"})
	m.commits = append(m.commits, commit{SHA: "deadbeef", Short: "deadbee", Author: "someone", Age: "3d", Subject: long})
	m.status.Entries = append(m.status.Entries, statusEntry{Path: long, Code: "M"})
	m.branches = append(m.branches, branch{Name: long, Subject: long})
	m.stashes = append(m.stashes, stash{Ref: "stash@{0}", Message: long})
	for _, p := range []paneID{paneStatus, paneBranches, paneCommits, paneStash} {
		m.lastLeft = p
		m = m.snapHeights()
		checkFrame(t, m.View(), 80, 24)
	}
}

func TestStageToggleArgs(t *testing.T) {
	m := loaded(t, 120, 40)
	e, ok := m.selectedEntry()
	if !ok || !e.Staged || e.Path != "staged.txt" {
		t.Fatalf("first entry should be the staged file, got %+v", e)
	}
	m, cmd := updateCmd(t, m, key("space"))
	if cmd == nil {
		t.Fatal("expected a git command")
	}
	if got := strings.Join(m.lastGit, " "); got != "restore --staged -- staged.txt" {
		t.Fatalf("unstage args = %q", got)
	}
	m, _ = m.setSel(paneStatus, 1)
	e, _ = m.selectedEntry()
	if e.Staged {
		t.Fatalf("second entry should be unstaged, got %+v", e)
	}
	m, _ = updateCmd(t, m, key("space"))
	if got := strings.Join(m.lastGit, " "); got != "add -- "+e.Path {
		t.Fatalf("stage args = %q", got)
	}
	if got := strings.Join(stageToggleArgs(statusEntry{Path: "a b.txt"}), "|"); got != "add|--|a b.txt" {
		t.Fatalf("path with space mangled: %q", got)
	}
}

func TestStageRoundTripOnRealRepo(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = m.setSel(paneStatus, 1)
	e, _ := m.selectedEntry()
	m, cmd := updateCmd(t, m, key("space"))
	msg, ok := firstAction(collect(cmd))
	if !ok || msg.err != nil {
		t.Fatalf("stage action: ok=%v err=%v", ok, msg.err)
	}
	m = update(t, m, msg)
	if m.toast == nil || !m.toast.ok || !strings.Contains(m.toast.text, e.Path) {
		t.Fatalf("expected success toast, got %+v", m.toast)
	}
	st := run(t, loadStatus(m.root)).(statusMsg)
	found := false
	for _, x := range st.st.Entries {
		if x.Path == e.Path && x.Staged {
			found = true
		}
	}
	if !found {
		t.Fatalf("%s should now be staged: %+v", e.Path, st.st.Entries)
	}
}

func TestStageAllAndUnstageAll(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = updateCmd(t, m, key("a"))
	if got := strings.Join(m.lastGit, " "); got != "add -A" {
		t.Fatalf("stage all = %q", got)
	}
	m, _ = updateCmd(t, m, key("A"))
	if got := strings.Join(m.lastGit, " "); got != "reset -q" {
		t.Fatalf("unstage all = %q", got)
	}
}

func TestHelpOverlayToggles(t *testing.T) {
	m := loaded(t, 80, 24)
	m = update(t, m, key("?"))
	if !m.showHelp {
		t.Fatal("help should be open")
	}
	view := m.View()
	checkFrame(t, view, 80, 24)
	if !strings.Contains(stripANSI(view), "keyboard reference") {
		t.Fatal("help content missing")
	}
	m, cmd := updateCmd(t, m, key("q"))
	if m.showHelp || cmd != nil {
		t.Fatal("q should close help without quitting")
	}
	m = update(t, m, key("?"))
	m = update(t, m, key("?"))
	if m.showHelp {
		t.Fatal("second ? should close help")
	}
}

func TestNotARepoRenders(t *testing.T) {
	m := newModel("", "/some/where/else")
	if !m.noRepo {
		t.Fatal("expected noRepo")
	}
	if m.Init() != nil {
		t.Fatal("no commands should run outside a repo")
	}
	for _, sz := range sizes {
		m = update(t, m, tea.WindowSizeMsg{Width: sz[0], Height: sz[1]})
		view := m.View()
		checkFrame(t, view, sz[0], sz[1])
		if !strings.Contains(stripANSI(view), "/some/where/else") {
			t.Fatal("path missing from no-repo screen")
		}
	}
	_, cmd := m.Update(key("q"))
	if cmd == nil {
		t.Fatal("q should quit")
	}
}

func TestFocusAndAccordion(t *testing.T) {
	m := loaded(t, 120, 40)
	m, cmd := updateCmd(t, m, key("3"))
	if m.focus != paneCommits || m.lastLeft != paneCommits || !m.animating || cmd == nil {
		t.Fatalf("focus=%v lastLeft=%v animating=%v", m.focus, m.lastLeft, m.animating)
	}
	for i := 0; i < 200 && m.animating; i++ {
		m = update(t, m, animMsg{})
		checkFrame(t, m.View(), 120, 40)
	}
	if m.animating {
		t.Fatal("spring never settled")
	}
	l := m.layout()
	if l.paneH[paneCommits] <= collapsedH || l.paneH[paneStatus] != collapsedH {
		t.Fatalf("accordion heights wrong: %+v", l.paneH)
	}
	total := 0
	for _, h := range l.paneH {
		total += h
	}
	if total != l.bodyH {
		t.Fatalf("pane heights sum %d != body %d", total, l.bodyH)
	}
	m = update(t, m, key("tab"))
	m = update(t, m, key("tab"))
	if m.focus != paneDiff {
		t.Fatalf("tab tab from commits should reach diff, got %v", m.focus)
	}
	m = update(t, m, key("h"))
	if m.focus != paneStash || m.lastLeft != paneStash {
		t.Fatalf("h should return to the last list (stash), got %v", m.focus)
	}
	m = update(t, m, key("shift+tab"))
	if m.focus != paneCommits {
		t.Fatalf("shift+tab should go to commits, got %v", m.focus)
	}
	m = update(t, m, key("l"))
	m = update(t, m, key("1"))
	if m.focus != paneStatus || m.lastLeft != paneStatus {
		t.Fatalf("1 should focus status from the diff pane, got %v", m.focus)
	}
}

func TestSelectionRequestsDiff(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("3"))
	c, _ := m.selectedCommit()
	if !strings.HasPrefix(m.diffKey, "commit:"+c.SHA) {
		t.Fatalf("diffKey = %q", m.diffKey)
	}
	seq := m.diffSeq
	m = update(t, m, key("j"))
	c2, _ := m.selectedCommit()
	if c2.SHA == c.SHA || m.diffSeq == seq || !m.diffLoading {
		t.Fatalf("moving selection should request a new diff (seq %d -> %d)", seq, m.diffSeq)
	}
	stale := diffMsg{seq: seq, key: "commit:" + c.SHA, text: "stale"}
	m = update(t, m, stale)
	if m.diff.text == "stale" {
		t.Fatal("stale diff result must be ignored")
	}
	req, _ := m.currentRequest()
	fresh := run(t, loadDiff(m.root, m.diffSeq, req)).(diffMsg)
	m = update(t, m, fresh)
	if m.diffLoading || !strings.Contains(stripANSI(m.View()), "initial commit") {
		t.Fatal("fresh commit diff should be shown")
	}
}

func TestBranchSelectionShowsLog(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("2"))
	req, ok := m.currentRequest()
	if !ok || req.kind != "log" || req.args[0] != "log" {
		t.Fatalf("branch request = %+v", req)
	}
	msg := run(t, loadDiff(m.root, m.diffSeq, req)).(diffMsg)
	m = update(t, m, msg)
	if !strings.Contains(stripANSI(m.View()), "expand readme") {
		t.Fatal("branch log not rendered")
	}
}

func TestUntrackedFileShowsContent(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("G"))
	e, _ := m.selectedEntry()
	if e.Code != "?" {
		t.Fatalf("last entry should be untracked, got %+v", e)
	}
	req, _ := m.currentRequest()
	msg := run(t, loadDiff(m.root, m.diffSeq, req)).(diffMsg)
	m = update(t, m, msg)
	if !strings.Contains(stripANSI(m.View()), "new file") {
		t.Fatal("untracked file content not rendered")
	}
}

func TestFuzzyFilter(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("/"))
	if !m.filterEditing {
		t.Fatal("/ should start filter editing")
	}
	for _, r := range "untr" {
		m = update(t, m, key(string(r)))
	}
	if n := len(m.visible(paneStatus)); n != 1 {
		t.Fatalf("filter should leave 1 entry, got %d", n)
	}
	e, _ := m.selectedEntry()
	if e.Path != "untracked.txt" {
		t.Fatalf("selected %+v", e)
	}
	checkFrame(t, m.View(), 120, 40)
	m = update(t, m, key("enter"))
	if m.filterEditing || m.lists[paneStatus].filter != "untr" {
		t.Fatal("enter should keep the filter and stop editing")
	}
	m = update(t, m, key("esc"))
	if m.lists[paneStatus].filter != "" || len(m.visible(paneStatus)) != 3 {
		t.Fatal("esc should clear the filter")
	}
}

func TestCommitFormFlow(t *testing.T) {
	m := loaded(t, 120, 40)
	m, cmd := updateCmd(t, m, key("c"))
	if m.modal != modalCommit || m.form == nil || cmd == nil {
		t.Fatal("c should open the commit form")
	}
	checkFrame(t, m.View(), 120, 40)
	if !strings.Contains(stripANSI(m.View()), "Commit message") {
		t.Fatal("form not rendered")
	}
	m, cmd = updateCmd(t, m, key("q"))
	if m.modal != modalCommit || cmd == nil {
		t.Fatal("q inside a form must type, not quit")
	}
	m = update(t, m, key("esc"))
	if m.modal != modalNone || m.form != nil {
		t.Fatal("esc should close the form")
	}
	m, cmd = updateCmd(t, m, key("ctrl+c"))
	if cmd == nil {
		t.Fatal("ctrl+c should quit once the form is closed")
	}
}

func TestCommitFormCompletion(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = updateCmd(t, m, key("c"))
	m.fv.title = "feat: from test"
	m.fv.yes = true
	m.form.State = huh.StateCompleted
	m, cmd := updateCmd(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if m.modal != modalNone {
		t.Fatal("form should close on completion")
	}
	if got := strings.Join(m.lastGit, " "); got != "commit -m feat: from test" {
		t.Fatalf("commit args = %q", got)
	}
	action, ok := firstAction(collect(cmd))
	if !ok || action.err != nil || len(action.out) < 7 {
		t.Fatalf("commit failed: %+v", action)
	}
	m = update(t, m, action)
	if m.toast == nil || !strings.HasPrefix(m.toast.text, "committed ") {
		t.Fatalf("toast = %+v", m.toast)
	}
	if !strings.Contains(git(t, m.root, "log", "--oneline", "-1"), "feat: from test") {
		t.Fatal("commit not created")
	}
}

func TestCommitWithNothingStaged(t *testing.T) {
	m := loaded(t, 120, 40)
	m.status.Entries = []statusEntry{{Path: "x", Code: "M"}}
	m, _ = updateCmd(t, m, key("c"))
	if m.modal != modalNone || m.toast == nil || m.toast.text != "nothing staged" || m.toast.ok {
		t.Fatalf("expected 'nothing staged' toast, got modal=%v toast=%+v", m.modal, m.toast)
	}
	checkFrame(t, m.View(), 120, 40)
}

func TestDiscardNeedsConfirm(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = m.setSel(paneStatus, 1)
	e, _ := m.selectedEntry()
	m, _ = updateCmd(t, m, key("d"))
	if m.modal != modalConfirm || m.confirm != confirmDiscard {
		t.Fatal("d should open the confirm dialog")
	}
	if !strings.Contains(stripANSI(m.View()), "Discard changes to "+e.Path) {
		t.Fatal("confirm question missing")
	}
	m.fv.yes = false
	m.form.State = huh.StateCompleted
	m, cmd := updateCmd(t, m, key("enter"))
	if m.modal != modalNone || cmd != nil || m.busy != 0 {
		t.Fatal("answering No must run nothing")
	}
	m, _ = updateCmd(t, m, key("d"))
	m.fv.yes = true
	m.form.State = huh.StateCompleted
	m, cmd = updateCmd(t, m, key("enter"))
	if got := strings.Join(m.lastGit, " "); got != "checkout -- "+e.Path {
		t.Fatalf("discard args = %q", got)
	}
	if a, ok := firstAction(collect(cmd)); !ok || a.err != nil {
		t.Fatalf("discard did not succeed: ok=%v err=%v", ok, a.err)
	}
	if strings.Contains(git(t, m.root, "status", "--porcelain"), " M "+e.Path) {
		t.Fatal("file still modified after discard")
	}
}

func TestBranchCheckoutAndCreate(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("2"))
	for i := 0; i < 3; i++ {
		if b, _ := m.selectedBranch(); !b.Current {
			break
		}
		m = update(t, m, key("j"))
	}
	b, _ := m.selectedBranch()
	if b.Current {
		t.Fatal("could not find a non-current branch")
	}
	m, _ = updateCmd(t, m, key("enter"))
	if got := strings.Join(m.lastGit, " "); got != "checkout "+b.Name {
		t.Fatalf("checkout args = %q", got)
	}
	m, _ = updateCmd(t, m, key("n"))
	if m.modal != modalBranch {
		t.Fatal("n should open the branch form")
	}
	m.fv.branch = "topic/new"
	m.form.State = huh.StateCompleted
	m, _ = updateCmd(t, m, key("enter"))
	if got := strings.Join(m.lastGit, " "); got != "checkout -b topic/new" {
		t.Fatalf("new branch args = %q", got)
	}
}

func TestGitErrorBecomesToast(t *testing.T) {
	m := loaded(t, 120, 40)
	msg := run(t, gitAction(m.root, "checkout", "", []string{"checkout", "does-not-exist"})).(actionMsg)
	if msg.err == nil {
		t.Fatal("expected an error from git")
	}
	m = update(t, m, msg)
	if m.toast == nil || m.toast.ok || !strings.Contains(m.toast.text, "checkout failed") {
		t.Fatalf("toast = %+v", m.toast)
	}
	view := m.View()
	checkFrame(t, view, 120, 40)
	if !strings.Contains(stripANSI(view), "checkout failed") {
		t.Fatal("toast not visible")
	}
	m = update(t, m, toastExpireMsg{id: m.toast.id})
	if m.toast != nil {
		t.Fatal("toast should expire")
	}
}

func TestStashAndPopArgs(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = updateCmd(t, m, key("s"))
	if m.modal != modalConfirm || m.confirm != confirmStash {
		t.Fatal("s should ask for confirmation")
	}
	m.fv.yes = true
	m.form.State = huh.StateCompleted
	m, cmd := updateCmd(t, m, key("enter"))
	if got := strings.Join(m.lastGit, " "); got != "stash push --include-untracked" {
		t.Fatalf("stash args = %q", got)
	}
	for _, x := range collect(cmd) {
		if a, ok := x.(actionMsg); ok && a.err != nil {
			t.Fatal(a.err)
		}
	}
	ss := run(t, loadStashes(m.root)).(stashesMsg)
	if len(ss.stashes) != 1 {
		t.Fatalf("expected one stash, got %+v", ss.stashes)
	}
	m = update(t, m, ss)
	m = update(t, m, key("4"))
	m, _ = updateCmd(t, m, key("enter"))
	if got := strings.Join(m.lastGit, " "); got != "stash pop "+ss.stashes[0].Ref {
		t.Fatalf("pop args = %q", got)
	}
}

func TestCopyShaCommand(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("3"))
	_, cmd := updateCmd(t, m, key("y"))
	if cmd == nil {
		t.Fatal("y should produce a clipboard command")
	}
	c, _ := m.selectedCommit()
	m = update(t, m, copiedMsg{sha: c.Short})
	if m.toast == nil || m.toast.text != "copied "+c.Short {
		t.Fatalf("toast = %+v", m.toast)
	}
}

func TestMouseClickSelectsAndFocuses(t *testing.T) {
	m := loaded(t, 120, 40)
	l := m.layout()
	click := func(x, y int) {
		m = update(t, m, tea.MouseMsg{X: x, Y: y, Action: tea.MouseActionPress, Button: tea.MouseButtonLeft})
	}
	click(l.leftW+5, 5)
	if m.focus != paneDiff {
		t.Fatalf("click on right pane should focus diff, got %v", m.focus)
	}
	rows, _ := m.rows(paneStatus, l.leftW-2)
	target := -1
	for i, r := range rows {
		if r.idx == 2 {
			target = i
		}
	}
	click(3, l.paneY[paneStatus]+1+target)
	if m.focus != paneStatus || m.lists[paneStatus].sel != 2 {
		t.Fatalf("click should focus status and select row 2, got focus=%v sel=%d", m.focus, m.lists[paneStatus].sel)
	}
	click(3, l.paneY[paneCommits]+1)
	if m.focus != paneCommits || m.lastLeft != paneCommits {
		t.Fatalf("click on collapsed pane should expand it, got %v", m.focus)
	}
	m = m.snapHeights()
	m = update(t, m, tea.MouseMsg{X: 3, Y: m.layout().paneY[paneCommits] + 1, Action: tea.MouseActionPress, Button: tea.MouseButtonWheelDown})
	if m.lists[paneCommits].sel != 1 {
		t.Fatalf("wheel should move commit selection, got %d", m.lists[paneCommits].sel)
	}
	before := m.vp.YOffset
	m = update(t, m, tea.MouseMsg{X: l.leftW + 3, Y: 5, Action: tea.MouseActionPress, Button: tea.MouseButtonWheelDown})
	if m.vp.YOffset < before {
		t.Fatal("wheel over diff should not scroll upward")
	}
}

func TestDiffPaneScrollKeys(t *testing.T) {
	m := loaded(t, 80, 24)
	var b strings.Builder
	for i := 0; i < 200; i++ {
		b.WriteString("+line\n")
	}
	m = update(t, m, diffMsg{seq: m.diffSeq, key: m.diffKey, kind: "diff", text: b.String()})
	m = update(t, m, key("l"))
	if m.focus != paneDiff {
		t.Fatal("l should focus the diff pane")
	}
	m = update(t, m, key("j"))
	if m.vp.YOffset != 1 {
		t.Fatalf("j should scroll one line, got %d", m.vp.YOffset)
	}
	m = update(t, m, key("G"))
	if !m.vp.AtBottom() {
		t.Fatal("G should reach the bottom")
	}
	m = update(t, m, key("g"))
	if m.vp.YOffset != 0 {
		t.Fatal("g should return to top")
	}
	m = update(t, m, tea.KeyMsg{Type: tea.KeyCtrlD})
	if m.vp.YOffset == 0 {
		t.Fatal("ctrl+d should page down")
	}
	checkFrame(t, m.View(), 80, 24)
}

func TestAutoRefreshPausesDuringModal(t *testing.T) {
	m := loaded(t, 120, 40)
	m.lastRefresh = time.Now().Add(-10 * time.Second)
	m, _ = updateCmd(t, m, key("c"))
	before := m.lastRefresh
	m = update(t, m, clockMsg(time.Now()))
	if !m.lastRefresh.Equal(before) {
		t.Fatal("auto-refresh must pause while a form is open")
	}
	m = update(t, m, key("esc"))
	m = update(t, m, clockMsg(time.Now()))
	if m.lastRefresh.Equal(before) {
		t.Fatal("auto-refresh should resume after the form closes")
	}
}

func TestEmptyRepoAndDetachedHead(t *testing.T) {
	dir := t.TempDir()
	git(t, dir, "init", "-q", "-b", "main")
	m := newModel(dir, dir)
	m = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 24})
	m = update(t, m, run(t, loadStatus(dir)))
	m = update(t, m, run(t, loadBranches(dir)))
	m = update(t, m, run(t, loadCommits(dir)))
	m = update(t, m, run(t, loadStashes(dir)))
	if !m.status.Unborn {
		t.Fatal("fresh repo should be unborn")
	}
	view := m.View()
	checkFrame(t, view, 80, 24)
	if !strings.Contains(stripANSI(view), "no commits") {
		t.Fatal("unborn state not shown")
	}
	for _, p := range []paneID{paneBranches, paneCommits, paneStash, paneDiff} {
		m, _ = m.focusPane(p)
		m = m.snapHeights()
		checkFrame(t, m.View(), 80, 24)
	}

	root := tempRepo(t)
	git(t, root, "checkout", "-q", "HEAD~1")
	d := newModel(root, root)
	d = update(t, d, tea.WindowSizeMsg{Width: 80, Height: 24})
	d = update(t, d, run(t, loadStatus(root)))
	d = update(t, d, run(t, loadBranches(root)))
	if !d.status.Detached {
		t.Fatal("expected detached HEAD")
	}
	view = d.View()
	checkFrame(t, view, 80, 24)
	if !strings.Contains(stripANSI(view), "HEAD detached") {
		t.Fatal("detached state not shown in header")
	}
}

func TestTooSmallTerminal(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, tea.WindowSizeMsg{Width: 40, Height: 10})
	view := m.View()
	checkFrame(t, view, 40, 10)
	if !strings.Contains(stripANSI(view), "needs at least") {
		t.Fatal("too-small message missing")
	}
}

func TestAnsiSliceAndTruncation(t *testing.T) {
	styled := stGold.Render("héllo") + " " + stError.Render("wörld")
	if got := stripANSI(ansiSlice(styled, 0, 5)); got != "héllo" {
		t.Fatalf("ansiSlice head = %q", got)
	}
	if got := stripANSI(ansiSlice(styled, 6, 11)); got != "wörld" {
		t.Fatalf("ansiSlice tail = %q", got)
	}
	if got := truncPlain("日本語テキスト", 7); runewidth.StringWidth(got) > 7 {
		t.Fatalf("wide-char truncation overflowed: %q", got)
	}
	if got := fitLine("abc", 6); got != "abc   " {
		t.Fatalf("fitLine = %q", got)
	}
	if got := shortAge("3 hours ago"); got != "3h" {
		t.Fatalf("shortAge = %q", got)
	}
}

// collect runs a command and flattens any tea.Batch result into messages.
func collect(cmd tea.Cmd) []tea.Msg {
	if cmd == nil {
		return nil
	}
	switch v := cmd().(type) {
	case tea.BatchMsg:
		var out []tea.Msg
		for _, c := range v {
			out = append(out, collect(c)...)
		}
		return out
	case nil:
		return nil
	default:
		return []tea.Msg{v}
	}
}

func firstAction(msgs []tea.Msg) (actionMsg, bool) {
	for _, msg := range msgs {
		if a, ok := msg.(actionMsg); ok {
			return a, true
		}
	}
	return actionMsg{}, false
}
