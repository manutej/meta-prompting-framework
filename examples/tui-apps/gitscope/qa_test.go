package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
)

// Adversarial QA tests: layout invariants under every overlay and size, git
// edge cases, state-machine abuse, concurrency ordering and error surfacing.

var qaSizes = [][2]int{{80, 24}, {100, 30}, {120, 40}, {200, 60}, {79, 23}, {90, 25}, {140, 35}, {60, 20}, {40, 12}}

// longPath is a 250-cell mix of ASCII, accented Latin and CJK.
var longPath = func() string {
	var b strings.Builder
	for b.Len() < 400 {
		b.WriteString("dir/sübdir/日本語テキスト/")
	}
	r := []rune(b.String())
	return string(r[:250])
}()

func bigDiff(n int) string {
	var b strings.Builder
	b.WriteString("diff --git a/big.txt b/big.txt\n--- a/big.txt\n+++ b/big.txt\n@@ -1,1000 +1,1000 @@\n")
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			fmt.Fprintf(&b, "+line %d %s\n", i, longPath)
		} else {
			fmt.Fprintf(&b, "-line %d\n", i)
		}
	}
	return b.String()
}

// checkChrome is checkFrame plus: when the layout is usable the header must
// be on the first line and the status bar on the last one (bubbletea drops
// the TOP lines of an over-tall frame, so any overflow would eat the header).
func checkChrome(t *testing.T, m model, what string) {
	t.Helper()
	view := m.View()
	checkFrame(t, view, m.width, m.height)
	l := m.layout()
	if !l.ok {
		return
	}
	sum := 0
	for _, h := range l.paneH {
		sum += h
	}
	if sum != l.bodyH {
		t.Fatalf("%s at %dx%d: pane heights %v sum %d != body %d", what, m.width, m.height, l.paneH, sum, l.bodyH)
	}
	lines := strings.Split(view, "\n")
	if !strings.Contains(stripANSI(lines[0]), "gitscope") {
		t.Fatalf("%s at %dx%d: header missing from first line: %q", what, m.width, m.height, stripANSI(lines[0]))
	}
	if m.modal == modalNone && !m.showHelp && !strings.Contains(stripANSI(lines[len(lines)-1]), "q quit") {
		t.Fatalf("%s at %dx%d: status bar missing from last line: %q", what, m.width, m.height, stripANSI(lines[len(lines)-1]))
	}
}

func heavyModel(t *testing.T, w, h int) model {
	t.Helper()
	m := loaded(t, w, h)
	m.status.Entries = append([]statusEntry{{Path: longPath, Code: "M", Staged: true}}, m.status.Entries...)
	m.lists[paneStatus].sel = 0
	m.branches = append(m.branches, branch{Name: "feature/ünïcode/日本", Short: "abc1234", Subject: "feat: 日本語 subject"})
	m.commits = append(m.commits, commit{SHA: "0123456789abcdef", Short: "0123456", Author: "誰か", Age: "2mo", Subject: ""})
	m.stashes = append(m.stashes, stash{Ref: "stash@{0}", Message: "WIP on main: " + longPath})
	m, _ = m.selectChanged(true, true)
	m = update(t, m, diffMsg{seq: m.diffSeq, key: m.diffKey, kind: "diff", title: "Staged · " + longPath, text: bigDiff(2000)})
	return m
}

func TestLayoutMatrix(t *testing.T) {
	for _, sz := range qaSizes {
		w, h := sz[0], sz[1]
		m := heavyModel(t, w, h)
		checkChrome(t, m, "initial")
		for _, p := range []paneID{paneBranches, paneCommits, paneStash, paneDiff, paneStatus} {
			m, _ = m.focusPane(p)
			for i := 0; i < 5; i++ { // mid-animation frames
				m = update(t, m, animMsg{})
				checkChrome(t, m, fmt.Sprintf("animating->%v frame %d", p, i))
			}
			m = m.snapHeights()
			checkChrome(t, m, fmt.Sprintf("focus %v", p))
			m = update(t, m, key("j"))
			m = update(t, m, key("G"))
			checkChrome(t, m, fmt.Sprintf("focus %v at end", p))
		}
		m = update(t, m, key("1"))
		m = m.snapHeights()
		m.diff = diffMsg{seq: m.diffSeq, key: m.diffKey, kind: "diff", title: "Staged · " + longPath, text: bigDiff(2000)}
		m = m.rerenderDiff()

		// toast
		mm, _ := m.showToast("stage failed: "+longPath, false)
		m = mm.(model)
		checkChrome(t, m, "toast")
		m.toast = nil

		// help
		m = update(t, m, key("?"))
		checkChrome(t, m, "help")
		m = update(t, m, key("?"))

		// forms
		m = update(t, m, key("c"))
		if m.layout().ok && m.modal != modalCommit {
			t.Fatalf("commit form did not open at %dx%d", w, h)
		}
		checkChrome(t, m, "commit form")
		m = update(t, m, key("x"))
		checkChrome(t, m, "commit form typed")
		m = update(t, m, key("esc"))
		m = update(t, m, key("d"))
		checkChrome(t, m, "confirm form")
		m = update(t, m, key("esc"))
		m = update(t, m, key("s"))
		checkChrome(t, m, "stash confirm form")
		m = update(t, m, key("esc"))
		m = update(t, m, key("2"))
		m = m.snapHeights()
		m = update(t, m, key("n"))
		checkChrome(t, m, "branch form")
		m = update(t, m, key("esc"))

		// filter editing with a long pattern
		m = update(t, m, key("/"))
		for _, r := range "日本語テキストfilterpattern" {
			m = update(t, m, key(string(r)))
		}
		checkChrome(t, m, "filter editing")
		m = update(t, m, key("esc"))
	}
}

// Switching panes again before the spring settles puts three panes in motion;
// the rounded heights must still tile the body exactly on every frame.
func TestAccordionThreeWayAnimationTilesBody(t *testing.T) {
	for _, sz := range [][2]int{{80, 24}, {100, 30}, {120, 40}, {200, 60}, {79, 23}, {60, 20}} {
		base := loaded(t, sz[0], sz[1])
		for k := 0; k < 40; k++ {
			m := base
			m = update(t, m, key("2"))
			for i := 0; i < k && m.animating; i++ {
				m = update(t, m, animMsg{})
			}
			m = update(t, m, key("3"))
			for i := 0; i < 300 && m.animating; i++ {
				m = update(t, m, animMsg{})
				checkChrome(t, m, fmt.Sprintf("3-way k=%d frame %d", k, i))
			}
			if m.animating {
				t.Fatalf("spring never settled (k=%d)", k)
			}
		}
	}
}

func TestAnimTickIsNilWhenIdle(t *testing.T) {
	m := loaded(t, 120, 40)
	if _, cmd := m.Update(animMsg{}); cmd != nil {
		t.Fatal("anim tick while idle must return a nil cmd")
	}
	m, cmd := updateCmd(t, m, key("3"))
	if cmd == nil || !m.animating {
		t.Fatal("focus change should start the animation")
	}
	var last tea.Cmd
	for i := 0; i < 300 && m.animating; i++ {
		m, last = updateCmd(t, m, animMsg{})
	}
	if m.animating || last != nil {
		t.Fatalf("after settling the tick must stop: animating=%v cmd=%v", m.animating, last != nil)
	}
	if _, cmd := m.Update(animMsg{}); cmd != nil {
		t.Fatal("a late tick after settling must not restart the loop")
	}
	// resize mid-animation snaps; the pending tick must then die.
	m, _ = updateCmd(t, m, key("2"))
	m = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	if _, cmd := m.Update(animMsg{}); cmd != nil || m.animating {
		t.Fatal("resize should snap and stop the animation")
	}
}

// A title whose grapheme width (runewidth) is smaller than its per-rune width
// (lipgloss) must not drive strings.Repeat negative.
func TestRenderPaneGraphemeClusterTitle(t *testing.T) {
	family := strings.Repeat("👨‍👩‍👧", 12) // ZWJ sequences
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("renderPane panicked: %v", r)
		}
	}()
	out := renderPane("Staged · "+family+".txt", []string{"x"}, 52, 5, true)
	for _, l := range strings.Split(out, "\n") {
		if lw := len([]rune(stripANSI(l))); lw > 400 {
			t.Fatalf("absurd line: %q", l)
		}
	}
	m := loaded(t, 80, 24)
	m.status.Entries = []statusEntry{{Path: family + ".txt", Code: "M", Staged: true}}
	m.diff = diffMsg{key: "x", kind: "diff", title: "Staged · " + family + ".txt", text: "+" + family + "\n"}
	m = m.rerenderDiff()
	checkFrame(t, m.View(), 80, 24)
}

// ---------------------------------------------------------------- git edge cases

func TestParseStatusQuotedPaths(t *testing.T) {
	out := "# branch.oid abc\n# branch.head main\n" +
		"1 M. N... 100644 100644 100644 a b \"q\\\"uote.txt\"\n" +
		"2 R. N... 100644 100644 100644 a b R100 \"renamed \\346\\227\\245.txt\"\t\"\\346\\227\\245\\346\\234\\254.txt\"\n" +
		"1 .M N... 100644 100644 100644 a b a b.txt\n" +
		"? \"new \\\"un.txt\"\n" +
		"? \"tab\\there.txt\"\n"
	st := parseStatus(out)
	want := []statusEntry{
		{Path: `q"uote.txt`, Code: "M", Staged: true},
		{Path: "renamed 日.txt", OrigPath: "日本.txt", Code: "R", Staged: true},
		{Path: "a b.txt", Code: "M"},
		{Path: `new "un.txt`, Code: "?"},
		{Path: "tab\there.txt", Code: "?"},
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

func TestParseStatusUnmergedAndSubmoduleRecords(t *testing.T) {
	out := "# branch.oid abc\n# branch.head main\n" +
		"u UU N... 100644 100644 100644 100644 a b c conflict.txt\n" +
		"1 .M S.M. 160000 160000 160000 a b sub\n" +
		"? sub/\n" +
		"u AA\n" + // truncated garbage
		"1 M.\n" +
		"2 R.\n" +
		"?\n" +
		"x\n"
	st := parseStatus(out)
	if len(st.Entries) != 3 || st.Entries[0].Code != "U" || st.Entries[0].Path != "conflict.txt" || st.Entries[1].Path != "sub" {
		t.Fatalf("entries = %+v", st.Entries)
	}
}

func TestWeirdFilenamesRoundTrip(t *testing.T) {
	dir := t.TempDir()
	git(t, dir, "init", "-q", "-b", "main")
	names := []string{"a b.txt", `q"uote.txt`, "日本語.txt", `back\slash.txt`}
	for _, n := range names {
		write(t, dir, n, "x\n")
	}
	git(t, dir, "add", "-A")
	git(t, dir, "commit", "-q", "-m", "init")
	for _, n := range names {
		write(t, dir, n, "y\n")
	}
	write(t, dir, `new "untracked 語.txt`, "u\n")
	m := newModel(dir, dir)
	m = update(t, m, tea.WindowSizeMsg{Width: 120, Height: 40})
	st := run(t, loadStatus(dir)).(statusMsg)
	if st.err != nil {
		t.Fatal(st.err)
	}
	m = update(t, m, st)
	seen := map[string]bool{}
	for _, e := range m.status.Entries {
		seen[e.Path] = true
		if strings.HasPrefix(e.Path, `"`) || strings.Contains(e.Path, `\3`) {
			t.Fatalf("path not unquoted: %q", e.Path)
		}
		full := filepath.Join(dir, e.Path)
		if _, err := os.Stat(full); err != nil {
			t.Fatalf("entry path %q does not exist on disk: %v", e.Path, err)
		}
	}
	for _, n := range append(names, `new "untracked 語.txt`) {
		if !seen[n] {
			t.Fatalf("%q missing from status: %+v", n, m.status.Entries)
		}
	}
	// stage each file through the UI and make sure git accepts the path.
	for i := range m.visible(paneStatus) {
		m, _ = m.setSel(paneStatus, i)
		e, _ := m.selectedEntry()
		if e.Staged {
			continue
		}
		m, cmd := updateCmd(t, m, key("space"))
		a, ok := firstAction(collect(cmd))
		if !ok || a.err != nil {
			t.Fatalf("staging %q: ok=%v err=%v", e.Path, ok, a.err)
		}
		req, _ := m.currentRequest()
		d := run(t, loadDiff(dir, m.diffSeq, req)).(diffMsg)
		if d.err != nil {
			t.Fatalf("diff for %q: %v", e.Path, d.err)
		}
	}
	st = run(t, loadStatus(dir)).(statusMsg)
	for _, e := range st.st.Entries {
		if !e.Staged {
			t.Fatalf("%q still unstaged after staging everything: %+v", e.Path, st.st.Entries)
		}
	}
	m = update(t, m, st)
	checkChrome(t, m, "weird names")
}

func TestRenameIsParsedFromRealRepo(t *testing.T) {
	root := tempRepo(t)
	if err := os.Mkdir(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	git(t, root, "mv", "README.md", "docs/README 2.md")
	st := run(t, loadStatus(root)).(statusMsg)
	var found *statusEntry
	for i := range st.st.Entries {
		if st.st.Entries[i].Code == "R" {
			found = &st.st.Entries[i]
		}
	}
	if found == nil || found.Path != "docs/README 2.md" || found.OrigPath != "README.md" {
		t.Fatalf("rename entry = %+v (all: %+v)", found, st.st.Entries)
	}
	m := newModel(root, root)
	m = update(t, m, tea.WindowSizeMsg{Width: 120, Height: 40})
	m = update(t, m, st)
	for i, e := range m.status.Entries {
		if e.Code == "R" {
			m, _ = m.setSel(paneStatus, i)
		}
	}
	req, _ := m.currentRequest()
	d := run(t, loadDiff(root, m.diffSeq, req)).(diffMsg)
	if d.err != nil {
		t.Fatalf("rename diff: %v", d.err)
	}
	m = update(t, m, d)
	if !strings.Contains(stripANSI(m.View()), "README.md → docs/README 2.md") {
		t.Fatal("rename row not rendered")
	}
}

func TestDetachedHeadBranchList(t *testing.T) {
	root := tempRepo(t)
	git(t, root, "checkout", "-q", "HEAD~1")
	bs := run(t, loadBranches(root)).(branchesMsg)
	if bs.err != nil {
		t.Fatal(bs.err)
	}
	for _, b := range bs.branches {
		if strings.HasPrefix(b.Name, "(") {
			t.Fatalf("pseudo branch leaked into the list: %+v", bs.branches)
		}
		if b.Current {
			t.Fatalf("no real branch is current when detached: %+v", bs.branches)
		}
	}
	if len(bs.branches) != 2 {
		t.Fatalf("branches = %+v", bs.branches)
	}
	m := newModel(root, root)
	m = update(t, m, tea.WindowSizeMsg{Width: 120, Height: 40})
	m = update(t, m, run(t, loadStatus(root)))
	m = update(t, m, bs)
	m = update(t, m, key("2"))
	req, ok := m.currentRequest()
	if !ok {
		t.Fatal("expected a branch request")
	}
	d := run(t, loadDiff(root, m.diffSeq, req)).(diffMsg)
	if d.err != nil {
		t.Fatalf("branch log failed: %v", d.err)
	}
}

func TestUnbornRepoActions(t *testing.T) {
	dir := t.TempDir()
	git(t, dir, "init", "-q", "-b", "main")
	write(t, dir, "first.txt", "hello\n")
	write(t, dir, "second.txt", "hello\n")
	git(t, dir, "add", "first.txt")
	m := newModel(dir, dir)
	m = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	m = update(t, m, run(t, loadStatus(dir)))
	m = update(t, m, run(t, loadBranches(dir)))
	m = update(t, m, run(t, loadCommits(dir)))
	m = update(t, m, run(t, loadStashes(dir)))
	if !m.status.Unborn || len(m.status.Entries) != 2 || !m.status.Entries[0].Staged {
		t.Fatalf("status = %+v", m.status)
	}
	checkChrome(t, m, "unborn")
	// diff of a staged file in an unborn repo
	req, _ := m.currentRequest()
	d := run(t, loadDiff(dir, m.diffSeq, req)).(diffMsg)
	if d.err != nil || !strings.Contains(d.text, "hello") {
		t.Fatalf("staged diff on unborn: err=%v text=%q", d.err, d.text)
	}
	// unstage via space
	m, cmd := updateCmd(t, m, key("space"))
	a, ok := firstAction(collect(cmd))
	if !ok || a.err != nil {
		t.Fatalf("unstage on unborn: %+v", a)
	}
	m = update(t, m, a)
	git(t, dir, "add", "first.txt")
	m = update(t, m, run(t, loadStatus(dir)))
	// discard the staged file via d + Yes
	m, _ = m.setSel(paneStatus, 0)
	e, _ := m.selectedEntry()
	if !e.Staged {
		t.Fatalf("expected staged entry, got %+v", e)
	}
	m, _ = updateCmd(t, m, key("d"))
	m.fv.yes = true
	m.form.State = huh.StateCompleted
	m, cmd = updateCmd(t, m, key("enter"))
	a, ok = firstAction(collect(cmd))
	if !ok || a.err != nil {
		t.Fatalf("discard of a staged file in an unborn repo failed: ok=%v err=%v args=%v", ok, a.err, m.lastGit)
	}
	m = update(t, m, a)
	// commit form and commit on unborn
	git(t, dir, "add", "-A")
	m = update(t, m, run(t, loadStatus(dir)))
	m, _ = updateCmd(t, m, key("c"))
	if m.modal != modalCommit {
		t.Fatal("commit form should open")
	}
	m.fv.title = "first"
	m.fv.yes = true
	m.form.State = huh.StateCompleted
	m, cmd = updateCmd(t, m, key("enter"))
	a, ok = firstAction(collect(cmd))
	if !ok || a.err != nil {
		t.Fatalf("first commit: %+v", a)
	}
	m = update(t, m, a)
	if m.busy != 0 || m.toast == nil || !m.toast.ok {
		t.Fatalf("after commit: busy=%d toast=%+v", m.busy, m.toast)
	}
	// branches/commits/stash are empty: enter must be inert
	m.branches, m.commits, m.stashes = nil, nil, nil
	for _, k := range []string{"2", "3", "4"} {
		m = update(t, m, key(k))
		m = m.snapHeights()
		mm, cmd := m.Update(key("enter"))
		m = mm.(model)
		if cmd != nil || m.busy != 0 {
			t.Fatalf("enter on empty pane %s ran something", k)
		}
		checkChrome(t, m, "empty pane "+k)
	}
}

func TestBinaryDiffAndEmptySubject(t *testing.T) {
	root := tempRepo(t)
	if err := os.WriteFile(filepath.Join(root, "blob.bin"), []byte{0, 1, 2, 3, 0xff}, 0o644); err != nil {
		t.Fatal(err)
	}
	git(t, root, "add", "blob.bin")
	git(t, root, "commit", "-q", "--allow-empty-message", "-m", "")
	if err := os.WriteFile(filepath.Join(root, "blob.bin"), []byte{0, 9, 9, 9, 0xfe}, 0o644); err != nil {
		t.Fatal(err)
	}
	m := newModel(root, root)
	m = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	m = update(t, m, run(t, loadStatus(root)))
	m = update(t, m, run(t, loadCommits(root)))
	if m.commits[0].Subject != "" {
		t.Fatalf("expected empty subject, got %+v", m.commits[0])
	}
	for i, e := range m.status.Entries {
		if e.Path == "blob.bin" {
			m, _ = m.setSel(paneStatus, i)
		}
	}
	req, _ := m.currentRequest()
	d := run(t, loadDiff(root, m.diffSeq, req)).(diffMsg)
	m = update(t, m, d)
	if !strings.Contains(stripANSI(m.View()), "Binary files") {
		t.Fatalf("binary diff not shown: %q", d.text)
	}
	m = update(t, m, key("3"))
	m = m.snapHeights()
	checkChrome(t, m, "empty subject commit")
	req, _ = m.currentRequest()
	d = run(t, loadDiff(root, m.diffSeq, req)).(diffMsg)
	m = update(t, m, d)
	checkChrome(t, m, "empty subject commit diff")
}

func TestNoUpstreamHidesAheadBehind(t *testing.T) {
	m := loaded(t, 100, 30)
	if m.status.Upstream != "" || strings.Contains(stripANSI(m.View()), "↑") {
		t.Fatal("no upstream: ahead/behind should be hidden")
	}
}

// ---------------------------------------------------------------- state machine

func TestKeysDuringInFlightActions(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = m.setSel(paneStatus, 1) // the unstaged modification
	m, c1 := updateCmd(t, m, key("space"))
	m, c2 := updateCmd(t, m, key("space"))
	if m.busy != 2 {
		t.Fatalf("busy = %d", m.busy)
	}
	a1, _ := firstAction(collect(c1))
	a2, _ := firstAction(collect(c2))
	if a1.err != nil || a2.err != nil {
		t.Fatalf("double stage: %v / %v", a1.err, a2.err)
	}
	m = update(t, m, a1)
	m = update(t, m, a2)
	if m.busy != 0 {
		t.Fatalf("busy after both results = %d", m.busy)
	}
	checkChrome(t, m, "after double space")
	m = update(t, m, run(t, loadStatus(m.root)))
	if !m.hasEntries(true) {
		t.Fatal("expected staged entries")
	}
	m, _ = updateCmd(t, m, key("c"))
	m, cmd := updateCmd(t, m, key("c"))
	if m.modal != modalCommit || cmd == nil {
		t.Fatal("second c must go to the form, not reopen it")
	}
	if m.fv.title != "c" {
		t.Fatalf("second c should be typed into the title, got %q", m.fv.title)
	}
	// keys that would act on panes must not leak while a form is open
	for _, k := range []string{"j", "a", "A", "d", "s", "1", "2", "tab", "space", "/", "?", "r"} {
		m = update(t, m, key(k))
		if m.modal != modalCommit || m.showHelp || m.filterEditing || m.focus != paneStatus || m.busy != 0 {
			t.Fatalf("key %q leaked out of the form", k)
		}
	}
	m, cmd = updateCmd(t, m, key("q"))
	if m.modal != modalCommit {
		t.Fatal("q must not quit inside a form")
	}
	if msgs := collect(cmd); len(msgs) > 0 {
		for _, x := range msgs {
			if _, isQuit := x.(tea.QuitMsg); isQuit {
				t.Fatal("q inside form produced QuitMsg")
			}
		}
	}
	m, cmd = updateCmd(t, m, key("ctrl+c"))
	for _, x := range collect(cmd) {
		if _, isQuit := x.(tea.QuitMsg); isQuit {
			t.Fatal("ctrl+c inside a form should abort the form, not the app")
		}
	}
	if m.modal != modalNone {
		t.Fatal("ctrl+c should abort the form")
	}
}

func TestFilterToZeroThenAct(t *testing.T) {
	m := loaded(t, 120, 40)
	m = update(t, m, key("/"))
	for _, r := range "zzzzqq" {
		m = update(t, m, key(string(r)))
	}
	if len(m.visible(paneStatus)) != 0 {
		t.Fatal("filter should match nothing")
	}
	checkChrome(t, m, "zero matches editing")
	m = update(t, m, key("enter"))
	for _, k := range []string{"enter", "space", "d", "j", "k", "G", "g", "y"} {
		mm, cmd := m.Update(key(k))
		m = mm.(model)
		if m.busy != 0 || m.modal != modalNone {
			t.Fatalf("key %q acted on an empty filtered list", k)
		}
		_ = cmd
	}
	checkChrome(t, m, "zero matches")
	if !strings.Contains(stripANSI(m.View()), "No files match") {
		t.Fatal("empty-filter message missing")
	}
	// refresh replaces the list while the filter is active: cursor must clamp
	m.lists[paneStatus].filter = "t"
	m.lists[paneStatus].sel = 2
	st := run(t, loadStatus(m.root)).(statusMsg)
	st.st.Entries = st.st.Entries[:1]
	m = update(t, m, st)
	if n := len(m.visible(paneStatus)); m.lists[paneStatus].sel >= n && n > 0 {
		t.Fatalf("cursor %d not clamped to %d", m.lists[paneStatus].sel, n)
	}
	checkChrome(t, m, "refresh with filter")
	// same for branches
	m = update(t, m, key("2"))
	m = m.snapHeights()
	m.lists[paneBranches].filter = "x"
	m.lists[paneBranches].sel = 5
	m = update(t, m, branchesMsg{branches: []branch{{Name: "main", Current: true}}})
	if _, ok := m.selectedBranch(); ok {
		t.Fatal("filter x should match nothing and selection must be invalid, not out of range")
	}
	mm, cmd := m.Update(key("enter"))
	m = mm.(model)
	if cmd != nil {
		t.Fatal("enter on a filtered-empty branch list must be inert")
	}
	checkChrome(t, m, "branches empty filter")
}

func TestMouseEdgeCases(t *testing.T) {
	m := loaded(t, 120, 40)
	press := func(x, y int, b tea.MouseButton) {
		mm, _ := m.Update(tea.MouseMsg{X: x, Y: y, Action: tea.MouseActionPress, Button: b})
		m = mm.(model)
	}
	before := m
	press(5, 0, tea.MouseButtonLeft)                  // header
	press(5, 39, tea.MouseButtonLeft)                 // status bar
	press(200, 200, tea.MouseButtonLeft)              // outside
	press(0, m.height-2, tea.MouseButtonLeft)         // bottom border
	press(m.layout().leftW-1, 1, tea.MouseButtonLeft) // top border corner
	if m.focus != before.focus || m.lists != before.lists {
		t.Fatal("clicks outside content changed state")
	}
	// wheel over the diff pane before any diff has loaded
	m.vp.SetContent("")
	press(m.layout().leftW+3, 5, tea.MouseButtonWheelDown)
	press(m.layout().leftW+3, 5, tea.MouseButtonWheelUp)
	checkChrome(t, m, "wheel empty diff")
	// click on a collapsed pane's summary line and its borders
	l := m.layout()
	press(2, l.paneY[paneStash], tea.MouseButtonLeft) // border of collapsed stash
	press(2, l.paneY[paneStash]+1, tea.MouseButtonLeft)
	if m.lastLeft != paneStash {
		t.Fatalf("click on collapsed pane should expand it, lastLeft=%v", m.lastLeft)
	}
	for i := 0; i < 3; i++ {
		m = update(t, m, animMsg{})
		press(2, 5, tea.MouseButtonLeft) // click during animation
		press(2, 30, tea.MouseButtonWheelDown)
		checkChrome(t, m, "mouse mid-animation")
	}
	m = update(t, m, key("4"))
	m = m.snapHeights()
	// wheel on a collapsed pane does nothing
	l = m.layout()
	sel := m.lists[paneStatus].sel
	press(2, l.paneY[paneStatus]+1, tea.MouseButtonWheelDown)
	if m.lists[paneStatus].sel != sel {
		t.Fatal("wheel on collapsed pane moved its selection")
	}
	// motion/release events are ignored
	mm, cmd := m.Update(tea.MouseMsg{X: 2, Y: 5, Action: tea.MouseActionMotion})
	if cmd != nil {
		t.Fatal("motion produced a command")
	}
	m = mm.(model)
	// too-small terminal: mouse must be inert
	m = update(t, m, tea.WindowSizeMsg{Width: 40, Height: 10})
	press(3, 3, tea.MouseButtonLeft)
	checkFrame(t, m.View(), 40, 10)
}

// ---------------------------------------------------------------- concurrency

func TestRefreshResultsDuringForm(t *testing.T) {
	m := loaded(t, 120, 40)
	m, _ = updateCmd(t, m, key("c"))
	m = update(t, m, key("h"))
	m = update(t, m, key("i"))
	form := m.form
	m.lastRefresh = time.Now().Add(-time.Minute)
	m = update(t, m, clockMsg(time.Now()))
	m = update(t, m, run(t, loadStatus(m.root)))
	m = update(t, m, run(t, loadBranches(m.root)))
	m = update(t, m, run(t, loadCommits(m.root)))
	m = update(t, m, run(t, loadStashes(m.root)))
	m = update(t, m, spinnerTick(m))
	if m.modal != modalCommit || m.form != form || m.fv.title != "hi" {
		t.Fatalf("background results reset the form: modal=%v title=%q", m.modal, m.fv.title)
	}
	checkChrome(t, m, "form + refresh")
	// an actionMsg (from an earlier action) arriving mid-form
	m = update(t, m, actionMsg{verb: "stage", ok: "staged x"})
	if m.modal != modalCommit || m.fv.title != "hi" {
		t.Fatal("action result reset the form")
	}
	checkChrome(t, m, "form + toast")
}

func spinnerTick(m model) tea.Msg {
	return m.spin.Tick()
}

func TestStaleResultsNeverOverwriteCurrentPane(t *testing.T) {
	m := loaded(t, 120, 40)
	seqStatus := m.diffSeq
	reqStatus, _ := m.currentRequest()
	m = update(t, m, key("3"))
	seqCommit := m.diffSeq
	reqCommit, _ := m.currentRequest()
	m = update(t, m, key("2"))
	// stale results arrive out of order
	m = update(t, m, diffMsg{seq: seqStatus, key: reqStatus.key, title: reqStatus.title, kind: "diff", text: "+STALE STATUS"})
	m = update(t, m, diffMsg{seq: seqCommit, key: reqCommit.key, title: reqCommit.title, kind: "diff", text: "+STALE COMMIT"})
	if v := stripANSI(m.View()); strings.Contains(v, "STALE") || m.diff.key == reqStatus.key || m.diff.key == reqCommit.key {
		t.Fatalf("stale diff overwrote the branch pane: key=%q", m.diff.key)
	}
	if !m.diffLoading {
		t.Fatal("still waiting for the branch log")
	}
	// silent auto-refresh bumps the sequence; the previous in-flight result is stale
	old := m.diffSeq
	m = update(t, m, run(t, loadBranches(m.root)))
	m = update(t, m, diffMsg{seq: old, key: m.diffKey, kind: "log", text: "old"})
	if m.diff.text == "old" {
		t.Fatal("superseded result applied")
	}
	req, _ := m.currentRequest()
	m = update(t, m, run(t, loadDiff(m.root, m.diffSeq, req)))
	if m.diffLoading || !strings.Contains(stripANSI(m.View()), "expand readme") {
		t.Fatal("fresh branch log should be visible")
	}
	// status result after switching to commits must not touch the diff
	m = update(t, m, key("3"))
	key := m.diffKey
	m = update(t, m, run(t, loadStatus(m.root)))
	if m.diffKey != key {
		t.Fatalf("status refresh changed the commit diff key to %q", m.diffKey)
	}
}

func TestSpinnerStopsWhenIdle(t *testing.T) {
	m := loaded(t, 120, 40)
	req, _ := m.currentRequest()
	m = update(t, m, run(t, loadDiff(m.root, m.diffSeq, req)))
	if m.diffLoading || m.busy != 0 || !m.loaded {
		t.Fatal("expected idle")
	}
	mm, cmd := m.Update(spinnerTick(m))
	m = mm.(model)
	if cmd != nil || m.spinning {
		t.Fatal("spinner tick while idle must stop the spinner")
	}
	m, cmd = updateCmd(t, m, key("a"))
	if cmd == nil || !m.spinning {
		t.Fatal("action should restart the spinner")
	}
	mm, cmd = m.Update(spinnerTick(m))
	if cmd == nil {
		t.Fatal("spinner should keep ticking while busy")
	}
}

// ---------------------------------------------------------------- errors

func TestCheckoutWithConflictingLocalChanges(t *testing.T) {
	root := tempRepo(t)
	git(t, root, "stash", "-q", "--include-untracked")
	git(t, root, "checkout", "-q", "feature/x")
	write(t, root, "main.go", "package main\n\nfunc main() { /* feature */ }\n")
	git(t, root, "commit", "-q", "-am", "feat: change main on feature")
	git(t, root, "checkout", "-q", "main")
	write(t, root, "main.go", "package main\n\nfunc main() { /* local */ }\n")
	m := newModel(root, root)
	m = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	m = update(t, m, run(t, loadStatus(root)))
	m = update(t, m, run(t, loadBranches(root)))
	m = update(t, m, key("2"))
	m = m.snapHeights()
	for i := range m.branches {
		if m.branches[i].Name == "feature/x" {
			m, _ = m.setSel(paneBranches, i)
		}
	}
	m, cmd := updateCmd(t, m, key("enter"))
	a, ok := firstAction(collect(cmd))
	if !ok || a.err == nil {
		t.Fatalf("checkout should fail: %+v", a)
	}
	m = update(t, m, a)
	if m.toast == nil || m.toast.ok || !strings.Contains(m.toast.text, "checkout failed") {
		t.Fatalf("toast = %+v", m.toast)
	}
	if m.busy != 0 {
		t.Fatalf("busy = %d", m.busy)
	}
	view := m.View()
	checkChrome(t, m, "checkout conflict")
	if !strings.Contains(stripANSI(view), "checkout failed") {
		t.Fatal("red toast not visible")
	}
	if out := git(t, root, "branch", "--show-current"); strings.TrimSpace(out) != "main" {
		t.Fatalf("still on %q", out)
	}
}

func TestGitMissingFromPath(t *testing.T) {
	root := tempRepo(t)
	t.Setenv("PATH", t.TempDir())
	m := newModel(root, root)
	m = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	for _, c := range []tea.Cmd{loadStatus(root), loadBranches(root), loadCommits(root), loadStashes(root)} {
		m = update(t, m, c())
	}
	if m.toast == nil || m.toast.ok {
		t.Fatalf("missing git must produce an error toast, got %+v", m.toast)
	}
	checkChrome(t, m, "no git")
	m, cmd := updateCmd(t, m, key("a"))
	if cmd != nil {
		for _, x := range collect(cmd) {
			if a, ok := x.(actionMsg); ok {
				m = update(t, m, a)
			}
		}
	}
	if m.toast == nil || m.toast.ok {
		t.Fatalf("action without git must show a red toast, got %+v", m.toast)
	}
	checkChrome(t, m, "no git action")
	req := diffReq{key: "k", kind: "diff", args: []string{"diff"}}
	m = update(t, m, run(t, loadDiff(root, m.diffSeq, req)))
	checkChrome(t, m, "no git diff")
}
