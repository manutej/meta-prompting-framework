package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/cursor"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"alembic/harness"
	"alembic/jev"
)

var sizes = [][2]int{{80, 24}, {100, 30}, {120, 40}, {200, 60}}

func TestMain(m *testing.M) {
	// render real ANSI so the width invariants exercise the escape-aware paths
	lipgloss.SetColorProfile(termenv.TrueColor)
	os.Setenv("TYPESAFE_API_KEY", "")
	os.Exit(m.Run())
}

// ---------- fixtures ----------

func initRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dir := filepath.Join(t.TempDir(), "repo")
	os.MkdirAll(dir, 0o755)
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
	run("init", "-q", "-b", "main")
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("a\n"), 0o644)
	run("add", "a.txt")
	run("-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
	return dir
}

type fixture struct {
	m      model
	repo   string
	wtPath string
	feed   string
	outbox string
	cfg    config
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	repo := initRepo(t)
	wtPath, err := harness.AddWorktree(repo, "feat/x-y", "")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	feed, outbox := filepath.Join(dir, "feed.jsonl"), filepath.Join(dir, "outbox.jsonl")
	if err := harness.NewDemo(feed, outbox).Seed(repo); err != nil {
		t.Fatal(err)
	}
	packs, _ := filepath.Abs("packs")
	cfg := config{Feed: feed, Outbox: outbox, Packs: packs, Receipts: filepath.Join(dir, "receipts"), Repo: repo, RepoOK: true, Poll: time.Second}
	m := quiet(newModel(cfg))
	m = resize(m, 120, 40)
	return &fixture{m: m, repo: repo, wtPath: wtPath, feed: feed, outbox: outbox, cfg: cfg}
}

// quiet makes a model deterministic for headless tests: no clipboard, no
// cursor blink and short timers so pumped Cmds resolve immediately.
func quiet(m model) model {
	toastTTL, demoInterval, wtInterval = time.Millisecond, time.Millisecond, time.Millisecond
	m.copier = func(string) {}
	for _, c := range []*cursor.Model{&m.composer.input.Cursor, &m.picker.query.Cursor, &m.palette.input.Cursor, &m.tasks.filter.Cursor, &m.jv.file.Cursor, &m.jv.text.Cursor} {
		c.SetMode(cursor.CursorStatic)
	}
	return m
}

func resize(m model, w, h int) model {
	mm, _ := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return mm.(model)
}

func keyMsg(k string) tea.KeyMsg {
	switch k {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEscape}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	case "shift+tab":
		return tea.KeyMsg{Type: tea.KeyShiftTab}
	case "up":
		return tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case "ctrl+k":
		return tea.KeyMsg{Type: tea.KeyCtrlK}
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case "backspace":
		return tea.KeyMsg{Type: tea.KeyBackspace}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)}
}

// press sends keys, running any returned Cmds and feeding their results back.
func (f *fixture) press(t *testing.T, keys ...string) {
	t.Helper()
	for _, k := range keys {
		mm, cmd := f.m.Update(keyMsg(k))
		f.m = mm.(model)
		f.pump(t, cmd, 0)
	}
}

func (f *fixture) typeText(t *testing.T, s string) {
	t.Helper()
	for _, r := range s {
		f.press(t, string(r))
	}
}

func (f *fixture) send(t *testing.T, msg tea.Msg) {
	t.Helper()
	mm, cmd := f.m.Update(msg)
	f.m = mm.(model)
	f.pump(t, cmd, 0)
}

// pump executes a Cmd with a timeout (ticks are left to expire) and feeds the
// resulting messages back into Update, so async work resolves deterministically.
func (f *fixture) pump(t *testing.T, cmd tea.Cmd, depth int) {
	t.Helper()
	if cmd == nil || depth > 8 {
		return
	}
	ch := make(chan tea.Msg, 1)
	go func() { ch <- cmd() }()
	var msg tea.Msg
	select {
	case msg = <-ch:
	case <-time.After(1500 * time.Millisecond):
		return
	}
	switch v := msg.(type) {
	case nil:
		return
	case tea.BatchMsg:
		for _, c := range v {
			f.pump(t, c, depth+1)
		}
		return
	case animTickMsg, pollTickMsg, demoTickMsg, wtTickMsg, toastGoneMsg:
		return
	}
	mm, next := f.m.Update(msg)
	f.m = mm.(model)
	f.pump(t, next, depth+1)
}

func (f *fixture) view() string { return stripANSI(f.m.View()) }

func checkFrame(t *testing.T, m model, w, h int, label string) {
	t.Helper()
	out := m.View()
	lines := strings.Split(out, "\n")
	if len(lines) != h {
		t.Errorf("%s @%dx%d: %d lines, want %d", label, w, h, len(lines), h)
	}
	for i, l := range lines {
		if lw := lipgloss.Width(l); lw > w {
			t.Errorf("%s @%dx%d: line %d is %d wide (> %d): %q", label, w, h, i, lw, w, stripANSI(l))
		}
	}
}

func readOutbox(t *testing.T, path string) []harness.Command {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []harness.Command
	for _, l := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if l == "" {
			continue
		}
		var c harness.Command
		if err := json.Unmarshal([]byte(l), &c); err != nil {
			t.Fatalf("bad outbox line %q: %v", l, err)
		}
		out = append(out, c)
	}
	return out
}

func (f *fixture) selectTask(t *testing.T, id string) {
	t.Helper()
	f.m.tasks.selID = id
	f.m.rebuildRows()
	if f.m.tasks.selID != id {
		t.Fatalf("task %s not selectable", id)
	}
}

func (f *fixture) gotoJevPack(t *testing.T, id string) {
	t.Helper()
	f.press(t, "3")
	for i, p := range f.m.packs {
		if p.ID == id {
			f.m.jv.packCursor = i
		}
	}
	f.m.jv.focus = jevFocusPacks
	f.pump(t, f.m.jevStateCmd(), 0)
	if f.m.selectedPack() == nil || f.m.selectedPack().ID != id {
		t.Fatalf("pack %s not selected", id)
	}
}

// ---------- layout invariants ----------

func TestFrameSizesEveryTab(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2") // load worktrees
	for _, sz := range sizes {
		f.m = resize(f.m, sz[0], sz[1])
		for _, tab := range []string{"1", "2", "3", "4"} {
			f.press(t, tab)
			checkFrame(t, f.m, sz[0], sz[1], "tab "+tab)
		}
	}
}

func TestFrameSizesOverlays(t *testing.T) {
	f := newFixture(t)
	f.gotoJevPack(t, "task-readiness")
	f.press(t, "enter")
	if f.m.jv.result == nil {
		t.Fatal("run did not produce a result")
	}
	for _, sz := range sizes {
		f.m = resize(f.m, sz[0], sz[1])
		f.press(t, "1", "ctrl+k")
		checkFrame(t, f.m, sz[0], sz[1], "palette")
		f.press(t, "esc", "?")
		checkFrame(t, f.m, sz[0], sz[1], "help")
		f.press(t, "j", "j")
		checkFrame(t, f.m, sz[0], sz[1], "help scrolled")
		f.press(t, "esc")
		f.selectTask(t, "T-1041")
		f.press(t, "x")
		if !f.m.confirm.open {
			t.Fatal("confirm should open")
		}
		checkFrame(t, f.m, sz[0], sz[1], "confirm")
		f.press(t, "n", "p")
		if !f.m.composer.open {
			t.Fatal("composer should open")
		}
		checkFrame(t, f.m, sz[0], sz[1], "composer")
		f.press(t, "esc", "P")
		checkFrame(t, f.m, sz[0], sz[1], "picker")
		f.press(t, "esc", "2", "n")
		if f.m.form == nil {
			t.Fatal("form should open")
		}
		checkFrame(t, f.m, sz[0], sz[1], "form")
		f.press(t, "esc", "3", "h")
		if !f.m.jv.history.open {
			t.Fatal("history should open")
		}
		checkFrame(t, f.m, sz[0], sz[1], "history")
		f.press(t, "c", "esc", "T")
		checkFrame(t, f.m, sz[0], sz[1], "jev task picker")
		f.press(t, "esc", "W")
		checkFrame(t, f.m, sz[0], sz[1], "jev worktree picker")
		f.press(t, "esc")
	}
}

func TestTooSmallTerminal(t *testing.T) {
	f := newFixture(t)
	f.m = resize(f.m, 10, 3)
	if !strings.Contains(f.m.View(), "too small") {
		t.Fatal("expected the too-small notice")
	}
}

// ---------- tasks ----------

func TestTasksGroupedAndSorted(t *testing.T) {
	f := newFixture(t)
	var got []string
	for _, r := range f.m.tasks.rows {
		if r.kind == rowHeader {
			got = append(got, "#"+r.wf.Name)
		} else {
			got = append(got, r.task.ID)
		}
	}
	want := []string{"#billing-pipeline", "T-0990", "T-0981", "T-0977", "#checkout-service", "T-1042", "T-1041", "#nexus-tui", "T-2202", "T-2201"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("rows:\n got %v\nwant %v", got, want)
	}
	v := f.view()
	for _, s := range []string{"billing-pipeline", "checkout-service", "nexus-tui", "T-1041", "Idempotency keys"} {
		if !strings.Contains(v, s) {
			t.Errorf("view missing %q", s)
		}
	}
}

func TestSelectionSurvivesFeedUpdate(t *testing.T) {
	f := newFixture(t)
	f.press(t, "j", "j", "j")
	sel := f.m.tasks.selID
	if sel != "T-1042" {
		t.Fatalf("selection after 3×j: %s", sel)
	}
	// a new urgent task in the same workflow lands above the selection
	now := time.Now().UTC()
	f.send(t, feedMsg{recs: []harness.FeedRecord{{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-9999", Workflow: "checkout", Title: "Hotfix", State: harness.StateRunning, Priority: 2, Created: now}}}})
	if f.m.tasks.selID != sel {
		t.Fatalf("selection moved to %s after feed update", f.m.tasks.selID)
	}
	if _, ok := f.m.snap.Tasks["T-9999"]; !ok {
		t.Fatal("record not applied")
	}
	f.send(t, feedMsg{recs: []harness.FeedRecord{{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-1042", Workflow: "checkout", Title: "Flaky test", State: harness.StateDone, Priority: 2}}}})
	if f.m.tasks.selID != sel || f.m.snap.Tasks["T-1042"].State != harness.StateDone {
		t.Fatal("selection should follow the task through a state change")
	}
}

func TestCycleWorkflowFilter(t *testing.T) {
	f := newFixture(t)
	f.press(t, "w")
	if f.m.tasks.wfFilter != "billing" || len(f.m.taskRows()) != 3 {
		t.Fatalf("after w: filter=%q rows=%d", f.m.tasks.wfFilter, len(f.m.taskRows()))
	}
	f.press(t, "w")
	if f.m.tasks.wfFilter != "checkout" || len(f.m.taskRows()) != 2 {
		t.Fatalf("after w w: filter=%q rows=%d", f.m.tasks.wfFilter, len(f.m.taskRows()))
	}
	f.press(t, "w", "w")
	if f.m.tasks.wfFilter != "" || len(f.m.taskRows()) != 7 {
		t.Fatalf("cycle should wrap to all: filter=%q rows=%d", f.m.tasks.wfFilter, len(f.m.taskRows()))
	}
	if !strings.Contains(f.view(), "Tasks · 7") {
		t.Fatal("title should show the count")
	}
}

func TestFuzzyFilterNarrows(t *testing.T) {
	f := newFixture(t)
	f.press(t, "/")
	if !f.m.tasks.filtering {
		t.Fatal("/ should start filtering")
	}
	f.typeText(t, "stripe")
	ids := f.m.taskRows()
	if len(ids) == 0 || len(ids) >= 7 {
		t.Fatalf("filter 'stripe' should narrow: %d rows", len(ids))
	}
	found := false
	for _, i := range ids {
		found = found || f.m.tasks.rows[i].task.ID == "T-0977"
	}
	if !found {
		t.Fatal("Stripe task should survive the filter")
	}
	f.typeText(t, " payouts nightly")
	if ids = f.m.taskRows(); len(ids) != 1 || f.m.tasks.rows[ids[0]].task.ID != "T-0977" || f.m.tasks.selID != "T-0977" {
		t.Fatalf("longer query should isolate T-0977: rows=%d sel=%s", len(ids), f.m.tasks.selID)
	}
	if !strings.Contains(f.view(), "nothing matches") == (len(ids) == 0) {
		t.Fatal("empty-filter message should track the row count")
	}
	f.press(t, "enter")
	if f.m.tasks.filtering || len(f.m.taskRows()) != 1 {
		t.Fatal("enter should keep the filter")
	}
	f.press(t, "/", "esc")
	if len(f.m.taskRows()) != 7 {
		t.Fatal("esc should clear the filter")
	}
}

func TestHideDoneToggle(t *testing.T) {
	f := newFixture(t)
	f.press(t, "e")
	if !f.m.tasks.hideDone || len(f.m.taskRows()) != 6 {
		t.Fatalf("hide done: %v rows=%d", f.m.tasks.hideDone, len(f.m.taskRows()))
	}
	f.press(t, "e")
	if len(f.m.taskRows()) != 7 {
		t.Fatal("show done again")
	}
}

func TestPingAppendsOutbox(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.press(t, "p")
	if !f.m.composer.open || f.m.composer.agentID != "coder-1" {
		t.Fatalf("composer: %+v", f.m.composer)
	}
	f.typeText(t, "status?")
	f.press(t, "enter")
	cmds := readOutbox(t, f.outbox)
	if len(cmds) != 1 || cmds[0].Type != "ping" || cmds[0].Agent != "coder-1" || cmds[0].TaskID != "T-1041" || cmds[0].Text != "status?" {
		t.Fatalf("outbox: %+v", cmds)
	}
	if !strings.Contains(f.view(), "ping sent") || f.m.pingsSent["coder-1"] != 1 {
		t.Fatal("expected the ping sent toast and counter")
	}
	// canned replies via tab
	f.press(t, "p", "tab", "enter")
	cmds = readOutbox(t, f.outbox)
	if len(cmds) != 2 || cmds[1].Text != cannedPings[0] {
		t.Fatalf("canned ping: %+v", cmds)
	}
}

func TestPingUnassignedTaskWarns(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-2202")
	f.press(t, "p")
	if f.m.composer.open || len(readOutbox(t, f.outbox)) != 0 {
		t.Fatal("unassigned task must not open the composer")
	}
	f.press(t, "P")
	if !f.m.picker.open {
		t.Fatal("P should open the agent picker")
	}
	f.typeText(t, "cod")
	f.press(t, "enter")
	if !f.m.composer.open || f.m.composer.agentID != "coder-1" || f.m.composer.taskID != "T-2202" {
		t.Fatalf("picker → composer: %+v", f.m.composer)
	}
}

func TestCancelAppendsOutbox(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.press(t, "x")
	if !f.m.confirm.open || !strings.Contains(f.view(), "Cancel T-1041") {
		t.Fatal("confirm should show the task")
	}
	f.press(t, "n")
	if len(readOutbox(t, f.outbox)) != 0 {
		t.Fatal("n must not send")
	}
	f.press(t, "x", "y")
	cmds := readOutbox(t, f.outbox)
	if len(cmds) != 1 || cmds[0].Type != "task.cancel" || cmds[0].TaskID != "T-1041" {
		t.Fatalf("outbox: %+v", cmds)
	}
}

func TestRetryAppendsOutbox(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.press(t, "R")
	if len(readOutbox(t, f.outbox)) != 0 {
		t.Fatal("retry on a running task must be refused")
	}
	f.selectTask(t, "T-0990")
	f.press(t, "R")
	cmds := readOutbox(t, f.outbox)
	if len(cmds) != 1 || cmds[0].Type != "task.retry" || cmds[0].TaskID != "T-0990" {
		t.Fatalf("outbox: %+v", cmds)
	}
}

func TestTaskReadinessWritesReceipt(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1042")
	f.press(t, "J")
	tj := f.m.taskJev["T-1042"]
	if tj == nil || tj.receipt == nil || tj.running {
		t.Fatalf("task jev: %+v", tj)
	}
	if tj.receipt.PackID != "task-readiness" || !tj.receipt.Mock || tj.receipt.StateRef != "T-1042" {
		t.Fatalf("receipt: %+v", tj.receipt)
	}
	entries, _ := os.ReadDir(f.cfg.Receipts)
	if len(entries) != 1 || !strings.HasSuffix(entries[0].Name(), "-task-readiness.json") {
		t.Fatalf("receipt files: %v", entries)
	}
	v := f.view()
	if !strings.Contains(v, "jev") || !strings.Contains(v, "task-readiness: "+strings.ToUpper(string(tj.receipt.Decision))) || !strings.Contains(v, "MOCK") {
		t.Fatalf("detail pane should show the jev line:\n%s", v)
	}
	// a fresh model indexes the receipt from disk
	m2 := newModel(f.cfg)
	if m2.taskJev["T-1042"] == nil {
		t.Fatal("receipt should be indexed at startup")
	}
}

func TestDetailFocusAndElementCopy(t *testing.T) {
	f := newFixture(t)
	var copied string
	f.m.copier = func(s string) { copied = s }
	f.selectTask(t, "T-1041")
	f.press(t, "tab", "j", "y")
	if copied != "https://github.com/acme/checkout/pull/412" {
		t.Fatalf("copied %q", copied)
	}
	if f.m.tasks.focus != focusDetail {
		t.Fatal("tab should focus the detail pane")
	}
	f.press(t, "tab")
	if f.m.tasks.focus != focusList {
		t.Fatal("tab should return focus to the list")
	}
}

// ---------- jev tab ----------

func TestJevTabRendersVisualizations(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.gotoJevPack(t, "task-readiness")
	v := f.view()
	for _, s := range []string{"Packs · 5", "State · events", "T-1041", "Result · task-readiness", "NOUL", "CHOICE", "SCORE", "task.autocontinue"} {
		if !strings.Contains(v, s) {
			t.Errorf("before run, view missing %q", s)
		}
	}
	if !f.m.jevStateLoaded() || len(f.m.jv.state) == 0 {
		t.Fatal("state should be loaded from the selected task")
	}
	f.press(t, "enter")
	r := f.m.jv.result
	if r == nil || f.m.jv.running {
		t.Fatal("run should complete synchronously through the pump")
	}
	if _, err := os.Stat(r.path); err != nil {
		t.Fatalf("receipt not written: %v", err)
	}
	v = f.view()
	for _, s := range []string{"no ◀", "▶ yes", "p=", "conf", "✓", "score", "MOCK — deterministic, not a model", "task.autocontinue", "refuse<0.30", "jev-mock", "tokens", "receipt "} {
		if !strings.Contains(v, s) {
			t.Errorf("after run, view missing %q\n%s", s, v)
		}
	}
	dec := "▌" + strings.ToUpper(string(r.receipt.Decision))
	if !strings.Contains(v, dec) {
		t.Errorf("verdict banner %q missing", dec)
	}
	for _, id := range []string{"stuck", "needs_human", "next", "risk"} {
		if !strings.Contains(v, id) {
			t.Errorf("question %s missing from result", id)
		}
	}
	// the run against a task also updates the detail pane's jev line
	if f.m.taskJev["T-1041"] == nil || f.m.taskJev["T-1041"].receipt == nil {
		t.Fatal("task jev line should be updated")
	}
}

func TestJevAnswerRenderers(t *testing.T) {
	p := 0.9
	lines := renderAnswer("q", jev.Answer{Type: jev.Noul, Noul: &p}, 40)
	if len(lines) != 3 || !strings.Contains(stripANSI(lines[1]), "no ◀ ") || !strings.Contains(stripANSI(lines[2]), "p=0.90") {
		t.Fatalf("noul: %q", lines)
	}
	if plain := stripANSI(lines[1]); strings.Count(plain, "█") == 0 || !strings.HasSuffix(strings.TrimRight(plain, " "), "░ ▶ yes") {
		t.Fatalf("noul bar should lean to yes with room left on the right: %q", plain)
	}
	low := 0.1
	plain := stripANSI(renderAnswer("q", jev.Answer{Type: jev.Noul, Noul: &low}, 40)[1])
	bar := strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(plain), "no ◀ "), " ▶ yes")
	left, right := bar[:len(bar)/2], bar[len(bar)/2:]
	if !strings.Contains(left, "█") || strings.Contains(right, "█") {
		t.Fatalf("noul bar should lean to no (mass left of centre): %q", plain)
	}
	c := 0.81
	lines = renderAnswer("next", jev.Answer{Type: jev.Choice, Choice: "billing", Probabilities: map[string]float64{"billing": 0.88, "ui": 0.12}, Confidence: &c}, 40)
	got := stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(got, "✓ billing") || !strings.Contains(got, "0.88") || !strings.Contains(got, "conf 0.81") {
		t.Fatalf("choice: %s", got)
	}
	if strings.Index(got, "billing ") > strings.Index(got, "ui ") {
		t.Fatalf("options should sort by probability desc: %s", got)
	}
	s := 1.05
	lines = renderAnswer("risk", jev.Answer{Type: jev.Score, Score: &s, Confidence: &c, Legend: map[string]string{"0": "None", "1": "Low", "2": "High"}, Probabilities: map[string]float64{"0": 0.2, "1": 0.55, "2": 0.25}}, 40)
	got = stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(got, "score 1.05") || !strings.Contains(got, "Low") {
		t.Fatalf("score: %s", got)
	}
	for _, l := range lines {
		pl := stripANSI(l)
		if strings.HasPrefix(pl, "Low") && !strings.Contains(pl, "◀") {
			t.Fatalf("nearest level should carry the marker: %q", pl)
		}
		if strings.HasPrefix(pl, "None") && strings.Contains(pl, "◀") {
			t.Fatalf("other levels must not carry the marker: %q", pl)
		}
	}
	if !strings.Contains(stripANSI(strings.Join(renderVerdict(&jev.Pack{}, jev.Receipt{}, 40), "\n")), "informational") {
		t.Fatal("no gate → informational")
	}
}

func TestJevHistoryAndCompare(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.gotoJevPack(t, "task-readiness")
	f.press(t, "h")
	if f.m.jv.history.open {
		t.Fatal("history with no receipts should only warn")
	}
	f.press(t, "enter")
	first := f.m.jv.result.receipt
	time.Sleep(3 * time.Millisecond)
	f.m.jv.taskID = "T-1042"
	f.pump(t, f.m.jevStateCmd(), 0)
	f.press(t, "enter")
	if f.m.jv.result.receipt.ID == first.ID {
		t.Fatal("second run should produce a new receipt")
	}
	f.press(t, "h")
	if !f.m.jv.history.open || len(f.m.jv.history.list) != 2 {
		t.Fatalf("history: %+v", f.m.jv.history)
	}
	v := f.view()
	if !strings.Contains(v, "History · Task readiness · 2 receipts") || !strings.Contains(v, "MOCK") {
		t.Fatalf("history overlay:\n%s", v)
	}
	f.press(t, "c", "j")
	v = f.view()
	if !strings.Contains(v, "compare") || !strings.Contains(v, "◆") {
		t.Fatalf("compare section missing:\n%s", v)
	}
	if !strings.Contains(v, "+0.") && !strings.Contains(v, "-0.") {
		t.Fatalf("compare should show signed deltas:\n%s", v)
	}
	f.press(t, "enter")
	if f.m.jv.history.open || f.m.jv.result == nil || f.m.jv.result.receipt.ID != first.ID || !f.m.jv.result.fromHistory {
		t.Fatalf("enter should load the older receipt: %+v", f.m.jv.result)
	}
	if !strings.Contains(f.view(), "(history)") {
		t.Fatal("result title should say it came from history")
	}
	f.press(t, "h", "esc")
	if f.m.jv.history.open {
		t.Fatal("esc closes history")
	}
}

func TestJevCopyAndSendReceipt(t *testing.T) {
	f := newFixture(t)
	var copied string
	f.m.copier = func(s string) { copied = s }
	f.selectTask(t, "T-1041")
	f.gotoJevPack(t, "task-readiness")
	f.press(t, "y", "s")
	if copied != "" || len(readOutbox(t, f.outbox)) != 0 {
		t.Fatal("nothing to copy or send before a run")
	}
	f.press(t, "enter", "y")
	if copied != f.m.jv.result.path {
		t.Fatalf("copied %q", copied)
	}
	f.press(t, "s")
	cmds := readOutbox(t, f.outbox)
	if len(cmds) != 1 || cmds[0].Type != "jev.receipt" || cmds[0].TaskID != "T-1041" {
		t.Fatalf("outbox: %+v", cmds)
	}
	data, _ := json.Marshal(cmds[0].Data)
	var rc jev.Receipt
	if json.Unmarshal(data, &rc) != nil || rc.ID != f.m.jv.result.receipt.ID {
		t.Fatalf("receipt payload: %s", data)
	}
	f.press(t, "S")
	if !f.m.composer.open || f.m.composer.kind != composerJev {
		t.Fatal("S opens the note composer")
	}
	f.typeText(t, "fyi")
	f.press(t, "enter")
	if cmds = readOutbox(t, f.outbox); len(cmds) != 2 || cmds[1].Text != "fyi" {
		t.Fatalf("outbox after S: %+v", cmds)
	}
}

func TestJevFocusAndPackNavigation(t *testing.T) {
	f := newFixture(t)
	f.press(t, "3")
	if f.m.tab != tabJev || f.m.jv.focus != jevFocusPacks {
		t.Fatal("jev tab should start focused on packs")
	}
	f.press(t, "tab")
	if f.m.jv.focus != jevFocusState {
		t.Fatal("tab → state")
	}
	f.press(t, "tab")
	if f.m.jv.focus != jevFocusResult {
		t.Fatal("tab → result")
	}
	f.press(t, "tab")
	if f.m.jv.focus != jevFocusPacks {
		t.Fatal("tab wraps to packs")
	}
	f.press(t, "j", "j")
	if f.m.selectedPack().ID != "pr-triage" {
		t.Fatalf("j j → %s", f.m.selectedPack().ID)
	}
	if !f.m.jevStateLoaded() {
		t.Fatal("moving packs reloads the state")
	}
	f.press(t, "G")
	if f.m.selectedPack().ID != "task-readiness" {
		t.Fatal("G → last pack")
	}
	f.press(t, "k", "k", "k", "k", "k")
	if f.m.jv.packCursor != 0 {
		t.Fatal("k clamps at 0")
	}
}

func TestJevTextSourceInlineEditor(t *testing.T) {
	f := newFixture(t)
	f.gotoJevPack(t, "ping-priority")
	if !strings.Contains(f.view(), "state is empty") {
		t.Fatal("empty text state should warn")
	}
	f.press(t, "enter")
	if f.m.jv.result != nil {
		t.Fatal("enter must not run on an empty state")
	}
	f.press(t, "e")
	if !f.m.jv.editing || f.m.jv.focus != jevFocusState {
		t.Fatal("e enters editing")
	}
	f.typeText(t, "prod is down, stop now")
	f.press(t, "esc")
	if f.m.jv.editing || string(f.m.jv.state) != "prod is down, stop now" {
		t.Fatalf("state after editing: %q editing=%v", f.m.jv.state, f.m.jv.editing)
	}
	if !strings.Contains(f.view(), "22 bytes") {
		t.Fatal("byte count should show")
	}
	f.press(t, "enter")
	if f.m.jv.result == nil || f.m.jv.result.receipt.StateSource != "text" {
		t.Fatal("text pack should run")
	}
	if !strings.Contains(f.view(), "informational") {
		t.Fatal("ping-priority has no gate → informational")
	}
}

func TestJevDiffSourceAndWorktreePicker(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2") // list worktrees
	f.gotoJevPack(t, "pr-triage")
	if src, ref := f.m.jevSource(); src != "working_diff" || ref != f.repo {
		t.Fatalf("source %s ref %s", src, ref)
	}
	if !strings.Contains(f.view(), "state is empty") {
		t.Fatal("clean worktree → empty diff warning")
	}
	os.WriteFile(filepath.Join(f.wtPath, "a.txt"), []byte("a\nb\n"), 0o644)
	f.press(t, "W")
	if !f.m.picker.open {
		t.Fatal("W opens the worktree picker")
	}
	f.typeText(t, "x-y")
	f.press(t, "enter")
	if f.m.jv.wtPath != f.wtPath || !strings.Contains(string(f.m.jv.state), "+b") {
		t.Fatalf("diff state: wt=%s state=%q", f.m.jv.wtPath, f.m.jv.state)
	}
	f.press(t, "enter")
	if f.m.jv.result == nil || f.m.jv.result.receipt.StateSource != "working_diff" || f.m.jv.result.receipt.StateRef != f.wtPath {
		t.Fatalf("result: %+v", f.m.jv.result)
	}
	if !strings.Contains(f.view(), "pr.automerge") {
		t.Fatal("verdict should name the gate action")
	}
}

func TestJevTaskPicker(t *testing.T) {
	f := newFixture(t)
	f.gotoJevPack(t, "proposal-review")
	f.press(t, "T")
	f.typeText(t, "2202")
	f.press(t, "enter")
	if f.m.jv.taskID != "T-2202" || !strings.Contains(string(f.m.jv.state), "T-2202") {
		t.Fatalf("task picker: %s", f.m.jv.taskID)
	}
	if strings.Contains(string(f.m.jv.state), `"events"`) {
		t.Fatal("task source must not include events")
	}
}

func TestWorktreeJRunsDiffPack(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2")
	os.WriteFile(filepath.Join(f.repo, "a.txt"), []byte("changed\n"), 0o644)
	f.press(t, "J")
	if !f.m.picker.open {
		t.Fatal("J asks staged vs working")
	}
	f.press(t, "w")
	if f.m.tab != tabJev || f.m.selectedPack().ID != "pr-triage" {
		t.Fatalf("expected the working-diff pack on the Jev tab, got tab=%d", f.m.tab)
	}
	if f.m.jv.result == nil || f.m.jv.running {
		t.Fatal("run should complete even though the state was not loaded yet")
	}
}

func TestInvalidAndFileSourcePacks(t *testing.T) {
	f := newFixture(t)
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "broken.json"), []byte(`{"id":"b"`), 0o644)
	os.WriteFile(filepath.Join(dir, "filepack.json"), []byte(`{"id":"filepack","name":"File pack","description":"reads a file","state_source":"file","questions":{"ok":{"type":"noul","instructions":"Is it fine?"}}}`), 0o644)
	f.m.cfg.Packs = dir
	f.press(t, "3", "r")
	if len(f.m.packs) != 1 || len(f.m.packErrs) != 1 {
		t.Fatalf("packs=%d errs=%d", len(f.m.packs), len(f.m.packErrs))
	}
	v := f.view()
	if !strings.Contains(v, "✖ broken.json") || !strings.Contains(v, "Packs · 1") || !strings.Contains(v, "1 invalid") {
		t.Fatalf("invalid pack should be listed in the pane and toast:\n%s", v)
	}
	state := filepath.Join(dir, "state.txt")
	os.WriteFile(state, []byte("hello world\n"), 0o644)
	f.press(t, "e")
	f.typeText(t, state)
	f.press(t, "enter")
	if string(f.m.jv.state) != "hello world\n" {
		t.Fatalf("file state: %q", f.m.jv.state)
	}
	f.press(t, "enter")
	if f.m.jv.result == nil || f.m.jv.result.receipt.StateRef != state {
		t.Fatal("file pack should run")
	}
	f.press(t, "e")
	f.typeText(t, "-missing")
	f.press(t, "enter")
	if f.m.jv.stateErr == nil || !strings.Contains(f.view(), "no such file") {
		t.Fatalf("missing file should surface as an error:\n%s", f.view())
	}
}

func TestEmptyPacksDir(t *testing.T) {
	f := newFixture(t)
	f.m.cfg.Packs = t.TempDir()
	f.press(t, "3", "r")
	if len(f.m.packs) != 0 {
		t.Fatal("no packs expected")
	}
	v := f.view()
	if !strings.Contains(v, "no packs found") || !strings.Contains(v, "select a pack") {
		t.Fatalf("empty states:\n%s", v)
	}
	f.press(t, "enter", "h", "W", "j", "k")
	if f.m.jv.result != nil || f.m.picker.open || f.m.jv.history.open {
		t.Fatal("nothing should run or open")
	}
	f.press(t, "1", "J")
	if !strings.Contains(f.view(), "not loaded") {
		t.Fatal("J without the pack should fail loudly")
	}
}

// ---------- worktrees ----------

func TestWorktreesTableListsBoth(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2")
	if len(f.m.wt.list) != 2 {
		t.Fatalf("worktrees: %+v", f.m.wt.list)
	}
	v := f.view()
	for _, s := range []string{"Worktrees · 2", "main", "feat/x-y", "clean"} {
		if !strings.Contains(v, s) {
			t.Errorf("view missing %q\n%s", s, v)
		}
	}
	// a task linked to the main worktree shows up in the tasks column
	f.send(t, feedMsg{recs: []harness.FeedRecord{{Type: "task.upsert", Task: &harness.Task{ID: "T-7", Workflow: "nexus", Title: "x", State: harness.StateRunning, Worktree: f.wtPath}}}})
	if !strings.Contains(f.view(), "T-7") {
		t.Fatal("linked task should be listed")
	}
	f.press(t, "j")
	if f.m.wt.cursor != 1 {
		t.Fatal("j moves")
	}
	f.press(t, "d")
	if !f.m.confirm.open {
		t.Fatal("d confirms")
	}
	f.press(t, "y")
	if len(f.m.wt.list) != 1 {
		t.Fatalf("worktree should be removed: %+v", f.m.wt.list)
	}
	f.press(t, "d")
	if f.m.confirm.open {
		t.Fatal("main worktree cannot be removed")
	}
}

func TestNewWorktreeForm(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2", "n")
	if f.m.form == nil || f.m.formKind != "worktree" {
		t.Fatal("n opens the form")
	}
	if !strings.Contains(f.view(), "New worktree") {
		t.Fatal("form overlay should render")
	}
	f.typeText(t, "feat/from-form")
	f.press(t, "enter") // → path
	f.press(t, "enter") // submit
	if f.m.form != nil {
		t.Fatal("form should complete")
	}
	want := filepath.Join(filepath.Dir(f.repo), "repo-feat-from-form")
	if _, err := os.Stat(filepath.Join(want, "a.txt")); err != nil {
		t.Fatalf("worktree not created at %s: %v", want, err)
	}
	if len(f.m.wt.list) != 3 || !strings.Contains(f.view(), "feat/from-form") {
		t.Fatalf("list after add: %+v", f.m.wt.list)
	}
	f.press(t, "n", "esc")
	if f.m.form != nil {
		t.Fatal("esc cancels")
	}
}

func TestNotARepoDoesNotPanic(t *testing.T) {
	dir := t.TempDir()
	packs, _ := filepath.Abs("packs")
	cfg := config{Feed: filepath.Join(dir, "feed.jsonl"), Outbox: filepath.Join(dir, "outbox.jsonl"), Packs: packs, Receipts: filepath.Join(dir, "r"), Repo: dir, RepoOK: false, Poll: time.Second}
	f := &fixture{m: quiet(newModel(cfg)), cfg: cfg}
	f.m = resize(f.m, 100, 30)
	if !f.m.feedMissing {
		t.Fatal("missing feed should be flagged")
	}
	for _, sz := range sizes {
		f.m = resize(f.m, sz[0], sz[1])
		for _, tab := range []string{"1", "2", "3", "4"} {
			f.press(t, tab)
			checkFrame(t, f.m, sz[0], sz[1], "no-repo tab "+tab)
		}
	}
	f.press(t, "1")
	v := f.view()
	if !strings.Contains(v, "no tasks yet") || !strings.Contains(v, "no feed") {
		t.Fatalf("empty task state:\n%s", v)
	}
	f.press(t, "j", "k", "p", "x", "R", "J", "enter", "w", "e")
	f.press(t, "2")
	if !strings.Contains(f.view(), "not a git repo") {
		t.Fatal("worktrees tab should explain")
	}
	f.press(t, "n", "d", "r", "J", "j", "enter")
	if f.m.form != nil || f.m.confirm.open || f.m.picker.open {
		t.Fatal("no overlays outside a repo")
	}
	f.press(t, "3", "W", "T", "enter", "j", "j", "enter")
	if f.m.picker.open || f.m.jv.running {
		t.Fatal("no worktree/task to pick")
	}
	f.press(t, "4")
	if !strings.Contains(f.view(), "no agents in feed") {
		t.Fatal("agents empty state")
	}
	f.press(t, "j", "p", "enter", "ctrl+k")
	f.typeText(t, "new")
	f.press(t, "enter")
	if f.m.form != nil {
		t.Fatal("New worktree must refuse outside a repo")
	}
}

// ---------- agents ----------

func TestAgentsTab(t *testing.T) {
	f := newFixture(t)
	f.press(t, "4")
	v := f.view()
	for _, s := range []string{"Agents · 4", "Coder", "Reviewer", "Tester", "SRE", "busy", "idle", "claude-opus-5-5", "T-1041", "0 / 0"} {
		if !strings.Contains(v, s) {
			t.Errorf("agents view missing %q\n%s", s, v)
		}
	}
	if f.m.selectedAgent().Name != "Coder" {
		t.Fatalf("first agent %s", f.m.selectedAgent().Name)
	}
	f.press(t, "p")
	if !f.m.composer.open || f.m.composer.agentID != "coder-1" || f.m.composer.taskID != "" {
		t.Fatalf("p should ping the agent without a task: %+v", f.m.composer)
	}
	f.typeText(t, "hi")
	f.press(t, "enter")
	f.send(t, feedMsg{recs: []harness.FeedRecord{{Type: "ping.ack", PingAck: &harness.PingAck{PingID: "x", Agent: "coder-1", Text: "ack"}}}})
	if !strings.Contains(f.view(), "1 / 1") {
		t.Fatalf("pings/acks column:\n%s", f.view())
	}
	f.press(t, "j", "j", "j")
	if f.m.selectedAgent().Name != "Tester" {
		t.Fatalf("j j j → %s", f.m.selectedAgent().Name)
	}
	f.press(t, "enter")
	if f.m.tab != tabTasks || f.m.tasks.selID != "T-1042" {
		t.Fatalf("enter should jump to the agent's task: tab=%d sel=%s", f.m.tab, f.m.tasks.selID)
	}
}

func TestAgentJumpClearsFilter(t *testing.T) {
	f := newFixture(t)
	f.press(t, "w") // billing only
	f.press(t, "4", "enter")
	if f.m.tab != tabTasks || f.m.tasks.selID != "T-1041" || f.m.tasks.wfFilter != "" {
		t.Fatalf("jump should clear the workflow filter: sel=%s filter=%q", f.m.tasks.selID, f.m.tasks.wfFilter)
	}
}

// ---------- palette & help ----------

func TestPaletteFuzzyReloadPacks(t *testing.T) {
	f := newFixture(t)
	f.press(t, "ctrl+k")
	if !f.m.palette.open || len(f.m.palette.matches) != len(f.m.palette.cmds) {
		t.Fatal("palette opens with every command")
	}
	f.typeText(t, "read")
	if len(f.m.palette.matches) == 0 || f.m.palette.cmds[f.m.palette.matches[0]].name != "Reload packs" {
		t.Fatalf("fuzzy 'read' top match: %v", f.m.palette.matches)
	}
	if !strings.Contains(f.view(), "Reload packs") {
		t.Fatal("palette should render the match")
	}
	f.press(t, "enter")
	if f.m.palette.open || !strings.Contains(f.view(), "packs reloaded · 5 valid") {
		t.Fatalf("enter runs the command:\n%s", f.view())
	}
	f.press(t, ":")
	f.typeText(t, "zzzz")
	if len(f.m.palette.matches) != 0 || !strings.Contains(f.view(), "no matching commands") {
		t.Fatal("no matches state")
	}
	f.press(t, "esc")
	if f.m.palette.open {
		t.Fatal("esc closes")
	}
}

func TestPaletteEveryCommandRuns(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2")
	f.selectTask(t, "T-0990")
	for i, c := range f.m.palette.cmds {
		if c.name == "Quit" {
			continue
		}
		f.press(t, "ctrl+k")
		f.m.palette.cursor = i
		f.press(t, "enter")
		checkFrame(t, f.m, 120, 40, "after "+c.name)
		f.press(t, "esc", "esc")
		f.m.composer.open, f.m.confirm.open, f.m.picker.open, f.m.help, f.m.form = false, false, false, false, nil
	}
	f.press(t, "ctrl+k")
	f.typeText(t, "quit")
	mm, cmd := f.m.Update(keyMsg("enter"))
	f.m = mm.(model)
	if cmd == nil {
		t.Fatal("Quit should return tea.Quit")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("Quit should return tea.Quit")
	}
}

func TestPaletteGoToTabs(t *testing.T) {
	f := newFixture(t)
	for _, tc := range []struct {
		q   string
		tab tab
	}{{"go to jev", tabJev}, {"go to agents", tabAgents}, {"go to work", tabWorktrees}, {"go to tasks", tabTasks}} {
		f.press(t, "ctrl+k")
		f.typeText(t, tc.q)
		f.press(t, "enter")
		if f.m.tab != tc.tab {
			t.Fatalf("%q → tab %d, want %d", tc.q, f.m.tab, tc.tab)
		}
	}
}

func TestHelpOverlay(t *testing.T) {
	f := newFixture(t)
	f.press(t, "?")
	if !f.m.help {
		t.Fatal("? opens help")
	}
	v := f.view()
	for _, s := range []string{"alembic — keys", "Tasks", "Worktrees", "Jev", "Agents", "ctrl+k / :", "task-readiness"} {
		if !strings.Contains(v, s) {
			t.Errorf("help missing %q", s)
		}
	}
	f.press(t, "?")
	if f.m.help {
		t.Fatal("? closes help")
	}
	f.m = resize(f.m, 80, 24)
	f.press(t, "?")
	if !strings.Contains(f.view(), "more below") {
		t.Fatal("short terminals should offer scrolling")
	}
	f.press(t, "j", "j", "j")
	if f.m.helpOff != 3 {
		t.Fatal("j scrolls")
	}
	f.press(t, "esc")
	if f.m.help || f.m.helpOff != 0 {
		t.Fatal("esc closes and resets")
	}
}

func TestQuitKeys(t *testing.T) {
	f := newFixture(t)
	for _, k := range []string{"q", "ctrl+c"} {
		_, cmd := f.m.Update(keyMsg(k))
		if cmd == nil {
			t.Fatalf("%s should quit", k)
		}
		if _, ok := cmd().(tea.QuitMsg); !ok {
			t.Fatalf("%s should quit", k)
		}
	}
	// ctrl+c also quits from inside the composer
	f.press(t, "p")
	_, cmd := f.m.Update(keyMsg("ctrl+c"))
	if cmd == nil {
		t.Fatal("ctrl+c in composer should quit")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("ctrl+c in composer should quit")
	}
}

// ---------- misc ----------

func TestHeaderShowsMockAndCounts(t *testing.T) {
	f := newFixture(t)
	v := f.view()
	for _, s := range []string{"ORMUS", "alembic", "jev: MOCK", "3 workflows", "7 tasks", "4 agents"} {
		if !strings.Contains(v, s) {
			t.Errorf("header missing %q", s)
		}
	}
}

func TestAnimationOnlyWhenNeeded(t *testing.T) {
	f := newFixture(t)
	if !f.m.needsAnim() {
		t.Fatal("running tasks on the Tasks tab animate the spinner")
	}
	f.press(t, "4")
	if f.m.needsAnim() {
		t.Fatal("nothing animates on the Agents tab")
	}
	mm, cmd := f.m.Update(animTickMsg{})
	if cmd != nil {
		t.Fatal("anim tick must stop when nothing animates")
	}
	f.m = mm.(model)
	if f.m.kickAnim() != nil {
		t.Fatal("kickAnim is a no-op when idle")
	}
}

func TestMouseClicks(t *testing.T) {
	f := newFixture(t)
	f.send(t, tea.MouseMsg{X: 12, Y: 1, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress})
	if f.m.tab != tabWorktrees {
		t.Fatalf("clicking the second tab: %d", f.m.tab)
	}
	f.press(t, "1")
	f.send(t, tea.MouseMsg{Button: tea.MouseButtonWheelDown, Action: tea.MouseActionPress})
	if f.m.tasks.selID != "T-1042" {
		t.Fatalf("wheel scroll: %s", f.m.tasks.selID)
	}
	f.send(t, tea.MouseMsg{X: 2, Y: 4, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress})
	if f.m.tasks.selID != "T-0990" {
		t.Fatalf("click row: %s", f.m.tasks.selID)
	}
	f.send(t, tea.MouseMsg{X: 100, Y: 10, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress})
	if f.m.tasks.focus != focusDetail {
		t.Fatal("click detail focuses it")
	}
}

func TestParseFlagsAndDefaults(t *testing.T) {
	t.Setenv("ALEMBIC_FEED", "")
	t.Setenv("ALEMBIC_OUTBOX", "")
	t.Setenv("ALEMBIC_PACKS", "")
	cfg, err := parseFlags([]string{"--poll", "1s", "--repo", t.TempDir(), "--receipts", "/tmp/r"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Poll != time.Second || cfg.RepoOK || cfg.Receipts != "/tmp/r" || !strings.HasSuffix(cfg.Packs, "packs") {
		t.Fatalf("cfg: %+v", cfg)
	}
	if !strings.HasSuffix(cfg.Feed, filepath.Join(".ormus", "feed.jsonl")) {
		t.Fatalf("feed default: %s", cfg.Feed)
	}
	cfg, err = parseFlags([]string{"--repo", "."})
	if err != nil || !cfg.RepoOK {
		t.Fatalf("the module dir is inside a repo: %v %+v", err, cfg)
	}
	if _, err := parseFlags([]string{"--nope"}); err == nil {
		t.Fatal("unknown flag should error")
	}
}

func TestDemoSetupSeedsFeed(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	cfg := config{Repo: initRepo(t), RepoOK: true, Poll: time.Second, Demo: true}
	if err := setupDemo(&cfg); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(cfg.Feed, filepath.Join(home, ".alembic", "demo")) {
		t.Fatalf("demo feed: %s", cfg.Feed)
	}
	cfg.Packs, _ = filepath.Abs("packs")
	cfg.Receipts = filepath.Join(home, "r")
	t.Setenv("TYPESAFE_API_KEY", "not-a-real-key")
	m := newModel(cfg)
	if len(m.snap.Tasks) != 7 || m.demo == nil || m.selectedPack().ID != "task-readiness" {
		t.Fatalf("demo model: tasks=%d demo=%v pack=%v", len(m.snap.Tasks), m.demo != nil, m.selectedPack())
	}
	if !m.client.IsMock() {
		t.Fatal("--demo must use the mock unless --live")
	}
	cfg.Live = true
	if newModel(cfg).client.IsMock() {
		t.Fatal("--demo --live keeps the real client")
	}
	cfg.Live = false
	f := &fixture{m: resize(quiet(m), 120, 40), cfg: cfg}
	before := f.m.snap.Records
	for i := 0; i < 12 && f.m.snap.Records == before; i++ {
		f.send(t, demoTickMsg{})
		f.send(t, pollTickMsg{})
	}
	if f.m.snap.Records == before {
		t.Fatal("demo steps should append records that the poll picks up")
	}
}
