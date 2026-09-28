package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"alembic/harness"
	"alembic/jev"
)

// Adversarial QA: crashes, freezes, outbox/receipt integrity and visible
// layout defects. Every fixture uses the MOCK Jev client (no API key).

var awkwardSizes = [][2]int{{80, 24}, {100, 30}, {120, 40}, {200, 60}, {79, 23}, {90, 25}, {140, 35}, {60, 20}, {40, 12}}

// isQuit runs a Cmd with a timeout and reports whether it yields tea.QuitMsg.
func isQuit(cmd tea.Cmd) bool {
	if cmd == nil {
		return false
	}
	ch := make(chan tea.Msg, 1)
	go func() { ch <- cmd() }()
	select {
	case msg := <-ch:
		if _, ok := msg.(tea.QuitMsg); ok {
			return true
		}
		if b, ok := msg.(tea.BatchMsg); ok {
			for _, c := range b {
				if isQuit(c) {
					return true
				}
			}
		}
	case <-time.After(300 * time.Millisecond):
	}
	return false
}

// countMsgs runs a Cmd (recursing into batches) and counts messages that
// satisfy pred within the deadline.
func countMsgs(cmd tea.Cmd, pred func(tea.Msg) bool, wait time.Duration) int {
	if cmd == nil {
		return 0
	}
	ch := make(chan tea.Msg, 1)
	go func() { ch <- cmd() }()
	select {
	case msg := <-ch:
		if b, ok := msg.(tea.BatchMsg); ok {
			n := 0
			for _, c := range b {
				n += countMsgs(c, pred, wait)
			}
			return n
		}
		if pred(msg) {
			return 1
		}
	case <-time.After(wait):
	}
	return 0
}

// typeFast types without pumping the returned Cmds (huh/textinput blink
// timers), for long strings where only the resulting frame matters.
func (f *fixture) typeFast(s string) {
	for _, r := range s {
		mm, _ := f.m.Update(keyMsg(string(r)))
		f.m = mm.(model)
	}
}

func (f *fixture) toastIs(t *testing.T, style string, contains string) {
	t.Helper()
	var want string
	switch style {
	case "ok":
		want = toastOK.Render("x")
	case "warn":
		want = toastWarn.Render("x")
	case "err":
		want = toastErr.Render("x")
	}
	if !strings.Contains(f.m.toast.text, contains) {
		t.Fatalf("toast %q should contain %q", f.m.toast.text, contains)
	}
	if got := f.m.toast.style.Render("x"); got != want {
		t.Fatalf("toast %q has the wrong style (want %s)", f.m.toast.text, style)
	}
}

// ---------- bugs ----------

// A receipt that cannot be saved must leave the red toast on screen; the
// decision toast used to overwrite it in the same Update.
func TestQAReceiptSaveFailureToastStays(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	blocker := filepath.Join(t.TempDir(), "not-a-dir")
	os.WriteFile(blocker, []byte("x"), 0o644)
	f.m.cfg.Receipts = filepath.Join(blocker, "receipts") // MkdirAll fails even as root
	f.selectTask(t, "T-1041")
	f.gotoJevPack(t, "task-readiness")
	f.press(t, "enter")
	if f.m.jv.result == nil || f.m.jv.running {
		t.Fatal("the run itself should still complete")
	}
	f.toastIs(t, "err", "receipt not saved")
	checkFrame(t, f.m, 120, 40, "after failed save")
	// the task-side run (J) too
	f.press(t, "1")
	f.selectTask(t, "T-1042")
	f.press(t, "J")
	f.toastIs(t, "err", "receipt not saved")
}

// Init schedules the spinner tick without marking it pending, so the first
// feed message started a second 90 ms loop: two loops for the whole session.
func TestQAInitStartsExactlyOneAnimLoop(t *testing.T) {
	f := newFixture(t)
	if !f.m.needsAnim() {
		t.Fatal("fixture should animate (running tasks on the Tasks tab)")
	}
	loops := countMsgs(f.m.Init(), func(m tea.Msg) bool { _, ok := m.(animTickMsg); return ok }, 500*time.Millisecond)
	if f.m.kickAnim() != nil {
		loops++
	}
	if loops != 1 {
		t.Fatalf("Init + first kickAnim started %d animation loops, want exactly 1", loops)
	}
	// and the loop keeps going only while something animates
	mm, cmd := f.m.Update(animTickMsg{})
	f.m = mm.(model)
	if cmd == nil {
		t.Fatal("tick should continue while tasks run")
	}
	f.press(t, "4")
	mm, cmd = f.m.Update(animTickMsg{})
	f.m = mm.(model)
	if cmd != nil || f.m.kickAnim() != nil {
		t.Fatal("tick must stop on an idle tab")
	}
}

// jev.receipt carries task_id only when the state came from a task; for a
// diff or file pack the state ref is a path and must not be sent as a task id.
func TestQASendReceiptTaskIDOnlyForTaskSources(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	f.press(t, "2")
	os.WriteFile(filepath.Join(f.repo, "a.txt"), []byte("changed\n"), 0o644)
	f.gotoJevPack(t, "pr-triage")
	f.press(t, "enter")
	if f.m.jv.result == nil || f.m.jv.result.receipt.StateSource != "working_diff" {
		t.Fatalf("diff pack should run: %+v", f.m.jv.result)
	}
	f.press(t, "s", "S")
	f.typeText(t, "note")
	f.press(t, "enter")
	cmds := readOutbox(t, f.outbox)
	if len(cmds) != 2 {
		t.Fatalf("outbox: %+v", cmds)
	}
	for _, c := range cmds {
		if c.Type != "jev.receipt" || c.TaskID != "" || c.Data == nil {
			t.Fatalf("diff receipt must not carry a path as task_id: %+v", c)
		}
	}
	// task-sourced receipts keep the task id
	f.selectTask(t, "T-1042")
	f.gotoJevPack(t, "task-readiness")
	f.press(t, "enter", "s")
	cmds = readOutbox(t, f.outbox)
	if last := cmds[len(cmds)-1]; last.TaskID != "T-1042" {
		t.Fatalf("task receipt should carry the task id: %+v", last)
	}
}

// A response that lacks an answer, names a choice outside its probabilities,
// or scores outside the legend must render (n/a for the missing one), never panic.
func TestQAMalformedAnswersRender(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.gotoJevPack(t, "task-readiness")
	p := f.m.selectedPack()
	half, conf, score := 0.5, 0.7, 9.0
	resp := &jev.Response{Model: "jev-mock", Mock: true, Answers: map[string]jev.Answer{
		"stuck": {Type: jev.Noul, Noul: &half},
		"next":  {Type: jev.Choice, Choice: "teleport", Probabilities: map[string]float64{"wait": 0.5, "ping": 0.5}, Confidence: &conf},
		"risk":  {Type: jev.Score, Score: &score, Legend: map[string]string{"0": "None", "1": "Low"}, Probabilities: map[string]float64{"0": 0.5, "1": 0.5}, Confidence: &conf},
		"bogus": {Type: "weird"},
	}}
	rc := jev.NewReceipt(p, "T-1041", []byte("{}"), resp, p.Evaluate(resp))
	f.m.jv.result = &jevResult{packID: p.ID, receipt: rc, path: "/nope"}
	f.m.jv.focus = jevFocusResult
	v := f.view()
	if !strings.Contains(v, "needs_human") || !strings.Contains(v, "n/a") {
		t.Fatalf("unanswered question should show as n/a:\n%s", v)
	}
	for _, s := range []string{"teleport", "9.00", "bogus"} {
		if !strings.Contains(v, s) {
			t.Errorf("result missing %q", s)
		}
	}
	for _, sz := range awkwardSizes {
		f.m = resize(f.m, sz[0], sz[1])
		checkFrame(t, f.m, sz[0], sz[1], "malformed answers")
	}
	f.press(t, "h") // history with the compare view over these answers
	if f.m.jv.history.open {
		f.press(t, "c", "j")
		checkFrame(t, f.m, 40, 12, "compare malformed")
		f.press(t, "esc")
	}
}

// Feed text is untrusted: a newline in a title, status line, event or agent
// name must not add screen lines (bubbletea drops the TOP of an over-tall view).
func TestQANewlinesInFeedTextKeepFrame(t *testing.T) {
	f := newFixture(t)
	now := time.Now().UTC()
	f.send(t, feedMsg{recs: []harness.FeedRecord{
		{TS: now, Type: "workflow.upsert", Workflow: &harness.Workflow{ID: "checkout", Name: "checkout\nservice", Env: "prod"}},
		{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-1041", Workflow: "checkout", Title: "Line one\nline two", Agent: "coder-1", State: harness.StateRunning, StatusLine: "x\r\ny\rz", Priority: 1}},
		{TS: now, Type: "task.event", Event: &harness.Event{TaskID: "T-1041", Level: harness.LevelWarn, Text: "multi\nline\nevent"}},
		{TS: now, Type: "agent.upsert", Agent: &harness.Agent{ID: "coder-1", Name: "Co\nder", State: "busy"}},
	}})
	f.selectTask(t, "T-1041")
	for _, sz := range awkwardSizes {
		f.m = resize(f.m, sz[0], sz[1])
		for _, tab := range []string{"1", "4", "1"} {
			f.press(t, tab)
			checkFrame(t, f.m, sz[0], sz[1], "newline text tab "+tab)
			lines := strings.Split(f.view(), "\n")
			if last := lines[len(lines)-1]; !strings.Contains(last, "j/k") {
				t.Fatalf("@%dx%d tab %s: status bar pushed off screen; last line %q", sz[0], sz[1], tab, last)
			}
			if strings.Contains(f.view(), "\r") {
				t.Fatalf("@%dx%d: carriage return leaked into the frame", sz[0], sz[1])
			}
		}
		f.press(t, "tab")
		checkFrame(t, f.m, sz[0], sz[1], "newline text detail focus")
		f.press(t, "tab", "x")
		checkFrame(t, f.m, sz[0], sz[1], "newline text confirm")
		f.press(t, "n", "p")
		checkFrame(t, f.m, sz[0], sz[1], "newline text composer")
		lines := strings.Split(f.view(), "\n")
		if !strings.Contains(lines[len(lines)-1], "enter send") {
			t.Fatalf("composer pushed the status bar off: %q", lines[len(lines)-1])
		}
		f.press(t, "esc")
	}
}

// ---------- layout stress ----------

func cjkTitle(n int) string {
	var b strings.Builder
	for b.Len() < n*3 {
		b.WriteString("日本語テキスト🚀🔥 lorem ")
	}
	return string([]rune(b.String())[:n])
}

// bigFixture loads a world designed to overflow every column and pane.
func bigFixture(t *testing.T) *fixture {
	t.Helper()
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	packs := t.TempDir()
	src, _ := filepath.Abs("packs")
	entries, _ := os.ReadDir(src)
	for _, e := range entries {
		data, _ := os.ReadFile(filepath.Join(src, e.Name()))
		os.WriteFile(filepath.Join(packs, e.Name()), data, 0o644)
	}
	opts := map[string]string{}
	for i := 0; i < 30; i++ {
		opts[fmt.Sprintf("option-number-%02d", i)] = "an option"
	}
	levels := make([]string, 10)
	for i := range levels {
		levels[i] = fmt.Sprintf("Level %d of ten with a long name", i)
	}
	big, _ := json.Marshal(map[string]any{
		"id": "big", "name": strings.Repeat("Big pack ", 10), "description": strings.Repeat("very long description ", 10), "state_source": "text",
		"questions": map[string]any{
			"pick":  map[string]any{"type": "choice", "instructions": strings.Repeat("which? ", 40), "criteria": opts},
			"grade": map[string]any{"type": "score", "instructions": "how much?", "criteria": levels},
			"ok":    map[string]any{"type": "noul", "instructions": "is it?"},
		},
		"gate": map[string]any{"action": "x.y", "favorable": map[string]string{"ok": "yes"}, "minProbability": 0.5, "autoConfidence": 0.5, "refuseBelow": 0.1},
	})
	os.WriteFile(filepath.Join(packs, "big.json"), big, 0o644)
	f.m.cfg.Packs = packs
	f.m.loadPacks()

	now := time.Now().UTC()
	longPath := filepath.Join(f.repo, strings.Repeat("deep-directory-name/", 9), "leaf")
	var recs []harness.FeedRecord
	for i := 0; i < 15; i++ {
		recs = append(recs, harness.FeedRecord{TS: now, Type: "workflow.upsert", Workflow: &harness.Workflow{ID: fmt.Sprintf("wf%02d", i), Name: fmt.Sprintf("workflow-%02d-%s", i, strings.Repeat("x", 40)), Env: "prod", State: "degraded"}})
		recs = append(recs, harness.FeedRecord{TS: now, Type: "task.upsert", Task: &harness.Task{ID: fmt.Sprintf("W-%02d", i), Workflow: fmt.Sprintf("wf%02d", i), Title: "t", State: harness.StateQueued, StatusLine: "s"}})
	}
	var elems []harness.Element
	for i := 0; i < 40; i++ {
		elems = append(elems, harness.Element{Kind: harness.ElemFile, Ref: fmt.Sprintf("path/to/file-%02d.go", i), Line: i, Label: strings.Repeat("label ", 10)})
	}
	recs = append(recs, harness.FeedRecord{TS: now, Type: "task.upsert", Task: &harness.Task{
		ID: "T-BIG", Workflow: "checkout", Title: cjkTitle(300), Agent: "coder-1", State: harness.StateRunning, Progress: 0.5,
		StatusLine: strings.Repeat("status line that never ends · ", 17)[:500], Worktree: longPath, Branch: strings.Repeat("feat/long-branch-", 6), Priority: 2, Elements: elems}})
	for i := 0; i < 60; i++ {
		recs = append(recs, harness.FeedRecord{TS: now.Add(time.Duration(i) * time.Second), Type: "task.upsert", Task: &harness.Task{ID: fmt.Sprintf("C-%03d", i), Workflow: "checkout", Title: fmt.Sprintf("checkout task %d", i), Agent: "coder-1", State: harness.StateRunning, StatusLine: "working"}})
		recs = append(recs, harness.FeedRecord{TS: now, Type: "task.event", Event: &harness.Event{TaskID: "T-BIG", Level: harness.LevelInfo, Text: strings.Repeat("event ", 60)}})
	}
	recs = append(recs, harness.FeedRecord{TS: now, Type: "agent.upsert", Agent: &harness.Agent{ID: "long-1", Name: strings.Repeat("LongAgentName", 5), Model: strings.Repeat("model-", 10), State: "busy", CurrentTask: "T-BIG"}})
	f.send(t, feedMsg{recs: recs})

	// 50 receipts for task-readiness
	p := f.m.findPack("task-readiness")
	resp, _ := f.m.client.Ask(contextBG(), "s", p.Questions)
	for i := 0; i < 50; i++ {
		rc := jev.NewReceipt(p, "T-BIG", []byte("s"), resp, p.Evaluate(resp))
		rc.At = now.Add(-time.Duration(i) * time.Minute)
		rc.ID = fmt.Sprintf("%s-%02d-task-readiness", rc.At.Format("20060102T150405"), i)
		if _, err := jev.SaveReceipt(f.m.cfg.Receipts, rc); err != nil {
			t.Fatal(err)
		}
	}
	return f
}

func TestQALayoutStressEveryTabFocusOverlay(t *testing.T) {
	f := bigFixture(t)
	f.press(t, "2") // list worktrees
	f.selectTask(t, "T-BIG")
	f.m.tasks.elemCursor = 39
	for _, sz := range awkwardSizes {
		f.m = resize(f.m, sz[0], sz[1])
		label := func(s string) string { return fmt.Sprintf("%s @%dx%d", s, sz[0], sz[1]) }
		chk := func(s string) { t.Helper(); checkFrame(t, f.m, sz[0], sz[1], label(s)) }

		f.press(t, "1")
		chk("tasks list")
		f.press(t, "G")
		chk("tasks end")
		f.selectTask(t, "T-BIG")
		f.press(t, "tab")
		chk("tasks detail")
		f.press(t, "G")
		chk("tasks detail last element")
		f.press(t, "tab", "/")
		f.typeFast("zzzz")
		chk("tasks filter no match")
		f.press(t, "esc", "w")
		chk("tasks workflow filter")
		f.m.tasks.wfFilter = ""
		f.m.rebuildRows()
		f.press(t, "ctrl+k")
		chk("palette")
		f.typeFast("qqq")
		chk("palette zero matches")
		f.press(t, "enter")
		if !f.m.palette.open {
			t.Fatal("enter on zero matches keeps the palette open")
		}
		f.press(t, "esc", "?")
		chk("help")
		f.press(t, "j", "j", "j", "j", "j")
		chk("help scrolled")
		f.press(t, "esc")
		f.selectTask(t, "T-BIG")
		f.press(t, "x")
		chk("confirm")
		f.press(t, "n", "p")
		chk("composer")
		f.typeFast(strings.Repeat("ping text ", 25))
		chk("composer full")
		f.press(t, "esc", "P")
		chk("picker")
		f.press(t, "esc")

		f.press(t, "2")
		chk("worktrees")
		f.press(t, "n")
		chk("form")
		f.typeFast(strings.Repeat("feat/very-long-branch-", 4))
		chk("form typed")
		f.press(t, "esc", "J")
		chk("diff picker")
		f.press(t, "esc")

		f.gotoJevPack(t, "big")
		chk("jev packs")
		f.press(t, "tab")
		chk("jev state")
		f.press(t, "e")
		chk("jev editing")
		f.typeFast("hello")
		f.press(t, "esc", "tab")
		chk("jev result questions")
		f.press(t, "enter")
		if f.m.jv.result == nil {
			t.Fatal("big pack should run")
		}
		chk("jev result receipt")
		f.press(t, "G")
		chk("jev result scrolled")
		f.press(t, "j", "j", "ctrl+d")
		chk("jev result scrolled more")

		f.gotoJevPack(t, "pr-triage")
		f.m.jv.state = bytes.Repeat([]byte("+ a diff line that is fairly long and repeats itself over and over\n"), 32*1024)
		f.m.jv.stateFor = f.m.jevStateKey()
		chk("jev 2MB state")
		f.gotoJevPack(t, "task-readiness")
		f.press(t, "h")
		if !f.m.jv.history.open || len(f.m.jv.history.list) != 50 {
			t.Fatalf("history: open=%v n=%d", f.m.jv.history.open, len(f.m.jv.history.list))
		}
		chk("history 50")
		f.press(t, "c", "j", "j")
		chk("history compare")
		f.press(t, "G")
		chk("history end")
		f.press(t, "esc", "T")
		chk("jev task picker")
		f.press(t, "esc", "W")
		chk("jev worktree picker")
		f.press(t, "esc")

		f.press(t, "4")
		chk("agents")
		f.press(t, "G")
		chk("agents end")
	}
}

// ---------- feed robustness ----------

func TestQAFeedRobustness(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	now := time.Now().UTC()
	f.press(t, "p")
	f.typeText(t, "keep me")
	f.send(t, feedMsg{recs: []harness.FeedRecord{
		{TS: now, Type: "something.new"},
		{TS: now, Type: "task.upsert"},
		{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-NOSTATE", Workflow: "checkout", Title: "no state"}},
		{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-LATE", Workflow: "later", Title: "wf later", State: harness.StateQueued}},
		{TS: now, Type: "task.event", Event: &harness.Event{TaskID: "T-UNKNOWN", Level: harness.LevelError, Text: "orphan"}},
		{TS: now, Type: "ping.ack", PingAck: &harness.PingAck{PingID: "never-sent", Agent: "ghost", Text: "ack"}},
		{TS: now, Type: "workflow.upsert", Workflow: &harness.Workflow{ID: "later", Name: "Later WF"}},
	}})
	if !f.m.composer.open || f.m.composer.input.Value() != "keep me" {
		t.Fatalf("composer text lost across a feed update: %+v", f.m.composer.input.Value())
	}
	f.press(t, "esc")
	v := f.view()
	for _, s := range []string{"T-NOSTATE", "Later WF", "T-LATE", "ghost: ack"} {
		if !strings.Contains(v, s) {
			t.Errorf("view missing %q", s)
		}
	}
	if f.m.tasks.selID != "T-1041" {
		t.Fatalf("selection moved to %s", f.m.tasks.selID)
	}
	if _, ok := f.m.snap.Tasks["T-UNKNOWN"]; ok {
		t.Fatal("an event must not invent a task")
	}
	f.selectTask(t, "T-NOSTATE")
	f.press(t, "x", "n", "R", "J", "tab", "j", "o", "tab")
	checkFrame(t, f.m, 120, 40, "task without state")

	// feed truncated to zero and rewritten while running
	os.WriteFile(f.feed, nil, 0o644)
	f.send(t, pollTickMsg{})
	f.m.feed.Poll()
	if err := harness.NewDemo(f.feed, f.outbox).Seed(f.repo); err != nil {
		t.Fatal(err)
	}
	f.send(t, pollTickMsg{})
	if f.m.tasks.selID == "" || f.m.selectedTask() == nil {
		t.Fatalf("selection after truncation: %q", f.m.tasks.selID)
	}
	checkFrame(t, f.m, 120, 40, "after truncation")

	// 10,000 records in one poll: events and new tasks
	var recs []harness.FeedRecord
	for i := 0; i < 5000; i++ {
		recs = append(recs, harness.FeedRecord{TS: now, Type: "task.event", Event: &harness.Event{TaskID: "T-1041", Level: harness.LevelInfo, Text: fmt.Sprintf("event %d", i)}})
		recs = append(recs, harness.FeedRecord{TS: now, Type: "task.upsert", Task: &harness.Task{ID: fmt.Sprintf("T-%05d", i), Workflow: fmt.Sprintf("wf-%d", i%15), Title: "bulk", State: harness.StateQueued, StatusLine: "x"}})
	}
	start := time.Now()
	f.send(t, feedMsg{recs: recs})
	_ = f.m.View()
	f.press(t, "G", "g", "j")
	if d := time.Since(start); d > time.Second {
		t.Fatalf("10,000 records took %v", d)
	}
	if len(f.m.snap.Tasks) < 5000 {
		t.Fatal("records not applied")
	}
}

// ---------- keys / state machine ----------

func TestQAKeysNeverQuitInsideInputs(t *testing.T) {
	f := newFixture(t)
	open := map[string]func(){
		"composer": func() { f.selectTask(t, "T-1041"); f.press(t, "1", "p") },
		"picker":   func() { f.press(t, "1", "P") },
		"palette":  func() { f.press(t, "ctrl+k") },
		"filter":   func() { f.press(t, "1", "/") },
		"form":     func() { f.press(t, "2", "n") },
		"editor":   func() { f.gotoJevPack(t, "ping-priority"); f.press(t, "e") },
		"history":  func() { f.gotoJevPack(t, "task-readiness"); f.press(t, "enter", "h") },
		"confirm":  func() { f.press(t, "1"); f.selectTask(t, "T-1041"); f.press(t, "x") },
		"help":     func() { f.press(t, "?") },
	}
	for name, o := range open {
		o()
		mm, cmd := f.m.Update(keyMsg("q"))
		f.m = mm.(model)
		if isQuit(cmd) {
			t.Fatalf("q inside %s must not quit", name)
		}
		_, cmd = f.m.Update(keyMsg("ctrl+c"))
		if !isQuit(cmd) {
			t.Fatalf("ctrl+c inside %s must quit", name)
		}
		// esc closes exactly this layer and nothing else
		before := f.m.tab
		f.press(t, "esc")
		if f.m.anyOverlay() || f.m.composer.open || f.m.tasks.filtering || f.m.jv.editing {
			// history/help/confirm close on q already; anything else must be gone after esc
			t.Fatalf("esc did not close %s", name)
		}
		if f.m.tab != before {
			t.Fatalf("esc changed the tab after %s", name)
		}
		checkFrame(t, f.m, 120, 40, "after "+name)
	}
	f.press(t, "1", "?", "?")
	if f.m.help {
		t.Fatal("? twice should close help")
	}
}

func TestQATabCyclesAndMouse(t *testing.T) {
	f := newFixture(t)
	f.press(t, "1")
	f.press(t, "tab", "tab")
	if f.m.tasks.focus != focusList {
		t.Fatal("tasks tab cycle should return to the list")
	}
	f.press(t, "shift+tab", "shift+tab")
	if f.m.tasks.focus != focusList {
		t.Fatal("shift+tab cycle should return")
	}
	f.press(t, "3", "tab", "tab", "tab")
	if f.m.jv.focus != jevFocusPacks {
		t.Fatal("jev tab cycle should return to packs")
	}
	f.press(t, "shift+tab")
	if f.m.jv.focus != jevFocusResult {
		t.Fatal("shift+tab goes backwards")
	}
	f.press(t, "2", "tab", "4", "tab")
	checkFrame(t, f.m, 120, 40, "tab on tabs without focus")

	click := func(x, y int) {
		f.send(t, tea.MouseMsg{X: x, Y: y, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress})
	}
	wheel := func(down bool) {
		b := tea.MouseButtonWheelUp
		if down {
			b = tea.MouseButtonWheelDown
		}
		f.send(t, tea.MouseMsg{X: 5, Y: 5, Button: b, Action: tea.MouseActionPress})
	}
	x := 0
	for i, name := range tabNames {
		click(x+1, 1)
		if f.m.tab != tab(i) {
			t.Fatalf("clicking %s selected tab %d", name, f.m.tab)
		}
		x += len(name) + 4
	}
	click(x+5, 1) // past the last tab
	click(3, 0)   // header
	click(3, f.m.height-1)
	click(3, f.m.height+5)
	click(f.m.width+10, 3)
	for _, tab := range []string{"1", "2", "3", "4"} {
		f.press(t, tab)
		for i := 0; i < 5; i++ {
			wheel(true)
		}
		for i := 0; i < 8; i++ {
			wheel(false)
		}
		click(2, 4)
		click(f.m.width-2, f.m.height-3)
		checkFrame(t, f.m, 120, 40, "after mouse on tab "+tab)
	}
	f.press(t, "1", "tab")
	for i := 0; i < 30; i++ {
		wheel(true)
	}
	checkFrame(t, f.m, 120, 40, "detail wheel")
	f.press(t, "3", "tab", "tab")
	for i := 0; i < 30; i++ {
		wheel(true)
	}
	checkFrame(t, f.m, 120, 40, "result wheel")
	// collapsed detail pane: clicking where it would be must not break the list
	f.m = resize(f.m, 40, 12)
	f.press(t, "1")
	click(38, 5)
	f.press(t, "j", "k", "o")
	checkFrame(t, f.m, 40, 12, "collapsed detail click")
}

func TestQAEmptyThingsWarnInsteadOfCrash(t *testing.T) {
	f := newFixture(t)
	now := time.Now().UTC()
	f.send(t, feedMsg{recs: []harness.FeedRecord{
		{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-BARE", Workflow: "checkout", Title: "bare", State: harness.StateQueued}},
		{TS: now, Type: "agent.upsert", Agent: &harness.Agent{ID: "idle-9", Name: "Idler", State: "idle"}},
	}})
	f.selectTask(t, "T-BARE")
	f.press(t, "J")
	if tj := f.m.taskJev["T-BARE"]; tj == nil || tj.receipt == nil {
		t.Fatal("J on a task with zero events should still run")
	}
	f.press(t, "p")
	if f.m.composer.open {
		t.Fatal("p without agent must not open the composer")
	}
	f.toastIs(t, "warn", "no agent")
	f.press(t, "P")
	f.typeText(t, "idl")
	f.press(t, "enter")
	if !f.m.composer.open || f.m.composer.agentID != "idle-9" || f.m.composer.taskID != "T-BARE" {
		t.Fatalf("P should pick the agent: %+v", f.m.composer)
	}
	f.press(t, "esc", "o")
	f.toastIs(t, "warn", "no elements")
	f.press(t, "tab", "o", "y", "tab")
	f.press(t, "4")
	for i := 0; i < 10; i++ {
		f.press(t, "j")
	}
	if f.m.selectedAgent().ID != "idle-9" && f.m.selectedAgent().Name != "Tester" {
		// agents are sorted by name; find the idle one explicitly
		for i, a := range f.m.agentList() {
			if a.ID == "idle-9" {
				f.m.agents.cursor = i
			}
		}
	}
	for i, a := range f.m.agentList() {
		if a.ID == "idle-9" {
			f.m.agents.cursor = i
		}
	}
	f.press(t, "enter")
	if f.m.tab != tabAgents {
		t.Fatal("enter on an agent without a task must stay on the Agents tab")
	}
	f.toastIs(t, "warn", "no current task")
	f.press(t, "ctrl+k")
	f.typeText(t, "zzzzzz")
	f.press(t, "enter")
	if !f.m.palette.open {
		t.Fatal("enter with zero palette matches should be a no-op")
	}
	f.press(t, "esc")
}

// ---------- jev edges ----------

func TestQAJevSourcesAndRunGuards(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	// staged_diff on a worktree with nothing staged
	f.press(t, "2")
	f.gotoJevPack(t, "commit-safety")
	if !strings.Contains(f.view(), "state is empty") {
		t.Fatal("no staged changes should warn")
	}
	f.press(t, "enter")
	if f.m.jv.result != nil || f.m.jv.running {
		t.Fatal("enter must not send an empty state")
	}
	f.toastIs(t, "warn", "empty")
	// staged_diff outside any repo
	f.m.cfg.RepoOK = false
	f.m.wt.list = nil
	f.m.jv.wtPath = ""
	f.pump(t, f.m.jevStateCmd(), 0)
	if src, ref := f.m.jevSource(); src != "staged_diff" || ref != "" {
		t.Fatalf("source %s ref %q", src, ref)
	}
	if !strings.Contains(f.view(), "no worktree") {
		t.Fatalf("no-worktree state should warn:\n%s", f.view())
	}
	f.press(t, "enter")
	if f.m.jv.result != nil {
		t.Fatal("enter without a worktree must not run")
	}
	f.m.cfg.RepoOK = true

	// file source pointing nowhere
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "filepack.json"), []byte(`{"id":"filepack","name":"File pack","state_source":"file","questions":{"ok":{"type":"noul","instructions":"Is it fine?"}}}`), 0o644)
	f.m.cfg.Packs = dir
	f.press(t, "r")
	f.gotoJevPack(t, "filepack")
	f.press(t, "e")
	f.typeText(t, filepath.Join(dir, "missing.json"))
	f.press(t, "enter")
	if f.m.jv.stateErr == nil {
		t.Fatal("missing file should be an error")
	}
	f.press(t, "enter")
	if f.m.jv.result != nil || f.m.jv.running {
		t.Fatal("enter on a missing file must not run")
	}

	// text source: paste 100 KB
	f.m.cfg.Packs, _ = filepath.Abs("packs")
	f.press(t, "r")
	f.gotoJevPack(t, "ping-priority")
	f.press(t, "e")
	start := time.Now()
	f.send(t, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(strings.Repeat("paste ", 17000))})
	if d := time.Since(start); d > time.Second {
		t.Fatalf("100 KB paste took %v", d)
	}
	f.press(t, "esc")
	if len(f.m.jv.state) == 0 {
		t.Fatal("pasted state should be kept")
	}
	checkFrame(t, f.m, 120, 40, "after paste")
	f.press(t, "enter")
	if f.m.jv.result == nil {
		t.Fatal("pasted text should run")
	}

	// running twice quickly: the second enter is refused, the first result lands
	f.gotoJevPack(t, "task-readiness")
	f.selectTask(t, "T-1041")
	f.pump(t, f.m.jevStateCmd(), 0)
	mm, first := f.m.Update(keyMsg("enter"))
	f.m = mm.(model)
	if !f.m.jv.running {
		t.Fatal("first enter should start a run")
	}
	runID := f.m.jv.runID
	mm, second := f.m.Update(keyMsg("enter"))
	f.m = mm.(model)
	f.toastIs(t, "warn", "already")
	f.pump(t, second, 0)
	// a stale done message from a superseded run must be ignored
	stale := jevDoneMsg{runID: runID - 100, taskID: "T-1041", pack: f.m.selectedPack(), stateRef: "T-1041", resp: &jev.Response{Answers: map[string]jev.Answer{}}}
	f.send(t, stale)
	if !f.m.jv.running || f.m.jv.result != nil && f.m.jv.result.packID == "task-readiness" {
		t.Fatal("stale run result must not clobber the in-flight run")
	}
	f.pump(t, first, 0)
	if f.m.jv.running || f.m.jv.result == nil || f.m.jv.result.packID != "task-readiness" {
		t.Fatalf("first run should land: running=%v result=%+v", f.m.jv.running, f.m.jv.result)
	}
	// history with a single mark and cursor on it: no compare, no crash
	f.press(t, "h", "c")
	if !f.m.jv.history.open || strings.Contains(f.view(), "  →  ") {
		t.Fatal("marking the cursor row alone must not show a compare section")
	}
	checkFrame(t, f.m, 120, 40, "single mark")
	f.press(t, "esc")
}

// ---------- worktrees ----------

func TestQAWorktreeEdges(t *testing.T) {
	f := newFixture(t)
	f.press(t, "2", "n")
	f.typeText(t, "feat x y")
	f.press(t, "enter", "enter")
	if f.m.form == nil {
		t.Fatal("an invalid branch name must keep the form open")
	}
	if !strings.Contains(f.view(), "not a valid git ref") {
		t.Fatalf("form should show the validation error:\n%s", f.view())
	}
	// a refresh landing while the form is open must not reset it
	f.send(t, wtTickMsg{})
	f.send(t, wtListMsg{wts: f.m.wt.list})
	if f.m.form == nil || f.m.formVals.branch != "feat x y" {
		t.Fatalf("refresh reset the form: %+v", f.m.formVals)
	}
	f.press(t, "esc")
	if _, err := harness.AddWorktree(f.repo, "feat x y", ""); err == nil || !strings.Contains(err.Error(), "invalid branch") {
		t.Fatalf("AddWorktree should reject spaces: %v", err)
	}
	// d on main refuses; d on dirty arms --force
	f.press(t, "g", "d")
	if f.m.confirm.open {
		t.Fatal("main worktree must not be removable")
	}
	f.toastIs(t, "warn", "main worktree")
	os.WriteFile(filepath.Join(f.wtPath, "dirty.txt"), []byte("d"), 0o644)
	f.press(t, "r", "j", "d")
	if !f.m.confirm.open || !strings.Contains(f.view(), "uncommitted") {
		t.Fatalf("dirty removal should confirm with a warning:\n%s", f.view())
	}
	f.press(t, "y")
	if len(f.m.wt.list) != 2 {
		t.Fatal("first confirmation only arms --force")
	}
	f.press(t, "d")
	if !f.m.confirm.open || !strings.Contains(f.view(), "FORCE") {
		t.Fatal("second d should ask for a forced removal")
	}
	f.press(t, "n", "r")
	if len(f.m.wt.list) != 2 {
		t.Fatal("n must not remove")
	}
	if err := harness.RemoveWorktree(f.repo, f.wtPath, false); err == nil {
		t.Fatal("dirty removal without force must fail")
	}
}

func TestQAGitMissingFromPath(t *testing.T) {
	f := newFixture(t)
	t.Setenv("PATH", t.TempDir())
	f.press(t, "2", "r")
	if f.m.wt.err == nil || !strings.Contains(f.view(), "git error") {
		t.Fatalf("missing git should show a friendly error:\n%s", f.view())
	}
	checkFrame(t, f.m, 120, 40, "git missing")
	f.press(t, "n")
	f.typeText(t, "feat/nogit")
	f.press(t, "enter", "enter")
	if f.m.form != nil {
		t.Fatal("form should submit")
	}
	f.toastIs(t, "err", "")
	f.press(t, "J", "w")
	checkFrame(t, f.m, 120, 40, "git missing diff pack")
	f.press(t, "3", "W", "esc")
	f.gotoJevPack(t, "commit-safety")
	f.press(t, "enter")
	if f.m.jv.result != nil {
		t.Fatal("no git → no diff → no run")
	}
	checkFrame(t, f.m, 120, 40, "git missing jev")
}

// ---------- outbox / receipts ----------

func TestQAOutboxIntegrity(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	f.selectTask(t, "T-1041")
	f.press(t, "p")
	f.typeText(t, "one")
	f.press(t, "enter", "p")
	f.typeText(t, "two")
	f.press(t, "enter")
	cmds := readOutbox(t, f.outbox)
	if len(cmds) != 2 || cmds[0].ID == cmds[1].ID || cmds[0].ID == "" {
		t.Fatalf("two pings need two distinct ids: %+v", cmds)
	}
	f.selectTask(t, "T-2201") // done
	f.press(t, "x")
	if f.m.confirm.open {
		t.Fatal("cancel on a done task must not confirm")
	}
	f.toastIs(t, "warn", "already done")
	f.press(t, "y")
	if len(readOutbox(t, f.outbox)) != 2 {
		t.Fatal("nothing must be written for a refused cancel")
	}
	// receipt round trip with every answer type
	f.gotoJevPack(t, "task-readiness")
	f.selectTask(t, "T-1041")
	f.pump(t, f.m.jevStateCmd(), 0)
	f.press(t, "enter")
	rc := f.m.jv.result.receipt
	loaded := jev.LoadReceipts(f.m.cfg.Receipts, "task-readiness")
	if len(loaded) != 1 {
		t.Fatalf("receipts: %d", len(loaded))
	}
	a, _ := json.Marshal(rc)
	b, _ := json.Marshal(loaded[0])
	if string(a) != string(b) {
		t.Fatalf("receipt did not round-trip:\n%s\n%s", a, b)
	}
	types := map[jev.QuestionType]bool{}
	for _, ans := range loaded[0].Answers {
		types[ans.Type] = true
	}
	if !types[jev.Noul] || !types[jev.Choice] || !types[jev.Score] {
		t.Fatalf("all answer types expected: %v", types)
	}
}

// ---------- animation / ticks ----------

func TestQATicksStopAndContinueCorrectly(t *testing.T) {
	f := newFixture(t)
	f.gotoJevPack(t, "task-readiness")
	f.selectTask(t, "T-1041")
	f.pump(t, f.m.jevStateCmd(), 0)
	f.press(t, "enter", "4")
	for i := 0; i < 3; i++ {
		mm, cmd := f.m.Update(animTickMsg{})
		f.m = mm.(model)
		if cmd != nil {
			t.Fatal("settled: the animation tick must return nil")
		}
	}
	if _, cmd := f.m.Update(pollTickMsg{}); cmd == nil {
		t.Fatal("the poll tick must continue")
	}
	if _, cmd := f.m.Update(demoTickMsg{}); cmd != nil {
		t.Fatal("no demo → the demo tick must stop")
	}
	if _, cmd := f.m.Update(wtTickMsg{}); cmd != nil {
		t.Fatal("off the Worktrees tab the worktree tick must stop")
	}
}
