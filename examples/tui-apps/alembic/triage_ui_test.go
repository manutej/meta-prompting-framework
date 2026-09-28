package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"alembic/harness"
	"alembic/jev"
)

// Triage in the TUI: scheduler, sort, chips, detail lines, what-next routing
// and the headless subcommand. Every fixture uses the MOCK client.

func triageCmds(t *testing.T, path string) []harness.Command {
	t.Helper()
	var out []harness.Command
	for _, c := range readOutbox(t, path) {
		if c.Type == "triage" {
			out = append(out, c)
		}
	}
	return out
}

// tick runs one scheduled triage tick to completion.
func (f *fixture) tick(t *testing.T) {
	t.Helper()
	f.send(t, triageTickMsg{})
	if f.m.triage.running || f.m.triage.last == nil {
		t.Fatalf("tick did not land: running=%v last=%v", f.m.triage.running, f.m.triage.last != nil)
	}
}

// inject applies a hand-built triage as if the scheduler produced it.
func (f *fixture) inject(t *testing.T, tr jev.Triage) {
	t.Helper()
	f.m.triage.running = true
	f.m.triage.seq++
	f.send(t, triageDoneMsg{seq: f.m.triage.seq, t: tr})
}

func TestTriageTickAppliesAndLogs(t *testing.T) {
	f := newFixture(t)
	if f.m.triage.last != nil || f.m.triage.tr == nil {
		t.Fatal("no triage before the first tick; one Triager per session")
	}
	f.tick(t)
	tr := f.m.triage.last
	if len(tr.Tasks) != 7 || tr.Tasks[0].Rank != 1 || tr.Tasks[6].Rank != 7 {
		t.Fatalf("triage: %+v", tr.Tasks)
	}
	if !tr.Mock || tr.JevCalls == 0 {
		t.Fatalf("the fixture's mock client should be consulted: mock=%v calls=%d", tr.Mock, tr.JevCalls)
	}
	logged := jev.LoadTriage(f.m.cfg.TriageLog, 10)
	if len(logged) != 1 || len(logged[0].Tasks) != 7 {
		t.Fatalf("log at %s: %d entries", f.m.cfg.TriageLog, len(logged))
	}
	if !strings.HasPrefix(f.m.cfg.TriageLog, filepath.Dir(f.cfg.Receipts)) {
		t.Fatalf("log should live beside the receipts dir: %s", f.m.cfg.TriageLog)
	}
	f.tick(t)
	if len(jev.LoadTriage(f.m.cfg.TriageLog, 10)) != 2 {
		t.Fatal("every tick appends")
	}
	// a stale result from a superseded run is ignored
	before := f.m.triage.last.At
	f.send(t, triageDoneMsg{seq: f.m.triage.seq - 1, t: jev.Triage{At: time.Now().Add(time.Hour)}})
	if !f.m.triage.last.At.Equal(before) {
		t.Fatal("stale run must not replace the current triage")
	}
}

func TestTriageOutboxOnlyOnChange(t *testing.T) {
	f := newFixture(t)
	f.tick(t)
	if n := len(triageCmds(t, f.outbox)); n != 1 {
		t.Fatalf("first tick → one triage command, got %d", n)
	}
	f.tick(t)
	if n := len(triageCmds(t, f.outbox)); n != 1 {
		t.Fatalf("unchanged ranking must not write: %d", n)
	}
	// the top task finishes: the top changes and so does its Next
	top := f.m.triage.last.Next().ID
	now := time.Now().UTC()
	task := *f.m.snap.Tasks[top]
	task.State, task.Progress = harness.StateDone, 1
	f.send(t, feedMsg{recs: []harness.FeedRecord{{TS: now, Type: "task.upsert", Task: &task}}})
	f.tick(t)
	cmds := triageCmds(t, f.outbox)
	if len(cmds) != 2 || f.m.triage.last.Next().ID == top {
		t.Fatalf("a changed ranking writes exactly one more command: %d (top %s → %s)", len(cmds), top, f.m.triage.last.Next().ID)
	}
	data, _ := json.Marshal(cmds[1].Data)
	var payload jev.Triage
	if json.Unmarshal(data, &payload) != nil || len(payload.Tasks) != 7 || payload.Tasks[0].Rank != 1 {
		t.Fatalf("triage payload: %s", data)
	}
	// only a Next change, same top
	prev := *f.m.triage.last
	mod := prev
	mod.Tasks = append([]jev.TaskTriage{}, prev.Tasks...)
	mod.Tasks[3].Next = jev.ActCancel
	if !triageChanged(&prev, mod) || triageChanged(&prev, prev) {
		t.Fatal("triageChanged: a Next change counts, an identical triage does not")
	}
}

func TestTriageSortToggleSmartOrder(t *testing.T) {
	f := newFixture(t)
	if f.m.triage.sort != sortStatus || !strings.Contains(f.view(), "Tasks · 7 · status") {
		t.Fatalf("status sort before any triage:\n%s", f.view())
	}
	f.tick(t)
	if f.m.triage.sort != sortSmart || !strings.Contains(f.view(), "Tasks · 7 · smart") {
		t.Fatal("the first triage switches to smart sort")
	}
	// within each workflow tasks ascend by rank; workflows ascend by their best rank
	lastRank, lastBest := 0, 0
	for _, r := range f.m.tasks.rows {
		switch r.kind {
		case rowHeader:
			lastRank = 0
		case rowTask:
			rank := f.m.triage.last.Get(r.task.ID).Rank
			if rank <= lastRank {
				t.Fatalf("smart order broken at %s: rank %d after %d", r.task.ID, rank, lastRank)
			}
			if lastRank == 0 {
				if rank <= lastBest {
					t.Fatalf("workflow order broken at %s: best %d after %d", r.wf.Name, rank, lastBest)
				}
				lastBest = rank
			}
			lastRank = rank
		}
	}
	if f.m.tasks.rows[1].task.ID != f.m.triage.last.Next().ID {
		t.Fatalf("the first row should be the top-ranked task, got %s", f.m.tasks.rows[1].task.ID)
	}
	f.press(t, "s")
	if f.m.triage.sort != sortStatus || !f.m.triage.sortSet {
		t.Fatal("s toggles to status")
	}
	var got []string
	for _, r := range f.m.tasks.rows {
		if r.kind == rowTask {
			got = append(got, r.task.ID)
		}
	}
	if want := "T-0990 T-0981 T-0977 T-1042 T-1041 T-2202 T-2201"; strings.Join(got, " ") != want {
		t.Fatalf("status order after toggle: %v", got)
	}
	f.toastIs(t, "ok", "sort: status")
	f.press(t, "s")
	if f.m.triage.sort != sortSmart || !strings.Contains(f.view(), "· smart") {
		t.Fatal("s toggles back to smart")
	}
	// an explicit choice survives the next tick
	f.press(t, "s")
	f.tick(t)
	if f.m.triage.sort != sortStatus {
		t.Fatal("the operator's sort must not be overridden by a tick")
	}
}

func TestTriageChipsNeverExceedWidth(t *testing.T) {
	f := newFixture(t)
	f.tick(t)
	v := f.view()
	if !strings.Contains(v, "▰") || !strings.Contains(v, "◆") {
		t.Fatalf("chips missing from the list:\n%s", v)
	}
	if !regexp.MustCompile(`[▰▱]{4} [·✉◉✖↻]`).MatchString(v) {
		t.Fatalf("chip shape (bar + next glyph) missing:\n%s", v)
	}
	for _, sz := range awkwardSizes {
		f.m = resize(f.m, sz[0], sz[1])
		f.press(t, "1")
		checkFrame(t, f.m, sz[0], sz[1], "chips list")
		f.press(t, "G")
		checkFrame(t, f.m, sz[0], sz[1], "chips end")
		f.press(t, "tab")
		checkFrame(t, f.m, sz[0], sz[1], "chips detail focus")
		f.press(t, "tab")
	}
	// a very long title is truncated before the chip, never after
	f.m = resize(f.m, 80, 24)
	now := time.Now().UTC()
	f.send(t, feedMsg{recs: []harness.FeedRecord{{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-LONG", Workflow: "checkout", Title: strings.Repeat("very long title ", 20), State: harness.StateBlocked, Priority: 2}}}})
	f.tick(t)
	checkFrame(t, f.m, 80, 24, "long title chip")
	for _, l := range strings.Split(f.view(), "\n") {
		if strings.Contains(l, "T-LONG") && !regexp.MustCompile(`[▰▱]{4} [·✉◉✖↻]◆? ?│`).MatchString(l) {
			t.Fatalf("chip should sit at the right edge of the row: %q", l)
		}
	}
}

func TestTriageDetailLines(t *testing.T) {
	f := newFixture(t)
	f.selectTask(t, "T-1042")
	v := f.view()
	if !strings.Contains(v, "─ triage ─") || !strings.Contains(v, "not run yet · t runs it") {
		t.Fatalf("detail should say triage has not run:\n%s", v)
	}
	f.tick(t)
	tt := f.m.triage.last.Get("T-1042")
	v = f.view()
	for _, s := range []string{"─ triage ─", "#" + itoa(tt.Rank), "next: " + string(tt.Next), "blocked", "priority 2"} {
		if !strings.Contains(v, s) {
			t.Errorf("detail missing %q:\n%s", s, v)
		}
	}
	if tt.JevUsed && !strings.Contains(v, "◆ jev") {
		t.Fatal("jev-backed rows say so")
	}
	if tt.NextConf > 0 && !regexp.MustCompile(`next: \w+ \(0\.\d\d\)`).MatchString(v) {
		t.Fatalf("confidence should follow the action:\n%s", v)
	}
	// a task that appeared after the tick
	now := time.Now().UTC()
	f.send(t, feedMsg{recs: []harness.FeedRecord{{TS: now, Type: "task.upsert", Task: &harness.Task{ID: "T-NEW", Workflow: "checkout", Title: "new", State: harness.StateQueued}}}})
	f.selectTask(t, "T-NEW")
	if !strings.Contains(f.view(), "not in the last tick") {
		t.Fatal("a task missing from the last tick says so")
	}
}

func itoa(n int) string { return string(rune('0' + n)) }

func TestTriageManualKeyToasts(t *testing.T) {
	f := newFixture(t)
	f.press(t, "t")
	if f.m.triage.last == nil {
		t.Fatal("t runs a tick")
	}
	f.toastIs(t, "ok", "triage: 7 tasks · jev ")
	if !strings.Contains(f.m.toast.text, "$0.") || !strings.Contains(f.m.toast.text, "mock") {
		t.Fatalf("summary should carry cost and the mock marker: %q", f.m.toast.text)
	}
	// budget exhausted → deterministic
	f.m.triage.tr.Budget.MaxCallsPerHour = 1
	f.press(t, "t")
	f.toastIs(t, "ok", "triage: deterministic (budget)")
	// detail focus does not steal t, and a run in flight is refused
	f.press(t, "tab", "t")
	if !strings.Contains(f.m.toast.text, "triage:") {
		t.Fatal("t from the detail pane must not run a tick")
	}
	f.press(t, "tab")
	f.m.triage.running = true
	f.press(t, "t")
	f.toastIs(t, "warn", "already running")
	f.m.triage.running = false
	// the scheduled tick does not toast
	f.m.toast = toast{}
	f.tick(t)
	if f.m.toast.text != "" {
		t.Fatalf("scheduled ticks are silent, got %q", f.m.toast.text)
	}
}

func fixedTriage(top string, next jev.Action, ids ...string) jev.Triage {
	tr := jev.Triage{At: time.Now(), Deterministic: true}
	tr.Tasks = append(tr.Tasks, jev.TaskTriage{ID: top, Rank: 1, Score: 0.9, Base: 0.9, Next: next, NextConf: 0.85, Reasons: []string{"blocked", "jev: stuck 0.90"}, JevUsed: true})
	for i, id := range ids {
		tr.Tasks = append(tr.Tasks, jev.TaskTriage{ID: id, Rank: i + 2, Score: 0.5 - float64(i)*0.1, Next: jev.ActWait, Reasons: []string{"running"}})
	}
	return tr
}

func TestWhatNextArmsRecommendedAction(t *testing.T) {
	f := newFixture(t)
	f.press(t, "N")
	f.toastIs(t, "warn", "no triage yet")
	// ping → composer with the canned text
	f.press(t, "4") // from another tab
	f.inject(t, fixedTriage("T-1042", jev.ActPing, "T-1041"))
	f.press(t, "1", "N")
	if f.m.tab != tabTasks || f.m.tasks.selID != "T-1042" || !f.m.composer.open || f.m.composer.kind != composerPing {
		t.Fatalf("N should select the top task and open the ping composer: tab=%d sel=%s composer=%+v", f.m.tab, f.m.tasks.selID, f.m.composer.open)
	}
	if f.m.composer.agentID != "tester-1" || f.m.composer.input.Value() != triagePingText {
		t.Fatalf("composer: agent=%s text=%q", f.m.composer.agentID, f.m.composer.input.Value())
	}
	if !strings.Contains(f.m.toast.text, "T-1042 → ping") || !strings.Contains(f.m.toast.text, "jev: stuck 0.90") {
		t.Fatalf("reasons toast: %q", f.m.toast.text)
	}
	f.press(t, "enter")
	if cmds := readOutbox(t, f.outbox); cmds[len(cmds)-1].Text != triagePingText {
		t.Fatalf("armed ping should send as-is: %+v", cmds)
	}
	// retry → confirm, then y sends
	f.inject(t, fixedTriage("T-0990", jev.ActRetry, "T-1041"))
	f.press(t, "N")
	if !f.m.confirm.open || !strings.Contains(f.view(), "Retry T-0990") {
		t.Fatalf("N should open the retry confirm:\n%s", f.view())
	}
	f.press(t, "y")
	if cmds := readOutbox(t, f.outbox); cmds[len(cmds)-1].Type != "task.retry" || cmds[len(cmds)-1].TaskID != "T-0990" {
		t.Fatalf("y should send the retry: %+v", cmds[len(cmds)-1])
	}
	// cancel → the existing confirm
	f.inject(t, fixedTriage("T-1041", jev.ActCancel, "T-0990"))
	f.press(t, "N")
	if !f.m.confirm.open || !strings.Contains(f.view(), "Cancel T-1041") {
		t.Fatal("N should open the cancel confirm")
	}
	f.press(t, "n")
	// review → detail focus on the first element
	f.press(t, "tab", "j", "j", "tab")
	f.inject(t, fixedTriage("T-0977", jev.ActReview, "T-1041"))
	f.press(t, "N")
	if f.m.tasks.focus != focusDetail || f.m.tasks.elemCursor != 0 || f.m.tasks.selID != "T-0977" {
		t.Fatalf("review should focus the detail pane on element 0: focus=%d cursor=%d sel=%s", f.m.tasks.focus, f.m.tasks.elemCursor, f.m.tasks.selID)
	}
	// wait → nothing to do
	f.inject(t, fixedTriage("T-1041", jev.ActWait, "T-0990"))
	f.press(t, "N")
	f.toastIs(t, "ok", "nothing needs you")
	if f.m.composer.open || f.m.confirm.open {
		t.Fatal("wait arms nothing")
	}
	// hidden by a workflow filter: N clears it
	f.press(t, "w") // billing only
	f.inject(t, fixedTriage("T-1042", jev.ActPing, "T-0990"))
	f.press(t, "N")
	if f.m.tasks.wfFilter != "" || f.m.tasks.selID != "T-1042" {
		t.Fatalf("N should clear the filter to land: filter=%q sel=%s", f.m.tasks.wfFilter, f.m.tasks.selID)
	}
	f.press(t, "esc")
	// palette route
	f.press(t, "ctrl+k")
	f.typeText(t, "what next")
	f.press(t, "enter")
	if !f.m.composer.open || f.m.composer.input.Value() != triagePingText {
		t.Fatal("palette What next? should route like N")
	}
	f.press(t, "esc")
	checkFrame(t, f.m, 120, 40, "after what next")
}

func TestHeaderShowsTriageAge(t *testing.T) {
	f := newFixture(t)
	if !strings.Contains(f.view(), "triage: pending") {
		t.Fatalf("scheduler on, no tick yet:\n%s", f.view())
	}
	f.tick(t)
	if !regexp.MustCompile(`triage \d+s ago`).MatchString(f.view()) {
		t.Fatalf("header should show the tick age:\n%s", f.view())
	}
	if strings.Contains(f.view(), "$0.") {
		t.Fatal("a mock tick spends nothing; no cost in the header")
	}
	f.m.triage.last.Mock, f.m.triage.last.CostUSD = false, 0.0002
	if !strings.Contains(f.view(), "$0.0002") {
		t.Fatalf("a paid tick shows its cost:\n%s", f.view())
	}
	f.m.triage.running = true
	if !strings.Contains(f.view(), "triage running") {
		t.Fatal("a run in flight is visible")
	}
	for _, sz := range awkwardSizes {
		f.m = resize(f.m, sz[0], sz[1])
		checkFrame(t, f.m, sz[0], sz[1], "header triage")
	}
}

func TestPendingPingCounting(t *testing.T) {
	f := newFixture(t)
	if len(f.m.pendingPings()) != 0 {
		t.Fatal("nothing pending at start")
	}
	f.selectTask(t, "T-1041")
	f.press(t, "p")
	f.typeText(t, "status?")
	f.press(t, "enter")
	if f.m.pendingPings()["T-1041"] != 1 {
		t.Fatalf("one unacked ping: %v", f.m.pendingPings())
	}
	f.tick(t)
	if !strings.Contains(strings.Join(f.m.triage.last.Get("T-1041").Reasons, ","), "1 ping unanswered") {
		t.Fatalf("the tick should see the pending ping: %v", f.m.triage.last.Get("T-1041").Reasons)
	}
	cmds := readOutbox(t, f.outbox)
	var pingID string
	for _, c := range cmds {
		if c.Type == "ping" {
			pingID = c.ID
		}
	}
	f.send(t, feedMsg{recs: []harness.FeedRecord{{Type: "ping.ack", PingAck: &harness.PingAck{PingID: pingID, Agent: "coder-1", Text: "ack"}}}})
	if n := f.m.pendingPings()["T-1041"]; n != 0 {
		t.Fatalf("ack should clear the pending ping: %d", n)
	}
	f.tick(t)
	if strings.Contains(strings.Join(f.m.triage.last.Get("T-1041").Reasons, ","), "unanswered") {
		t.Fatal("no unanswered reason after the ack")
	}
	// a ping without a task never counts against a task
	f.press(t, "4", "p")
	f.typeText(t, "hi")
	f.press(t, "enter")
	if len(f.m.pendingPings()) != 0 {
		t.Fatalf("agent-only pings are not task pings: %v", f.m.pendingPings())
	}
}

func TestTriageIntervalZeroDisables(t *testing.T) {
	f := newFixture(t)
	f.m.cfg.TriageInterval = 0
	if n := countMsgs(f.m.Init(), func(m tea.Msg) bool { _, ok := m.(triageTickMsg); return ok }, 100*time.Millisecond); n != 0 {
		t.Fatalf("Init scheduled %d triage ticks with interval 0", n)
	}
	if _, cmd := f.m.Update(triageTickMsg{}); cmd != nil {
		t.Fatal("a stray tick must not reschedule or run")
	}
	if !strings.Contains(f.view(), "triage: off") {
		t.Fatalf("header should say off:\n%s", f.view())
	}
	// the manual key still works with the scheduler off
	f.press(t, "t")
	if f.m.triage.last == nil || !regexp.MustCompile(`triage \d+s ago`).MatchString(f.view()) {
		t.Fatal("t runs a tick even when the scheduler is off")
	}
	// and with the scheduler on, Init schedules exactly one first tick
	f.m.cfg.TriageInterval = time.Minute
	if n := countMsgs(f.m.Init(), func(m tea.Msg) bool { _, ok := m.(triageTickMsg); return ok }, 100*time.Millisecond); n != 1 {
		t.Fatalf("Init scheduled %d triage ticks, want 1", n)
	}
}

func TestTriageCLIOnceTableAndOutboxRule(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	f := newFixture(t)
	logPath := filepath.Join(t.TempDir(), "log", "triage.jsonl")
	var out, errOut bytes.Buffer
	args := []string{"--feed", f.feed, "--outbox", f.outbox, "--log", logPath, "--once"}
	if code := runTriageCLI(args, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	v := out.String()
	for _, s := range []string{"rank score next", "T-1042", "Flaky test", "blocked · priority 2", "jev-mock"} {
		if !strings.Contains(v, s) {
			t.Errorf("table missing %q:\n%s", s, v)
		}
	}
	if !regexp.MustCompile(`(?m)^1\s+\d\.\d\d\s+(wait|ping|review|cancel|retry)\s+[◆-]\s+T-`).MatchString(v) {
		t.Fatalf("first row shape (rank score next jev id):\n%s", v)
	}
	if len(jev.LoadTriage(logPath, 5)) != 1 || len(triageCmds(t, f.outbox)) != 1 {
		t.Fatal("one tick → one log line and one triage command")
	}
	// second cron run: the previous tick comes from the log, nothing changed
	out.Reset()
	if code := runTriageCLI(args, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if len(jev.LoadTriage(logPath, 5)) != 2 || len(triageCmds(t, f.outbox)) != 1 {
		t.Fatal("an unchanged ranking under cron must not write a second command")
	}
	// pending pings are read from the outbox
	harness.NewOutbox(f.outbox).Ping("tester-1", "T-1042", "status?")
	out.Reset()
	runTriageCLI(append(args, "--json"), &out, &errOut)
	var tr jev.Triage
	if err := json.Unmarshal(out.Bytes(), &tr); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if !strings.Contains(strings.Join(tr.Get("T-1042").Reasons, ","), "1 ping unanswered") {
		t.Fatalf("cron form should count outbox pings without acks: %v", tr.Get("T-1042").Reasons)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatal(err)
	}
	if code := runTriageCLI([]string{"--bogus"}, &out, &errOut); code != 2 {
		t.Fatalf("unknown flag → exit 2, got %d", code)
	}
}

func TestTriageCLIDemoOnceJSONBinary(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	home := t.TempDir()
	bin := filepath.Join(home, "alembic-test")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin, "triage", "--demo", "--once", "--json")
	cmd.Env = append(os.Environ(), "HOME="+home, "TYPESAFE_API_KEY=")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	var tr jev.Triage
	if err := json.Unmarshal(out, &tr); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out)
	}
	if len(tr.Tasks) != 7 || !tr.Mock {
		t.Fatalf("demo triage: tasks=%d mock=%v", len(tr.Tasks), tr.Mock)
	}
	for i, tt := range tr.Tasks {
		if tt.Rank != i+1 {
			t.Fatalf("ranks must be 1..n in order: %+v", tr.Tasks)
		}
	}
	if tr.Get("T-2201").Score != 0 || tr.Tasks[0].Score <= 0 {
		t.Fatal("done scores 0; the top task scores above 0")
	}
	demo := filepath.Join(home, ".alembic", "demo")
	if len(jev.LoadTriage(filepath.Join(demo, "triage.jsonl"), 5)) != 1 {
		t.Fatal("--demo logs under the demo dir")
	}
	if cmds := triageCmds(t, filepath.Join(demo, "outbox.jsonl")); len(cmds) != 1 {
		t.Fatalf("--demo writes the triage command to the demo outbox: %d", len(cmds))
	}
	// the TUI flags still parse with the new ones
	help := exec.Command(bin, "--triage-interval", "0", "--triage-tasks", "2", "--triage-calls", "5", "--help")
	if out, _ := help.CombinedOutput(); !strings.Contains(string(out), "triage-interval") {
		t.Fatalf("help should list the triage flags:\n%s", out)
	}
}

func TestTriageFlagsParse(t *testing.T) {
	t.Setenv("ALEMBIC_FEED", "")
	t.Setenv("ALEMBIC_OUTBOX", "")
	cfg, err := parseFlags([]string{"--repo", t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TriageInterval != 60*time.Second || cfg.TriageTasks != 8 || cfg.TriageCalls != 60 {
		t.Fatalf("defaults: %+v", cfg)
	}
	cfg, err = parseFlags([]string{"--repo", t.TempDir(), "--triage-interval", "0", "--triage-tasks", "0", "--triage-calls", "3"})
	if err != nil || cfg.TriageInterval != 0 || cfg.TriageTasks != 0 || cfg.TriageCalls != 3 {
		t.Fatalf("flags: %v %+v", err, cfg)
	}
	m := newModel(cfg)
	if m.triage.tr.Budget.MaxTasksPerTick != 0 || m.triage.tr.Budget.MaxCallsPerHour != 3 || m.triage.tr.Budget.ChunkSize != jev.DefaultBudget().ChunkSize {
		t.Fatalf("budget: %+v", m.triage.tr.Budget)
	}
	if !strings.HasSuffix(m.cfg.TriageLog, filepath.Join(".alembic", "triage.jsonl")) {
		t.Fatalf("default log: %s", m.cfg.TriageLog)
	}
}

func TestTriageHelpAndHints(t *testing.T) {
	f := newFixture(t)
	f.m = resize(f.m, 200, 60)
	if v := f.view(); !strings.Contains(v, "t triage") || !strings.Contains(v, "N next") || !strings.Contains(v, "s sort") {
		t.Fatalf("hint strip missing the triage keys:\n%s", v)
	}
	f.press(t, "?")
	v := f.view()
	for _, s := range []string{"run a triage tick", "smart (triage rank)", "what next"} {
		if !strings.Contains(v, s) {
			t.Errorf("help missing %q", s)
		}
	}
}
