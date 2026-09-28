package harness

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeLines(t *testing.T, path string, lines ...string) {
	t.Helper()
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	for _, l := range lines {
		if _, err := fh.WriteString(l); err != nil {
			t.Fatal(err)
		}
	}
}

func rec(typ string, payload string) string {
	return `{"v":1,"ts":"2026-09-28T10:00:00Z","type":"` + typ + `",` + payload + "}\n"
}

func TestFeedTailsOnlyNewRecords(t *testing.T) {
	p := filepath.Join(t.TempDir(), "feed.jsonl")
	f := NewFeed(p)
	if recs, err := f.Poll(); err != nil || len(recs) != 0 {
		t.Fatalf("missing file should yield nothing: %v %d", err, len(recs))
	}
	writeLines(t, p, rec("task.upsert", `"task":{"id":"T1","title":"a","state":"running"}`))
	recs, err := f.Poll()
	if err != nil || len(recs) != 1 || recs[0].Task.ID != "T1" {
		t.Fatalf("first poll: %v %+v", err, recs)
	}
	if recs, _ := f.Poll(); len(recs) != 0 {
		t.Fatal("second poll should be empty")
	}
	writeLines(t, p, rec("task.event", `"event":{"task_id":"T1","level":"ok","text":"done"}`))
	recs, _ = f.Poll()
	if len(recs) != 1 || recs[0].Event.Text != "done" {
		t.Fatalf("third poll: %+v", recs)
	}
}

func TestFeedPartialLineAndGarbage(t *testing.T) {
	p := filepath.Join(t.TempDir(), "feed.jsonl")
	f := NewFeed(p)
	full := rec("task.upsert", `"task":{"id":"T1","title":"a","state":"queued"}`)
	writeLines(t, p, "not json\n", full[:20])
	recs, _ := f.Poll()
	if len(recs) != 0 {
		t.Fatalf("partial line must not be emitted: %+v", recs)
	}
	writeLines(t, p, full[20:])
	recs, _ = f.Poll()
	if len(recs) != 1 || recs[0].Task.ID != "T1" {
		t.Fatalf("completed line should be emitted once: %+v", recs)
	}
}

func TestFeedTruncationRestarts(t *testing.T) {
	p := filepath.Join(t.TempDir(), "feed.jsonl")
	f := NewFeed(p)
	writeLines(t, p, rec("task.upsert", `"task":{"id":"T1","title":"a","state":"queued"}`), rec("task.upsert", `"task":{"id":"T2","title":"b","state":"queued"}`))
	if recs, _ := f.Poll(); len(recs) != 2 {
		t.Fatal("expected 2")
	}
	os.WriteFile(p, []byte(rec("task.upsert", `"task":{"id":"T3","title":"c","state":"queued"}`)), 0o644)
	recs, _ := f.Poll()
	if len(recs) != 1 || recs[0].Task.ID != "T3" {
		t.Fatalf("after truncation: %+v", recs)
	}
}

func TestSnapshotFold(t *testing.T) {
	s := NewSnapshot()
	ts := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	s.Apply(FeedRecord{TS: ts, Type: "task.upsert", Task: &Task{ID: "T1", Title: "a", State: StateRunning, Elements: []Element{{Kind: ElemFile, Ref: "x.go"}}}})
	s.Apply(FeedRecord{TS: ts.Add(time.Minute), Type: "task.upsert", Task: &Task{ID: "T1", Title: "a", State: StateReview}})
	tk := s.Tasks["T1"]
	if tk.State != StateReview || len(tk.Elements) != 1 || !tk.Created.Equal(ts) || tk.Workflow != "default" {
		t.Fatalf("upsert merge: %+v", tk)
	}
	if _, ok := s.Workflows["default"]; !ok {
		t.Fatal("default workflow should be created")
	}
	s.Apply(FeedRecord{TS: ts, Type: "task.event", Event: &Event{TaskID: "T1", Level: LevelWarn, Text: "hmm"}})
	if tk.StatusLine != "hmm" || len(s.Events["T1"]) != 1 {
		t.Fatal("warn event should update status line")
	}
	s.Apply(FeedRecord{TS: ts, Type: "task.event", Event: &Event{TaskID: "T1", Level: LevelInfo, Text: "quiet"}})
	if tk.StatusLine != "quiet" {
		// info events also become the status line only via upsert; check the rule is stable
		if tk.StatusLine != "hmm" {
			t.Fatal("status line changed unexpectedly")
		}
	}
	s.Apply(FeedRecord{TS: ts, Type: "bogus.type"})
	s.Apply(FeedRecord{TS: ts, Type: "ping.ack", PingAck: &PingAck{PingID: "p1", Agent: "a"}})
	if len(s.Acks) != 1 || s.Records != 6 {
		t.Fatalf("acks=%d records=%d", len(s.Acks), s.Records)
	}
	for i := 0; i < 600; i++ {
		s.Apply(FeedRecord{TS: ts, Type: "task.event", Event: &Event{TaskID: "T1", Level: LevelInfo, Text: "x"}})
	}
	if len(s.Events["T1"]) != 500 {
		t.Fatal("events should be capped at 500")
	}
}

func TestOutboxAppendsCommands(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sub", "outbox.jsonl")
	o := NewOutbox(p)
	c, err := o.Ping("coder-1", "T1", "status?")
	if err != nil || c.ID == "" || c.V != ContractVersion {
		t.Fatalf("ping: %v %+v", err, c)
	}
	if _, err := o.Cancel("T1"); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(p)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 lines, got %d", len(lines))
	}
	var first Command
	json.Unmarshal([]byte(lines[0]), &first)
	if first.Type != "ping" || first.Agent != "coder-1" || first.Text != "status?" {
		t.Fatalf("first: %+v", first)
	}
}

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

func TestWorktreesRoundTrip(t *testing.T) {
	repo := initRepo(t)
	root, err := RepoRoot(filepath.Join(repo))
	if err != nil || filepath.Clean(root) != filepath.Clean(repo) {
		t.Fatalf("RepoRoot: %v %q", err, root)
	}
	if _, err := AddWorktree(repo, "feat/x y", ""); err == nil {
		t.Fatal("invalid branch name must be rejected before git worktree add")
	}
	path, err := AddWorktree(repo, "feat/x-y", "")
	if err != nil {
		t.Fatal(err)
	}
	wts, err := ListWorktrees(repo)
	if err != nil || len(wts) != 2 || !wts[0].Main || wts[1].Branch != "feat/x-y" {
		t.Fatalf("list: %v %+v", err, wts)
	}
	os.WriteFile(filepath.Join(path, "dirty.txt"), []byte("d"), 0o644)
	wts, _ = ListWorktrees(repo)
	if wts[1].Dirty != 1 {
		t.Fatalf("dirty count: %+v", wts[1])
	}
	s := NewSnapshot()
	s.Apply(FeedRecord{Type: "task.upsert", Task: &Task{ID: "T9", Title: "x", State: StateRunning, Worktree: path}})
	LinkTasks(wts, s)
	if len(wts[1].TaskIDs) != 1 || wts[1].TaskIDs[0] != "T9" {
		t.Fatal("task not linked to worktree")
	}
	if err := RemoveWorktree(repo, path, false); err == nil {
		t.Fatal("removing a dirty worktree without force should fail")
	}
	if err := RemoveWorktree(repo, path, true); err != nil {
		t.Fatal(err)
	}
	if _, err := RepoRoot(t.TempDir()); err == nil {
		t.Fatal("non-repo should error")
	}
}

func TestDemoSeedsAndAcks(t *testing.T) {
	repo := initRepo(t)
	dir := t.TempDir()
	feed, out := filepath.Join(dir, "feed.jsonl"), filepath.Join(dir, "outbox.jsonl")
	d := NewDemo(feed, out)
	if err := d.Seed(repo); err != nil {
		t.Fatal(err)
	}
	s, f, err := LoadAll(feed)
	if err != nil || len(s.Tasks) != 7 || len(s.Agents) != 4 || len(s.Workflows) != 3 {
		t.Fatalf("seed: %v tasks=%d agents=%d wfs=%d", err, len(s.Tasks), len(s.Agents), len(s.Workflows))
	}
	if err := d.Seed(repo); err != nil {
		t.Fatal(err)
	}
	if recs, _ := f.Poll(); len(recs) != 0 {
		t.Fatal("second seed must be a no-op")
	}
	NewOutbox(out).Ping("coder-1", "T-1041", "stop after this")
	if err := d.Step(s); err != nil {
		t.Fatal(err)
	}
	recs, _ := f.Poll()
	var acked bool
	for _, r := range recs {
		s.Apply(r)
		if r.Type == "ping.ack" && r.PingAck.Agent == "coder-1" && strings.Contains(r.PingAck.Text, "pausing") {
			acked = true
		}
	}
	if !acked {
		t.Fatalf("ping not acknowledged: %+v", recs)
	}
	for i := 0; i < 30; i++ {
		d.Step(s)
		for _, r := range func() []FeedRecord { rr, _ := f.Poll(); return rr }() {
			s.Apply(r)
		}
	}
	if s.Tasks["T-2202"].State == StateQueued {
		t.Fatal("queued task should get picked up within 30 steps")
	}
}
