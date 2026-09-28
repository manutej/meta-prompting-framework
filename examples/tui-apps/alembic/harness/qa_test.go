package harness

import (
	"os"
	"path/filepath"
	"testing"
)

// A complete record whose trailing newline has not been written yet must be
// delivered exactly once: not on the poll that sees it unterminated, and not
// twice once the newline lands.
func TestQAFeedUnterminatedLineDeliveredOnce(t *testing.T) {
	for _, withTS := range []bool{true, false} {
		p := filepath.Join(t.TempDir(), "feed.jsonl")
		f := NewFeed(p)
		line := `{"v":1,"type":"task.upsert","task":{"id":"T1","title":"a","state":"queued"}}`
		if withTS {
			line = `{"v":1,"ts":"2026-09-28T10:00:00Z","type":"task.upsert","task":{"id":"T1","title":"a","state":"queued"}}`
		}
		// first record is complete, second is complete but unterminated
		writeLines(t, p, line+"\n", line)
		total := 0
		for i := 0; i < 3; i++ {
			recs, err := f.Poll()
			if err != nil {
				t.Fatal(err)
			}
			total += len(recs)
		}
		writeLines(t, p, "\n")
		recs, _ := f.Poll()
		total += len(recs)
		if total != 2 {
			t.Fatalf("ts=%v: two records were written, %d delivered", withTS, total)
		}
		// a lone unterminated line (no newline at all in the buffer)
		p2 := filepath.Join(t.TempDir(), "feed.jsonl")
		f2 := NewFeed(p2)
		writeLines(t, p2, line)
		total = 0
		for i := 0; i < 3; i++ {
			recs, _ := f2.Poll()
			total += len(recs)
		}
		writeLines(t, p2, "\n")
		recs, _ = f2.Poll()
		total += len(recs)
		if total != 1 {
			t.Fatalf("ts=%v: one record was written, %d delivered", withTS, total)
		}
	}
}

// The demo harness answers task.cancel / task.retry with a task.upsert; per
// the contract every field except elements/created is replaced, so the
// upsert must carry the task's identity or the task jumps to a blank row in
// a "default" workflow.
func TestQADemoCancelRetryKeepTaskIdentity(t *testing.T) {
	repo := initRepo(t)
	dir := t.TempDir()
	feed, out := filepath.Join(dir, "feed.jsonl"), filepath.Join(dir, "outbox.jsonl")
	d := NewDemo(feed, out)
	if err := d.Seed(repo); err != nil {
		t.Fatal(err)
	}
	s, f, _ := LoadAll(feed)
	before := *s.Tasks["T-1041"]
	NewOutbox(out).Cancel("T-1041")
	NewOutbox(out).Retry("T-0990")
	if err := d.Step(s); err != nil {
		t.Fatal(err)
	}
	recs, _ := f.Poll()
	for _, r := range recs {
		s.Apply(r)
	}
	got := s.Tasks["T-1041"]
	if got.State != StateFailed {
		t.Fatalf("cancel should fail the task: %+v", got)
	}
	if got.Title != before.Title || got.Workflow != before.Workflow || got.Agent != before.Agent || got.Worktree != before.Worktree {
		t.Fatalf("cancel upsert lost identity:\n got %+v\nwant %+v", got, before)
	}
	if _, ok := s.Workflows["default"]; ok {
		t.Fatal("a phantom 'default' workflow appeared")
	}
	if r := s.Tasks["T-0990"]; r.State != StateQueued || r.Title == "" || r.Workflow != "billing" {
		t.Fatalf("retry upsert lost identity: %+v", r)
	}
	// unknown task ids must not create blank tasks
	NewOutbox(out).Cancel("T-NOPE")
	d.Step(s)
	recs, _ = f.Poll()
	for _, r := range recs {
		s.Apply(r)
	}
	if _, ok := s.Tasks["T-NOPE"]; ok {
		t.Fatal("cancelling an unknown task must not invent one")
	}
	_ = os.Remove
}
