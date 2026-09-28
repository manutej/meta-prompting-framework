package jev

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"alembic/harness"
)

var t0 = time.Date(2026, 9, 28, 12, 0, 0, 0, time.UTC)

func snapWith(tasks ...*harness.Task) *harness.Snapshot {
	s := harness.NewSnapshot()
	for _, t := range tasks {
		s.Apply(harness.FeedRecord{TS: t0, Type: "task.upsert", Task: t})
	}
	return s
}

func task(id string, st harness.TaskState, prio int, age time.Duration) *harness.Task {
	return &harness.Task{ID: id, Title: "t " + id, State: st, Priority: prio, Updated: t0.Add(-age), Progress: 0.5}
}

type fakeAsker struct {
	calls   int
	lastQ   map[string]Question
	answers func(q map[string]Question) map[string]Answer
	err     error
}

func (f *fakeAsker) IsMock() bool { return true }
func (f *fakeAsker) Ask(_ context.Context, _ any, q map[string]Question) (*Response, error) {
	f.calls++
	f.lastQ = q
	if f.err != nil {
		return nil, f.err
	}
	return &Response{Model: "fake", Answers: f.answers(q), Usage: Usage{InputTokens: 1000}, Mock: true}, nil
}

func fp(v float64) *float64 { return &v }

func TestDeterministicOrdering(t *testing.T) {
	tr := NewTriager(nil, DefaultBudget())
	tr.Now = func() time.Time { return t0 }
	snap := snapWith(
		task("done", harness.StateDone, 2, 0),
		task("queued", harness.StateQueued, 0, 0),
		task("running-fresh", harness.StateRunning, 0, time.Minute),
		task("running-stale", harness.StateRunning, 0, 40*time.Minute),
		task("blocked", harness.StateBlocked, 0, time.Minute),
		task("failed-urgent", harness.StateFailed, 2, time.Minute),
	)
	snap.Apply(harness.FeedRecord{TS: t0, Type: "task.event", Event: &harness.Event{TaskID: "running-fresh", Level: harness.LevelError, Text: "boom"}})
	out := tr.Run(context.Background(), snap, map[string]int{"blocked": 2})
	order := []string{}
	for _, tt := range out.Tasks {
		order = append(order, tt.ID)
	}
	want := []string{"failed-urgent", "blocked", "running-stale", "running-fresh", "queued", "done"}
	if strings.Join(order, ",") != strings.Join(want, ",") {
		t.Fatalf("order %v, want %v", order, want)
	}
	if out.Skipped != "jev disabled" || !out.Deterministic {
		t.Fatalf("no client should mean deterministic: %+v", out.Skipped)
	}
	b := out.Get("blocked")
	if !strings.Contains(strings.Join(b.Reasons, ","), "2 pings unanswered") || b.Next != ActPing {
		t.Fatalf("blocked reasons/next: %+v", b)
	}
	if out.Get("done").Score != 0 || out.Next().ID != "failed-urgent" || out.Get("failed-urgent").Next != ActRetry {
		t.Fatal("done must score 0; Next must be the top task with retry")
	}
	if out.CostUSD != 0 || out.JevCalls != 0 {
		t.Fatal("no spend without jev")
	}
}

func TestStallDetectionAcrossTicks(t *testing.T) {
	tr := NewTriager(nil, DefaultBudget())
	now := t0
	tr.Now = func() time.Time { return now }
	snap := snapWith(task("r", harness.StateRunning, 0, time.Minute))
	first := tr.Run(context.Background(), snap, nil)
	// one-minute ticks: the baseline must survive them and fire at the 5-minute mark
	for i := 1; i <= 4; i++ {
		now = t0.Add(time.Duration(i) * time.Minute)
		snap.Tasks["r"].Updated = now.Add(-time.Minute)
		if got := tr.Run(context.Background(), snap, nil); strings.Contains(strings.Join(got.Get("r").Reasons, ","), "no progress") {
			t.Fatalf("stall fired too early at tick %d", i)
		}
	}
	now = t0.Add(5 * time.Minute)
	snap.Tasks["r"].Updated = now.Add(-time.Minute)
	fifth := tr.Run(context.Background(), snap, nil)
	if fifth.Get("r").Score <= first.Get("r").Score || !strings.Contains(strings.Join(fifth.Get("r").Reasons, ","), "no progress for 5m") {
		t.Fatalf("unchanged progress over 5 min of 1-min ticks should add a stall reason: %+v", fifth.Get("r"))
	}
	// progress moves: baseline resets, stall clears
	snap.Tasks["r"].Progress = 0.6
	now = t0.Add(6 * time.Minute)
	if got := tr.Run(context.Background(), snap, nil); strings.Contains(strings.Join(got.Get("r").Reasons, ","), "no progress") {
		t.Fatal("progress change must reset the stall baseline")
	}
}

func TestJevConsultedOnlyForAmbiguousAndMerged(t *testing.T) {
	fa := &fakeAsker{answers: func(q map[string]Question) map[string]Answer {
		a := map[string]Answer{}
		for id := range q {
			switch {
			case strings.HasSuffix(id, "__stuck"):
				a[id] = Answer{Type: Noul, Noul: fp(0.9)}
			case strings.HasSuffix(id, "__needs_human"):
				a[id] = Answer{Type: Noul, Noul: fp(0.1)}
			case strings.HasSuffix(id, "__next"):
				a[id] = Answer{Type: Choice, Choice: "cancel", Probabilities: map[string]float64{"cancel": 0.8, "wait": 0.2}, Confidence: fp(0.85)}
			}
		}
		return a
	}}
	tr := NewTriager(fa, DefaultBudget())
	tr.Now = func() time.Time { return t0 }
	snap := snapWith(
		task("blocked", harness.StateBlocked, 0, time.Minute),
		task("fresh", harness.StateRunning, 0, time.Minute),
		task("queued", harness.StateQueued, 0, 0),
	)
	out := tr.Run(context.Background(), snap, nil)
	if fa.calls != 1 || out.JevCalls != 1 || out.JevTasks != 1 || out.Deterministic {
		t.Fatalf("exactly one call for the one ambiguous task: calls=%d %+v", fa.calls, out)
	}
	if len(fa.lastQ) != 3 {
		t.Fatalf("3 questions per task, got %d", len(fa.lastQ))
	}
	b := out.Get("blocked")
	if !b.JevUsed || b.Stuck != 0.9 || b.Next != ActCancel || b.NextConf != 0.85 || b.Score <= b.Base {
		t.Fatalf("merge: %+v", b)
	}
	if !strings.Contains(strings.Join(b.Reasons, ","), "jev: stuck 0.90") {
		t.Fatalf("reason missing: %v", b.Reasons)
	}
	if out.Get("fresh").JevUsed || out.Get("queued").JevUsed {
		t.Fatal("unambiguous tasks must not be sent to jev")
	}
	if out.CostUSD != 0 || !out.Mock || out.Usage.InputTokens != 1000 {
		t.Fatalf("mock ticks are free but still report usage: cost=%v mock=%v usage=%+v", out.CostUSD, out.Mock, out.Usage)
	}
}

func TestBudgetAndChunking(t *testing.T) {
	fa := &fakeAsker{answers: func(q map[string]Question) map[string]Answer { return map[string]Answer{} }}
	tr := NewTriager(fa, Budget{MaxTasksPerTick: 8, MaxCallsPerHour: 3, ChunkSize: 3})
	now := t0
	tr.Now = func() time.Time { return now }
	var tasks []*harness.Task
	for i := 0; i < 10; i++ {
		tasks = append(tasks, task("b"+string(rune('a'+i)), harness.StateBlocked, 0, time.Minute))
	}
	snap := snapWith(tasks...)
	out := tr.Run(context.Background(), snap, nil)
	if fa.calls != 3 || out.JevCalls != 3 {
		t.Fatalf("8 tasks / chunk 3 = 3 calls, got %d", fa.calls)
	}
	out = tr.Run(context.Background(), snap, nil)
	if fa.calls != 3 || !strings.HasPrefix(out.Skipped, "budget") {
		t.Fatalf("hourly budget must block the second tick: calls=%d skipped=%q", fa.calls, out.Skipped)
	}
	now = t0.Add(61 * time.Minute)
	out = tr.Run(context.Background(), snap, nil)
	if fa.calls != 6 {
		t.Fatalf("budget window should roll over after an hour: calls=%d", fa.calls)
	}
	_ = out
}

func TestJevErrorFallsBackToDeterministic(t *testing.T) {
	fa := &fakeAsker{err: errors.New("429")}
	tr := NewTriager(fa, DefaultBudget())
	tr.Now = func() time.Time { return t0 }
	out := tr.Run(context.Background(), snapWith(task("b", harness.StateBlocked, 0, time.Minute)), nil)
	if !out.Deterministic || !strings.HasPrefix(out.Skipped, "jev error") || out.Get("b").Score != out.Get("b").Base {
		t.Fatalf("fallback: %+v", out)
	}
}

func TestTriageLogRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sub", "triage.jsonl")
	tr := NewTriager(nil, DefaultBudget())
	tr.Now = func() time.Time { return t0 }
	for i := 0; i < 3; i++ {
		if err := AppendTriage(p, tr.Run(context.Background(), snapWith(task("x", harness.StateReview, 0, 0)), nil)); err != nil {
			t.Fatal(err)
		}
	}
	got := LoadTriage(p, 2)
	if len(got) != 2 || got[0].Tasks[0].ID != "x" || got[0].Tasks[0].Next != ActReview {
		t.Fatalf("load: %+v", got)
	}
	b, _ := json.Marshal(got[0])
	if !strings.Contains(string(b), `"deterministic":true`) {
		t.Fatal("json shape")
	}
}

func TestQuestionIDsAreSafe(t *testing.T) {
	byID := map[string]*harness.Task{"T-1041/x": {ID: "T-1041/x", Title: `q"uote`}}
	qs := TriageQuestions([]string{"T-1041/x"}, byID)
	for id := range qs {
		if strings.ContainsAny(id, "-/ ") || !strings.HasPrefix(id, "T_1041_x__") {
			t.Fatalf("unsafe id %q", id)
		}
	}
	for _, p := range []*Pack{{ID: "p", Name: "p", StateSource: "text", Questions: qs}} {
		if err := p.Validate(); err != nil {
			t.Fatalf("triage questions must be valid jev questions: %v", err)
		}
	}
}
