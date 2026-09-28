package harness

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Demo is a stand-in harness: it writes a believable feed and acknowledges
// pings from the outbox. It exists so alembic can be demoed with no harness
// attached; every record it writes is a normal feed record.
type Demo struct {
	FeedPath   string
	OutboxPath string
	outOffset  int64
	rng        *rand.Rand
	tick       int
}

func NewDemo(feed, outbox string) *Demo {
	return &Demo{FeedPath: feed, OutboxPath: outbox, rng: rand.New(rand.NewSource(7))}
}

func (d *Demo) write(recs ...FeedRecord) error {
	if err := os.MkdirAll(filepath.Dir(d.FeedPath), 0o755); err != nil {
		return err
	}
	fh, err := os.OpenFile(d.FeedPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer fh.Close()
	for _, r := range recs {
		r.V = ContractVersion
		if r.TS.IsZero() {
			r.TS = time.Now().UTC()
		}
		b, _ := json.Marshal(r)
		if _, err := fh.Write(append(b, '\n')); err != nil {
			return err
		}
	}
	return nil
}

var demoWorkflows = []Workflow{
	{ID: "checkout", Name: "checkout-service", Env: "prod", State: "healthy"},
	{ID: "billing", Name: "billing-pipeline", Env: "prod", State: "degraded"},
	{ID: "nexus", Name: "nexus-tui", Env: "dev", State: "healthy"},
}

var demoAgents = []Agent{
	{ID: "coder-1", Name: "Coder", Model: "claude-opus-5-5", State: "busy"},
	{ID: "reviewer-1", Name: "Reviewer", Model: "claude-sonnet-5", State: "busy"},
	{ID: "tester-1", Name: "Tester", Model: "claude-haiku-4-5", State: "idle"},
	{ID: "sre-1", Name: "SRE", Model: "claude-sonnet-5", State: "busy"},
}

func demoTasks(repo string) []Task {
	now := time.Now().UTC()
	wt := func(name string) string { return filepath.Join(filepath.Dir(repo), filepath.Base(repo)+"-"+name) }
	return []Task{
		{ID: "T-1041", Workflow: "checkout", Title: "Idempotency keys on /orders POST", Agent: "coder-1", State: StateRunning, Progress: 0.55,
			StatusLine: "writing dedupe middleware · 3 files touched", Worktree: wt("idempotency"), Branch: "feat/idempotency-keys", Priority: 1,
			Elements: []Element{{Kind: ElemFile, Ref: "internal/orders/handler.go", Line: 88}, {Kind: ElemPR, Ref: "acme/checkout#412", Label: "PR #412"}, {Kind: ElemFile, Ref: "docs/adr/0007-idempotency.md"}},
			Created:  now.Add(-42 * time.Minute)},
		{ID: "T-1042", Workflow: "checkout", Title: "Flaky test: TestCartMerge_Concurrent", Agent: "tester-1", State: StateBlocked, Progress: 0.3,
			StatusLine: "waiting: needs decision on retry budget", Worktree: wt("cart-merge"), Branch: "fix/cart-merge-flake", Priority: 2,
			Elements: []Element{{Kind: ElemFile, Ref: "internal/cart/merge_test.go", Line: 214}, {Kind: ElemLog, Ref: "ci/run-88123.log", Label: "CI log"}},
			Created:  now.Add(-3 * time.Hour)},
		{ID: "T-0977", Workflow: "billing", Title: "Reconcile Stripe payouts nightly", Agent: "sre-1", State: StateReview, Progress: 0.95,
			StatusLine: "PR ready · 2 approvals needed", Worktree: wt("payouts"), Branch: "feat/payout-reconcile",
			Elements: []Element{{Kind: ElemPR, Ref: "acme/billing#221", Label: "PR #221"}, {Kind: ElemURL, Ref: "https://grafana.acme.internal/d/payouts", Label: "Payouts dashboard"}},
			Created:  now.Add(-26 * time.Hour)},
		{ID: "T-0981", Workflow: "billing", Title: "Ledger: reject float amounts at the boundary", Agent: "reviewer-1", State: StateRunning, Progress: 0.7,
			StatusLine: "reviewing iteration 2 · quality 0.78", Worktree: wt("ledger-decimal"), Branch: "fix/ledger-decimal",
			Elements: []Element{{Kind: ElemFile, Ref: "internal/ledger/post.go", Line: 41}},
			Created:  now.Add(-55 * time.Minute)},
		{ID: "T-0990", Workflow: "billing", Title: "Rotate payout webhook secret", Agent: "sre-1", State: StateFailed, Progress: 0.4,
			StatusLine: "failed: vault write denied (403)", Elements: []Element{{Kind: ElemLog, Ref: "runs/T-0990.log", Label: "run log"}},
			Created: now.Add(-2 * time.Hour)},
		{ID: "T-2201", Workflow: "nexus", Title: "gitscope: quoted paths in porcelain v2", Agent: "coder-1", State: StateDone, Progress: 1,
			StatusLine: "shipped · 50 tests green", Worktree: repo, Branch: "claude/nextgen-tui-startup-plan",
			Elements: []Element{{Kind: ElemFile, Ref: "examples/tui-apps/gitscope/git.go", Line: 155}},
			Created:  now.Add(-5 * time.Hour)},
		{ID: "T-2202", Workflow: "nexus", Title: "alembic: Jev results pane", Agent: "", State: StateQueued, Progress: 0,
			StatusLine: "queued · waiting for a free agent", Created: now.Add(-10 * time.Minute)},
	}
}

// Seed writes the initial world if the feed is empty.
func (d *Demo) Seed(repo string) error {
	if st, err := os.Stat(d.FeedPath); err == nil && st.Size() > 0 {
		return nil
	}
	var recs []FeedRecord
	for i := range demoWorkflows {
		recs = append(recs, FeedRecord{Type: "workflow.upsert", Workflow: &demoWorkflows[i]})
	}
	for i := range demoAgents {
		a := demoAgents[i]
		a.Seen = time.Now().UTC()
		recs = append(recs, FeedRecord{Type: "agent.upsert", Agent: &a})
	}
	for _, t := range demoTasks(repo) {
		t := t
		t.Updated = time.Now().UTC()
		recs = append(recs, FeedRecord{Type: "task.upsert", Task: &t})
		recs = append(recs, FeedRecord{Type: "task.event", Event: &Event{TaskID: t.ID, Level: LevelInfo, Text: "task created"}})
		recs = append(recs, FeedRecord{Type: "task.event", Event: &Event{TaskID: t.ID, Level: LevelInfo, Text: t.StatusLine}})
	}
	return d.write(recs...)
}

var demoLines = map[TaskState][]string{
	StateRunning: {"reading %s", "editing %s", "running go test ./...", "tests green · 24 passed", "iteration %d · quality 0.%d", "waiting on LLM (streaming)", "refactoring for review notes"},
	StateBlocked: {"still waiting for a decision", "re-checked CI: same failure", "pinged owner in #checkout"},
	StateReview:  {"1 approval received", "addressing review comment on %s", "CI green on latest push"},
}

// Step advances the demo world: rotates status lines, occasionally moves a
// task between states, and acknowledges any new pings in the outbox.
func (d *Demo) Step(s *Snapshot) error {
	d.tick++
	var recs []FeedRecord
	ids := make([]string, 0, len(s.Tasks))
	for id := range s.Tasks {
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil
	}
	sortStrings(ids)
	t := s.Tasks[ids[d.rng.Intn(len(ids))]]
	if lines := demoLines[t.State]; len(lines) > 0 {
		line := lines[d.rng.Intn(len(lines))]
		file := "handler.go"
		if len(t.Elements) > 0 {
			file = filepath.Base(t.Elements[0].Ref)
		}
		switch strings.Count(line, "%") {
		case 1:
			line = fmt.Sprintf(line, file)
		case 2:
			line = fmt.Sprintf(line, 2+d.rng.Intn(3), 70+d.rng.Intn(25))
		}
		nt := *t
		nt.StatusLine = line
		if nt.State == StateRunning {
			nt.Progress = minF(0.98, nt.Progress+0.03+d.rng.Float64()*0.05)
		}
		nt.Updated = time.Now().UTC()
		recs = append(recs, FeedRecord{Type: "task.upsert", Task: &nt}, FeedRecord{Type: "task.event", Event: &Event{TaskID: nt.ID, Level: LevelInfo, Text: line}})
	}
	if d.tick%9 == 0 {
		for _, id := range ids {
			t := s.Tasks[id]
			if t.State == StateRunning && t.Progress > 0.9 {
				nt := *t
				nt.State, nt.Progress, nt.StatusLine = StateReview, 1, "PR opened · awaiting review"
				nt.Updated = time.Now().UTC()
				recs = append(recs, FeedRecord{Type: "task.upsert", Task: &nt}, FeedRecord{Type: "task.event", Event: &Event{TaskID: nt.ID, Level: LevelOK, Text: nt.StatusLine}})
				break
			}
			if t.State == StateQueued && t.Agent == "" {
				nt := *t
				nt.State, nt.Agent, nt.Progress, nt.StatusLine = StateRunning, "tester-1", 0.05, "picked up by Tester · reading task"
				nt.Updated = time.Now().UTC()
				recs = append(recs, FeedRecord{Type: "task.upsert", Task: &nt}, FeedRecord{Type: "task.event", Event: &Event{TaskID: nt.ID, Level: LevelOK, Text: nt.StatusLine}})
				break
			}
		}
	}
	acks, err := d.ackPings(s)
	if err == nil {
		recs = append(recs, acks...)
	}
	if len(recs) == 0 {
		return nil
	}
	return d.write(recs...)
}

func (d *Demo) ackPings(s *Snapshot) ([]FeedRecord, error) {
	fh, err := os.Open(d.OutboxPath)
	if err != nil {
		return nil, nil
	}
	defer fh.Close()
	if _, err := fh.Seek(d.outOffset, 0); err != nil {
		return nil, err
	}
	var recs []FeedRecord
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		line := sc.Bytes()
		d.outOffset += int64(len(line) + 1)
		var c Command
		if json.Unmarshal(line, &c) != nil {
			continue
		}
		switch c.Type {
		case "ping":
			reply := "ack — on it"
			if strings.Contains(strings.ToLower(c.Text), "stop") {
				reply = "ack — pausing after current step"
			}
			recs = append(recs, FeedRecord{Type: "ping.ack", PingAck: &PingAck{PingID: c.ID, Agent: c.Agent, Text: reply}})
			if c.TaskID != "" {
				recs = append(recs, FeedRecord{Type: "task.event", Event: &Event{TaskID: c.TaskID, Level: LevelWarn, Text: "operator: " + c.Text}})
			}
		case "task.cancel":
			// an upsert replaces every field but elements/created, so it
			// must carry the whole task, not just the changed fields
			t, ok := s.Tasks[c.TaskID]
			if !ok {
				continue
			}
			nt := *t
			nt.State, nt.StatusLine, nt.Updated = StateFailed, "cancelled by operator", time.Now().UTC()
			recs = append(recs, FeedRecord{Type: "task.event", Event: &Event{TaskID: c.TaskID, Level: LevelError, Text: "cancelled by operator"}})
			recs = append(recs, FeedRecord{Type: "task.upsert", Task: &nt})
		case "task.retry":
			t, ok := s.Tasks[c.TaskID]
			if !ok {
				continue
			}
			nt := *t
			nt.State, nt.Progress, nt.StatusLine, nt.Updated = StateQueued, 0, "queued for retry", time.Now().UTC()
			recs = append(recs, FeedRecord{Type: "task.event", Event: &Event{TaskID: c.TaskID, Level: LevelOK, Text: "retry requested by operator"}})
			recs = append(recs, FeedRecord{Type: "task.upsert", Task: &nt})
		}
	}
	return recs, nil
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
