package jev

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"alembic/harness"
)

// Triage is the observability layer: a cheap deterministic score for every
// task on every tick, with Jev consulted only for the ambiguous ones, under a
// budget declared before anything is spent. Its output ranks tasks and
// recommends the next operator action.

// PricePerMillionInput is TypeSafe's published list price (USD); output is free.
const PricePerMillionInput = 0.042

type Budget struct {
	MaxTasksPerTick int // tasks sent to Jev per tick (0 = deterministic only)
	MaxCallsPerHour int // Jev HTTP calls in any rolling hour
	ChunkSize       int // tasks per Jev call (3 questions each)
}

func DefaultBudget() Budget { return Budget{MaxTasksPerTick: 8, MaxCallsPerHour: 60, ChunkSize: 6} }

type Action string

const (
	ActWait   Action = "wait"
	ActPing   Action = "ping"
	ActReview Action = "review"
	ActCancel Action = "cancel"
	ActRetry  Action = "retry"
)

type TaskTriage struct {
	ID         string             `json:"id"`
	Rank       int                `json:"rank"`
	Score      float64            `json:"score"`     // 0..1, higher = look at it sooner
	Base       float64            `json:"base"`      // deterministic part
	Reasons    []string           `json:"reasons"`   // human-readable, deterministic first
	Next       Action             `json:"next"`      // recommended operator action
	NextConf   float64            `json:"next_conf"` // 0 when deterministic
	JevUsed    bool               `json:"jev_used"`
	Stuck      float64            `json:"stuck,omitempty"`       // P(stuck) from Jev
	NeedsHuman float64            `json:"needs_human,omitempty"` // P(needs human) from Jev
	NextProbs  map[string]float64 `json:"next_probs,omitempty"`
}

type Triage struct {
	At            time.Time     `json:"at"`
	Tasks         []TaskTriage  `json:"tasks"`
	JevCalls      int           `json:"jev_calls"`
	JevTasks      int           `json:"jev_tasks"`
	Mock          bool          `json:"mock"`
	Usage         Usage         `json:"usage"`
	CostUSD       float64       `json:"cost_usd"`
	Latency       time.Duration `json:"latency"`
	Skipped       string        `json:"skipped,omitempty"` // why Jev was not consulted
	Deterministic bool          `json:"deterministic"`     // true when no Jev answer was used
}

// Next is the top-ranked task, or nil when nothing needs attention.
func (t Triage) Next() *TaskTriage {
	for i := range t.Tasks {
		if t.Tasks[i].Score > 0 {
			return &t.Tasks[i]
		}
	}
	return nil
}

func (t Triage) Get(id string) *TaskTriage {
	for i := range t.Tasks {
		if t.Tasks[i].ID == id {
			return &t.Tasks[i]
		}
	}
	return nil
}

type Asker interface {
	Ask(ctx context.Context, state any, questions map[string]Question) (*Response, error)
	IsMock() bool
}

type Triager struct {
	Client Asker
	Budget Budget
	Now    func() time.Time
	calls  []time.Time
	base   map[string]baseline // progress baseline per task, for stall detection
}

// baseline is the progress value first seen and when; it resets only when
// progress changes, so a stall accumulates across ticks of any interval.
type baseline struct {
	progress float64
	since    time.Time
}

func NewTriager(c Asker, b Budget) *Triager {
	return &Triager{Client: c, Budget: b, Now: time.Now, base: map[string]baseline{}}
}

var stateBase = map[harness.TaskState]float64{
	harness.StateBlocked: 0.75, harness.StateFailed: 0.70, harness.StateReview: 0.55,
	harness.StateRunning: 0.35, harness.StateQueued: 0.20, harness.StateDone: 0,
}

var defaultNext = map[harness.TaskState]Action{
	harness.StateBlocked: ActPing, harness.StateFailed: ActRetry, harness.StateReview: ActReview,
	harness.StateRunning: ActWait, harness.StateQueued: ActWait, harness.StateDone: ActWait,
}

// Deterministic scores one task with no model call. pendingPings is the
// number of unacknowledged pings for the task (0 if unknown).
func (tr *Triager) Deterministic(t *harness.Task, events []harness.Event, pendingPings int) TaskTriage {
	now := tr.Now()
	tt := TaskTriage{ID: t.ID, Next: defaultNext[t.State]}
	s := stateBase[t.State]
	tt.Reasons = append(tt.Reasons, string(t.State))
	if t.State == harness.StateDone {
		tt.Score, tt.Base = 0, 0
		return tt
	}
	if t.Priority > 0 {
		s += 0.10 * float64(min(2, t.Priority))
		tt.Reasons = append(tt.Reasons, fmt.Sprintf("priority %d", t.Priority))
	}
	age := now.Sub(t.Updated)
	switch {
	case t.State == harness.StateRunning && age > 30*time.Minute:
		s += 0.25
		tt.Reasons = append(tt.Reasons, "stale "+shortDur(age))
	case t.State == harness.StateRunning && age > 10*time.Minute:
		s += 0.15
		tt.Reasons = append(tt.Reasons, "stale "+shortDur(age))
	case (t.State == harness.StateBlocked || t.State == harness.StateReview) && age > 30*time.Minute:
		s += 0.10
		tt.Reasons = append(tt.Reasons, "waiting "+shortDur(age))
	}
	errs, warns := 0, 0
	for i := max(0, len(events)-10); i < len(events); i++ {
		switch events[i].Level {
		case harness.LevelError:
			errs++
		case harness.LevelWarn:
			warns++
		}
	}
	if errs > 0 {
		s += min(0.15, 0.05*float64(errs))
		tt.Reasons = append(tt.Reasons, fmt.Sprintf("%d error%s", errs, plural(errs)))
	}
	if warns > 0 {
		s += min(0.06, 0.02*float64(warns))
	}
	if pendingPings > 0 {
		s += 0.10
		tt.Reasons = append(tt.Reasons, fmt.Sprintf("%d ping%s unanswered", pendingPings, plural(pendingPings)))
	}
	if b, ok := tr.base[t.ID]; ok && t.State == harness.StateRunning && b.progress == t.Progress && now.Sub(b.since) >= 5*time.Minute {
		s += 0.10
		tt.Reasons = append(tt.Reasons, "no progress for "+shortDur(now.Sub(b.since)))
	}
	tt.Base = clamp(s, 0, 1)
	tt.Score = tt.Base
	return tt
}

// ambiguous reports whether the deterministic view leaves real doubt worth a
// model call: the task is alive and either waiting on something, stale, or
// throwing errors.
func ambiguous(t *harness.Task, tt TaskTriage) bool {
	if t.State == harness.StateDone || t.State == harness.StateQueued {
		return false
	}
	if t.State == harness.StateBlocked || t.State == harness.StateReview || t.State == harness.StateFailed {
		return true
	}
	for _, r := range tt.Reasons {
		if strings.HasPrefix(r, "stale") || strings.HasSuffix(r, "unanswered") || strings.Contains(r, "error") || strings.HasPrefix(r, "no progress") {
			return true
		}
	}
	return false
}

// Run performs one tick. It never returns an error for Jev failures: the
// deterministic ranking is always produced and Skipped says why Jev was not
// used.
func (tr *Triager) Run(ctx context.Context, snap *harness.Snapshot, pendingPings map[string]int) Triage {
	start := tr.Now()
	out := Triage{At: start, Deterministic: true}
	ids := make([]string, 0, len(snap.Tasks))
	for id := range snap.Tasks {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	byID := map[string]*harness.Task{}
	for _, id := range ids {
		t := snap.Tasks[id]
		byID[id] = t
		out.Tasks = append(out.Tasks, tr.Deterministic(t, snap.Events[id], pendingPings[id]))
	}
	tr.sortAndRank(&out)

	var candidates []string
	for _, tt := range out.Tasks {
		if ambiguous(byID[tt.ID], tt) {
			candidates = append(candidates, tt.ID)
		}
	}
	switch {
	case tr.Client == nil || tr.Budget.MaxTasksPerTick <= 0:
		out.Skipped = "jev disabled"
	case len(candidates) == 0:
		out.Skipped = "nothing ambiguous"
	case !tr.withinBudget(start):
		out.Skipped = fmt.Sprintf("budget: %d calls/hour reached", tr.Budget.MaxCallsPerHour)
	default:
		if len(candidates) > tr.Budget.MaxTasksPerTick {
			candidates = candidates[:tr.Budget.MaxTasksPerTick]
		}
		chunk := tr.Budget.ChunkSize
		if chunk <= 0 {
			chunk = 6
		}
		for i := 0; i < len(candidates); i += chunk {
			if !tr.withinBudget(tr.Now()) {
				out.Skipped = "budget reached mid-tick"
				break
			}
			end := min(len(candidates), i+chunk)
			tr.askChunk(ctx, &out, candidates[i:end], byID, snap)
		}
		if out.JevCalls > 0 {
			tr.sortAndRank(&out)
		}
	}
	next := map[string]baseline{}
	for _, t := range byID {
		if b, ok := tr.base[t.ID]; ok && b.progress == t.Progress {
			next[t.ID] = b
		} else {
			next[t.ID] = baseline{progress: t.Progress, since: start}
		}
	}
	tr.base = next
	out.Latency = tr.Now().Sub(start)
	if !out.Mock {
		out.CostUSD = float64(out.Usage.InputTokens) / 1e6 * PricePerMillionInput
	}
	return out
}

func (tr *Triager) withinBudget(now time.Time) bool {
	cutoff := now.Add(-time.Hour)
	kept := tr.calls[:0]
	for _, c := range tr.calls {
		if c.After(cutoff) {
			kept = append(kept, c)
		}
	}
	tr.calls = kept
	return tr.Budget.MaxCallsPerHour <= 0 || len(tr.calls) < tr.Budget.MaxCallsPerHour
}

type compactTask struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	State       string   `json:"state"`
	Agent       string   `json:"agent,omitempty"`
	Progress    float64  `json:"progress"`
	StatusLine  string   `json:"status_line"`
	MinutesIdle int      `json:"minutes_since_update"`
	Events      []string `json:"recent_events"`
}

func (tr *Triager) compact(t *harness.Task, events []harness.Event) compactTask {
	c := compactTask{ID: t.ID, Title: t.Title, State: string(t.State), Agent: t.Agent, Progress: t.Progress,
		StatusLine: t.StatusLine, MinutesIdle: int(tr.Now().Sub(t.Updated).Minutes())}
	for i := max(0, len(events)-5); i < len(events); i++ {
		c.Events = append(c.Events, string(events[i].Level)+": "+events[i].Text)
	}
	return c
}

func qid(taskID, q string) string {
	safe := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			return r
		}
		return '_'
	}, taskID)
	return safe + "__" + q
}

// TriageQuestions builds the namespaced question set for a batch of tasks.
func TriageQuestions(ids []string, byID map[string]*harness.Task) map[string]Question {
	qs := map[string]Question{}
	for _, id := range ids {
		t := byID[id]
		ref := fmt.Sprintf("Task %s (%q)", t.ID, t.Title)
		qs[qid(id, "stuck")] = Question{Type: Noul, Instructions: ref + ": the agent is looping or has stopped making progress.",
			Criteria: json.RawMessage(`{"true":"repeated identical attempts, or no forward movement for a long period","false":"events show forward movement"}`)}
		qs[qid(id, "needs_human")] = Question{Type: Noul, Instructions: ref + ": a human decision is required before the agent can continue.",
			Criteria: json.RawMessage(`{"true":"waiting on approval, credentials, or a product decision","false":"the agent can proceed on its own"}`)}
		qs[qid(id, "next")] = Question{Type: Choice, Instructions: ref + ": what should the operator do next?",
			Criteria: json.RawMessage(`{"wait":"nothing; the agent is progressing","ping":"send a short nudge or clarification","review":"open the output and review it now","cancel":"stop the task; it is off course","retry":"restart from the last good state"}`)}
	}
	return qs
}

func (tr *Triager) askChunk(ctx context.Context, out *Triage, ids []string, byID map[string]*harness.Task, snap *harness.Snapshot) {
	state := make([]compactTask, 0, len(ids))
	for _, id := range ids {
		state = append(state, tr.compact(byID[id], snap.Events[id]))
	}
	tr.calls = append(tr.calls, tr.Now())
	resp, err := tr.Client.Ask(ctx, state, TriageQuestions(ids, byID))
	if err != nil {
		out.Skipped = "jev error: " + err.Error()
		return
	}
	out.JevCalls++
	out.Mock = out.Mock || resp.Mock
	out.Usage.InputTokens += resp.Usage.InputTokens
	out.Usage.OutputTokens += resp.Usage.OutputTokens
	for _, id := range ids {
		tt := out.Get(id)
		if tt == nil {
			continue
		}
		stuck, ok1 := resp.Answers[qid(id, "stuck")]
		human, ok2 := resp.Answers[qid(id, "needs_human")]
		next, ok3 := resp.Answers[qid(id, "next")]
		if !ok1 && !ok2 && !ok3 {
			continue
		}
		tt.JevUsed = true
		out.JevTasks++
		out.Deterministic = false
		s := tt.Base
		if ok1 {
			tt.Stuck = stuck.Probability()
			s += 0.15 * tt.Stuck
			if tt.Stuck >= 0.7 {
				tt.Reasons = append(tt.Reasons, fmt.Sprintf("jev: stuck %.2f", tt.Stuck))
			}
		}
		if ok2 {
			tt.NeedsHuman = human.Probability()
			s += 0.15 * tt.NeedsHuman
			if tt.NeedsHuman >= 0.7 {
				tt.Reasons = append(tt.Reasons, fmt.Sprintf("jev: needs human %.2f", tt.NeedsHuman))
			}
		}
		if ok3 && next.Choice != "" {
			tt.NextProbs = next.Probabilities
			if c := next.Conf(); c >= 0.6 {
				tt.Next, tt.NextConf = Action(next.Choice), c
			}
		}
		tt.Score = clamp(s, 0, 1)
	}
}

func (tr *Triager) sortAndRank(out *Triage) {
	sort.SliceStable(out.Tasks, func(i, j int) bool {
		if out.Tasks[i].Score != out.Tasks[j].Score {
			return out.Tasks[i].Score > out.Tasks[j].Score
		}
		return out.Tasks[i].ID < out.Tasks[j].ID
	})
	for i := range out.Tasks {
		out.Tasks[i].Rank = i + 1
	}
}

// AppendTriage writes one tick to an append-only JSONL log.
func AppendTriage(path string, t Triage) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer fh.Close()
	b, err := json.Marshal(t)
	if err != nil {
		return err
	}
	_, err = fh.Write(append(b, '\n'))
	return err
}

// LoadTriage returns the last n ticks, oldest first.
func LoadTriage(path string, n int) []Triage {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	out := make([]Triage, 0, len(lines))
	for _, l := range lines {
		var t Triage
		if json.Unmarshal([]byte(l), &t) == nil {
			out = append(out, t)
		}
	}
	return out
}

func shortDur(d time.Duration) string {
	switch {
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
	return fmt.Sprintf("%dd", int(d.Hours()/24))
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
