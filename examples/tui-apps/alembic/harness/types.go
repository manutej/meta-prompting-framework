// Package harness defines the adapter contract between an agent harness and
// alembic: an append-only JSONL feed the harness writes, and an append-only
// JSONL outbox alembic writes. Both sides only ever append.
package harness

import "time"

const ContractVersion = 1

type TaskState string

const (
	StateQueued  TaskState = "queued"
	StateRunning TaskState = "running"
	StateBlocked TaskState = "blocked"
	StateReview  TaskState = "review"
	StateDone    TaskState = "done"
	StateFailed  TaskState = "failed"
)

type ElementKind string

const (
	ElemFile ElementKind = "file"
	ElemPR   ElementKind = "pr"
	ElemURL  ElementKind = "url"
	ElemLog  ElementKind = "log"
	ElemDir  ElementKind = "dir"
)

// Element is something a task points at that an operator may want to open.
type Element struct {
	Kind  ElementKind `json:"kind"`
	Ref   string      `json:"ref"`             // path, URL, or "owner/repo#123"
	Label string      `json:"label,omitempty"` // display name
	Line  int         `json:"line,omitempty"`  // for files
}

// Task is the unit shown as one row with a status line.
type Task struct {
	ID         string    `json:"id"`
	Workflow   string    `json:"workflow"`
	Title      string    `json:"title"`
	Agent      string    `json:"agent,omitempty"`
	State      TaskState `json:"state"`
	Progress   float64   `json:"progress,omitempty"` // 0..1
	StatusLine string    `json:"status_line"`        // one line, what the agent is doing right now
	Worktree   string    `json:"worktree,omitempty"` // absolute path
	Branch     string    `json:"branch,omitempty"`
	Elements   []Element `json:"elements,omitempty"`
	Priority   int       `json:"priority,omitempty"` // 0 normal, 1 high, 2 urgent
	Created    time.Time `json:"created"`
	Updated    time.Time `json:"updated"`
}

type Agent struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Model       string    `json:"model,omitempty"`
	State       string    `json:"state"` // idle | busy | offline
	CurrentTask string    `json:"current_task,omitempty"`
	Seen        time.Time `json:"seen"`
}

type Workflow struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Env   string `json:"env,omitempty"` // prod | staging | dev
	State string `json:"state,omitempty"`
}

type EventLevel string

const (
	LevelInfo  EventLevel = "info"
	LevelWarn  EventLevel = "warn"
	LevelError EventLevel = "error"
	LevelOK    EventLevel = "ok"
)

// Event is one line in a task's history.
type Event struct {
	TaskID string     `json:"task_id"`
	Level  EventLevel `json:"level"`
	Text   string     `json:"text"`
	At     time.Time  `json:"at"`
}

// PingAck is the harness telling alembic an agent received a ping.
type PingAck struct {
	PingID string    `json:"ping_id"`
	Agent  string    `json:"agent"`
	Text   string    `json:"text,omitempty"`
	At     time.Time `json:"at"`
}

// FeedRecord is one JSONL line the harness appends to the feed.
// Exactly one of the payload fields is set, chosen by Type.
type FeedRecord struct {
	V        int       `json:"v"`
	TS       time.Time `json:"ts"`
	Type     string    `json:"type"` // task.upsert | task.event | agent.upsert | workflow.upsert | ping.ack
	Task     *Task     `json:"task,omitempty"`
	Event    *Event    `json:"event,omitempty"`
	Agent    *Agent    `json:"agent,omitempty"`
	Workflow *Workflow `json:"workflow,omitempty"`
	PingAck  *PingAck  `json:"ping_ack,omitempty"`
}

// Command is one JSONL line alembic appends to the outbox for the harness.
type Command struct {
	V      int       `json:"v"`
	ID     string    `json:"id"`
	TS     time.Time `json:"ts"`
	Type   string    `json:"type"` // ping | task.cancel | task.retry | task.open | jev.receipt
	Agent  string    `json:"agent,omitempty"`
	TaskID string    `json:"task_id,omitempty"`
	Text   string    `json:"text,omitempty"`
	Data   any       `json:"data,omitempty"`
}

// Snapshot is alembic's in-memory view, rebuilt by folding the feed.
type Snapshot struct {
	Tasks     map[string]*Task
	Agents    map[string]*Agent
	Workflows map[string]*Workflow
	Events    map[string][]Event // by task id, oldest first
	Acks      []PingAck
	Records   int
}

func NewSnapshot() *Snapshot {
	return &Snapshot{
		Tasks:     map[string]*Task{},
		Agents:    map[string]*Agent{},
		Workflows: map[string]*Workflow{},
		Events:    map[string][]Event{},
	}
}

// Apply folds one record into the snapshot. Unknown types are ignored so a
// newer harness never breaks an older alembic.
func (s *Snapshot) Apply(r FeedRecord) {
	s.Records++
	switch r.Type {
	case "task.upsert":
		if r.Task != nil && r.Task.ID != "" {
			t := *r.Task
			if prev, ok := s.Tasks[t.ID]; ok {
				if t.Created.IsZero() {
					t.Created = prev.Created
				}
				if t.Elements == nil {
					t.Elements = prev.Elements
				}
			}
			if t.Created.IsZero() {
				t.Created = r.TS
			}
			if t.Updated.IsZero() {
				t.Updated = r.TS
			}
			if t.Workflow == "" {
				t.Workflow = "default"
			}
			if _, ok := s.Workflows[t.Workflow]; !ok {
				s.Workflows[t.Workflow] = &Workflow{ID: t.Workflow, Name: t.Workflow}
			}
			s.Tasks[t.ID] = &t
		}
	case "task.event":
		if r.Event != nil && r.Event.TaskID != "" {
			e := *r.Event
			if e.At.IsZero() {
				e.At = r.TS
			}
			evs := append(s.Events[e.TaskID], e)
			if len(evs) > 500 {
				evs = evs[len(evs)-500:]
			}
			s.Events[e.TaskID] = evs
			if t, ok := s.Tasks[e.TaskID]; ok && e.Level != LevelInfo {
				t.StatusLine = e.Text
				t.Updated = e.At
			}
		}
	case "agent.upsert":
		if r.Agent != nil && r.Agent.ID != "" {
			a := *r.Agent
			if a.Seen.IsZero() {
				a.Seen = r.TS
			}
			s.Agents[a.ID] = &a
		}
	case "workflow.upsert":
		if r.Workflow != nil && r.Workflow.ID != "" {
			w := *r.Workflow
			s.Workflows[w.ID] = &w
		}
	case "ping.ack":
		if r.PingAck != nil {
			a := *r.PingAck
			if a.At.IsZero() {
				a.At = r.TS
			}
			s.Acks = append(s.Acks, a)
		}
	}
}
