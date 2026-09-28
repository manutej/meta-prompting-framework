package harness

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Outbox appends commands for the harness. Append-only, one JSON object per
// line, fsync'd, so a crash never leaves a half-written command.
type Outbox struct{ Path string }

func NewOutbox(path string) *Outbox { return &Outbox{Path: path} }

func newID() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return time.Now().UTC().Format("20060102T150405") + "-" + hex.EncodeToString(b)
}

// Send appends one command and returns it with ID and timestamp filled in.
func (o *Outbox) Send(c Command) (Command, error) {
	c.V = ContractVersion
	if c.ID == "" {
		c.ID = newID()
	}
	c.TS = time.Now().UTC()
	if err := os.MkdirAll(filepath.Dir(o.Path), 0o755); err != nil {
		return c, err
	}
	fh, err := os.OpenFile(o.Path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return c, err
	}
	defer fh.Close()
	line, err := json.Marshal(c)
	if err != nil {
		return c, err
	}
	if _, err := fh.Write(append(line, '\n')); err != nil {
		return c, err
	}
	return c, fh.Sync()
}

func (o *Outbox) Ping(agent, taskID, text string) (Command, error) {
	return o.Send(Command{Type: "ping", Agent: agent, TaskID: taskID, Text: text})
}

func (o *Outbox) Cancel(taskID string) (Command, error) {
	return o.Send(Command{Type: "task.cancel", TaskID: taskID})
}

func (o *Outbox) Retry(taskID string) (Command, error) {
	return o.Send(Command{Type: "task.retry", TaskID: taskID})
}
