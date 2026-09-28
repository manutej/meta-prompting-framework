package jev

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Receipt records one pack run: the request that was sent, the answers, and
// the verdict. It is written once and never edited; a digest ties the fields
// together (a digest is not a signature — it detects accidents, not attackers).
type Receipt struct {
	Schema      int               `json:"schema"`
	ID          string            `json:"id"`
	At          time.Time         `json:"at"`
	PackID      string            `json:"pack_id"`
	PackName    string            `json:"pack_name"`
	StateSource string            `json:"state_source"`
	StateRef    string            `json:"state_ref,omitempty"` // task id / file / worktree
	StateSHA    string            `json:"state_sha256"`
	StateBytes  int               `json:"state_bytes"`
	Model       string            `json:"model"`
	Mock        bool              `json:"mock"`
	LatencyMs   int64             `json:"latency_ms"`
	Usage       Usage             `json:"usage"`
	Answers     map[string]Answer `json:"answers"`
	Decision    Decision          `json:"decision"`
	Reason      string            `json:"reason"`
	Unfavorable []string          `json:"unfavorable,omitempty"`
	Digest      string            `json:"digest"`
}

func NewReceipt(p *Pack, stateRef string, state []byte, r *Response, v Verdict) Receipt {
	sum := sha256.Sum256(state)
	rc := Receipt{
		Schema: 1, At: time.Now().UTC(), PackID: p.ID, PackName: p.Name,
		StateSource: p.StateSource, StateRef: stateRef,
		StateSHA: hex.EncodeToString(sum[:]), StateBytes: len(state),
		Model: r.Model, Mock: r.Mock, LatencyMs: r.Latency.Milliseconds(), Usage: r.Usage,
		Answers: r.Answers, Decision: v.Decision, Reason: v.Reason, Unfavorable: v.Unfavorable,
	}
	rc.ID = rc.At.Format("20060102T150405.000") + "-" + p.ID
	body, _ := json.Marshal(struct {
		Receipt
		Digest string `json:"-"`
	}{Receipt: rc})
	d := sha256.Sum256(body)
	rc.Digest = hex.EncodeToString(d[:8])
	return rc
}

func SaveReceipt(dir string, rc Receipt) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, rc.ID+".json")
	data, err := json.MarshalIndent(rc, "", "  ")
	if err != nil {
		return "", err
	}
	return path, os.WriteFile(path, data, 0o644)
}

// LoadReceipts returns receipts newest first, optionally filtered by pack.
func LoadReceipts(dir, packID string) []Receipt {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []Receipt
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var rc Receipt
		if json.Unmarshal(data, &rc) != nil || (packID != "" && rc.PackID != packID) {
			continue
		}
		out = append(out, rc)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].At.After(out[j].At) })
	return out
}
