// Package jev calls TypeSafe AI's System One model (Jev) and, when no API key
// is present, a deterministic mock that is always labelled as such.
package jev

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"time"
)

const (
	DefaultEndpoint = "https://api.typesafe.ai/v1/systemone"
	DefaultModel    = "jev-latest"
)

type QuestionType string

const (
	Noul   QuestionType = "noul"
	Choice QuestionType = "choice"
	Score  QuestionType = "score"
)

// Question mirrors the wire format. Criteria is a map for noul/choice and an
// ordered list for score, so it is kept as raw JSON and validated by type.
type Question struct {
	Type         QuestionType    `json:"type"`
	Instructions string          `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// Options returns the choice option names in a stable order.
func (q Question) Options() []string {
	var m map[string]string
	if err := json.Unmarshal(q.Criteria, &m); err != nil {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Levels returns the ordered score levels.
func (q Question) Levels() []string {
	var l []string
	_ = json.Unmarshal(q.Criteria, &l)
	return l
}

type Request struct {
	Model     string              `json:"model"`
	State     any                 `json:"state"`
	Questions map[string]Question `json:"questions"`
}

type Answer struct {
	Type          QuestionType       `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// Probability is the headline number for an answer: P(yes) for noul, the
// winning option's probability for choice, and confidence for score.
func (a Answer) Probability() float64 {
	switch a.Type {
	case Noul:
		if a.Noul != nil {
			return *a.Noul
		}
	case Choice:
		if p, ok := a.Probabilities[a.Choice]; ok {
			return p
		}
	case Score:
		if a.Confidence != nil {
			return *a.Confidence
		}
	}
	return 0
}

// Conf is calibrated confidence: max(p, 1-p) for noul, the reported
// confidence otherwise.
func (a Answer) Conf() float64 {
	if a.Type == Noul && a.Noul != nil {
		p := *a.Noul
		if p < 0.5 {
			return 1 - p
		}
		return p
	}
	if a.Confidence != nil {
		return *a.Confidence
	}
	return 0
}

// Level returns the nearest score level name.
func (a Answer) Level() string {
	if a.Type != Score || a.Score == nil {
		return ""
	}
	idx := int(*a.Score + 0.5)
	if name, ok := a.Legend[fmt.Sprint(idx)]; ok {
		return name
	}
	return fmt.Sprintf("%.2f", *a.Score)
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Response struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
	Latency time.Duration     `json:"-"`
	Mock    bool              `json:"-"`
}

type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string {
	switch e.Status {
	case 401:
		return "jev: missing or invalid API key (TYPESAFE_API_KEY)"
	case 422:
		return "jev: request rejected (422): " + e.Body
	case 429:
		return "jev: rate limited (429) — retry with backoff"
	case 529:
		return "jev: service overloaded (529) — retry with backoff"
	}
	return fmt.Sprintf("jev: HTTP %d: %s", e.Status, e.Body)
}

type Client struct {
	Endpoint string
	APIKey   string
	Model    string
	HTTP     *http.Client
}

// NewFromEnv builds a client from TYPESAFE_API_KEY / TYPESAFE_ENDPOINT /
// TYPESAFE_MODEL. With no key the client runs in mock mode.
func NewFromEnv() *Client {
	c := &Client{
		Endpoint: os.Getenv("TYPESAFE_ENDPOINT"),
		APIKey:   os.Getenv("TYPESAFE_API_KEY"),
		Model:    os.Getenv("TYPESAFE_MODEL"),
		HTTP:     &http.Client{Timeout: 30 * time.Second},
	}
	if c.Endpoint == "" {
		c.Endpoint = DefaultEndpoint
	}
	if c.Model == "" {
		c.Model = DefaultModel
	}
	return c
}

func (c *Client) IsMock() bool { return c.APIKey == "" }

// Ask evaluates all questions in one call (they run in parallel server-side).
func (c *Client) Ask(ctx context.Context, state any, questions map[string]Question) (*Response, error) {
	if len(questions) == 0 {
		return nil, errors.New("jev: no questions")
	}
	start := time.Now()
	if c.IsMock() {
		r := mockAnswer(state, questions)
		r.Latency = time.Since(start)
		return r, nil
	}
	body, err := json.Marshal(Request{Model: c.Model, State: state, Questions: questions})
	if err != nil {
		return nil, err
	}
	var last error
	for attempt := 0; attempt < 3; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := c.HTTP.Do(req)
		if err != nil {
			last = err
		} else {
			data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			resp.Body.Close()
			if resp.StatusCode == 200 {
				var out Response
				if err := json.Unmarshal(data, &out); err != nil {
					return nil, fmt.Errorf("jev: bad response: %w", err)
				}
				out.Latency = time.Since(start)
				return &out, nil
			}
			last = &APIError{Status: resp.StatusCode, Body: string(bytes.TrimSpace(data))}
			if resp.StatusCode != 429 && resp.StatusCode != 529 {
				return nil, last
			}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Duration(500*(1<<attempt)) * time.Millisecond):
		}
	}
	return nil, last
}

// mockAnswer is deterministic in (state, question) so demos are repeatable.
func mockAnswer(state any, questions map[string]Question) *Response {
	sb, _ := json.Marshal(state)
	r := &Response{Model: "jev-mock", Answers: map[string]Answer{}, Mock: true}
	r.Usage.InputTokens = len(sb) / 4
	for id, q := range questions {
		h := sha256.Sum256(append(append([]byte(id), '|'), sb...))
		u := func(i int) float64 { return float64(binary.BigEndian.Uint16(h[i*2:])) / 65535 }
		switch q.Type {
		case Noul:
			p := 0.05 + 0.9*u(0)
			r.Answers[id] = Answer{Type: Noul, Noul: &p}
		case Choice:
			opts := q.Options()
			if len(opts) == 0 {
				continue
			}
			probs, best, bestP := map[string]float64{}, "", 0.0
			total := 0.0
			for i, o := range opts {
				w := 0.05 + u(i%16)
				probs[o] = w
				total += w
			}
			for o := range probs {
				probs[o] /= total
				if probs[o] > bestP {
					best, bestP = o, probs[o]
				}
			}
			conf := 0.5 + 0.45*u(15)
			r.Answers[id] = Answer{Type: Choice, Choice: best, Probabilities: probs, Confidence: &conf}
		case Score:
			levels := q.Levels()
			if len(levels) == 0 {
				continue
			}
			legend, probs := map[string]string{}, map[string]float64{}
			total := 0.0
			for i, l := range levels {
				legend[fmt.Sprint(i)] = l
				w := 0.02 + u(i%16)*u((i+3)%16)
				probs[fmt.Sprint(i)] = w
				total += w
			}
			score := 0.0
			for k := range probs {
				probs[k] /= total
				var idx int
				fmt.Sscan(k, &idx)
				score += float64(idx) * probs[k]
			}
			conf := 0.5 + 0.45*u(14)
			r.Answers[id] = Answer{Type: Score, Score: &score, Legend: legend, Probabilities: probs, Confidence: &conf}
		}
		r.Usage.OutputTokens += 12
	}
	return r
}
