package jev

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Pack is a named set of questions designed for one core task, plus the gate
// that turns answers into a decision. Packs are JSON files in a directory.
type Pack struct {
	ID          string              `json:"id"`
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	StateSource string              `json:"state_source"` // task | events | staged_diff | working_diff | file | text
	Questions   map[string]Question `json:"questions"`
	Order       []string            `json:"order,omitempty"` // display order; defaults to sorted ids
	Gate        *Gate               `json:"gate,omitempty"`
	Path        string              `json:"-"`
}

// Gate follows the aurum-gate vocabulary: per-action thresholds on the
// headline probability and calibrated confidence, plus which answers count
// as favorable. Decision is auto / escalate / refuse.
type Gate struct {
	Action         string            `json:"action"`
	Favorable      map[string]string `json:"favorable"` // question id -> "yes"|"no"|option|level
	MinProbability float64           `json:"minProbability"`
	AutoConfidence float64           `json:"autoConfidence"`
	RefuseBelow    float64           `json:"refuseBelow"`
}

type Decision string

const (
	DecisionAuto     Decision = "auto"
	DecisionEscalate Decision = "escalate"
	DecisionRefuse   Decision = "refuse"
)

// Verdict is the gate applied to one response.
type Verdict struct {
	Decision    Decision
	Unfavorable []string // question ids that answered against the favorable value
	MinConf     float64  // lowest calibrated confidence across gated questions
	MinProb     float64  // lowest favorable-side probability
	Reason      string
}

func (p *Pack) QuestionIDs() []string {
	if len(p.Order) > 0 {
		return p.Order
	}
	ids := make([]string, 0, len(p.Questions))
	for id := range p.Questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (p *Pack) Validate() error {
	if p.ID == "" || p.Name == "" {
		return errors.New("pack needs id and name")
	}
	if len(p.Questions) == 0 {
		return errors.New("pack has no questions")
	}
	for id, q := range p.Questions {
		switch q.Type {
		case Noul:
		case Choice:
			if n := len(q.Options()); n < 2 || n > 255 {
				return fmt.Errorf("question %q: choice needs 2..255 options", id)
			}
		case Score:
			if n := len(q.Levels()); n < 2 || n > 10 {
				return fmt.Errorf("question %q: score needs 2..10 levels", id)
			}
		default:
			return fmt.Errorf("question %q: unknown type %q", id, q.Type)
		}
		if strings.TrimSpace(q.Instructions) == "" {
			return fmt.Errorf("question %q: empty instructions", id)
		}
	}
	if g := p.Gate; g != nil {
		if g.RefuseBelow < 0 || g.MinProbability > 1 || g.AutoConfidence > 1 || g.RefuseBelow > g.MinProbability {
			return fmt.Errorf("gate thresholds must satisfy 0 <= refuseBelow <= minProbability <= 1 and autoConfidence <= 1")
		}
		for id := range g.Favorable {
			if _, ok := p.Questions[id]; !ok {
				return fmt.Errorf("gate references unknown question %q", id)
			}
		}
	}
	return nil
}

// LoadDir reads every *.json pack in dir, sorted by id. Invalid packs are
// returned as errors keyed by filename rather than aborting the load.
func LoadDir(dir string) ([]*Pack, map[string]error) {
	errs := map[string]error{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		errs[dir] = err
		return nil, errs
	}
	var packs []*Pack
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			errs[e.Name()] = err
			continue
		}
		var p Pack
		if err := json.Unmarshal(data, &p); err != nil {
			errs[e.Name()] = err
			continue
		}
		p.Path = path
		if err := p.Validate(); err != nil {
			errs[e.Name()] = err
			continue
		}
		packs = append(packs, &p)
	}
	sort.Slice(packs, func(i, j int) bool { return packs[i].ID < packs[j].ID })
	return packs, errs
}

// Evaluate applies the pack's gate to a response.
func (p *Pack) Evaluate(r *Response) Verdict {
	v := Verdict{Decision: DecisionAuto, MinConf: 1, MinProb: 1}
	if p.Gate == nil {
		v.Reason = "no gate defined; informational"
		return v
	}
	g := p.Gate
	for id, want := range g.Favorable {
		a, ok := r.Answers[id]
		if !ok {
			v.Unfavorable = append(v.Unfavorable, id)
			v.MinProb, v.MinConf = 0, 0
			continue
		}
		fav, prob := favorable(a, want)
		if !fav {
			v.Unfavorable = append(v.Unfavorable, id)
		}
		if prob < v.MinProb {
			v.MinProb = prob
		}
		if c := a.Conf(); c < v.MinConf {
			v.MinConf = c
		}
	}
	sort.Strings(v.Unfavorable)
	switch {
	case len(v.Unfavorable) > 0 && v.MinProb < g.RefuseBelow:
		v.Decision = DecisionRefuse
		v.Reason = fmt.Sprintf("%s below refuse floor %.2f", strings.Join(v.Unfavorable, ", "), g.RefuseBelow)
	case len(v.Unfavorable) > 0:
		v.Decision = DecisionEscalate
		v.Reason = "unfavorable: " + strings.Join(v.Unfavorable, ", ")
	case v.MinProb < g.MinProbability:
		v.Decision = DecisionEscalate
		v.Reason = fmt.Sprintf("probability %.2f < minProbability %.2f", v.MinProb, g.MinProbability)
	case v.MinConf < g.AutoConfidence:
		v.Decision = DecisionEscalate
		v.Reason = fmt.Sprintf("confidence %.2f < autoConfidence %.2f", v.MinConf, g.AutoConfidence)
	default:
		v.Reason = fmt.Sprintf("all favorable · p ≥ %.2f · conf ≥ %.2f", v.MinProb, v.MinConf)
	}
	return v
}

// favorable reports whether answer a matches the favorable value and the
// probability mass on the favorable side.
func favorable(a Answer, want string) (bool, float64) {
	switch a.Type {
	case Noul:
		p := a.Probability()
		if strings.EqualFold(want, "no") {
			return p < 0.5, 1 - p
		}
		return p >= 0.5, p
	case Choice:
		p := a.Probabilities[want]
		return a.Choice == want, p
	case Score:
		// favorable value is a level name or "<=name" / ">=name"
		op, name := "", want
		if strings.HasPrefix(want, "<=") || strings.HasPrefix(want, ">=") {
			op, name = want[:2], strings.TrimSpace(want[2:])
		}
		target := -1
		for k, v := range a.Legend {
			if strings.EqualFold(v, name) {
				fmt.Sscan(k, &target)
			}
		}
		if target < 0 || a.Score == nil {
			return false, 0
		}
		idx := int(*a.Score + 0.5)
		mass := 0.0
		for k, p := range a.Probabilities {
			var i int
			fmt.Sscan(k, &i)
			if (op == "<=" && i <= target) || (op == ">=" && i >= target) || (op == "" && i == target) {
				mass += p
			}
		}
		switch op {
		case "<=":
			return idx <= target, mass
		case ">=":
			return idx >= target, mass
		}
		return idx == target, mass
	}
	return false, 0
}
