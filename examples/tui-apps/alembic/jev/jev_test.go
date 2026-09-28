package jev

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func f(v float64) *float64 { return &v }

func choiceProbs(choice string, cp float64) map[string]float64 {
	m := map[string]float64{choice: cp}
	if choice != "ui" {
		m["ui"] = 1 - cp
	} else {
		m["api"] = 1 - cp
	}
	return m
}

func TestLoadPacksDir(t *testing.T) {
	packs, errs := LoadDir("../packs")
	if len(errs) != 0 {
		t.Fatalf("errors: %v", errs)
	}
	if len(packs) != 5 {
		t.Fatalf("want 5 packs, got %d", len(packs))
	}
	for _, p := range packs {
		if err := p.Validate(); err != nil {
			t.Fatalf("%s: %v", p.ID, err)
		}
		if len(p.QuestionIDs()) != len(p.Questions) {
			t.Fatalf("%s: order incomplete", p.ID)
		}
	}
}

func TestInvalidPacksAreReportedNotFatal(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "bad.json"), []byte(`{"id":"b","name":"b","questions":{"q":{"type":"score","instructions":"x","criteria":["only one"]}}}`), 0o644)
	os.WriteFile(filepath.Join(dir, "junk.json"), []byte(`{`), 0o644)
	os.WriteFile(filepath.Join(dir, "gate.json"), []byte(`{"id":"g","name":"g","questions":{"q":{"type":"noul","instructions":"x"}},"gate":{"action":"a","favorable":{"nope":"yes"},"minProbability":0.5,"autoConfidence":0.9,"refuseBelow":0.7}}`), 0o644)
	os.WriteFile(filepath.Join(dir, "ok.json"), []byte(`{"id":"ok","name":"ok","state_source":"text","questions":{"q":{"type":"noul","instructions":"x"}}}`), 0o644)
	packs, errs := LoadDir(dir)
	if len(packs) != 1 || packs[0].ID != "ok" {
		t.Fatalf("packs: %+v", packs)
	}
	for _, name := range []string{"bad.json", "junk.json", "gate.json"} {
		if errs[name] == nil {
			t.Fatalf("%s should be rejected", name)
		}
	}
}

func TestMockIsDeterministicAndWellFormed(t *testing.T) {
	c := &Client{}
	packs, _ := LoadDir("../packs")
	for _, p := range packs {
		r1, err := c.Ask(context.Background(), "some state", p.Questions)
		if err != nil || !r1.Mock {
			t.Fatalf("%s: %v", p.ID, err)
		}
		r2, _ := c.Ask(context.Background(), "some state", p.Questions)
		a, _ := json.Marshal(r1.Answers)
		b, _ := json.Marshal(r2.Answers)
		if string(a) != string(b) {
			t.Fatalf("%s: mock not deterministic", p.ID)
		}
		for id, q := range p.Questions {
			ans, ok := r1.Answers[id]
			if !ok || ans.Type != q.Type {
				t.Fatalf("%s/%s: missing or wrong type", p.ID, id)
			}
			switch q.Type {
			case Choice:
				sum := 0.0
				for _, v := range ans.Probabilities {
					sum += v
				}
				if sum < 0.999 || sum > 1.001 || ans.Probabilities[ans.Choice] == 0 {
					t.Fatalf("%s/%s: choice probs sum %v", p.ID, id, sum)
				}
			case Score:
				if ans.Score == nil || len(ans.Legend) != len(q.Levels()) || ans.Level() == "" {
					t.Fatalf("%s/%s: score shape %+v", p.ID, id, ans)
				}
			case Noul:
				if ans.Noul == nil || *ans.Noul < 0 || *ans.Noul > 1 || ans.Conf() < 0.5 {
					t.Fatalf("%s/%s: noul %+v", p.ID, id, ans)
				}
			}
		}
	}
}

func TestGateDecisions(t *testing.T) {
	p := &Pack{ID: "t", Name: "t", Questions: map[string]Question{
		"risky": {Type: Noul, Instructions: "x"},
		"area":  {Type: Choice, Instructions: "x", Criteria: json.RawMessage(`{"ui":"","api":""}`)},
		"blast": {Type: Score, Instructions: "x", Criteria: json.RawMessage(`["Cosmetic","Minor","Major"]`)},
	}, Gate: &Gate{Action: "a", Favorable: map[string]string{"risky": "no", "area": "ui", "blast": "<=Minor"}, MinProbability: 0.7, AutoConfidence: 0.9, RefuseBelow: 0.3}}
	legend := map[string]string{"0": "Cosmetic", "1": "Minor", "2": "Major"}
	mk := func(risky float64, choice string, cp float64, score float64, sprobs map[string]float64, conf float64) *Response {
		return &Response{Answers: map[string]Answer{
			"risky": {Type: Noul, Noul: f(risky)},
			"area":  {Type: Choice, Choice: choice, Probabilities: choiceProbs(choice, cp), Confidence: f(conf)},
			"blast": {Type: Score, Score: f(score), Legend: legend, Probabilities: sprobs, Confidence: f(conf)},
		}}
	}
	lowBlast := map[string]float64{"0": 0.6, "1": 0.35, "2": 0.05}
	// "api" wins with 0.6, "ui" keeps 0.4: unfavorable but above the refuse floor
	if v := p.Evaluate(mk(0.05, "ui", 0.95, 0.4, lowBlast, 0.95)); v.Decision != DecisionAuto {
		t.Fatalf("all favorable should be auto: %+v", v)
	}
	if v := p.Evaluate(mk(0.05, "ui", 0.95, 0.4, lowBlast, 0.85)); v.Decision != DecisionEscalate || v.MinConf > 0.86 {
		t.Fatalf("low confidence should escalate: %+v", v)
	}
	if v := p.Evaluate(mk(0.6, "ui", 0.95, 0.4, lowBlast, 0.95)); v.Decision != DecisionEscalate || v.Unfavorable[0] != "risky" {
		t.Fatalf("risky=yes should escalate: %+v", v)
	}
	if v := p.Evaluate(mk(0.05, "api", 0.6, 0.4, lowBlast, 0.95)); v.Decision != DecisionEscalate || v.Unfavorable[0] != "area" {
		t.Fatalf("wrong area should escalate: %+v", v)
	}
	if v := p.Evaluate(mk(0.05, "api", 0.95, 0.4, lowBlast, 0.95)); v.Decision != DecisionRefuse {
		t.Fatalf("wrong area with 0.05 favorable mass should refuse: %+v", v)
	}
	highBlast := map[string]float64{"0": 0.05, "1": 0.1, "2": 0.85}
	if v := p.Evaluate(mk(0.05, "ui", 0.95, 1.9, highBlast, 0.95)); v.Decision != DecisionRefuse {
		t.Fatalf("blast Major with 0.15 favorable mass should refuse: %+v", v)
	}
	if v := p.Evaluate(&Response{Answers: map[string]Answer{}}); v.Decision != DecisionRefuse {
		t.Fatalf("missing answers should refuse: %+v", v)
	}
	if v := (&Pack{Questions: p.Questions}).Evaluate(mk(0.9, "api", 0.9, 2, highBlast, 0.5)); v.Decision != DecisionAuto {
		t.Fatal("no gate means informational auto")
	}
}

func TestClientHTTPAndRetry(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if r.Header.Get("Authorization") != "Bearer k" {
			w.WriteHeader(401)
			return
		}
		var req Request
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "jev-latest" || len(req.Questions) != 1 {
			w.WriteHeader(422)
			w.Write([]byte(`{"error":"bad"}`))
			return
		}
		if n == 1 {
			w.WriteHeader(429)
			return
		}
		w.Write([]byte(`{"model":"jev-1.13.0","answers":{"q":{"type":"score","score":1.05,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.0,"1":0.95,"2":0.05},"confidence":0.92}},"usage":{"input_tokens":304,"output_tokens":18}}`))
	}))
	defer srv.Close()
	c := &Client{Endpoint: srv.URL, APIKey: "k", Model: DefaultModel, HTTP: srv.Client()}
	q := map[string]Question{"q": {Type: Score, Instructions: "x", Criteria: json.RawMessage(`["Calm","Frustrated","Very angry"]`)}}
	r, err := c.Ask(context.Background(), "state", q)
	if err != nil || r.Mock || atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected retry after 429: %v calls=%d", err, calls)
	}
	a := r.Answers["q"]
	if a.Level() != "Frustrated" || a.Conf() != 0.92 || r.Usage.InputTokens != 304 {
		t.Fatalf("parse: %+v", a)
	}
	bad := &Client{Endpoint: srv.URL, APIKey: "wrong", Model: DefaultModel, HTTP: srv.Client()}
	if _, err := bad.Ask(context.Background(), "s", q); err == nil || err.(*APIError).Status != 401 {
		t.Fatalf("401 expected: %v", err)
	}
}

func TestReceiptsRoundTrip(t *testing.T) {
	packs, _ := LoadDir("../packs")
	p := packs[0]
	c := &Client{}
	r, _ := c.Ask(context.Background(), "state", p.Questions)
	v := p.Evaluate(r)
	rc := NewReceipt(p, "T-1", []byte("state"), r, v)
	if rc.Digest == "" || rc.StateSHA == "" || !rc.Mock {
		t.Fatalf("receipt: %+v", rc)
	}
	dir := t.TempDir()
	if _, err := SaveReceipt(dir, rc); err != nil {
		t.Fatal(err)
	}
	got := LoadReceipts(dir, p.ID)
	if len(got) != 1 || got[0].ID != rc.ID || got[0].Decision != v.Decision {
		t.Fatalf("load: %+v", got)
	}
	if len(LoadReceipts(dir, "other")) != 0 {
		t.Fatal("filter by pack")
	}
}
