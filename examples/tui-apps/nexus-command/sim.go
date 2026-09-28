package main

import (
	"fmt"
	"math/rand"
	"time"
)

type agentState int

const (
	stIdle agentState = iota
	stQueued
	stRunning
	stDone
	stError
	stKilled
)

func (s agentState) String() string {
	return [...]string{"idle", "queued", "running", "done", "error", "killed"}[s]
}

type agent struct {
	name    string
	role    string
	model   string
	state   agentState
	target  float64 // progress goal 0..1
	tokens  int
	costUSD float64
	started time.Time
	elapsed time.Duration
	lastMsg string
}

type level int

const (
	lvInfo level = iota
	lvOK
	lvWarn
	lvErr
	lvLLM
)

type logEntry struct {
	at       time.Time
	lvl      level
	agent    string
	text     string
	revealed int // chars revealed for streaming entries
	stream   bool
}

// A step is one unit of scripted pipeline work.
type step struct {
	agent    string
	lvl      level
	text     string
	stream   bool
	delayMs  int
	progress float64 // agent progress after this step
	quality  float64 // >0 sets pipeline quality
	tokens   int
	finish   bool // agent transitions to done after this step
	start    bool // agent transitions to running before this step
	iter     int  // >0 sets iteration counter
	failable bool // step where injected failure lands
}

const (
	aResearcher = "Researcher"
	aPlanner    = "Planner"
	aCoder      = "Coder"
	aReviewer   = "Reviewer"
	aTester     = "Tester"
	aDeployer   = "Deployer"
)

func newAgents() []agent {
	return []agent{
		{name: aResearcher, role: "context & prior art", model: "claude-haiku-4-5"},
		{name: aPlanner, role: "complexity & strategy", model: "claude-sonnet-5"},
		{name: aCoder, role: "Bubble Tea generation", model: "claude-opus-5-5"},
		{name: aReviewer, role: "quality assessment", model: "claude-sonnet-5"},
		{name: aTester, role: "compile & headless tests", model: "claude-haiku-4-5"},
		{name: aDeployer, role: "package & ship", model: "claude-haiku-4-5"},
	}
}

func pipelineScript(task string) []step {
	return []step{
		{agent: aResearcher, start: true, lvl: lvInfo, text: "Task received: " + task, delayMs: 300},
		{agent: aResearcher, lvl: lvLLM, stream: true, text: "Scanning Charmbracelet ecosystem: bubbletea, bubbles/list, bubbles/viewport, lipgloss…", delayMs: 900, progress: 0.35, tokens: 812},
		{agent: aResearcher, lvl: lvInfo, text: "Found 3 prior patterns (glow, lazygit, superfile) — extracting key-binding conventions", delayMs: 700, progress: 0.7, tokens: 1204},
		{agent: aResearcher, lvl: lvOK, text: "Context bundle ready (2.1k tokens)", delayMs: 400, progress: 1, finish: true},

		{agent: aPlanner, start: true, lvl: lvInfo, text: "Analyzing complexity: scope 0.62 · ambiguity 0.41 · deps 0.50 · domain 0.30", delayMs: 800, progress: 0.4, tokens: 640},
		{agent: aPlanner, lvl: lvWarn, text: "Overall complexity 0.72 → strategy: ITERATIVE (target quality ≥ 0.85, max 5 iterations)", delayMs: 600, progress: 0.8, tokens: 910},
		{agent: aPlanner, lvl: lvOK, text: "Plan: list+viewport split · fuzzy filter · vim keys · Gold/Navy theme", delayMs: 500, progress: 1, finish: true},

		// iteration 1
		{agent: aCoder, start: true, iter: 1, lvl: lvInfo, text: "Iteration 1 — generating from plan", delayMs: 400},
		{agent: aCoder, lvl: lvLLM, stream: true, text: "package main … type model struct { list list.Model; preview viewport.Model } … func (m model) Update(msg tea.Msg) …", delayMs: 1800, progress: 0.5, tokens: 3120},
		{agent: aCoder, lvl: lvOK, text: "Emitted main.go (214 lines) — handing to Reviewer", delayMs: 500, progress: 1, finish: true},
		{agent: aReviewer, start: true, lvl: lvInfo, text: "Assessing: functionality 0.65 · aesthetics 0.58 · code 0.70 · perf 0.62", delayMs: 900, progress: 0.6, tokens: 1480},
		{agent: aReviewer, lvl: lvWarn, text: "Quality 0.62 < 0.85 — issues: fuzzy filter missing, theme not applied, no preview pane", delayMs: 600, progress: 1, finish: true, quality: 0.62},

		// iteration 2
		{agent: aCoder, start: true, iter: 2, lvl: lvInfo, text: "Iteration 2 — refining with reviewer context (215 chars)", delayMs: 400},
		{agent: aCoder, lvl: lvLLM, stream: true, text: "func fuzzyMatch(pattern, s string) bool … preview.SetContent(…) … lipgloss.NewStyle().Foreground(lipgloss.Color(\"178\"))", delayMs: 1600, progress: 0.55, tokens: 3890},
		{agent: aCoder, lvl: lvOK, text: "Emitted main.go (298 lines)", delayMs: 400, progress: 1, finish: true},
		{agent: aReviewer, start: true, lvl: lvInfo, text: "Assessing: functionality 0.82 · aesthetics 0.75 · code 0.78 · perf 0.74", delayMs: 900, progress: 0.6, tokens: 1510},
		{agent: aReviewer, lvl: lvWarn, text: "Quality 0.78 < 0.85 — issues: theme colors inconsistent, vim keys partial", delayMs: 600, progress: 1, finish: true, quality: 0.78},

		// iteration 3
		{agent: aCoder, start: true, iter: 3, lvl: lvInfo, text: "Iteration 3 — refining with reviewer context (189 chars)", delayMs: 400},
		{agent: aCoder, lvl: lvLLM, stream: true, text: "case \"g\": m.cursor = 0 … case \"G\": … focusPane = lipgloss.NewStyle().BorderForeground(gold) …", delayMs: 1500, progress: 0.6, tokens: 4210},
		{agent: aCoder, lvl: lvOK, text: "Emitted main.go (341 lines)", delayMs: 400, progress: 1, finish: true},
		{agent: aReviewer, start: true, lvl: lvInfo, text: "Assessing: functionality 0.90 · aesthetics 0.86 · code 0.88 · perf 0.85", delayMs: 900, progress: 0.7, tokens: 1530},
		{agent: aReviewer, lvl: lvOK, text: "Quality 0.88 ≥ 0.85 — threshold met, releasing to Tester", delayMs: 500, progress: 1, finish: true, quality: 0.88},

		{agent: aTester, start: true, lvl: lvInfo, text: "go build ./... && go vet ./...", delayMs: 900, progress: 0.35, tokens: 210},
		{agent: aTester, lvl: lvInfo, text: "Running 24 headless tests (Update/View driven, no TTY)", delayMs: 1100, progress: 0.7, tokens: 380, failable: true},
		{agent: aTester, lvl: lvOK, text: "ok  file-browser  0.31s — 24 passed, 0 failed", delayMs: 500, progress: 1, finish: true},

		{agent: aDeployer, start: true, lvl: lvInfo, text: "Packaging output/file-browser/ (main.go, go.mod, README.md)", delayMs: 700, progress: 0.5, tokens: 140},
		{agent: aDeployer, lvl: lvOK, text: "Shipped ✓ — 3 iterations · quality 0.88 · $0.41 · 46s", delayMs: 400, progress: 1, finish: true},
	}
}

var failureSteps = []step{
	{agent: aTester, lvl: lvErr, text: "FAIL TestFuzzyFilter_Unicode — index out of range [-1]", delayMs: 500},
	{agent: aTester, lvl: lvErr, text: "1 of 24 tests failed — returning to Coder with failure context", delayMs: 600, progress: 1, finish: true},
	{agent: aCoder, start: true, lvl: lvInfo, text: "Iteration 4 — repairing from test failure (stack trace 312 chars)", delayMs: 400, iter: 4},
	{agent: aCoder, lvl: lvLLM, stream: true, text: "if m.cursor >= len(m.filtered) { m.cursor = max(0, len(m.filtered)-1) }", delayMs: 1200, progress: 0.6, tokens: 1720},
	{agent: aCoder, lvl: lvOK, text: "Patched main.go (+3 −1)", delayMs: 400, progress: 1, finish: true},
	{agent: aTester, start: true, lvl: lvInfo, text: "Re-running 24 headless tests", delayMs: 1000, progress: 0.7, tokens: 380},
}

func jitter(ms int) time.Duration {
	f := 0.75 + rand.Float64()*0.5
	return time.Duration(float64(ms)*f) * time.Millisecond
}

func costFor(model string, tokens int) float64 {
	per1k := map[string]float64{"claude-haiku-4-5": 0.004, "claude-sonnet-5": 0.015, "claude-opus-5-5": 0.075}[model]
	return float64(tokens) / 1000 * per1k
}

func fmtDuration(d time.Duration) string {
	d = d.Round(time.Second)
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}
