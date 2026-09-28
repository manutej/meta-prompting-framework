package main

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

func key(s string) tea.KeyMsg {
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEscape}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	case "ctrl+k":
		return tea.KeyMsg{Type: tea.KeyCtrlK}
	case "up":
		return tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case " ":
		return tea.KeyMsg{Type: tea.KeySpace}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
}

func sized(w, h int) model {
	m := newModel("demo task")
	mm, _ := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return mm.(model)
}

func press(m model, keys ...string) (model, tea.Cmd) {
	var cmd tea.Cmd
	for _, k := range keys {
		var mm tea.Model
		mm, cmd = m.Update(key(k))
		m = mm.(model)
	}
	return m, cmd
}

// runScript applies every scripted step synchronously.
func runScript(m model) model {
	m, _ = m.startRun()
	for i := 0; i < len(m.script); i++ {
		m, _ = m.applyStep(i)
	}
	return m
}

func assertFits(t *testing.T, m model, w, h int) {
	t.Helper()
	v := m.View()
	lines := strings.Split(v, "\n")
	if len(lines) != h {
		t.Fatalf("%dx%d: view has %d lines, want %d", w, h, len(lines), h)
	}
	for i, l := range lines {
		if lw := len([]rune(stripANSI(l))); lw > w {
			t.Fatalf("%dx%d: line %d is %d cols (> %d): %q", w, h, i, lw, w, stripANSI(l))
		}
	}
}

func TestRunStartsPipeline(t *testing.T) {
	m := sized(120, 40)
	m, cmd := press(m, "r")
	if !m.running || cmd == nil {
		t.Fatalf("r should start pipeline and return cmds")
	}
	if !strings.Contains(stripANSI(m.View()), "pipeline started") {
		t.Fatalf("log line missing")
	}
}

func TestFullPipelineReachesQuality(t *testing.T) {
	m := runScript(sized(120, 40))
	if m.running {
		t.Fatal("pipeline should be finished")
	}
	if m.quality != 0.88 || m.iter != 3 {
		t.Fatalf("quality=%v iter=%d", m.quality, m.iter)
	}
	for _, a := range m.agents {
		if a.state != stDone {
			t.Fatalf("%s state %s", a.name, a.state)
		}
	}
	if m.tokens == 0 || m.cost == 0 {
		t.Fatal("tokens/cost not accumulated")
	}
}

func TestInjectedFailureSelfHeals(t *testing.T) {
	m := sized(120, 40)
	m, _ = m.startRun()
	m, _ = m.injectFailure()
	for i := 0; i < len(m.script); i++ {
		m, _ = m.applyStep(i)
	}
	v := stripANSI(strings.Join(logTexts(m), "\n"))
	if !strings.Contains(v, "FAIL TestFuzzyFilter") || !strings.Contains(v, "Iteration 4") {
		t.Fatalf("failure path not executed:\n%s", v)
	}
	if m.iter != 4 || m.running {
		t.Fatalf("iter=%d running=%v", m.iter, m.running)
	}
	if m.agents[m.agentIndex(aDeployer)].state != stDone {
		t.Fatal("deployer should finish after self-heal")
	}
}

func logTexts(m model) []string {
	out := make([]string, len(m.logs))
	for i, e := range m.logs {
		out[i] = e.text
	}
	return out
}

func TestPauseHoldsStep(t *testing.T) {
	m := sized(120, 40)
	m, _ = m.startRun()
	m, _ = m.applyStep(0)
	m, _ = m.togglePause()
	m, cmd := m.applyStep(1)
	if cmd != nil || m.heldStep != 1 {
		t.Fatalf("paused step should be held: held=%d", m.heldStep)
	}
	m, cmd = m.togglePause()
	if cmd == nil || m.heldStep != -1 {
		t.Fatal("resume should reschedule held step")
	}
}

func TestKillAndResume(t *testing.T) {
	m := sized(120, 40)
	m, _ = m.startRun()
	m, _ = m.applyStep(0) // researcher running
	m, _ = press(m, "k")
	if !m.confirmOpen {
		t.Fatal("kill should open confirm")
	}
	m, _ = press(m, "y")
	if m.agents[0].state != stKilled {
		t.Fatal("agent not killed")
	}
	m, cmd := m.applyStep(1)
	if m.running || cmd != nil {
		t.Fatal("pipeline should halt on killed agent")
	}
	m, _ = m.resumeAfterKill()
	if !m.running || m.agents[0].state != stRunning {
		t.Fatal("resume should revive agent")
	}
}

func TestKillDeclined(t *testing.T) {
	m := sized(120, 40)
	m, _ = m.startRun()
	m, _ = m.applyStep(0)
	m, _ = press(m, "k", "n")
	if m.confirmOpen || m.agents[0].state != stRunning {
		t.Fatal("decline should leave agent running")
	}
}

func TestPaletteFuzzyAndRun(t *testing.T) {
	m := sized(120, 40)
	m, _ = press(m, "ctrl+k")
	if !m.paletteOpen || len(m.matches) != len(m.cmds) {
		t.Fatal("palette should open with all commands")
	}
	m, _ = press(m, "m", "e", "t")
	if len(m.matches) == 0 || m.cmds[m.matches[0]].name != "Go to Metrics" {
		t.Fatalf("fuzzy 'met' top match = %q", m.cmds[m.matches[0]].name)
	}
	m, _ = press(m, "enter")
	if m.paletteOpen || m.tab != tabMetrics {
		t.Fatal("enter should run the command and close palette")
	}
	if !strings.Contains(stripANSI(sizedView(m)), "Metrics") {
		t.Fatal("metrics tab not rendered")
	}
}

func sizedView(m model) string { return m.View() }

func TestPaletteNoMatch(t *testing.T) {
	m := sized(120, 40)
	m, _ = press(m, "ctrl+k", "z", "z", "z", "z")
	if len(m.matches) != 0 {
		t.Fatal("expected no matches")
	}
	m, _ = press(m, "enter")
	if !m.paletteOpen {
		t.Fatal("enter with no matches should keep palette open")
	}
	if !strings.Contains(stripANSI(m.View()), "no matching commands") {
		t.Fatal("empty state missing")
	}
}

func TestHelpOverlay(t *testing.T) {
	m := sized(120, 40)
	m, _ = press(m, "?")
	if !m.helpOpen || !strings.Contains(stripANSI(m.View()), "keys") {
		t.Fatal("help not shown")
	}
	m, _ = press(m, "esc")
	if m.helpOpen {
		t.Fatal("esc should close help")
	}
}

func TestTabsAndFocus(t *testing.T) {
	m := sized(120, 40)
	for i, k := range []string{"1", "2", "3", "4"} {
		m, _ = press(m, k)
		if m.tab != i {
			t.Fatalf("tab %s -> %d", k, m.tab)
		}
		assertFits(t, m, 120, 40)
	}
	m, _ = press(m, "tab", "tab", "tab")
	if m.focus != focusAgents {
		t.Fatal("tab should cycle back")
	}
}

func TestViewFitsAllSizes(t *testing.T) {
	for _, sz := range [][2]int{{80, 24}, {100, 30}, {120, 40}, {200, 60}, {60, 20}} {
		m := runScript(sized(sz[0], sz[1]))
		for tab := 0; tab < 4; tab++ {
			m.tab = tab
			assertFits(t, m, sz[0], sz[1])
		}
		m.tab = tabOverview
		m, _ = press(m, "ctrl+k")
		assertFits(t, m, sz[0], sz[1])
		m, _ = press(m, "esc", "?")
		assertFits(t, m, sz[0], sz[1])
	}
}

func TestTinyTerminal(t *testing.T) {
	m := sized(15, 5)
	if v := m.View(); !strings.Contains(v, "too small") {
		t.Fatal("tiny terminal should show message")
	}
}

func TestLogScrollAndFollow(t *testing.T) {
	m := runScript(sized(120, 40))
	m, _ = press(m, "tab") // focus logs
	if !m.follow {
		t.Fatal("follow should start on")
	}
	m, _ = press(m, "g")
	if m.follow || m.logOffset != 0 {
		t.Fatal("g should jump to top and disable follow")
	}
	m, _ = press(m, "G")
	if !m.follow {
		t.Fatal("G should re-enable follow")
	}
	m, _ = press(m, "e")
	for _, e := range m.visibleLogs() {
		if e.lvl != lvErr && e.lvl != lvWarn {
			t.Fatal("errors-only filter leaked an info line")
		}
	}
}

func TestStreamingReveals(t *testing.T) {
	m := sized(120, 40)
	m, _ = m.startRun()
	m, _ = m.applyStep(0)
	m, _ = m.applyStep(1) // streaming LLM line
	e := m.logs[len(m.logs)-1]
	if !e.stream || e.revealed != 0 {
		t.Fatal("stream entry should start hidden")
	}
	mm, cmd := m.Update(streamMsg{})
	m = mm.(model)
	if m.logs[len(m.logs)-1].revealed != 3 || cmd == nil {
		t.Fatal("stream should reveal 3 runes per tick and continue")
	}
	for i := 0; i < 200; i++ {
		mm, cmd = m.Update(streamMsg{})
		m = mm.(model)
		if cmd == nil {
			break
		}
	}
	if m.logs[len(m.logs)-1].revealed != len([]rune(e.text)) {
		t.Fatal("stream should finish fully revealed")
	}
}

func TestSpringSettles(t *testing.T) {
	m := sized(120, 40)
	m.agents[0].target = 1
	for i := 0; i < 300; i++ {
		mm, _ := m.Update(animMsg(time.Now()))
		m = mm.(model)
	}
	if m.pos[0] < 0.99 || m.animating {
		t.Fatalf("spring should settle at target: pos=%v animating=%v", m.pos[0], m.animating)
	}
}

func TestMouseClickSelectsAgentAndTab(t *testing.T) {
	m := sized(120, 40)
	ar, _, _ := m.layoutRects()
	mm, _ := m.Update(tea.MouseMsg{X: ar.x + 2, Y: ar.y + 1 + 2*3, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress})
	m = mm.(model)
	if m.selected != 3 {
		t.Fatalf("click row 3 -> selected %d", m.selected)
	}
	mm, _ = m.Update(tea.MouseMsg{X: 14, Y: 1, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress})
	m = mm.(model)
	if m.tab != tabAgents {
		t.Fatalf("click on second tab -> %d", m.tab)
	}
}

func TestToastExpires(t *testing.T) {
	m := sized(120, 40)
	m, _ = press(m, "p") // nothing running -> error toast
	if m.toast == "" {
		t.Fatal("toast expected")
	}
	id := m.toastID
	mm, _ := m.Update(toastGoneMsg{id})
	if mm.(model).toast != "" {
		t.Fatal("toast should clear")
	}
}

func TestQuitKeys(t *testing.T) {
	m := sized(120, 40)
	for _, k := range []string{"q"} {
		_, cmd := press(m, k)
		if cmd == nil {
			t.Fatalf("%s should quit", k)
		}
	}
}

func TestFitAndCutCols(t *testing.T) {
	if got := fit("日本語テキスト", 5); len([]rune(stripANSI(got))) > 5 || lipglossWidth(got) != 5 {
		t.Fatalf("fit CJK: %q", got)
	}
	if got := cutCols("abcdef", 2, 4); got != "cd" {
		t.Fatalf("cutCols = %q", got)
	}
	if got := gauge(1.5, 4, gold); strings.Count(stripANSI(got), "█") != 4 {
		t.Fatal("gauge should clamp")
	}
	if got := sparkline(nil, 5, 0, 1, gold); len(got) != 5 {
		t.Fatal("empty sparkline should be padded")
	}
}

func lipglossWidth(s string) int { return lipgloss.Width(s) }
