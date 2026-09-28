package main

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/muesli/termenv"
)

// isQuit reports whether cmd is tea.Quit (bubbletea has no exported sentinel,
// so we execute the cmd and inspect the resulting message).
func isQuit(t *testing.T, cmd tea.Cmd) bool {
	t.Helper()
	if cmd == nil {
		return false
	}
	_, ok := cmd().(tea.QuitMsg)
	return ok
}

func asModel(t *testing.T, tm tea.Model) model {
	t.Helper()
	m, ok := tm.(model)
	if !ok {
		t.Fatalf("Update returned %T, want model", tm)
	}
	return m
}

func key(s string) tea.KeyMsg {
	switch s {
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEscape}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
}

func TestInitialModel(t *testing.T) {
	m := initialModel()
	if m.percent != 0 {
		t.Errorf("percent = %v, want 0", m.percent)
	}
	if m.duration != 30*time.Second {
		t.Errorf("duration = %v, want 30s", m.duration)
	}
	if time.Since(m.startTime) > time.Second {
		t.Errorf("startTime not recent: %v", m.startTime)
	}
	if m.progress.Width != maxWidth-padding*2-4 {
		t.Errorf("progress width = %d, want %d", m.progress.Width, maxWidth-padding*2-4)
	}
	if m.Init() == nil {
		t.Error("Init() returned nil cmd; expected a tick")
	}
}

func TestQuitKeys(t *testing.T) {
	for _, k := range []string{"q", "ctrl+c", "esc"} {
		t.Run(k, func(t *testing.T) {
			m := initialModel()
			_, cmd := m.Update(key(k))
			if !isQuit(t, cmd) {
				t.Errorf("key %q did not return tea.Quit", k)
			}
		})
	}
}

func TestNonQuitKeysDoNothing(t *testing.T) {
	m := initialModel()
	for _, k := range []string{"x", "Q", "enter", " "} {
		nm, cmd := m.Update(key(k))
		if isQuit(t, cmd) {
			t.Errorf("key %q unexpectedly quit", k)
		}
		if cmd != nil {
			t.Errorf("key %q returned a cmd; want nil", k)
		}
		if asModel(t, nm).percent != m.percent {
			t.Errorf("key %q mutated percent", k)
		}
	}
}

func TestResetRestoresFreshState(t *testing.T) {
	m := initialModel()
	m.percent = 0.75
	m.startTime = time.Now().Add(-20 * time.Second)

	nm, cmd := m.Update(key("r"))
	if isQuit(t, cmd) {
		t.Fatal("reset must not quit")
	}
	r := asModel(t, nm)
	if r.percent != 0 {
		t.Errorf("after reset percent = %v, want 0", r.percent)
	}
	if time.Since(r.startTime) > time.Second {
		t.Errorf("after reset startTime not refreshed: %v", r.startTime)
	}
	if r.duration != 30*time.Second {
		t.Errorf("after reset duration = %v, want 30s", r.duration)
	}
}

// BUG: reset must not forget the terminal size the user already told us about.
func TestResetKeepsWindowWidth(t *testing.T) {
	m := initialModel()
	nm, _ := m.Update(tea.WindowSizeMsg{Width: 20, Height: 10})
	m = asModel(t, nm)
	want := m.progress.Width
	if want != 20-padding*2-4 {
		t.Fatalf("precondition: width = %d, want %d", want, 20-padding*2-4)
	}

	nm, _ = m.Update(key("r"))
	r := asModel(t, nm)
	if r.progress.Width != want {
		t.Errorf("reset discarded window size: progress width = %d, want %d", r.progress.Width, want)
	}
}

// BUG: the tick chain started by Init() is self-sustaining (every tickMsg
// returns another tickCmd) and only ends with tea.Quit. If 'r' also returns a
// tickCmd, every reset adds another concurrent 50ms tick loop forever.
func TestResetDoesNotMultiplyTickLoops(t *testing.T) {
	var cur tea.Model = initialModel()
	pending := []tea.Cmd{cur.Init()}

	// Simulate the runtime: press 'r' three times, then run the loop a couple
	// of rounds, counting how many live tick chains we end up with.
	for i := 0; i < 3; i++ {
		nm, cmd := cur.Update(key("r"))
		cur = nm
		if cmd != nil {
			pending = append(pending, cmd)
		}
	}
	for round := 0; round < 2; round++ {
		var next []tea.Cmd
		for _, cmd := range pending {
			msg := cmd()
			if _, ok := msg.(tickMsg); !ok {
				t.Fatalf("expected tickMsg, got %T", msg)
			}
			nm, c := cur.Update(msg)
			cur = nm
			if c != nil {
				next = append(next, c)
			}
		}
		pending = next
	}
	if len(pending) != 1 {
		t.Errorf("after 3 resets there are %d concurrent tick loops, want exactly 1", len(pending))
	}
}

func TestTickUpdatesPercent(t *testing.T) {
	m := initialModel()
	m.startTime = time.Now().Add(-15 * time.Second)
	nm, cmd := m.Update(tickMsg(time.Now()))
	if isQuit(t, cmd) {
		t.Fatal("tick at 50% must not quit")
	}
	if cmd == nil {
		t.Fatal("tick must schedule the next tick")
	}
	got := asModel(t, nm).percent
	if got < 0.49 || got > 0.51 {
		t.Errorf("percent = %v, want ~0.5", got)
	}
}

func TestTickClampsAndQuitsWhenDone(t *testing.T) {
	for _, age := range []time.Duration{30 * time.Second, 31 * time.Second, 10 * time.Minute, 200 * 365 * 24 * time.Hour} {
		m := initialModel()
		m.startTime = time.Now().Add(-age)
		nm, cmd := m.Update(tickMsg(time.Now()))
		r := asModel(t, nm)
		if r.percent != 1.0 {
			t.Errorf("age %v: percent = %v, want exactly 1.0", age, r.percent)
		}
		if !isQuit(t, cmd) {
			t.Errorf("age %v: did not quit when complete", age)
		}
		v := r.View()
		if !strings.Contains(v, "100.0%") || !strings.Contains(v, "COMPLETE") {
			t.Errorf("age %v: completed view missing 100%%/COMPLETE:\n%s", age, v)
		}
	}
}

func TestTickWithFutureStartTimeDoesNotGoNegative(t *testing.T) {
	// Clock skew / monotonic weirdness: startTime in the future.
	m := initialModel()
	m.startTime = time.Now().Add(10 * time.Second)
	nm, cmd := m.Update(tickMsg(time.Now()))
	r := asModel(t, nm)
	if isQuit(t, cmd) {
		t.Fatal("must not quit")
	}
	if r.percent > 0 {
		t.Errorf("percent = %v, want <= 0", r.percent)
	}
	// View must still render and its bar must not blow up on negative input.
	if v := r.View(); v == "" {
		t.Error("empty view")
	}
}

func TestWindowSizeMsg(t *testing.T) {
	cases := []struct {
		w, want int
	}{
		{0, 0 - padding*2 - 4},
		{1, 1 - padding*2 - 4},
		{8, 0},
		{9, 1},
		{20, 12},
		{80, 72},
		{88, 80},
		{89, maxWidth},
		{100000, maxWidth},
		{-5, -5 - padding*2 - 4},
	}
	for _, c := range cases {
		m := initialModel()
		nm, cmd := m.Update(tea.WindowSizeMsg{Width: c.w, Height: 24})
		if cmd != nil {
			t.Errorf("w=%d: WindowSizeMsg returned a cmd", c.w)
		}
		r := asModel(t, nm)
		if r.progress.Width > maxWidth {
			t.Errorf("w=%d: progress width %d exceeds maxWidth", c.w, r.progress.Width)
		}
		if r.progress.Width != c.want {
			t.Errorf("w=%d: progress width = %d, want %d", c.w, r.progress.Width, c.want)
		}
		// Must render without panicking for any size.
		if v := r.View(); v == "" {
			t.Errorf("w=%d: empty view", c.w)
		}
	}
}

func TestViewNeverPanicsOrEmptyAcrossWidths(t *testing.T) {
	for w := 10; w <= 200; w++ {
		for _, pct := range []float64{0, 0.001, 0.5, 0.999, 1.0} {
			m := initialModel()
			nm, _ := m.Update(tea.WindowSizeMsg{Width: w, Height: 24})
			r := asModel(t, nm)
			r.percent = pct
			func() {
				defer func() {
					if rec := recover(); rec != nil {
						t.Fatalf("View panicked at width %d pct %v: %v", w, pct, rec)
					}
				}()
				v := r.View()
				if strings.TrimSpace(v) == "" {
					t.Fatalf("View empty at width %d pct %v", w, pct)
				}
			}()
		}
	}
}

func TestViewContainsTitleAndPercentage(t *testing.T) {
	m := initialModel()
	m.percent = 0.42
	v := m.View()
	for _, want := range []string{"PROGRESS TIMER", "Progress: 42.0%", "42%", "Time:", "/ 30s", "Remaining:", "r: reset", "q/esc: quit"} {
		if !strings.Contains(v, want) {
			t.Errorf("View missing %q:\n%s", want, v)
		}
	}
	if strings.Contains(v, "COMPLETE") {
		t.Error("View shows COMPLETE at 42%")
	}
}

func TestViewPercentFormatting(t *testing.T) {
	for _, pct := range []float64{0, 0.1, 0.5, 1.0} {
		m := initialModel()
		m.percent = pct
		want := fmt.Sprintf("Progress: %.1f%%", pct*100)
		if v := m.View(); !strings.Contains(v, want) {
			t.Errorf("pct %v: View missing %q", pct, want)
		}
	}
}

func TestFrameMsgIsHarmless(t *testing.T) {
	m := initialModel()
	nm, cmd := m.Update(progress.FrameMsg{})
	if cmd != nil {
		t.Error("stray FrameMsg produced a cmd")
	}
	if asModel(t, nm).percent != 0 {
		t.Error("stray FrameMsg mutated percent")
	}
}

func TestUnknownMsgIsIgnored(t *testing.T) {
	m := initialModel()
	nm, cmd := m.Update(struct{}{})
	if cmd != nil {
		t.Error("unknown msg produced a cmd")
	}
	if asModel(t, nm).percent != m.percent {
		t.Error("unknown msg mutated model")
	}
}

// BUG: the bar is supposed to be Gold (ANSI 178) on Navy (ANSI 24), but the
// model is built with WithDefaultGradient(), which puts progress.Model into
// ramp mode. In ramp mode FullColor is ignored entirely and the filled cells
// render the library's default purple->pink (#5A56E0 -> #EE6FF8) gradient.
// We force a TrueColor profile so escape sequences are emitted headlessly.
func TestProgressBarUsesGoldNavyTheme(t *testing.T) {
	m := initialModel()
	p := m.progress
	progress.WithColorProfile(termenv.TrueColor)(&p)

	out := p.ViewAs(0.5)

	const gold = "38;5;178" // ANSI-256 fg 178
	const navy = "38;5;24"  // ANSI-256 fg 24
	if !strings.Contains(out, gold) {
		t.Errorf("filled portion does not use gold (fg %s); got:\n%q", gold, out)
	}
	if !strings.Contains(out, navy) {
		t.Errorf("empty portion does not use navy (fg %s); got:\n%q", navy, out)
	}
	if strings.Contains(out, "38;2;") {
		t.Errorf("bar contains 24-bit gradient colors (default purple/pink ramp) instead of the theme:\n%q", out)
	}
	if p.FullColor != string(goldColor) || p.EmptyColor != string(navyColor) {
		t.Errorf("FullColor/EmptyColor = %q/%q, want %q/%q", p.FullColor, p.EmptyColor, goldColor, navyColor)
	}
}
