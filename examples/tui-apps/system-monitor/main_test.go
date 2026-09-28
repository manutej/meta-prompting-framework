package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ---------- helpers ----------

var ansiRE = regexp.MustCompile("\x1b\\[[0-9;]*[A-Za-z]")

func stripANSI(s string) string { return ansiRE.ReplaceAllString(s, "") }

func key(s string) tea.KeyMsg {
	switch s {
	case " ":
		return tea.KeyMsg{Type: tea.KeySpace}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
}

func sized(w, h int) model {
	m := model{interval: time.Second, lastProcTime: map[int]uint64{}}
	nm, _ := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return nm.(model)
}

func mustNotPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s panicked: %v", what, r)
		}
	}()
	f()
}

func fakeProcs(n int, name string) []proc {
	out := make([]proc, n)
	for i := range out {
		out[i] = proc{pid: 1000 + i, name: name, rss: uint64(i) << 20, cpu: float64(i)}
	}
	return out
}

// ---------- View ----------

func TestViewBeforeWindowSize(t *testing.T) {
	m := initialModel()
	if got := m.View(); got != "loading..." {
		t.Fatalf("View() before WindowSizeMsg = %q, want %q", got, "loading...")
	}
}

func TestViewSizesNoPanic(t *testing.T) {
	sizes := [][2]int{{20, 5}, {80, 24}, {300, 100}, {1, 1}, {40, 0}}
	for _, sz := range sizes {
		m := sized(sz[0], sz[1])
		m.cur.procs = fakeProcs(50, "a-fairly-long-process-name-xyz")
		m.cur.memUsed, m.cur.memTotal = 1<<30, 4<<30
		m.cpuHist = make([]float64, 60)
		var out string
		mustNotPanic(t, fmt.Sprintf("View() at %dx%d", sz[0], sz[1]), func() { out = m.View() })
		if strings.TrimSpace(out) == "" {
			t.Fatalf("View() at %dx%d is empty", sz[0], sz[1])
		}
		if !utf8.ValidString(out) {
			t.Fatalf("View() at %dx%d produced invalid UTF-8", sz[0], sz[1])
		}
	}
}

// Coordinator-reported bug: at 100x30 each process row was 1 cell wider than
// the pane's content width, so lipgloss force-wrapped the CPU% column onto its
// own line. Also: the total view was 2 lines taller than the terminal, and the
// renderer drops the TOP lines (header) in that case.
func TestViewProcessRowsFitPane(t *testing.T) {
	const w, h = 100, 30
	m := sized(w, h)
	m.cur.procs = fakeProcs(40, "rcu_preempt")
	m.paused = true
	out := m.View()
	lines := strings.Split(stripANSI(out), "\n")

	inner := max(30, w-6)
	paneW := inner + 2 // + rounded border
	orphan := regexp.MustCompile(`^│\s+\d+\.\d%\s*│$`)
	for i, ln := range lines {
		if lw := lipgloss.Width(ln); lw > w {
			t.Errorf("line %d is %d cells wide, terminal is %d: %q", i, lw, w, ln)
		}
		if strings.HasPrefix(ln, "│") && lipgloss.Width(ln) != paneW {
			t.Errorf("pane line %d has width %d, want %d: %q", i, lipgloss.Width(ln), paneW, ln)
		}
		if orphan.MatchString(ln) {
			t.Errorf("line %d is a wrapped CPU%% orphan: %q", i, ln)
		}
	}
	// Every process row must carry its own CPU% on the same line.
	rows := 0
	for _, ln := range lines {
		if regexp.MustCompile(`^│\s*\d{4} `).MatchString(ln) {
			rows++
			if !strings.Contains(ln, ".0%") && !strings.Contains(ln, "%") {
				t.Errorf("process row missing CPU%% on same line: %q", ln)
			}
		}
	}
	if rows == 0 {
		t.Fatalf("no process rows rendered:\n%s", stripANSI(out))
	}
	// Whole view must fit the terminal height, else the renderer drops the header.
	if len(lines) > h {
		t.Errorf("View() is %d lines tall at height %d; header would be cut off", len(lines), h)
	}
	if !strings.Contains(lines[0], "SYSTEM MONITOR") || !strings.Contains(lines[0], "[PAUSED]") {
		t.Errorf("first line is not the header: %q", lines[0])
	}
}

func TestViewLongNameTruncatedAtNarrowWidth(t *testing.T) {
	m := sized(20, 5)
	m.cur.procs = fakeProcs(3, strings.Repeat("x", 200))
	var out string
	mustNotPanic(t, "View() with long name at 20x5", func() { out = m.View() })
	if !strings.Contains(out, "…") {
		t.Errorf("long name was not truncated with ellipsis")
	}
	if strings.Contains(stripANSI(out), strings.Repeat("x", 30)) {
		t.Errorf("long name leaked untruncated into the view")
	}
}

// Kernel comm may contain multibyte runes (TASK_COMM_LEN is bytes). Byte slicing
// name[:n] can split a rune and emit invalid UTF-8.
func TestViewMultibyteNameTruncationIsRuneSafe(t *testing.T) {
	for w := 20; w <= 45; w++ {
		m := sized(w, 24)
		m.cur.procs = fakeProcs(2, "日本語日本語日本")
		var out string
		mustNotPanic(t, fmt.Sprintf("View() multibyte at width %d", w), func() { out = m.View() })
		if !utf8.ValidString(out) {
			t.Fatalf("width %d: View() contains invalid UTF-8 (rune split by byte slicing)", w)
		}
		if strings.Contains(out, "�") {
			t.Fatalf("width %d: View() contains replacement char", w)
		}
	}
}

// ---------- Update / keys ----------

func TestPauseTogglesAndTickWhilePausedDoesNotAppendHistory(t *testing.T) {
	m := sized(80, 24)
	nm, _ := m.Update(key("p"))
	m = nm.(model)
	if !m.paused {
		t.Fatal("p did not pause")
	}
	before := len(m.cpuHist)
	nm, cmd := m.Update(tickMsg(time.Now()))
	m = nm.(model)
	if len(m.cpuHist) != before || len(m.memHist) != before {
		t.Fatalf("tick while paused appended history: cpu %d mem %d, want %d", len(m.cpuHist), len(m.memHist), before)
	}
	if cmd == nil {
		t.Fatal("tick while paused must still reschedule the next tick")
	}
	nm, _ = m.Update(key(" "))
	m = nm.(model)
	if m.paused {
		t.Fatal("space did not unpause")
	}
	nm, _ = m.Update(tickMsg(time.Now()))
	m = nm.(model)
	if len(m.cpuHist) != before+1 || len(m.memHist) != before+1 {
		t.Fatalf("tick while running did not append exactly one sample: cpu %d mem %d", len(m.cpuHist), len(m.memHist))
	}
}

func TestIntervalClamp(t *testing.T) {
	m := sized(80, 24)
	for i := 0; i < 20; i++ {
		nm, _ := m.Update(key("+"))
		m = nm.(model)
		if m.interval < 250*time.Millisecond {
			t.Fatalf("interval fell below 250ms: %s", m.interval)
		}
	}
	if m.interval != 250*time.Millisecond {
		t.Fatalf("after many '+', interval = %s, want 250ms", m.interval)
	}
	for i := 0; i < 20; i++ {
		nm, _ := m.Update(key("-"))
		m = nm.(model)
		if m.interval > 8*time.Second {
			t.Fatalf("interval exceeded 8s: %s", m.interval)
		}
	}
	if m.interval != 8*time.Second {
		t.Fatalf("after many '-', interval = %s, want 8s", m.interval)
	}
	// aliases
	nm, _ := m.Update(key("="))
	if nm.(model).interval != 4*time.Second {
		t.Fatalf("'=' did not halve interval: %s", nm.(model).interval)
	}
	nm, _ = nm.(model).Update(key("_"))
	if nm.(model).interval != 8*time.Second {
		t.Fatalf("'_' did not double interval: %s", nm.(model).interval)
	}
}

func TestQuitKeys(t *testing.T) {
	for _, k := range []string{"q", "ctrl+c", "esc"} {
		m := sized(80, 24)
		_, cmd := m.Update(key(k))
		if cmd == nil {
			t.Fatalf("key %q returned nil cmd, want tea.Quit", k)
		}
		if _, ok := cmd().(tea.QuitMsg); !ok {
			t.Fatalf("key %q did not return tea.Quit", k)
		}
	}
	// a non-quit key must not quit
	m := sized(80, 24)
	if _, cmd := m.Update(key("x")); cmd != nil {
		t.Fatal("unbound key returned a cmd")
	}
}

func TestInitReturnsTick(t *testing.T) {
	if initialModel().Init() == nil {
		t.Fatal("Init() returned nil cmd")
	}
}

// ---------- pure helpers ----------

func TestPushHistCaps(t *testing.T) {
	var h []float64
	for i := 0; i < 500; i++ {
		h = pushHist(h, float64(i))
		if len(h) > historyLen {
			t.Fatalf("history exceeded cap at push %d: len %d", i, len(h))
		}
	}
	if len(h) != historyLen || h[0] != 440 || h[historyLen-1] != 499 {
		t.Fatalf("history window wrong: len %d first %v last %v", len(h), h[0], h[len(h)-1])
	}
}

func TestGaugeNoPanicAndFilledNeverExceedsWidth(t *testing.T) {
	pcts := []float64{0, 50, 100, 150, -5, -50, math.NaN(), math.Inf(1), math.Inf(-1)}
	widths := []int{0, 1, 2, 3, 4, 10, 33, 100}
	for _, p := range pcts {
		for _, w := range widths {
			var out string
			mustNotPanic(t, fmt.Sprintf("gauge(%v,%d)", p, w), func() { out = gauge(p, w) })
			eff := max(4, w)
			filled := strings.Count(out, "█")
			empty := strings.Count(out, "░")
			if filled > eff {
				t.Errorf("gauge(%v,%d): filled %d > width %d", p, w, filled, eff)
			}
			if filled+empty != eff {
				t.Errorf("gauge(%v,%d): filled+empty = %d, want %d", p, w, filled+empty, eff)
			}
		}
	}
}

func TestSparkline(t *testing.T) {
	cases := []struct {
		n, width int
	}{{0, 10}, {1, 10}, {60, 60}, {200, 60}, {60, 10}, {5, 60}, {3, 0}, {0, 0}}
	for _, c := range cases {
		h := make([]float64, c.n)
		for i := range h {
			h[i] = float64(i % 101)
		}
		var out string
		mustNotPanic(t, fmt.Sprintf("sparkline(n=%d,w=%d)", c.n, c.width), func() { out = sparkline(h, c.width) })
		if got := lipgloss.Width(stripANSI(out)); got != c.width {
			t.Errorf("sparkline(n=%d,w=%d) rendered width %d, want %d", c.n, c.width, got, c.width)
		}
	}
	// out-of-range values must clamp to the rune table
	mustNotPanic(t, "sparkline out-of-range", func() { sparkline([]float64{-100, 1e9, math.NaN()}, 3) })
}

func TestHumanBytes(t *testing.T) {
	cases := map[uint64]string{
		0:              "0 B",
		1023:           "1023 B",
		1024:           "1.0 KB",
		1536:           "1.5 KB",
		1 << 20:        "1.0 MB",
		1 << 40:        "1.0 TB",
		math.MaxUint64: "16.0 EB",
	}
	for n, want := range cases {
		if got := humanBytes(n); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestClamp(t *testing.T) {
	if clamp(-1, 0, 100) != 0 || clamp(101, 0, 100) != 100 || clamp(42, 0, 100) != 42 {
		t.Fatal("clamp broken")
	}
}

// ---------- /proc readers ----------

func requireLinux(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("requires /proc")
	}
	if _, err := os.Stat("/proc/stat"); err != nil {
		t.Skip("no /proc/stat")
	}
}

func TestReadCPUReal(t *testing.T) {
	requireLinux(t)
	c, err := readCPU()
	if err != nil {
		t.Fatal(err)
	}
	if c.total == 0 || c.idle > c.total {
		t.Fatalf("implausible cpu times: %+v", c)
	}
}

func TestReadMemReal(t *testing.T) {
	requireLinux(t)
	used, total, err := readMem()
	if err != nil {
		t.Fatal(err)
	}
	if total == 0 || used > total {
		t.Fatalf("implausible mem: used %d total %d", used, total)
	}
}

func TestReadLoadReal(t *testing.T) {
	requireLinux(t)
	l1, l5, l15, err := readLoad()
	if err != nil {
		t.Fatal(err)
	}
	if l1 < 0 || l5 < 0 || l15 < 0 {
		t.Fatalf("negative load: %v %v %v", l1, l5, l15)
	}
}

func TestReadUptimeReal(t *testing.T) {
	requireLinux(t)
	up, err := readUptime()
	if err != nil {
		t.Fatal(err)
	}
	if up <= 0 {
		t.Fatalf("uptime %s not positive", up)
	}
}

func TestReadProcsTwiceNonNegative(t *testing.T) {
	requireLinux(t)
	m := initialModel()
	first := m.readProcs(1)
	if len(first) == 0 {
		t.Fatal("no processes read")
	}
	if len(m.lastProcTime) == 0 {
		t.Fatal("lastProcTime not populated")
	}
	// burn a little CPU so at least our own pid has a delta
	deadline := time.Now().Add(30 * time.Millisecond)
	x := 0
	for time.Now().Before(deadline) {
		x++
	}
	_ = x
	second := m.readProcs(0.05)
	self := os.Getpid()
	found := false
	for _, p := range second {
		if p.cpu < 0 || math.IsNaN(p.cpu) || math.IsInf(p.cpu, 0) {
			t.Fatalf("pid %d has bad cpu %v", p.pid, p.cpu)
		}
		if p.pid == self {
			found = true
		}
	}
	if !found {
		t.Fatal("own pid not found in second readProcs")
	}
	for i := 1; i < len(second); i++ {
		a, b := second[i-1], second[i]
		if a.cpu < b.cpu || (a.cpu == b.cpu && a.rss < b.rss) {
			t.Fatalf("procs not sorted by cpu desc then rss desc at %d: %+v > %+v", i, a, b)
		}
	}
}

// exec -a does NOT change the kernel comm (it's taken from the executable's
// basename), so we run a shell script whose filename is the weird name instead.
func TestReadProcsWeirdName(t *testing.T) {
	requireLinux(t)
	sh, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("no sh")
	}
	dir := t.TempDir()
	const weird = "weird (name) x" // 14 bytes, fits TASK_COMM_LEN-1
	script := filepath.Join(dir, weird)
	if err := os.WriteFile(script, []byte("#!"+sh+"\nsleep 5\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(script)
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start weird-named process: %v", err)
	}
	defer func() { _ = cmd.Process.Kill(); _, _ = cmd.Process.Wait() }()

	// verify the kernel actually applied the comm we expect
	comm, err := os.ReadFile(fmt.Sprintf("/proc/%d/comm", cmd.Process.Pid))
	if err != nil {
		t.Skip("cannot read comm")
	}
	if strings.TrimSpace(string(comm)) != weird {
		t.Skipf("kernel comm is %q, not %q; cannot exercise parser", strings.TrimSpace(string(comm)), weird)
	}

	m := initialModel()
	var got *proc
	for _, p := range m.readProcs(1) {
		if p.pid == cmd.Process.Pid {
			pp := p
			got = &pp
			break
		}
	}
	if got == nil {
		t.Fatal("weird-named process not found in readProcs")
	}
	if got.name != weird {
		t.Fatalf("parsed name %q, want %q", got.name, weird)
	}
	if got.rss == 0 {
		t.Errorf("rss parsed as 0 for a live process (field misalignment after ')' in name?)")
	}
}

// ---------- collect() counter wrap ----------

func TestCollectCounterWrapDoesNotProduceGarbage(t *testing.T) {
	requireLinux(t)
	m := initialModel()
	m.cur.cpuPct = 12.5 // sentinel from a "previous" good sample
	// Simulate a counter reset/wrap: previous total is far larger than what
	// /proc/stat reports now. uint64 subtraction would underflow to ~2^63.
	m.last = cpuTimes{idle: 0, total: 1 << 63}
	m.lastSample = time.Now().Add(-time.Second)
	mustNotPanic(t, "collect() after wrap", m.collect)
	if m.cur.cpuPct != 12.5 {
		t.Fatalf("counter wrap produced garbage cpuPct %.2f (want previous value 12.5 kept)", m.cur.cpuPct)
	}
	if m.last.total == 1<<63 {
		t.Fatal("m.last was not resynced to current counters after wrap")
	}
	// second collect must work normally from the resynced baseline
	m.lastSample = time.Now().Add(-time.Second)
	m.collect()
	if m.cur.cpuPct < 0 || m.cur.cpuPct > 100 || math.IsNaN(m.cur.cpuPct) {
		t.Fatalf("cpuPct out of range after resync: %v", m.cur.cpuPct)
	}
	if len(m.cpuHist) != 2 {
		t.Fatalf("history len %d, want 2", len(m.cpuHist))
	}
}

func TestCollectPopulatesSample(t *testing.T) {
	requireLinux(t)
	m := initialModel()
	m.lastSample = time.Now().Add(-100 * time.Millisecond)
	m.collect()
	if m.err != "" {
		t.Fatalf("collect err: %s", m.err)
	}
	if m.cur.memTotal == 0 || m.cur.uptime == 0 || len(m.cur.procs) == 0 {
		t.Fatalf("sample not populated: %+v", m.cur)
	}
	if len(m.cpuHist) != 1 || len(m.memHist) != 1 {
		t.Fatalf("history not pushed: cpu %d mem %d", len(m.cpuHist), len(m.memHist))
	}
	if m.memHist[0] < 0 || m.memHist[0] > 100 {
		t.Fatalf("memPct out of range: %v", m.memHist[0])
	}
}
