package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	gold  = lipgloss.Color("178")
	navy  = lipgloss.Color("24")
	gray  = lipgloss.Color("240")
	green = lipgloss.Color("76")
	amber = lipgloss.Color("220")
	red   = lipgloss.Color("196")

	titleStyle = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true).Padding(0, 1)
	labelStyle = lipgloss.NewStyle().Foreground(gold).Bold(true)
	valueStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	dimStyle   = lipgloss.NewStyle().Foreground(gray)
	paneStyle  = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(navy).Padding(0, 1)
)

const historyLen = 60

type cpuTimes struct{ idle, total uint64 }

type sample struct {
	cpuPct   float64
	memUsed  uint64
	memTotal uint64
	load1    float64
	load5    float64
	load15   float64
	uptime   time.Duration
	procs    []proc
}

type proc struct {
	pid  int
	name string
	rss  uint64 // bytes
	cpu  float64
}

type tickMsg time.Time

type model struct {
	width, height int
	cpuHist       []float64
	memHist       []float64
	last          cpuTimes
	lastProcTime  map[int]uint64
	lastSample    time.Time
	cur           sample
	paused        bool
	interval      time.Duration
	err           string
}

func initialModel() model {
	m := model{
		interval:     time.Second,
		lastProcTime: map[int]uint64{},
	}
	m.last, _ = readCPU()
	m.lastSample = time.Now()
	return m
}

func tick(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) Init() tea.Cmd { return tick(200 * time.Millisecond) }

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		case "p", " ":
			m.paused = !m.paused
		case "+", "=":
			if m.interval > 250*time.Millisecond {
				m.interval /= 2
			}
		case "-", "_":
			if m.interval < 8*time.Second {
				m.interval *= 2
			}
		}
	case tickMsg:
		if !m.paused {
			m.collect()
		}
		return m, tick(m.interval)
	}
	return m, nil
}

func (m *model) collect() {
	m.err = ""
	now := time.Now()
	elapsed := now.Sub(m.lastSample).Seconds()
	if elapsed <= 0 {
		elapsed = 1
	}

	cur, err := readCPU()
	if err != nil {
		m.err = err.Error()
	} else if cur.total < m.last.total || cur.idle < m.last.idle {
		// counters went backwards (wrap/reset): resync baseline, keep last pct
		m.last = cur
	} else {
		dTotal := float64(cur.total - m.last.total)
		dIdle := float64(cur.idle - m.last.idle)
		if dTotal > 0 {
			m.cur.cpuPct = clamp(100*(1-dIdle/dTotal), 0, 100)
		}
		m.last = cur
	}

	if used, total, err := readMem(); err == nil {
		m.cur.memUsed, m.cur.memTotal = used, total
	}
	if l1, l5, l15, err := readLoad(); err == nil {
		m.cur.load1, m.cur.load5, m.cur.load15 = l1, l5, l15
	}
	if up, err := readUptime(); err == nil {
		m.cur.uptime = up
	}
	m.cur.procs = m.readProcs(elapsed)

	m.cpuHist = pushHist(m.cpuHist, m.cur.cpuPct)
	memPct := 0.0
	if m.cur.memTotal > 0 {
		memPct = 100 * float64(m.cur.memUsed) / float64(m.cur.memTotal)
	}
	m.memHist = pushHist(m.memHist, memPct)
	m.lastSample = now
}

func pushHist(h []float64, v float64) []float64 {
	h = append(h, v)
	if len(h) > historyLen {
		h = h[len(h)-historyLen:]
	}
	return h
}

func readCPU() (cpuTimes, error) {
	f, err := os.Open("/proc/stat")
	if err != nil {
		return cpuTimes{}, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 5 || fields[0] != "cpu" {
			continue
		}
		var t cpuTimes
		for i, fld := range fields[1:] {
			v, _ := strconv.ParseUint(fld, 10, 64)
			t.total += v
			if i == 3 || i == 4 { // idle, iowait
				t.idle += v
			}
		}
		return t, nil
	}
	return cpuTimes{}, fmt.Errorf("no cpu line in /proc/stat")
}

func readMem() (used, total uint64, err error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	var avail uint64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		v, _ := strconv.ParseUint(fields[1], 10, 64)
		switch fields[0] {
		case "MemTotal:":
			total = v * 1024
		case "MemAvailable:":
			avail = v * 1024
		}
	}
	if total == 0 {
		return 0, 0, fmt.Errorf("MemTotal missing")
	}
	return total - avail, total, nil
}

func readLoad() (float64, float64, float64, error) {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, 0, 0, err
	}
	f := strings.Fields(string(b))
	if len(f) < 3 {
		return 0, 0, 0, fmt.Errorf("bad loadavg")
	}
	l1, _ := strconv.ParseFloat(f[0], 64)
	l5, _ := strconv.ParseFloat(f[1], 64)
	l15, _ := strconv.ParseFloat(f[2], 64)
	return l1, l5, l15, nil
}

func readUptime() (time.Duration, error) {
	b, err := os.ReadFile("/proc/uptime")
	if err != nil {
		return 0, err
	}
	f := strings.Fields(string(b))
	if len(f) < 1 {
		return 0, fmt.Errorf("bad uptime")
	}
	s, _ := strconv.ParseFloat(f[0], 64)
	return time.Duration(s) * time.Second, nil
}

var clkTck = float64(100)

func (m *model) readProcs(elapsed float64) []proc {
	dirents, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	pageSize := uint64(os.Getpagesize())
	out := make([]proc, 0, 64)
	seen := map[int]uint64{}
	for _, d := range dirents {
		pid, err := strconv.Atoi(d.Name())
		if err != nil {
			continue
		}
		statB, err := os.ReadFile("/proc/" + d.Name() + "/stat")
		if err != nil {
			continue
		}
		s := string(statB)
		lp, rp := strings.IndexByte(s, '('), strings.LastIndexByte(s, ')')
		if lp < 0 || rp < 0 || rp < lp {
			continue
		}
		name := s[lp+1 : rp]
		rest := strings.Fields(s[rp+2:])
		if len(rest) < 22 {
			continue
		}
		utime, _ := strconv.ParseUint(rest[11], 10, 64)
		stime, _ := strconv.ParseUint(rest[12], 10, 64)
		rssPages, _ := strconv.ParseUint(rest[21], 10, 64)
		total := utime + stime
		seen[pid] = total
		cpu := 0.0
		if prev, ok := m.lastProcTime[pid]; ok && total >= prev {
			cpu = 100 * float64(total-prev) / clkTck / elapsed
		}
		out = append(out, proc{pid: pid, name: name, rss: rssPages * pageSize, cpu: cpu})
	}
	m.lastProcTime = seen
	sort.Slice(out, func(i, j int) bool {
		if out[i].cpu != out[j].cpu {
			return out[i].cpu > out[j].cpu
		}
		return out[i].rss > out[j].rss
	})
	return out
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func pctColor(p float64) lipgloss.Color {
	switch {
	case p >= 85:
		return red
	case p >= 60:
		return amber
	default:
		return green
	}
}

func gauge(pct float64, width int) string {
	if width < 4 {
		width = 4
	}
	filled := int(pct/100*float64(width) + 0.5)
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}
	bar := lipgloss.NewStyle().Foreground(pctColor(pct)).Render(strings.Repeat("█", filled)) +
		lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("░", width-filled))
	return bar + " " + labelStyle.Render(fmt.Sprintf("%5.1f%%", pct))
}

var sparkRunes = []rune("▁▂▃▄▅▆▇█")

func sparkline(h []float64, width int) string {
	if len(h) == 0 {
		return dimStyle.Render(strings.Repeat("·", width))
	}
	if len(h) > width {
		h = h[len(h)-width:]
	}
	var b strings.Builder
	for i := 0; i < width-len(h); i++ {
		b.WriteRune(' ')
	}
	for _, v := range h {
		idx := int(v / 100 * float64(len(sparkRunes)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sparkRunes) {
			idx = len(sparkRunes) - 1
		}
		b.WriteString(lipgloss.NewStyle().Foreground(pctColor(v)).Render(string(sparkRunes[idx])))
	}
	return b.String()
}

func humanBytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := uint64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

func (m model) View() string {
	if m.width == 0 {
		return "loading..."
	}
	inner := max(30, m.width-6)
	gaugeW := max(10, inner-10)

	memPct := 0.0
	if m.cur.memTotal > 0 {
		memPct = 100 * float64(m.cur.memUsed) / float64(m.cur.memTotal)
	}

	host, _ := os.Hostname()
	header := titleStyle.Render("📊 SYSTEM MONITOR") + " " +
		dimStyle.Render(fmt.Sprintf("%s • %s/%s • %d cores • up %s",
			host, runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), m.cur.uptime.Round(time.Second)))
	if m.paused {
		header += " " + lipgloss.NewStyle().Foreground(amber).Bold(true).Render("[PAUSED]")
	}
	if m.err != "" {
		header += " " + lipgloss.NewStyle().Foreground(red).Render(m.err)
	}

	cpuPane := paneStyle.Width(inner).Render(lipgloss.JoinVertical(lipgloss.Left,
		labelStyle.Render("CPU"),
		gauge(m.cur.cpuPct, gaugeW),
		sparkline(m.cpuHist, inner-2),
	))
	memPane := paneStyle.Width(inner).Render(lipgloss.JoinVertical(lipgloss.Left,
		labelStyle.Render("MEMORY")+"  "+valueStyle.Render(humanBytes(m.cur.memUsed)+" / "+humanBytes(m.cur.memTotal)),
		gauge(memPct, gaugeW),
		sparkline(m.memHist, inner-2),
	))
	loadPane := paneStyle.Width(inner).Render(
		labelStyle.Render("LOAD") + "  " +
			valueStyle.Render(fmt.Sprintf("1m %.2f   5m %.2f   15m %.2f", m.cur.load1, m.cur.load5, m.cur.load15)))

	// fixed chrome: header 1 + cpu 5 + mem 5 + load 3 + proc borders/header 3 + help 1 = 18
	procRows := max(3, m.height-18)
	// pane content width is inner-2: 7+1 + nameW + 1+8 + 1+7 = inner-2  =>  nameW = inner-27
	nameW := inner - 27
	var pb strings.Builder
	pb.WriteString(labelStyle.Render(fmt.Sprintf("%-7s %-*s %8s %7s", "PID", nameW, "NAME", "RSS", "CPU%")) + "\n")
	for i, p := range m.cur.procs {
		if i >= procRows {
			break
		}
		name := p.name
		if r := []rune(name); len(r) > nameW {
			name = string(r[:nameW-1]) + "…"
		}
		pb.WriteString(valueStyle.Render(fmt.Sprintf("%-7d %-*s %8s ", p.pid, nameW, name, humanBytes(p.rss))))
		pb.WriteString(lipgloss.NewStyle().Foreground(pctColor(p.cpu)).Render(fmt.Sprintf("%6.1f%%", p.cpu)) + "\n")
	}
	procPane := paneStyle.Width(inner).Render(strings.TrimRight(pb.String(), "\n"))

	help := dimStyle.Render(fmt.Sprintf("p/space pause • +/- interval (%s) • q quit", m.interval))

	return lipgloss.JoinVertical(lipgloss.Left, header, cpuPane, memPane, loadPane, procPane, help)
}

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
