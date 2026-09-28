package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
	"github.com/muesli/reflow/truncate"

	"alembic/harness"
	"alembic/jev"
)

var (
	gold     = lipgloss.Color("#d4a017")
	goldHi   = lipgloss.Color("#d29e3d")
	navy     = lipgloss.Color("#0e1830")
	darkNavy = lipgloss.Color("#1a1407")
	fgText   = lipgloss.Color("#eceae3")
	muted    = lipgloss.Color("#9ca3af")
	dim      = lipgloss.Color("#2a2a2a")
	green    = lipgloss.Color("#22c55e")
	amber    = lipgloss.Color("#fbbf24")
	red      = lipgloss.Color("#ef4444")
	cyan     = lipgloss.Color("#60a5fa")
	white    = lipgloss.Color("#eceae3")

	badgeStyle    = lipgloss.NewStyle().Foreground(darkNavy).Background(gold).Bold(true).Padding(0, 1)
	headerStyle   = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true)
	headerDim     = lipgloss.NewStyle().Foreground(fgText).Background(navy)
	tabActive     = lipgloss.NewStyle().Foreground(darkNavy).Background(gold).Bold(true).Padding(0, 2)
	tabInactive   = lipgloss.NewStyle().Foreground(muted).Padding(0, 2)
	titleStyle    = lipgloss.NewStyle().Foreground(gold).Bold(true)
	textStyle     = lipgloss.NewStyle().Foreground(fgText)
	mutedStyle    = lipgloss.NewStyle().Foreground(muted)
	dimStyle      = lipgloss.NewStyle().Foreground(dim)
	selStyle      = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true)
	successStyle  = lipgloss.NewStyle().Foreground(green).Bold(true)
	warnStyle     = lipgloss.NewStyle().Foreground(amber).Bold(true)
	errorStyle    = lipgloss.NewStyle().Foreground(red).Bold(true)
	cyanStyle     = lipgloss.NewStyle().Foreground(cyan)
	keyStyle      = lipgloss.NewStyle().Foreground(gold).Bold(true)
	statusStyle   = lipgloss.NewStyle().Foreground(muted).Background(lipgloss.Color("#1a1a1a"))
	toastOK       = lipgloss.NewStyle().Foreground(darkNavy).Background(green).Bold(true).Padding(0, 1)
	toastWarn     = lipgloss.NewStyle().Foreground(darkNavy).Background(amber).Bold(true).Padding(0, 1)
	toastErr      = lipgloss.NewStyle().Foreground(white).Background(red).Bold(true).Padding(0, 1)
	matchStyle    = lipgloss.NewStyle().Foreground(goldHi).Bold(true).Underline(true)
	overlayBorder = lipgloss.NewStyle().Border(lipgloss.DoubleBorder()).BorderForeground(gold).Padding(0, 1)
	sectionStyle  = lipgloss.NewStyle().Foreground(navy).Bold(true)
)

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]`)

func stripANSI(s string) string { return ansiRe.ReplaceAllString(s, "") }

// oneLine flattens control characters that would add screen lines or move
// the cursor: feed text (titles, status lines, events) is untrusted.
func oneLine(s string) string {
	if !strings.ContainsAny(s, "\t\r\n") {
		return s
	}
	return strings.NewReplacer("\t", "  ", "\r\n", " ", "\n", " ", "\r", " ").Replace(s)
}

// fit truncates (ANSI-aware) and pads s to exactly w columns.
func fit(s string, w int) string {
	if w <= 0 {
		return ""
	}
	s = oneLine(s)
	if lipgloss.Width(s) > w {
		s = truncate.StringWithTail(s, uint(w), "…")
	}
	if pad := w - lipgloss.Width(s); pad > 0 {
		s += strings.Repeat(" ", pad)
	}
	return s
}

// fitLines returns exactly h lines, each exactly w columns.
func fitLines(lines []string, w, h int) []string {
	out := make([]string, 0, h)
	for i := 0; i < h; i++ {
		if i < len(lines) {
			out = append(out, fit(lines[i], w))
		} else {
			out = append(out, strings.Repeat(" ", w))
		}
	}
	return out
}

// pane draws a titled rounded box of total size w x h.
func pane(title string, lines []string, w, h int, focused bool) string {
	if w < 6 || h < 3 {
		return strings.Join(fitLines(nil, max(w, 0), max(h, 0)), "\n")
	}
	bc := navy
	if focused {
		bc = gold
	}
	border := lipgloss.NewStyle().Foreground(bc)
	title = oneLine(title)
	t := titleStyle.Render(title)
	if !focused {
		t = mutedStyle.Bold(true).Render(title)
	}
	tw := lipgloss.Width(t)
	fill := w - 5 - tw
	if fill < 0 {
		t = fit(t, w-5)
		fill = 0
	}
	top := border.Render("╭─ ") + t + border.Render(" "+strings.Repeat("─", fill)+"╮")
	body := fitLines(lines, w-2, h-2)
	var b strings.Builder
	b.WriteString(top)
	for _, l := range body {
		b.WriteString("\n" + border.Render("│") + l + border.Render("│"))
	}
	b.WriteString("\n" + border.Render("╰"+strings.Repeat("─", w-2)+"╯"))
	return b.String()
}

// overlay composites box centered onto base, dimming the base.
func overlay(base, box string, w, h int) string {
	baseLines := strings.Split(base, "\n")
	boxLines := strings.Split(box, "\n")
	bw := lipgloss.Width(box)
	bh := len(boxLines)
	x := max(0, (w-bw)/2)
	y := max(0, (h-bh)/2)
	out := make([]string, len(baseLines))
	for i, l := range baseLines {
		plain := fit(stripANSI(l), w)
		if i >= y && i < y+bh {
			left := cutCols(plain, 0, x)
			right := cutCols(plain, x+bw, w)
			out[i] = dimStyle.Render(left) + fit(boxLines[i-y], bw) + dimStyle.Render(right)
		} else {
			out[i] = dimStyle.Render(plain)
		}
	}
	return strings.Join(out, "\n")
}

// cutCols returns the substring of plain (no ANSI) occupying columns [from, to).
func cutCols(plain string, from, to int) string {
	var b strings.Builder
	col := 0
	for _, r := range plain {
		rw := runewidth.RuneWidth(r)
		if col >= from && col+rw <= to {
			b.WriteRune(r)
		}
		col += rw
		if col >= to {
			break
		}
	}
	return b.String()
}

func gauge(pct float64, width int, c lipgloss.Color) string {
	if width < 1 {
		return ""
	}
	pct = clamp(pct, 0, 1)
	filled := int(pct*float64(width) + 0.5)
	return lipgloss.NewStyle().Foreground(c).Render(strings.Repeat("█", filled)) +
		lipgloss.NewStyle().Foreground(darkNavy).Render(strings.Repeat("░", width-filled))
}

var sparkRunes = []rune("▁▂▃▄▅▆▇█")

func sparkline(h []float64, width int, lo, hi float64, c lipgloss.Color) string {
	if width < 1 {
		return ""
	}
	if len(h) > width {
		h = h[len(h)-width:]
	}
	var b strings.Builder
	b.WriteString(strings.Repeat(" ", width-len(h)))
	span := hi - lo
	if span <= 0 {
		span = 1
	}
	st := lipgloss.NewStyle().Foreground(c)
	for _, v := range h {
		idx := int(clamp((v-lo)/span, 0, 1) * float64(len(sparkRunes)-1))
		b.WriteString(st.Render(string(sparkRunes[idx])))
	}
	return b.String()
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

// splitRow lays left and right on one line of exactly w columns.
func splitRow(left, right string, w int) string {
	gap := w - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 0 {
		left = fit(left, max(0, w-lipgloss.Width(right)-1))
		gap = 1
	}
	return fit(left+strings.Repeat(" ", gap)+right, w)
}

// boxWidth is the inner width of an overlay box for the current terminal.
func boxWidth(want, termW int) int {
	return max(20, min(want, termW-6))
}

// highlight renders s with the fuzzy-matched rune indexes in gold.
func highlight(s string, idx []int) string {
	if len(idx) == 0 {
		return textStyle.Render(s)
	}
	set := map[int]bool{}
	for _, i := range idx {
		set[i] = true
	}
	var b strings.Builder
	for i, r := range []rune(s) {
		if set[i] {
			b.WriteString(matchStyle.Render(string(r)))
		} else {
			b.WriteString(textStyle.Render(string(r)))
		}
	}
	return b.String()
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func spinnerAt(frame int) string { return spinnerFrames[frame%len(spinnerFrames)] }

func stateColor(s harness.TaskState) lipgloss.Color {
	switch s {
	case harness.StateRunning:
		return gold
	case harness.StateBlocked:
		return amber
	case harness.StateReview:
		return cyan
	case harness.StateDone:
		return green
	case harness.StateFailed:
		return red
	}
	return muted
}

// stateGlyph is the one-cell state marker; running animates with frame.
func stateGlyph(s harness.TaskState, frame int) string {
	var g string
	switch s {
	case harness.StateRunning:
		g = spinnerAt(frame)
	case harness.StateBlocked:
		g = "◼"
	case harness.StateReview:
		g = "◆"
	case harness.StateDone:
		g = "●"
	case harness.StateFailed:
		g = "✖"
	default:
		g = "○"
	}
	return lipgloss.NewStyle().Foreground(stateColor(s)).Render(g)
}

func stateOrder(s harness.TaskState) int {
	switch s {
	case harness.StateBlocked:
		return 0
	case harness.StateFailed:
		return 1
	case harness.StateRunning:
		return 2
	case harness.StateReview:
		return 3
	case harness.StateQueued:
		return 4
	}
	return 5
}

func priorityMark(p int) string {
	switch {
	case p >= 2:
		return errorStyle.Render("!!")
	case p == 1:
		return warnStyle.Render("!")
	}
	return ""
}

func levelGlyph(l harness.EventLevel) string {
	switch l {
	case harness.LevelWarn:
		return warnStyle.Render("▲")
	case harness.LevelError:
		return errorStyle.Render("✖")
	case harness.LevelOK:
		return successStyle.Render("✓")
	}
	return mutedStyle.Render("·")
}

func kindIcon(k harness.ElementKind) string {
	switch k {
	case harness.ElemFile:
		return "▤"
	case harness.ElemPR:
		return "⎇"
	case harness.ElemURL:
		return "⌘"
	case harness.ElemLog:
		return "≡"
	case harness.ElemDir:
		return "▸"
	}
	return "·"
}

func workflowStateStyle(state string) lipgloss.Style {
	switch state {
	case "healthy":
		return successStyle
	case "degraded":
		return warnStyle
	case "down", "failed":
		return errorStyle
	}
	return mutedStyle
}

func agentStateStyle(state string) lipgloss.Style {
	switch state {
	case "busy":
		return lipgloss.NewStyle().Foreground(gold).Bold(true)
	case "idle":
		return successStyle
	case "offline":
		return errorStyle
	}
	return mutedStyle
}

func decisionStyle(d jev.Decision) lipgloss.Style {
	switch d {
	case jev.DecisionAuto:
		return successStyle
	case jev.DecisionRefuse:
		return errorStyle
	}
	return warnStyle
}

func confColor(c float64) lipgloss.Color {
	switch {
	case c >= 0.8:
		return green
	case c >= 0.6:
		return amber
	}
	return red
}

func qtypeBadge(t jev.QuestionType) string {
	switch t {
	case jev.Noul:
		return lipgloss.NewStyle().Foreground(darkNavy).Background(cyan).Bold(true).Render(" NOUL ")
	case jev.Choice:
		return lipgloss.NewStyle().Foreground(darkNavy).Background(gold).Bold(true).Render(" CHOICE ")
	case jev.Score:
		return lipgloss.NewStyle().Foreground(darkNavy).Background(green).Bold(true).Render(" SCORE ")
	}
	return mutedStyle.Render(" ? ")
}

// relTime renders "2s", "4m", "3h", "2d" for a duration in the past.
func relTime(now, t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := now.Sub(t)
	if d < 0 {
		d = 0
	}
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
	return fmt.Sprintf("%dd", int(d.Hours()/24))
}

// shortPath replaces the home directory prefix with "~".
func shortPath(p string) string {
	if p == "" {
		return ""
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if p == home {
			return "~"
		}
		if strings.HasPrefix(p, home+string(filepath.Separator)) {
			return "~" + p[len(home):]
		}
	}
	return p
}

func shortHead(h string) string {
	if len(h) > 7 {
		return h[:7]
	}
	return h
}

func plural(n int, word string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, word)
	}
	return fmt.Sprintf("%d %ss", n, word)
}
