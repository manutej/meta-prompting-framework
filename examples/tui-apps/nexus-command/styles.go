package main

import (
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
	"github.com/muesli/reflow/truncate"
)

var (
	gold     = lipgloss.Color("178")
	goldHi   = lipgloss.Color("221")
	navy     = lipgloss.Color("24")
	darkNavy = lipgloss.Color("17")
	text     = lipgloss.Color("252")
	muted    = lipgloss.Color("240")
	dim      = lipgloss.Color("238")
	green    = lipgloss.Color("76")
	amber    = lipgloss.Color("220")
	red      = lipgloss.Color("196")
	cyan     = lipgloss.Color("51")
	white    = lipgloss.Color("255")

	badgeStyle    = lipgloss.NewStyle().Foreground(darkNavy).Background(gold).Bold(true).Padding(0, 1)
	headerStyle   = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true)
	headerDim     = lipgloss.NewStyle().Foreground(text).Background(navy)
	tabActive     = lipgloss.NewStyle().Foreground(darkNavy).Background(gold).Bold(true).Padding(0, 2)
	tabInactive   = lipgloss.NewStyle().Foreground(muted).Padding(0, 2)
	titleStyle    = lipgloss.NewStyle().Foreground(gold).Bold(true)
	textStyle     = lipgloss.NewStyle().Foreground(text)
	mutedStyle    = lipgloss.NewStyle().Foreground(muted)
	dimStyle      = lipgloss.NewStyle().Foreground(dim)
	selStyle      = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true)
	successStyle  = lipgloss.NewStyle().Foreground(green).Bold(true)
	warnStyle     = lipgloss.NewStyle().Foreground(amber).Bold(true)
	errorStyle    = lipgloss.NewStyle().Foreground(red).Bold(true)
	cyanStyle     = lipgloss.NewStyle().Foreground(cyan)
	keyStyle      = lipgloss.NewStyle().Foreground(gold).Bold(true)
	statusStyle   = lipgloss.NewStyle().Foreground(muted).Background(lipgloss.Color("235"))
	toastOK       = lipgloss.NewStyle().Foreground(darkNavy).Background(green).Bold(true).Padding(0, 1)
	toastErr      = lipgloss.NewStyle().Foreground(white).Background(red).Bold(true).Padding(0, 1)
	matchStyle    = lipgloss.NewStyle().Foreground(goldHi).Bold(true).Underline(true)
	overlayBorder = lipgloss.NewStyle().Border(lipgloss.DoubleBorder()).BorderForeground(gold).Padding(0, 1)
)

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]`)

func stripANSI(s string) string { return ansiRe.ReplaceAllString(s, "") }

// fit truncates (ANSI-aware) and pads s to exactly w columns.
func fit(s string, w int) string {
	if w <= 0 {
		return ""
	}
	s = strings.ReplaceAll(s, "\t", "  ")
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

func qualityColor(q float64) lipgloss.Color {
	switch {
	case q >= 0.85:
		return green
	case q >= 0.7:
		return amber
	case q > 0:
		return red
	default:
		return muted
	}
}
