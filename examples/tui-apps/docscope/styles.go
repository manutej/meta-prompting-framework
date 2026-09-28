package main

import (
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/ansi"
	"github.com/muesli/reflow/truncate"
)

var (
	styleBadge      = lipgloss.NewStyle().Foreground(colGold).Background(colNavy).Bold(true).Padding(0, 1)
	styleHeader     = lipgloss.NewStyle().Foreground(colText)
	styleHeaderDim  = lipgloss.NewStyle().Foreground(colMuted)
	styleHeaderGold = lipgloss.NewStyle().Foreground(colGold).Bold(true)
	styleTitleFocus = lipgloss.NewStyle().Foreground(colGold).Bold(true)
	styleTitleBlur  = lipgloss.NewStyle().Foreground(colText)
	styleBorderFoc  = lipgloss.NewStyle().Foreground(colGold)
	styleBorderBlur = lipgloss.NewStyle().Foreground(colNavy)
	styleSelected   = lipgloss.NewStyle().Foreground(colGold).Background(colNavy).Bold(true)
	styleCurrent    = lipgloss.NewStyle().Foreground(colGold)
	styleDir        = lipgloss.NewStyle().Foreground(colGold)
	styleFile       = lipgloss.NewStyle().Foreground(colText)
	styleMuted      = lipgloss.NewStyle().Foreground(colMuted)
	styleText       = lipgloss.NewStyle().Foreground(colText)
	styleKey        = lipgloss.NewStyle().Foreground(colGold).Bold(true)
	styleHighlight  = lipgloss.NewStyle().Foreground(colDarkNavy).Background(colGold).Bold(true)
	styleHighlitCur = lipgloss.NewStyle().Foreground(colDarkNavy).Background(colWarn).Bold(true)
	styleBarFill    = lipgloss.NewStyle().Foreground(colGold)
	styleBarEmpty   = lipgloss.NewStyle().Foreground(colNavy)
	styleToastOK    = lipgloss.NewStyle().Foreground(colDarkNavy).Background(colSuccess).Bold(true).Padding(0, 1)
	styleToastWarn  = lipgloss.NewStyle().Foreground(colDarkNavy).Background(colWarn).Bold(true).Padding(0, 1)
	styleToastErr   = lipgloss.NewStyle().Foreground(colText).Background(colError).Bold(true).Padding(0, 1)
	styleError      = lipgloss.NewStyle().Foreground(colError).Bold(true)
	styleSpinner    = lipgloss.NewStyle().Foreground(colGold)
	styleBackdrop   = lipgloss.NewStyle().Foreground(colMuted)
)

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)`)

func stripANSI(s string) string { return ansiRe.ReplaceAllString(s, "") }

func visibleWidth(s string) int { return ansi.PrintableRuneWidth(s) }

// fit truncates (ANSI-aware) and pads s so it is exactly w cells wide.
func fit(s string, w int) string {
	if w <= 0 {
		return ""
	}
	s = strings.ReplaceAll(s, "\t", "    ")
	if visibleWidth(s) > w {
		s = truncate.String(s, uint(w))
	}
	if pad := w - visibleWidth(s); pad > 0 {
		s += strings.Repeat(" ", pad)
	}
	return s
}

// clip truncates s to at most w cells, adding an ellipsis when it was cut.
func clip(s string, w int) string {
	if w <= 0 {
		return ""
	}
	if visibleWidth(s) <= w {
		return s
	}
	if w == 1 {
		return truncate.String(s, 1)
	}
	return truncate.StringWithTail(s, uint(w), "…")
}

// pane draws a titled, bordered box of exactly w x h cells around body lines.
// The inner content area is (w-4) x (h-2): one border and one padding cell per side.
func pane(title string, body []string, w, h int, focused bool) string {
	if w < 4 || h < 2 {
		return strings.TrimRight(strings.Repeat(strings.Repeat(" ", max(w, 0))+"\n", max(h, 0)), "\n")
	}
	border := styleBorderBlur
	ts := styleTitleBlur
	if focused {
		border = styleBorderFoc
		ts = styleTitleFocus
	}
	innerW := w - 4
	innerH := h - 2

	t := ""
	if title != "" {
		t = " " + clip(title, w-6) + " "
	}
	tw := visibleWidth(t)
	top := border.Render("╭─") + ts.Render(t) + border.Render(strings.Repeat("─", max(w-4-tw, 0))+"─╮")

	lines := make([]string, 0, h)
	lines = append(lines, top)
	for i := 0; i < innerH; i++ {
		content := ""
		if i < len(body) {
			content = body[i]
		}
		lines = append(lines, border.Render("│")+" "+fit(content, innerW)+" "+border.Render("│"))
	}
	lines = append(lines, border.Render("╰"+strings.Repeat("─", w-2)+"╯"))
	return strings.Join(lines, "\n")
}

// overlay centres box on top of base, dimming everything else.
func overlay(base, box string, w, h int) string {
	baseLines := strings.Split(base, "\n")
	for len(baseLines) < h {
		baseLines = append(baseLines, "")
	}
	boxLines := strings.Split(box, "\n")
	bw := 0
	for _, l := range boxLines {
		bw = max(bw, visibleWidth(l))
	}
	bh := len(boxLines)
	x := max((w-bw)/2, 0)
	y := max((h-bh)/2, 0)

	out := make([]string, 0, h)
	for i := 0; i < h; i++ {
		plain := stripANSI(baseLines[i])
		if i < y || i >= y+bh {
			out = append(out, styleBackdrop.Render(fit(plain, w)))
			continue
		}
		left := fit(plain, x)
		rest := ""
		if visibleWidth(plain) > x+bw {
			rest = dropCells(plain, x+bw)
		}
		right := fit(rest, max(w-x-bw, 0))
		out = append(out, styleBackdrop.Render(left)+fit(boxLines[i-y], bw)+styleBackdrop.Render(right))
	}
	return strings.Join(out, "\n")
}

// dropCells removes the first n cells of a plain (ANSI-free) string.
func dropCells(s string, n int) string {
	w := 0
	for i, r := range s {
		if w >= n {
			return s[i:]
		}
		w += visibleWidth(string(r))
	}
	return ""
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
