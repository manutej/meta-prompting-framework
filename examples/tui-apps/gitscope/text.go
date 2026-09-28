package main

import (
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

var ansiRe = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07`)

func stripANSI(s string) string {
	return ansiRe.ReplaceAllString(s, "")
}

// ansiSlice keeps the printable columns [from, to) of s while preserving every
// escape sequence, so styling state survives the cut.
func ansiSlice(s string, from, to int) string {
	if to <= from {
		return ""
	}
	var b strings.Builder
	col := 0
	sawEsc := false
	for i := 0; i < len(s); {
		if s[i] == '\x1b' {
			j := i + 1
			if j < len(s) && s[j] == '[' {
				j++
				for j < len(s) && !(s[j] >= 0x40 && s[j] <= 0x7e) {
					j++
				}
				if j < len(s) {
					j++
				}
			}
			b.WriteString(s[i:j])
			sawEsc = true
			i = j
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		w := runewidth.RuneWidth(r)
		if col >= from && col+w <= to {
			b.WriteRune(r)
		}
		col += w
		i += size
		if col >= to && !sawEsc {
			break
		}
	}
	if sawEsc {
		b.WriteString("\x1b[0m")
	}
	return b.String()
}

func truncPlain(s string, w int) string {
	if w <= 0 {
		return ""
	}
	s = tabReplacer.Replace(s)
	if runewidth.StringWidth(s) <= w {
		return s
	}
	if w == 1 {
		return "…"
	}
	return runewidth.Truncate(s, w, "…")
}

// truncLeft keeps the tail of s (the most specific part of a path).
func truncLeft(s string, w int) string {
	if w <= 0 {
		return ""
	}
	s = tabReplacer.Replace(s)
	sw := runewidth.StringWidth(s)
	if sw <= w {
		return s
	}
	if w == 1 {
		return "…"
	}
	return runewidth.TruncateLeft(s, sw-(w-1), "…")
}

func padRight(s string, w int) string {
	if pw := lipgloss.Width(s); pw < w {
		return s + strings.Repeat(" ", w-pw)
	}
	return s
}

func fitLine(s string, w int) string {
	if lipgloss.Width(s) > w {
		s = ansiSlice(s, 0, w)
	}
	return padRight(s, w)
}

// fit forces a rendered view to exactly h lines of at most w columns.
func fit(view string, w, h int) string {
	lines := strings.Split(view, "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	for i, l := range lines {
		lines[i] = fitLine(l, w)
	}
	for len(lines) < h {
		lines = append(lines, strings.Repeat(" ", w))
	}
	return strings.Join(lines, "\n")
}

func dimView(view string) string {
	lines := strings.Split(view, "\n")
	for i, l := range lines {
		lines[i] = stDim.Render(stripANSI(l))
	}
	return strings.Join(lines, "\n")
}

// overlay paints box onto base at column x, row y.
func overlay(base, box string, x, y, width int) string {
	lines := strings.Split(base, "\n")
	bw := lipgloss.Width(box)
	for j, bl := range strings.Split(box, "\n") {
		i := y + j
		if i < 0 || i >= len(lines) {
			continue
		}
		line := lines[i]
		left := padRight(ansiSlice(line, 0, x), x)
		right := ansiSlice(line, x+bw, width)
		lines[i] = left + padRight(bl, bw) + right
	}
	return strings.Join(lines, "\n")
}

func overlayCentered(base, box string, width, height int) string {
	bw, bh := lipgloss.Width(box), lipgloss.Height(box)
	x := (width - bw) / 2
	y := (height - bh) / 2
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	return overlay(base, box, x, y, width)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
