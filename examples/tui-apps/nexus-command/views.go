package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/sahilm/fuzzy"
)

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func spinnerFrame() string {
	return spinnerFrames[(time.Now().UnixMilli()/80)%int64(len(spinnerFrames))]
}

// layoutRects returns the agents and logs regions for the current tab/size.
func (m model) layoutRects() (agents, logs, metrics rect) {
	w, h := m.width, m.height
	bodyY, bodyH := 2, h-3
	switch m.tab {
	case tabOverview:
		if w >= 120 {
			aw, mw := 34, 30
			if w >= 160 {
				aw, mw = 38, 34
			}
			agents = rect{0, bodyY, aw, bodyH}
			logs = rect{aw, bodyY, w - aw - mw, bodyH}
			metrics = rect{w - mw, bodyY, mw, bodyH}
		} else {
			aw := min(36, w/2)
			mh := 9
			agents = rect{0, bodyY, aw, bodyH - mh}
			logs = rect{aw, bodyY, w - aw, bodyH - mh}
			metrics = rect{0, bodyY + bodyH - mh, w, mh}
		}
	case tabAgents:
		agents = rect{0, bodyY, w, bodyH}
	case tabLogs:
		logs = rect{0, bodyY, w, bodyH}
	case tabMetrics:
		metrics = rect{0, bodyY, w, bodyH}
	}
	return
}

func (m model) View() string {
	if m.width < 20 || m.height < 8 {
		return "terminal too small"
	}
	w, h := m.width, m.height
	var rows []string
	rows = append(rows, m.viewHeader(w))
	rows = append(rows, m.viewTabs(w))

	ar, lr, mr := m.layoutRects()
	switch m.tab {
	case tabOverview:
		if w >= 120 {
			rows = append(rows, lipgloss.JoinHorizontal(lipgloss.Top,
				m.viewAgents(ar, false), m.viewLogs(lr), m.viewMetrics(mr)))
		} else {
			rows = append(rows, lipgloss.JoinHorizontal(lipgloss.Top, m.viewAgents(ar, false), m.viewLogs(lr)))
			rows = append(rows, m.viewMetrics(mr))
		}
	case tabAgents:
		rows = append(rows, m.viewAgents(ar, true))
	case tabLogs:
		rows = append(rows, m.viewLogs(lr))
	case tabMetrics:
		rows = append(rows, m.viewMetrics(mr))
	}
	rows = append(rows, m.viewStatus(w))
	base := strings.Join(rows, "\n")

	switch {
	case m.paletteOpen:
		return overlay(base, m.viewPalette(), w, h)
	case m.confirmOpen:
		return overlay(base, m.viewConfirm(), w, h)
	case m.helpOpen:
		return overlay(base, m.viewHelp(), w, h)
	}
	return base
}

func (m model) runState() (string, lipgloss.Style) {
	switch {
	case m.running && m.paused:
		return "PAUSED", warnStyle
	case m.running:
		return "RUNNING", successStyle
	case !m.runEnd.IsZero():
		return "DONE", successStyle
	}
	return "IDLE", mutedStyle
}

func (m model) viewHeader(w int) string {
	left := badgeStyle.Render("ORMUS") + headerStyle.Render(" NEXUS · command center ")
	state, st := m.runState()
	elapsed := ""
	if m.running {
		elapsed = fmtDuration(time.Since(m.runStart))
	} else if !m.runEnd.IsZero() {
		elapsed = fmtDuration(m.runEnd.Sub(m.runStart))
	}
	right := headerDim.Render(" "+elapsed+" ") + st.Background(navy).Render(" "+state+" ") + headerDim.Render(" "+m.now.Format("15:04:05")+" ")
	mid := w - lipgloss.Width(left) - lipgloss.Width(right)
	task := headerDim.Render(fit("  "+m.task, max(0, mid)))
	return left + task + right
}

func (m model) viewTabs(w int) string {
	var b strings.Builder
	for i, n := range tabNames {
		if i == m.tab {
			b.WriteString(tabActive.Render(n))
		} else {
			b.WriteString(tabInactive.Render(n))
		}
	}
	return fit(b.String(), w)
}

func (m model) viewAgents(r rect, detailed bool) string {
	if r.w == 0 {
		return ""
	}
	iw := r.w - 2
	var lines []string
	if detailed {
		lines = append(lines, mutedStyle.Render(fit(fmt.Sprintf("  %-11s %-8s %-22s %8s %8s %7s", "AGENT", "STATE", "MODEL", "TOKENS", "COST", "TIME"), iw)))
	}
	for i, a := range m.agents {
		glyph, gs := stateGlyph(a.state)
		el := a.elapsed
		if a.state == stRunning {
			el = time.Since(a.started)
		}
		pct := clamp(m.pos[i], 0, 1)
		var l1 string
		if detailed {
			l1 = fmt.Sprintf("%s %-11s %-8s %-22s %8s %8s %7s", gs.Render(glyph), a.name, a.state, a.model,
				fmtTokens(a.tokens), fmt.Sprintf("$%.3f", a.costUSD), fmtDuration(el))
		} else {
			l1 = fmt.Sprintf("%s %-11s %-8s %6s", gs.Render(glyph), a.name, a.state, fmtDuration(el))
		}
		if i == m.selected {
			l1 = selStyle.Render(fit(stripANSI(l1), iw))
		} else {
			l1 = fit(l1, iw)
		}
		gw := iw - 22
		if detailed {
			gw = iw - 30
		}
		bar := gauge(pct, max(4, gw), stateColor(a.state))
		l2 := fmt.Sprintf("  %s %3d%%  %s", bar, int(pct*100+0.5), mutedStyle.Render(fmtTokens(a.tokens)+" tok"))
		if detailed {
			l2 = fmt.Sprintf("  %s %3d%%  %s", bar, int(pct*100+0.5), mutedStyle.Render(fit(a.role+" · "+a.lastMsg, iw-gw-10)))
		}
		lines = append(lines, l1, l2)
		if detailed {
			lines = append(lines, "")
		}
	}
	title := fmt.Sprintf("Agents %d/%d", m.doneCount(), len(m.agents))
	return pane(title, lines, r.w, r.h, m.focus == focusAgents)
}

func (m model) doneCount() int {
	n := 0
	for _, a := range m.agents {
		if a.state == stDone {
			n++
		}
	}
	return n
}

func stateGlyph(s agentState) (string, lipgloss.Style) {
	switch s {
	case stRunning:
		return spinnerFrame(), keyStyle
	case stDone:
		return "●", successStyle
	case stError, stKilled:
		return "✖", errorStyle
	case stQueued:
		return "◌", warnStyle
	}
	return "○", mutedStyle
}

func stateColor(s agentState) lipgloss.Color {
	switch s {
	case stRunning:
		return gold
	case stDone:
		return green
	case stError, stKilled:
		return red
	}
	return navy
}

func fmtTokens(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprint(n)
}

func (m model) viewLogs(r rect) string {
	if r.w == 0 {
		return ""
	}
	iw, ih := r.w-2, r.h-2
	entries := m.visibleLogs()
	n := len(entries)
	offset := m.logOffset
	if m.follow {
		offset = max(0, n-ih)
	}
	offset = max(0, min(offset, max(0, n-ih)))
	lines := make([]string, 0, ih)
	for i := offset; i < n && i < offset+ih; i++ {
		lines = append(lines, renderLog(entries[i], iw))
	}
	title := fmt.Sprintf("Logs %d", n)
	if m.errorsOnly {
		title += " · errors only"
	}
	if m.follow {
		title += " · follow"
	} else {
		title += fmt.Sprintf(" · %d%%", int(100*float64(offset+ih)/float64(max(1, n))))
	}
	return pane(title, lines, r.w, r.h, m.focus == focusLogs)
}

func renderLog(e logEntry, w int) string {
	ts := dimStyle.Render(e.at.Format("15:04:05"))
	var mark, body string
	txt := e.text
	if e.stream {
		runes := []rune(e.text)
		txt = string(runes[:min(len(runes), e.revealed)])
		if e.revealed < len(runes) {
			txt += "▌"
		}
	}
	switch e.lvl {
	case lvOK:
		mark, body = successStyle.Render("✓"), textStyle.Render(txt)
	case lvWarn:
		mark, body = warnStyle.Render("▲"), warnStyle.Render(txt)
	case lvErr:
		mark, body = errorStyle.Render("✖"), errorStyle.Render(txt)
	case lvLLM:
		mark, body = cyanStyle.Render("⟩"), cyanStyle.Render(txt)
	default:
		mark, body = mutedStyle.Render("·"), textStyle.Render(txt)
	}
	ag := agentStyle(e.agent).Render(fmt.Sprintf("%-10s", fit(e.agent, 10)))
	return fit(ts+" "+mark+" "+ag+" "+body, w)
}

func agentStyle(name string) lipgloss.Style {
	switch name {
	case "nexus":
		return keyStyle
	case "chaos":
		return errorStyle
	}
	return lipgloss.NewStyle().Foreground(text).Bold(true)
}

func (m model) viewMetrics(r rect) string {
	if r.w == 0 {
		return ""
	}
	iw := r.w - 2
	q := clamp(m.qPos, 0, 1)
	qc := qualityColor(m.quality)
	label := func(s string) string { return titleStyle.Render(fmt.Sprintf("%-10s", s)) }
	gw := max(6, min(iw-18, 40))

	lines := []string{
		label("quality") + gauge(q, gw, qc) + " " + lipgloss.NewStyle().Foreground(qc).Bold(true).Render(fmt.Sprintf("%.2f", q)) + mutedStyle.Render(" / 0.85"),
		label("iteration") + textStyle.Render(fmt.Sprintf("%d", m.iter)) + mutedStyle.Render(" / 5"),
		label("tokens") + textStyle.Render(fmtTokens(m.tokens)) + mutedStyle.Render("   cost ") + textStyle.Render(fmt.Sprintf("$%.2f", m.cost)),
		"",
		label("tok/s") + sparkline(m.tokHist, iw-11, 0, maxOf(m.tokHist, 1000), gold),
		label("quality") + sparkline(m.qualityHist, min(iw-11, 24), 0.4, 1.0, qc),
	}
	if r.h > 10 {
		lines = append(lines, "", mutedStyle.Render("strategy   iterative · threshold 0.85"), mutedStyle.Render("complexity 0.72 · scope 0.62 · deps 0.50"))
	}
	if r.h > 14 {
		lines = append(lines, "", titleStyle.Render("per-agent tokens"))
		for _, a := range m.agents {
			lines = append(lines, fmt.Sprintf("  %-11s %s %6s", a.name, gauge(float64(a.tokens)/float64(max(1, m.tokens)), max(4, iw-24), stateColor(a.state)), fmtTokens(a.tokens)))
		}
	}
	return pane("Metrics", lines, r.w, r.h, m.focus == focusMetrics)
}

func maxOf(h []float64, floor float64) float64 {
	m := floor
	for _, v := range h {
		if v > m {
			m = v
		}
	}
	return m
}

func (m model) viewStatus(w int) string {
	hint := func(k, d string) string { return keyStyle.Render(k) + mutedStyle.Render(" "+d+"  ") }
	var b strings.Builder
	b.WriteString(" ")
	if m.running {
		b.WriteString(hint("p", "pause") + hint("i", "inject failure") + hint("k", "kill"))
	} else {
		b.WriteString(hint("r", "run pipeline"))
	}
	b.WriteString(hint("ctrl+k", "commands") + hint("tab", "focus") + hint("1-4", "tabs") + hint("?", "help") + hint("q", "quit"))
	left := b.String()
	right := ""
	if m.toast != "" && time.Now().Before(m.toastUntil) {
		if m.toastOK {
			right = toastOK.Render(m.toast)
		} else {
			right = toastErr.Render(m.toast)
		}
	}
	gap := w - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 0 {
		left = fit(left, max(0, w-lipgloss.Width(right)))
		gap = 0
	}
	return left + strings.Repeat(" ", gap) + right
}

func (m model) viewPalette() string {
	w := min(56, m.width-6)
	var lines []string
	lines = append(lines, fit(m.palette.View(), w))
	lines = append(lines, lipgloss.NewStyle().Foreground(navy).Render(strings.Repeat("─", w)))
	names := make([]string, len(m.cmds))
	for i, c := range m.cmds {
		names[i] = c.name
	}
	q := strings.TrimSpace(m.palette.Value())
	var hits map[int][]int
	if q != "" {
		hits = map[int][]int{}
		for _, r := range fuzzy.Find(q, names) {
			hits[r.Index] = r.MatchedIndexes
		}
	}
	shown := 0
	for i, idx := range m.matches {
		if shown >= 9 {
			lines = append(lines, mutedStyle.Render(fmt.Sprintf("  … %d more", len(m.matches)-shown)))
			break
		}
		c := m.cmds[idx]
		name := highlight(c.name, hits[idx])
		row := "  " + name + "  " + mutedStyle.Render(c.desc)
		if i == m.palCursor {
			row = selStyle.Render(fit("▸ "+stripANSI(name), 26)) + selStyle.Render(fit(c.desc, w-26))
		}
		lines = append(lines, fit(row, w))
		shown++
	}
	if len(m.matches) == 0 {
		lines = append(lines, mutedStyle.Render("  no matching commands"))
	}
	lines = append(lines, "", mutedStyle.Render(fit("  ↑↓ move   enter run   esc close", w)))
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

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

func (m model) viewConfirm() string {
	w := min(44, m.width-6)
	lines := []string{
		errorStyle.Render(fit("⚠ "+m.confirmText, w)),
		"",
		fit(keyStyle.Render("y")+mutedStyle.Render(" yes   ")+keyStyle.Render("n")+mutedStyle.Render(" no"), w),
	}
	return overlayBorder.Render(strings.Join(lines, "\n"))
}

func (m model) viewHelp() string {
	w := min(60, m.width-6)
	k := func(key, desc string) string {
		return fit("  "+keyStyle.Render(fmt.Sprintf("%-10s", key))+textStyle.Render(desc), w)
	}
	lines := []string{
		titleStyle.Render(fit("NEXUS command center — keys", w)), "",
		k("r", "run the generation pipeline"),
		k("p / space", "pause / resume"),
		k("i", "inject a test failure (watch the self-heal loop)"),
		k("k / x", "kill the selected agent (confirms)"),
		k("ctrl+k / :", "command palette (fuzzy)"),
		k("1-4", "switch tab   ·   tab / shift+tab: cycle focus"),
		k("j / k", "move selection or scroll logs (by focus)"),
		k("g / G", "top / bottom   ·   ctrl+d / ctrl+u: half page"),
		k("f", "toggle follow logs   ·   e: errors & warnings only"),
		k("mouse", "click agents/tabs, wheel to scroll"),
		k("q", "quit"),
		"", mutedStyle.Render(fit("  press ? or esc to close", w)),
	}
	return overlayBorder.Render(strings.Join(lines, "\n"))
}
