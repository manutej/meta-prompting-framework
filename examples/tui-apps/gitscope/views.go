package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"
)

type row struct {
	plain  string
	styled string
	idx    int
}

func (m model) View() string {
	if m.noRepo {
		return m.viewNoRepo()
	}
	if m.width == 0 || m.height == 0 {
		return ""
	}
	l := m.layout()
	if !l.ok {
		msg := stGoldBold.Render("gitscope") + stMuted.Render(fmt.Sprintf(" needs at least %dx%d (now %dx%d)", minWidth, minHeight, m.width, m.height))
		return fit(lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, msg), m.width, m.height)
	}
	out := m.renderHeader() + "\n" + m.renderBody(l) + "\n" + m.renderStatusBar()
	if m.toast != nil {
		out = m.overlayToast(out)
	}
	switch {
	case m.modal != modalNone && m.form != nil:
		out = overlayCentered(dimView(out), m.modalBox(), m.width, m.height)
	case m.showHelp:
		out = overlayCentered(dimView(out), m.helpBox(), m.width, m.height)
	}
	return fit(out, m.width, m.height)
}

func (m model) viewNoRepo() string {
	lines := []string{
		stBadge.Render("gitscope"),
		"",
		stError.Render("Not a git repository"),
		stMuted.Render(m.cwd),
		"",
		stText.Render("Run gitscope inside a repository, or press ") + stGoldBold.Render("q") + stText.Render(" to quit."),
	}
	box := stOverlay.Render(strings.Join(lines, "\n"))
	if m.width == 0 || m.height == 0 {
		return box
	}
	if lipgloss.Width(box) > m.width || lipgloss.Height(box) > m.height {
		return fit(strings.Join(lines, "\n"), m.width, m.height)
	}
	return fit(lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, box), m.width, m.height)
}

// ---------------------------------------------------------------- chrome

func (m model) branchLabel() string {
	switch {
	case !m.loaded:
		return "…"
	case m.status.Detached:
		return "HEAD detached @ " + shortOID(m.status.OID)
	case m.status.Unborn:
		return m.status.Head + " (no commits)"
	default:
		return m.status.Head
	}
}

func shortOID(oid string) string {
	if len(oid) > 7 {
		return oid[:7]
	}
	return oid
}

func (m model) renderHeader() string {
	clock := stBarMuted.Render(m.now.Format("15:04:05") + " ")
	badge := stBadge.Render("gitscope")
	repo := stBarBold.Render("  " + truncPlain(m.repoName, maxInt(8, m.width/4)))
	var tail []string
	if m.status.Upstream != "" {
		tail = append(tail,
			stBarSuccess.Render(fmt.Sprintf("  ↑%d", m.status.Ahead)),
			stBarWarn.Render(fmt.Sprintf(" ↓%d", m.status.Behind)))
	}
	if m.busy > 0 {
		tail = append(tail, stBarGold.Render("  "+stripANSI(m.spin.View())+" working"))
	}
	tailStr := strings.Join(tail, "")
	used := lipgloss.Width(badge) + lipgloss.Width(repo) + lipgloss.Width(tailStr) + lipgloss.Width(clock) + 4 + 1
	label := truncPlain(m.branchLabel(), maxInt(4, m.width-used))
	left := badge + repo + stBarGold.Render("  ⎇ "+label) + tailStr
	gap := m.width - lipgloss.Width(left) - lipgloss.Width(clock)
	if gap < 1 {
		left = ansiSlice(left, 0, m.width-lipgloss.Width(clock)-1)
		gap = 1
	}
	return left + stBar.Render(strings.Repeat(" ", gap)) + clock
}

func (m model) renderStatusBar() string {
	right := stBarMuted.Render("q quit ")
	var left string
	if m.filterEditing {
		left = stBarGold.Render(" /") + stBar.Render(m.lists[m.lastLeft].filter) + stBarGold.Render("▌") +
			stBarMuted.Render("  enter keep · esc clear")
	} else {
		budget := m.width - lipgloss.Width(right) - 1
		var b strings.Builder
		for i, h := range hintsFor(m.focus) {
			seg := stBarGold.Render(h.key) + stBar.Render(" "+h.desc)
			sep := stBar.Render(" ")
			if i > 0 {
				sep = stBarMuted.Render(" • ")
			}
			if lipgloss.Width(b.String())+lipgloss.Width(sep)+lipgloss.Width(seg) > budget {
				break
			}
			b.WriteString(sep + seg)
		}
		left = b.String()
	}
	gap := m.width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 0 {
		left = ansiSlice(left, 0, m.width-lipgloss.Width(right))
		gap = 0
	}
	return left + stBar.Render(strings.Repeat(" ", gap)) + right
}

// ---------------------------------------------------------------- panes

func paneName(p paneID) string {
	switch p {
	case paneStatus:
		return "Status"
	case paneBranches:
		return "Branches"
	case paneCommits:
		return "Commits"
	case paneStash:
		return "Stash"
	}
	return "Diff"
}

func (m model) paneCountLabel(p paneID) int {
	switch p {
	case paneStatus:
		return len(m.status.Entries)
	case paneBranches:
		return len(m.branches)
	case paneCommits:
		return len(m.commits)
	case paneStash:
		return len(m.stashes)
	}
	return 0
}

func (m model) paneSummary(p paneID) string {
	n := m.paneCountLabel(p)
	switch p {
	case paneStatus:
		staged, unstaged := 0, 0
		for _, e := range m.status.Entries {
			if e.Staged {
				staged++
			} else {
				unstaged++
			}
		}
		if n == 0 {
			return "clean"
		}
		return fmt.Sprintf("%d staged · %d unstaged", staged, unstaged)
	case paneBranches:
		return plural(n, "branch", "branches")
	case paneCommits:
		return plural(n, "commit", "commits")
	default:
		if n == 0 {
			return "empty"
		}
		return plural(n, "stash", "stashes")
	}
}

func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}

// renderPane draws a bordered box with the title embedded in the top border.
func renderPane(title string, lines []string, w, h int, focused bool) string {
	iw, ih := w-2, h-2
	bs := stBorder
	if focused {
		bs = stBorderFocused
	}
	if lipgloss.Width(title) > iw-3 {
		title = truncPlain(stripANSI(title), maxInt(0, iw-3))
	}
	rest := iw - 3 - lipgloss.Width(title)
	var b strings.Builder
	b.WriteString(bs.Render("╭─ ") + title + bs.Render(" "+strings.Repeat("─", rest)+"╮"))
	for i := 0; i < ih; i++ {
		var l string
		if i < len(lines) {
			l = lines[i]
		}
		b.WriteString("\n" + bs.Render("│") + fitLine(l, iw) + bs.Render("│"))
	}
	b.WriteString("\n" + bs.Render("╰"+strings.Repeat("─", iw)+"╯"))
	return b.String()
}

func (m model) renderBody(l layout) string {
	parts := make([]string, 0, leftPanes)
	for i := 0; i < leftPanes; i++ {
		parts = append(parts, m.renderLeftPane(paneID(i), l))
	}
	left := strings.Join(parts, "\n")
	return lipgloss.JoinHorizontal(lipgloss.Top, left, m.renderRightPane(l))
}

func (m model) renderLeftPane(p paneID, l layout) string {
	w, h := l.leftW, l.paneH[p]
	iw, ih := w-2, h-2
	focused := m.focus == p
	titleText := fmt.Sprintf("[%d] %s (%d)", p+1, paneName(p), m.paneCountLabel(p))
	if f := m.lists[p].filter; f != "" {
		titleText += " /" + f
	}
	title := stText.Render(titleText)
	if focused {
		title = stGoldBold.Render(titleText)
	}
	var lines []string
	if ih <= 1 || (p != m.lastLeft && !m.animating) {
		lines = []string{stMuted.Render(" " + m.paneSummary(p))}
	} else {
		rows, cursor := m.rows(p, iw)
		off := viewOffset(m.lists[p].off, cursor, rows, ih)
		for r := off; r < len(rows) && r < off+ih; r++ {
			rw := rows[r]
			switch {
			case p == m.lastLeft && r == cursor && focused:
				lines = append(lines, stSelected.Render(padRight(rw.plain, iw)))
			case p == m.lastLeft && r == cursor:
				lines = append(lines, lipgloss.NewStyle().Foreground(colText).Background(colDarkNavy).Bold(true).Render(padRight(rw.plain, iw)))
			default:
				lines = append(lines, rw.styled)
			}
		}
	}
	return renderPane(title, lines, w, h, focused)
}

func (m model) renderRightPane(l layout) string {
	w, h := l.rightW, l.bodyH
	focused := m.focus == paneDiff
	titleText := "Diff"
	if m.diff.title != "" {
		titleText = m.diff.title
	}
	title := stText.Render(titleText)
	if focused {
		title = stGoldBold.Render(titleText)
	}
	loading := m.diffLoading || !m.loaded
	if loading {
		title += " " + m.spin.View()
	} else if m.vp.TotalLineCount() > m.vp.VisibleLineCount() {
		title += stMuted.Render(fmt.Sprintf(" %3.0f%%", m.vp.ScrollPercent()*100))
	}
	var lines []string
	switch {
	case !m.loaded:
		lines = []string{"", " " + m.spin.View() + stMuted.Render(" loading repository…")}
	case m.diffLoading && m.diff.key != m.diffKey:
		lines = []string{"", " " + m.spin.View() + stMuted.Render(" running git…")}
	default:
		lines = strings.Split(m.vp.View(), "\n")
	}
	return renderPane(title, lines, w, h, focused)
}

// viewOffset scrolls a list so the cursor row stays visible; when the row just
// above the cursor is a section header it is kept visible too.
func viewOffset(off, cursor int, rows []row, ih int) int {
	n := len(rows)
	if ih <= 0 || n == 0 {
		return 0
	}
	if cursor >= 0 {
		top := cursor
		if cursor > 0 && rows[cursor-1].idx < 0 {
			top = cursor - 1
		}
		if top < off {
			off = top
		}
		if cursor >= off+ih {
			off = cursor - ih + 1
		}
	}
	return clampInt(off, 0, maxInt(0, n-ih))
}

// rows builds the display rows for a list pane and returns the cursor row.
func (m model) rows(p paneID, iw int) ([]row, int) {
	vis := m.visible(p)
	sel := m.lists[p].sel
	var rows []row
	cursor := -1
	switch p {
	case paneStatus:
		if len(vis) == 0 {
			return []row{{plain: " " + m.emptyShort(p), styled: stMuted.Render(" " + m.emptyShort(p)), idx: -1}}, -1
		}
		var stagedRows, unstagedRows []row
		for i, idx := range vis {
			e := m.status.Entries[idx]
			r := statusRow(e, iw)
			r.idx = i
			if e.Staged {
				stagedRows = append(stagedRows, r)
			} else {
				unstagedRows = append(unstagedRows, r)
			}
		}
		rows = append(rows, headerRow("Staged", len(stagedRows), iw))
		rows = append(rows, stagedRows...)
		rows = append(rows, headerRow("Unstaged", len(unstagedRows), iw))
		rows = append(rows, unstagedRows...)
	case paneBranches:
		for i, idx := range vis {
			r := branchRow(m.branches[idx], iw)
			r.idx = i
			rows = append(rows, r)
		}
	case paneCommits:
		for i, idx := range vis {
			r := commitRow(m.commits[idx], iw)
			r.idx = i
			rows = append(rows, r)
		}
	case paneStash:
		for i, idx := range vis {
			r := stashRow(m.stashes[idx], iw)
			r.idx = i
			rows = append(rows, r)
		}
	}
	if len(rows) == 0 {
		return []row{{plain: " " + m.emptyShort(p), styled: stMuted.Render(" " + m.emptyShort(p)), idx: -1}}, -1
	}
	for i, r := range rows {
		if r.idx == sel {
			cursor = i
			break
		}
	}
	return rows, cursor
}

func (m model) emptyShort(p paneID) string {
	if m.lists[p].filter != "" {
		return "no matches"
	}
	switch p {
	case paneStatus:
		return "working tree clean"
	case paneBranches:
		return "no branches yet"
	case paneCommits:
		return "no commits yet"
	default:
		return "no stashes"
	}
}

func headerRow(name string, n, iw int) row {
	plain := truncPlain(fmt.Sprintf(" ▾ %s (%d)", name, n), iw)
	return row{plain: plain, styled: stSectionHeader.Render(plain), idx: -1}
}

func codeStyle(code string) lipgloss.Style {
	switch code {
	case "M", "T":
		return stWarn
	case "A":
		return stSuccess
	case "D":
		return stError
	case "R", "C":
		return stHunk
	case "U":
		return stError
	}
	return stMuted
}

func statusRow(e statusEntry, iw int) row {
	code := e.Code
	if code == "?" {
		code = "??"
	}
	code = fmt.Sprintf("%-2s", code)
	path := e.Path
	if e.OrigPath != "" {
		path = e.OrigPath + " → " + e.Path
	}
	path = truncLeft(path, iw-6)
	plain := "   " + code + " " + path
	return row{
		plain:  plain,
		styled: "   " + codeStyle(e.Code).Render(code) + " " + stText.Render(path),
	}
}

func branchRow(b branch, iw int) row {
	marker := "  "
	if b.Current {
		marker = "* "
	}
	name := truncPlain(b.Name, iw-3)
	plain := " " + marker + name
	subj := ""
	if room := iw - lipgloss.Width(plain) - 2; room >= 6 && b.Subject != "" {
		subj = truncPlain(b.Subject, room)
	}
	var styled string
	if b.Current {
		styled = " " + stGoldBold.Render(marker+name)
	} else {
		styled = " " + marker + stText.Render(name)
	}
	if subj != "" {
		plain += "  " + subj
		styled += "  " + stMuted.Render(subj)
	}
	return row{plain: plain, styled: styled}
}

func commitRow(c commit, iw int) row {
	head := " ● " + c.Short + " "
	age := c.Age
	avail := iw - runewidth.StringWidth(head) - runewidth.StringWidth(age) - 1
	if avail < 8 {
		age = ""
		avail = iw - runewidth.StringWidth(head)
	}
	subj := truncPlain(c.Subject, avail)
	plain := head + subj
	styled := " " + stGold.Render("●") + " " + stGoldBold.Render(c.Short) + " " + stText.Render(subj)
	if age != "" {
		pad := strings.Repeat(" ", maxInt(1, avail-runewidth.StringWidth(subj)+1))
		plain += pad + age
		styled += pad + stMuted.Render(age)
	}
	return row{plain: plain, styled: styled}
}

func stashRow(s stash, iw int) row {
	ref := s.Ref
	msg := truncPlain(s.Message, iw-runewidth.StringWidth(ref)-3)
	return row{
		plain:  " " + ref + " " + msg,
		styled: " " + stGoldBold.Render(ref) + " " + stText.Render(msg),
	}
}

// ---------------------------------------------------------------- overlays

func (m model) overlayToast(base string) string {
	st := stToastOK
	if !m.toast.ok {
		st = stToastErr
	}
	box := st.Render(truncPlain(m.toast.text, m.width-6))
	x := m.width - lipgloss.Width(box) - 1
	return overlay(base, box, maxInt(0, x), m.height-3, m.width)
}

func (m model) modalBox() string {
	content := stGoldBold.Render(m.modalTitle) + "\n\n" + m.form.View()
	box := stOverlay.Render(content)
	if lipgloss.Height(box) > m.height {
		lines := strings.Split(box, "\n")
		box = strings.Join(lines[:m.height], "\n")
	}
	return box
}

func (m model) helpBox() string {
	inner := minInt(m.width-4, 78) - 6
	colW := (inner - 3) / 2
	nav := renderHelpCol("Navigation", helpNavigation, colW)
	act := renderHelpCol("Actions", helpActions, colW)
	body := lipgloss.JoinHorizontal(lipgloss.Top, nav, "   ", act)
	content := stGoldBold.Render("gitscope") + stMuted.Render(" · keyboard reference") + "\n\n" +
		body + "\n\n" + stMuted.Render("press ? or esc to close")
	box := stOverlay.Render(content)
	if lipgloss.Height(box) > m.height {
		lines := strings.Split(box, "\n")
		box = strings.Join(lines[:m.height], "\n")
	}
	return box
}

func renderHelpCol(title string, hints []hint, colW int) string {
	keyW := 0
	for _, h := range hints {
		keyW = maxInt(keyW, runewidth.StringWidth(h.key))
	}
	keyW = minInt(keyW, maxInt(4, colW/2))
	lines := []string{stGoldBold.Render(truncPlain(title, colW))}
	for _, h := range hints {
		key := padRight(truncPlain(h.key, keyW), keyW)
		desc := truncPlain(h.desc, maxInt(0, colW-keyW-1))
		lines = append(lines, padRight(stGold.Render(key)+" "+stText.Render(desc), colW))
	}
	return strings.Join(lines, "\n")
}
