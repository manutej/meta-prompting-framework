package main

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/muesli/reflow/truncate"
	"github.com/muesli/termenv"
)

type heading struct {
	level int
	title string
	line  int // index into the rendered lines
}

type renderedDoc struct {
	width    int
	lines    []string // ANSI-styled, each at most width cells wide
	plain    []string // the same lines with escapes stripped
	headings []heading
}

// renderDoneMsg carries a finished background render. seq lets the model
// discard results that were superseded by a newer open or resize.
type renderDoneMsg struct {
	seq   int
	path  string
	raw   string
	width int
	doc   renderedDoc
	err   error
}

const maxDocBytes = 32 << 20

func readDoc(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if info.Size() > maxDocBytes {
		return "", fmt.Errorf("%s is %d MB; docscope opens files up to %d MB", info.Name(), info.Size()>>20, maxDocBytes>>20)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.ReplaceAll(string(b), "\r\n", "\n"), nil
}

func openFileCmd(path string, width, seq int) tea.Cmd {
	return func() tea.Msg {
		raw, err := readDoc(path)
		if err != nil {
			return renderDoneMsg{seq: seq, path: path, width: width, err: err}
		}
		doc, err := renderMarkdown(raw, width)
		return renderDoneMsg{seq: seq, path: path, raw: raw, width: width, doc: doc, err: err}
	}
}

func rerenderCmd(path, raw string, width, seq int) tea.Cmd {
	return func() tea.Msg {
		doc, err := renderMarkdown(raw, width)
		return renderDoneMsg{seq: seq, path: path, raw: raw, width: width, doc: doc, err: err}
	}
}

var (
	atxRe        = regexp.MustCompile(`^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$`)
	setextH1Re   = regexp.MustCompile(`^ {0,3}=+[ \t]*$`)
	setextH2Re   = regexp.MustCompile(`^ {0,3}-+[ \t]*$`)
	fenceRe      = regexp.MustCompile("^ {0,3}(`{3,}|~{3,})")
	inlineLinkRe = regexp.MustCompile(`!?\[([^\]]*)\]\([^)]*\)`)
	inlineTagRe  = regexp.MustCompile(`<[^>]+>`)
	inlineMarkRe = regexp.MustCompile("[*_`~]+")
)

type scanResult struct {
	headings []heading
	blocks   [][]string // content lines of each fenced code block, in order
}

// scanMarkdown finds H1-H3 headings and fenced code blocks in the source.
func scanMarkdown(raw string) scanResult {
	var res scanResult
	lines := strings.Split(raw, "\n")
	var fence string
	var block []string
	for i, line := range lines {
		if fence != "" {
			if m := fenceRe.FindStringSubmatch(line); m != nil && m[1][0] == fence[0] && len(m[1]) >= len(fence) {
				res.blocks = append(res.blocks, block)
				block = nil
				fence = ""
				continue
			}
			block = append(block, line)
			continue
		}
		if m := fenceRe.FindStringSubmatch(line); m != nil {
			fence = m[1]
			continue
		}
		if m := atxRe.FindStringSubmatch(line); m != nil {
			if lvl := len(m[1]); lvl <= 3 {
				if t := cleanTitle(m[2]); t != "" {
					res.headings = append(res.headings, heading{level: lvl, title: t, line: -1})
				}
			}
			continue
		}
		if i > 0 && strings.TrimSpace(lines[i-1]) != "" && !isBlockStart(lines[i-1]) {
			if setextH1Re.MatchString(line) || setextH2Re.MatchString(line) {
				lvl := 2
				if setextH1Re.MatchString(line) {
					lvl = 1
				}
				if t := cleanTitle(lines[i-1]); t != "" && (i < 2 || strings.TrimSpace(lines[i-2]) == "" || len(res.headings) == 0 || res.headings[len(res.headings)-1].title != t) {
					res.headings = append(res.headings, heading{level: lvl, title: t, line: -1})
				}
			}
		}
	}
	if fence != "" && len(block) > 0 {
		res.blocks = append(res.blocks, block)
	}
	return res
}

func isBlockStart(line string) bool {
	t := strings.TrimSpace(line)
	return t == "" || strings.HasPrefix(t, "#") || strings.HasPrefix(t, "- ") || strings.HasPrefix(t, "* ") ||
		strings.HasPrefix(t, "|") || strings.HasPrefix(t, ">") || strings.HasPrefix(t, "```")
}

func cleanTitle(s string) string {
	s = inlineLinkRe.ReplaceAllString(s, "$1")
	s = inlineTagRe.ReplaceAllString(s, "")
	s = inlineMarkRe.ReplaceAllString(s, "")
	return strings.Join(strings.Fields(s), " ")
}

var mdStyle = docscopeStyle()

// renderMarkdown runs glamour and post-processes the output: headings are
// located in the rendered text, fenced code lines get a navy left border,
// table header rows turn gold, and every line is clipped to width.
func renderMarkdown(raw string, width int) (renderedDoc, error) {
	if width < 10 {
		width = 10
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithStyles(mdStyle),
		glamour.WithWordWrap(width),
		glamour.WithColorProfile(termenv.ANSI256),
	)
	if err != nil {
		return renderedDoc{}, err
	}
	out, err := r.Render(raw)
	if err != nil {
		return renderedDoc{}, err
	}
	lines := strings.Split(strings.ReplaceAll(out, "\r\n", "\n"), "\n")
	for len(lines) > 0 && strings.TrimSpace(stripANSI(lines[len(lines)-1])) == "" {
		lines = lines[:len(lines)-1]
	}
	for len(lines) > 0 && strings.TrimSpace(stripANSI(lines[0])) == "" {
		lines = lines[1:]
	}
	if len(lines) == 0 {
		lines = []string{styleMuted.Render("  (empty document)")}
	}
	plain := make([]string, len(lines))
	for i, l := range lines {
		plain[i] = stripANSI(l)
	}

	scan := scanMarkdown(raw)
	headings := locateHeadings(scan.headings, plain)
	markCodeLines(lines, plain, scan.blocks)
	markTableHeaders(lines, plain)
	for i := range lines {
		lines[i] = truncate.String(strings.ReplaceAll(lines[i], "\t", "    "), uint(width))
		plain[i] = stripANSI(lines[i])
	}
	return renderedDoc{width: width, lines: lines, plain: plain, headings: headings}, nil
}

// squash removes all whitespace so rendered text (where inline code gains
// padding spaces) compares equal to the source heading.
func squash(s string) string { return strings.Join(strings.Fields(s), "") }

// locateHeadings maps each source heading to the first rendered line that
// carries its text, scanning forward so duplicates resolve in order.
func locateHeadings(hs []heading, plain []string) []heading {
	out := make([]heading, 0, len(hs))
	from := 0
	for _, h := range hs {
		prefix := strings.Repeat("#", h.level)
		if h.level == 1 {
			prefix = ""
		}
		want := squash(prefix + h.title)
		line := -1
		for i := from; i < len(plain); i++ {
			if squash(plain[i]) == want {
				line = i
				break
			}
		}
		if line < 0 {
			head := squash(prefix + firstCells(h.title, 16))
			for i := from; i < len(plain); i++ {
				if strings.HasPrefix(squash(plain[i]), head) {
					line = i
					break
				}
			}
		}
		if line < 0 {
			line = from
		}
		h.line = line
		out = append(out, h)
		from = min(line+1, len(plain))
	}
	return out
}

func firstCells(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return strings.TrimSpace(string(r[:n]))
}

var codeBar = " " + styleBorderBlur.Render("▌")

// markCodeLines finds fenced code lines in the rendered output and prefixes
// them with a navy bar. Blank or wrapped lines between two matched lines of
// the same block are marked too.
func markCodeLines(lines, plain []string, blocks [][]string) {
	marked := make([]bool, len(lines))
	from := 0
	for _, block := range blocks {
		first, last := -1, -1
		cursor := from
		for _, code := range block {
			want := strings.TrimSpace(code)
			if want == "" {
				continue
			}
			found := -1
			limit := min(cursor+400, len(plain))
			for i := cursor; i < limit; i++ {
				if strings.TrimSpace(plain[i]) == want {
					found = i
					break
				}
			}
			if found < 0 {
				head := firstCells(want, 24)
				for i := cursor; i < limit; i++ {
					if strings.HasPrefix(strings.TrimSpace(plain[i]), head) {
						found = i
						break
					}
				}
			}
			if found < 0 {
				continue
			}
			if first < 0 {
				first = found
			}
			last = found
			cursor = found + 1
		}
		if first < 0 {
			continue
		}
		for i := first; i <= last; i++ {
			marked[i] = true
		}
		from = last + 1
	}
	for i, ok := range marked {
		if ok {
			lines[i] = codeBar + lines[i]
			plain[i] = " ▌" + plain[i]
		}
	}
}

var tableRuleRe = regexp.MustCompile(`^\s*[─┼]+\s*$`)

func markTableHeaders(lines, plain []string) {
	for i := 1; i < len(plain); i++ {
		if strings.Contains(plain[i], "┼") && tableRuleRe.MatchString(plain[i]) && strings.TrimSpace(plain[i-1]) != "" {
			lines[i-1] = styleHeaderGold.Render(plain[i-1])
		}
	}
}

// ---- in-document search --------------------------------------------------

type matchPos struct {
	line       int
	start, end int // rune offsets into the plain line
}

func lowerRunes(s string) []rune {
	r := []rune(s)
	for i, c := range r {
		r[i] = unicode.ToLower(c)
	}
	return r
}

func findMatches(plain []string, query string) []matchPos {
	q := lowerRunes(query)
	if len(q) == 0 {
		return nil
	}
	var out []matchPos
	for li, line := range plain {
		l := lowerRunes(line)
		for i := 0; i+len(q) <= len(l); {
			if runesEqual(l[i:i+len(q)], q) {
				out = append(out, matchPos{line: li, start: i, end: i + len(q)})
				i += len(q)
				continue
			}
			i++
		}
	}
	return out
}

func runesEqual(a, b []rune) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// highlightLine rebuilds a plain line with its matches wrapped in the gold
// highlight; the match equal to cur uses the brighter "current" style.
func highlightLine(plain string, ms []matchPos, cur matchPos) string {
	r := []rune(plain)
	var b strings.Builder
	pos := 0
	for _, m := range ms {
		if m.start < pos || m.end > len(r) {
			continue
		}
		b.WriteString(styleText.Render(string(r[pos:m.start])))
		st := styleHighlight
		if m == cur {
			st = styleHighlitCur
		}
		b.WriteString(st.Render(string(r[m.start:m.end])))
		pos = m.end
	}
	b.WriteString(styleText.Render(string(r[pos:])))
	return b.String()
}
