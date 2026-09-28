package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/sahilm/fuzzy"
)

var (
	styleFinderRow      = lipgloss.NewStyle().Foreground(colText)
	styleFinderRowSel   = lipgloss.NewStyle().Foreground(colText).Background(colNavy)
	styleFinderMatch    = lipgloss.NewStyle().Foreground(colGold).Bold(true)
	styleFinderMatchSel = lipgloss.NewStyle().Foreground(colGold).Background(colNavy).Bold(true)
)

type finderResult struct {
	n    *node
	idxs map[int]bool // byte offsets of matched characters in n.rel
}

// finder is the command-palette style fuzzy file picker.
type finder struct {
	input   textinput.Model
	items   []*node
	results []finderResult
	cursor  int
	offset  int
}

func newFinder() finder {
	ti := textinput.New()
	ti.Prompt = "› "
	ti.Placeholder = "type to fuzzy-find a file"
	ti.PromptStyle = styleKey
	ti.TextStyle = styleText
	ti.PlaceholderStyle = styleMuted
	ti.Cursor.Style = styleKey
	ti.CharLimit = 128
	return finder{input: ti}
}

func (f *finder) open(items []*node) tea.Cmd {
	f.items = items
	f.input.Reset()
	f.cursor, f.offset = 0, 0
	f.filter()
	return f.input.Focus()
}

func (f *finder) close() { f.input.Blur() }

func (f *finder) filter() {
	q := strings.TrimSpace(f.input.Value())
	f.results = f.results[:0]
	if q == "" {
		for _, n := range f.items {
			f.results = append(f.results, finderResult{n: n})
		}
	} else {
		paths := make([]string, len(f.items))
		for i, n := range f.items {
			paths[i] = n.rel
		}
		for _, m := range fuzzy.Find(q, paths) {
			idx := make(map[int]bool, len(m.MatchedIndexes))
			for _, i := range m.MatchedIndexes {
				idx[i] = true
			}
			f.results = append(f.results, finderResult{n: f.items[m.Index], idxs: idx})
		}
	}
	f.cursor = clamp(f.cursor, 0, max(len(f.results)-1, 0))
}

func (f *finder) update(msg tea.Msg) tea.Cmd {
	before := f.input.Value()
	var cmd tea.Cmd
	f.input, cmd = f.input.Update(msg)
	if f.input.Value() != before {
		f.cursor, f.offset = 0, 0
		f.filter()
	}
	return cmd
}

func (f *finder) move(delta int) {
	if len(f.results) == 0 {
		return
	}
	f.cursor = clamp(f.cursor+delta, 0, len(f.results)-1)
}

func (f *finder) selected() *node {
	if f.cursor < 0 || f.cursor >= len(f.results) {
		return nil
	}
	return f.results[f.cursor].n
}

// view renders the finder box for a screen of w x h.
func (f *finder) view(w, h int) string {
	boxW := clamp(w-4, 20, 76)
	rows := clamp(h-8, 3, 16)
	rows = min(rows, max(len(f.results), 1))
	boxH := rows + 4
	innerW := boxW - 4

	if f.cursor < f.offset {
		f.offset = f.cursor
	}
	if f.cursor >= f.offset+rows {
		f.offset = f.cursor - rows + 1
	}

	f.input.Width = max(innerW-4, 4)
	body := []string{f.input.View(), styleBorderBlur.Render(strings.Repeat("╌", innerW))}
	if len(f.results) == 0 {
		body = append(body, styleMuted.Render("  no matches"))
	}
	for i := f.offset; i < len(f.results) && i < f.offset+rows; i++ {
		body = append(body, f.renderRow(f.results[i], i == f.cursor, innerW))
	}
	title := fmt.Sprintf("Find file · %d/%d", len(f.results), len(f.items))
	return pane(title, body, boxW, boxH, true)
}

func (f *finder) renderRow(r finderResult, selected bool, w int) string {
	base, match := styleFinderRow, styleFinderMatch
	marker := "  "
	if selected {
		base, match = styleFinderRowSel, styleFinderMatchSel
		marker = "▸ "
	}
	label := clip(r.n.rel, w-2)
	var b strings.Builder
	b.WriteString(base.Render(marker))
	seg := strings.Builder{}
	segMatch := false
	flush := func() {
		if seg.Len() == 0 {
			return
		}
		if segMatch {
			b.WriteString(match.Render(seg.String()))
		} else {
			b.WriteString(base.Render(seg.String()))
		}
		seg.Reset()
	}
	for i, ch := range label {
		isMatch := r.idxs[i]
		if isMatch != segMatch {
			flush()
			segMatch = isMatch
		}
		seg.WriteRune(ch)
	}
	flush()
	if pad := w - 2 - visibleWidth(label); pad > 0 {
		b.WriteString(base.Render(strings.Repeat(" ", pad)))
	}
	return b.String()
}
