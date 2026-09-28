package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"
)

// Tests run without a TTY, where lipgloss would silently drop every colour;
// force a real profile so views carry the escapes the app emits in a terminal.
func TestMain(m *testing.M) {
	lipgloss.SetColorProfile(termenv.ANSI256)
	os.Exit(m.Run())
}

// ---- fixture -----------------------------------------------------------------

const fillerCount = 30

func specDoc() string {
	var b strings.Builder
	b.WriteString(specHead)
	for i := 1; i <= fillerCount; i++ {
		fmt.Fprintf(&b, "Line filler %d.\n\n", i)
	}
	b.WriteString(specTail)
	return b.String()
}

const specHead = `# Technical Spec

The widget service. A Widget is a small thing; WIDGET in caps too.

## Architecture

Paragraph one about the architecture of the widget platform and how the
pieces fit together across services, queues and storage layers.

Paragraph two continues with more detail so the document is long enough to
scroll inside a forty-row terminal window during the tests.

` + "```go" + `
func widget() string {
	return "widget"
}
` + "```" + `

## Data Model

> Every widget has an id.

- id
- name
- created_at

### Fields

| Field | Type |
|-------|------|
| id    | int  |
| name  | text |

## API

GET /widgets lists every widget in the system, paginated by cursor so the
client never loads the whole collection at once.

POST /widgets creates one. Errors are returned as JSON problem documents.

`

const specTail = `## Deployment

Ship it with the usual pipeline. Final paragraph of the specification.
`

const cjkDoc = "# 日本語のガイド\n\n## はじめに\n\nこれは日本語のテキストです。とても長い行がここにあります。折り返しが必要ですから、幅の計算が正しいことを確かめます。\n\n## 使い方\n\n中文内容也在这里，한국어 텍스트도 있습니다.\n"

const noHeadingsDoc = "just a paragraph\n\nanother paragraph without any headings at all\n"

const crlfDoc = "# CRLF Doc\r\n\r\nline one\r\n\r\n## Second\r\n\r\nline two without trailing newline"

func bigDoc() string {
	var b strings.Builder
	for i := 0; b.Len() < 500*1024; i++ {
		fmt.Fprintf(&b, "## Section %d\n\nParagraph %d with **bold** and `code`.\n\n", i, i)
	}
	return b.String()
}

func fixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, content string) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("docs/TECHNICAL_SPEC.md", specDoc())
	write("docs/guide/cjk.md", cjkDoc)
	write("docs/noheadings.md", noHeadingsDoc)
	write("docs/crlf.md", crlfDoc)
	write("examples/big.md", bigDoc())
	write("docs/notes.txt", "not markdown")
	write("node_modules/pkg/README.md", "# excluded")
	write("vendor/lib/README.md", "# excluded")
	write(".git/README.md", "# excluded")
	return root
}

// ---- helpers -----------------------------------------------------------------

func keyMsg(k string) tea.KeyMsg {
	switch k {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	case "shift+tab":
		return tea.KeyMsg{Type: tea.KeyShiftTab}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	case "ctrl+p":
		return tea.KeyMsg{Type: tea.KeyCtrlP}
	case "ctrl+d":
		return tea.KeyMsg{Type: tea.KeyCtrlD}
	case "ctrl+u":
		return tea.KeyMsg{Type: tea.KeyCtrlU}
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case "up":
		return tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case "left":
		return tea.KeyMsg{Type: tea.KeyLeft}
	case "right":
		return tea.KeyMsg{Type: tea.KeyRight}
	case "pgup":
		return tea.KeyMsg{Type: tea.KeyPgUp}
	case "pgdown":
		return tea.KeyMsg{Type: tea.KeyPgDown}
	case "space":
		return tea.KeyMsg{Type: tea.KeySpace, Runes: []rune{' '}}
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)}
}

func press(m model, ks ...string) (model, tea.Cmd) {
	var cmd tea.Cmd
	for _, k := range ks {
		var next tea.Model
		next, cmd = m.Update(keyMsg(k))
		m = next.(model)
	}
	return m, cmd
}

func typeText(m model, s string) model {
	for _, r := range s {
		m, _ = press(m, string(r))
	}
	return m
}

// inject renders the current document synchronously and feeds the result in
// through the same message the background command would send.
func inject(t *testing.T, m model) model {
	t.Helper()
	if m.current == nil {
		t.Fatal("no current document")
	}
	w, _ := m.readerInner()
	raw, err := readDoc(m.current.path)
	if err != nil {
		t.Fatal(err)
	}
	doc, err := renderMarkdown(raw, w)
	if err != nil {
		t.Fatal(err)
	}
	next, _ := m.Update(renderDoneMsg{seq: m.renderSeq, path: m.current.path, raw: raw, width: w, doc: doc})
	return next.(model)
}

func resize(m model, w, h int) model {
	next, _ := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return next.(model)
}

func newTestModel(t *testing.T, root string, w, h int) model {
	t.Helper()
	m := resize(newModel(root), w, h)
	if m.current != nil {
		m = inject(t, m)
	}
	return m
}

func openRel(t *testing.T, m model, rel string) model {
	t.Helper()
	for _, f := range m.files {
		if f.rel == rel {
			cmd := m.openNode(f, false)
			if cmd == nil {
				t.Fatalf("openNode(%s) returned no command", rel)
			}
			return inject(t, m)
		}
	}
	t.Fatalf("file %s not in tree", rel)
	return m
}

func settle(m model) model {
	for i := 0; i < 300 && m.animating(); i++ {
		next, _ := m.Update(animTickMsg{})
		m = next.(model)
	}
	return m
}

func plainLines(view string) []string { return strings.Split(stripANSI(view), "\n") }

func assertFits(t *testing.T, name, view string, w, h int) {
	t.Helper()
	if view == "" {
		t.Fatalf("%s: empty view", name)
	}
	ls := plainLines(view)
	if len(ls) != h {
		t.Errorf("%s: view has %d lines, want %d", name, len(ls), h)
	}
	for i, l := range ls {
		if vw := visibleWidth(l); vw > w {
			t.Errorf("%s: line %d is %d cells wide (> %d): %q", name, i, vw, w, l)
		}
	}
}

func fileRels(m model) []string {
	out := make([]string, len(m.files))
	for i, f := range m.files {
		out[i] = f.rel
	}
	return out
}

func rowNames(m model) []string {
	out := make([]string, len(m.rows))
	for i, r := range m.rows {
		out[i] = r.rel
	}
	return out
}

// ---- tests -------------------------------------------------------------------

func TestTreeOrderingAndExclusions(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	want := []string{"docs/guide/cjk.md", "docs/crlf.md", "docs/noheadings.md", "docs/TECHNICAL_SPEC.md", "examples/big.md"}
	got := fileRels(m)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("files = %v, want %v", got, want)
	}
	rows := rowNames(m)
	wantRows := []string{"docs", "docs/guide", "docs/guide/cjk.md", "docs/crlf.md", "docs/noheadings.md", "docs/TECHNICAL_SPEC.md", "examples", "examples/big.md"}
	if strings.Join(rows, ",") != strings.Join(wantRows, ",") {
		t.Fatalf("rows = %v, want %v", rows, wantRows)
	}
	for _, r := range got {
		if strings.Contains(r, "notes.txt") || strings.Contains(r, "node_modules") || strings.Contains(r, "vendor") || strings.Contains(r, ".git") {
			t.Errorf("unexpected entry %s", r)
		}
	}
}

func TestViewFitsAllSizes(t *testing.T) {
	root := fixture(t)
	for _, sz := range [][2]int{{80, 24}, {120, 40}, {200, 60}} {
		w, h := sz[0], sz[1]
		m := newTestModel(t, root, w, h)
		m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
		assertFits(t, fmt.Sprintf("%dx%d base", w, h), m.View(), w, h)

		busy, _ := press(m, "ctrl+p")
		assertFits(t, fmt.Sprintf("%dx%d finder", w, h), busy.View(), w, h)
		help, _ := press(m, "?")
		assertFits(t, fmt.Sprintf("%dx%d help", w, h), help.View(), w, h)
		search, _ := press(m, "3", "/")
		search = typeText(search, "widget")
		assertFits(t, fmt.Sprintf("%dx%d search", w, h), search.View(), w, h)
		hidden, _ := press(m, "b")
		assertFits(t, fmt.Sprintf("%dx%d hidden-sidebar rendering", w, h), hidden.View(), w, h)
		hidden = inject(t, hidden)
		assertFits(t, fmt.Sprintf("%dx%d hidden-sidebar", w, h), hidden.View(), w, h)
		outline, _ := press(m, "2")
		outline = settle(outline)
		assertFits(t, fmt.Sprintf("%dx%d outline", w, h), outline.View(), w, h)
		big := openRel(t, resize(m, w, h), "docs/guide/cjk.md")
		assertFits(t, fmt.Sprintf("%dx%d cjk", w, h), big.View(), w, h)
	}
}

func TestViewAdaptsToResize(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "3", "G")
	if m.progress() < 0.99 {
		t.Fatalf("expected bottom, progress=%v", m.progress())
	}
	m = resize(m, 80, 24)
	if !m.rendering {
		t.Fatal("resize should trigger a re-render")
	}
	assertFits(t, "80x24 while re-rendering", m.View(), 80, 24)
	m = inject(t, m)
	w, _ := m.readerInner()
	if m.doc.width != w {
		t.Fatalf("doc width %d, want %d", m.doc.width, w)
	}
	if m.progress() < 0.99 {
		t.Fatalf("scroll percentage not preserved: %v", m.progress())
	}
	assertFits(t, "80x24 after re-render", m.View(), 80, 24)
	small := resize(m, 30, 8)
	if v := small.View(); !strings.Contains(stripANSI(v), "needs at least") {
		t.Fatalf("expected too-small notice, got %q", stripANSI(v))
	}
}

func TestOutlineExtraction(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	hs := m.doc.headings
	want := []struct {
		level int
		title string
	}{{1, "Technical Spec"}, {2, "Architecture"}, {2, "Data Model"}, {3, "Fields"}, {2, "API"}, {2, "Deployment"}}
	if len(hs) != len(want) {
		t.Fatalf("got %d headings %+v, want %d", len(hs), hs, len(want))
	}
	last := -1
	for i, h := range hs {
		if h.level != want[i].level || h.title != want[i].title {
			t.Errorf("heading %d = %d %q, want %d %q", i, h.level, h.title, want[i].level, want[i].title)
		}
		if h.line <= last {
			t.Errorf("heading %d line %d not after %d", i, h.line, last)
		}
		if !strings.Contains(squash(m.doc.plain[h.line]), squash(h.title)) {
			t.Errorf("heading %q maps to line %d %q", h.title, h.line, m.doc.plain[h.line])
		}
		last = h.line
	}
	m, _ = press(m, "2")
	m = settle(m)
	view := stripANSI(m.View())
	for _, s := range []string{"# Technical Spec", "## Architecture", "### Fields", "Outline · 6"} {
		if !strings.Contains(view, s) {
			t.Errorf("outline view missing %q", s)
		}
	}
}

func TestRenderPostProcessing(t *testing.T) {
	doc, err := renderMarkdown(specDoc(), 70)
	if err != nil {
		t.Fatal(err)
	}
	bars := 0
	for _, l := range doc.plain {
		if strings.HasPrefix(l, " ▌") {
			bars++
		}
	}
	if bars != 3 {
		t.Errorf("expected 3 code lines with a bar, got %d", bars)
	}
	joined := strings.Join(doc.plain, "\n")
	for _, s := range []string{"◆ id", "▌ Every widget has an id.", "FIELD", "┼"} {
		if !strings.Contains(joined, s) {
			t.Errorf("rendered output missing %q", s)
		}
	}
	for i, l := range doc.plain {
		if visibleWidth(l) > 70 {
			t.Errorf("line %d exceeds width: %d", i, visibleWidth(l))
		}
	}
}

func TestZeroHeadingsDoc(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/noheadings.md")
	if len(m.doc.headings) != 0 {
		t.Fatalf("expected no headings, got %+v", m.doc.headings)
	}
	m, _ = press(m, "2")
	m = settle(m)
	v := stripANSI(m.View())
	if !strings.Contains(v, "no headings") || !strings.Contains(v, "Outline · 0") {
		t.Fatalf("outline pane should say no headings: %s", v)
	}
	m, _ = press(m, "enter", "3", "]", "[")
	assertFits(t, "zero headings", m.View(), 120, 40)
}

func TestCJKDoc(t *testing.T) {
	m := newTestModel(t, fixture(t), 80, 24)
	m = openRel(t, m, "docs/guide/cjk.md")
	if len(m.doc.headings) != 3 {
		t.Fatalf("expected 3 headings, got %+v", m.doc.headings)
	}
	if m.doc.headings[0].title != "日本語のガイド" {
		t.Errorf("title = %q", m.doc.headings[0].title)
	}
	w, _ := m.readerInner()
	for i, l := range m.doc.plain {
		if visibleWidth(l) > w {
			t.Errorf("line %d wider than %d: %q", i, w, l)
		}
	}
	assertFits(t, "cjk", m.View(), 80, 24)
	if !strings.Contains(stripANSI(m.View()), "はじめに") {
		t.Error("CJK heading not visible in outline/reader")
	}
}

func TestCRLFDoc(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/crlf.md")
	if strings.Contains(m.raw, "\r") {
		t.Fatal("CR not normalised")
	}
	if len(m.doc.headings) != 2 || m.doc.headings[1].title != "Second" {
		t.Fatalf("headings = %+v", m.doc.headings)
	}
	if !strings.Contains(strings.Join(m.doc.plain, "\n"), "line two without trailing newline") {
		t.Fatal("last line without newline was lost")
	}
}

func TestNotFoundRoot(t *testing.T) {
	empty := t.TempDir()
	if err := os.WriteFile(filepath.Join(empty, "readme.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := newTestModel(t, empty, 100, 30)
	v := stripANSI(m.View())
	if !strings.Contains(v, "No markdown files found") {
		t.Fatalf("expected friendly message, got %q", v)
	}
	assertFits(t, "empty root", m.View(), 100, 30)
	_, cmd := press(m, "q")
	if cmd == nil {
		t.Fatal("q should quit on the empty screen")
	}

	missing := newTestModel(t, filepath.Join(empty, "nope"), 100, 30)
	if v := stripANSI(missing.View()); !strings.Contains(v, "Cannot read") {
		t.Fatalf("expected error screen, got %q", v)
	}
}

func TestFuzzyFinderFiltersAndOpens(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m, _ = press(m, "ctrl+p")
	if m.overlay != overlayFinder {
		t.Fatal("ctrl+p should open the finder")
	}
	if len(m.finder.results) != 5 {
		t.Fatalf("empty query should list all 5 files, got %d", len(m.finder.results))
	}
	m = typeText(m, "spec")
	if len(m.finder.results) != 1 || m.finder.results[0].n.rel != "docs/TECHNICAL_SPEC.md" {
		t.Fatalf("results = %+v", m.finder.results)
	}
	if len(m.finder.results[0].idxs) != 4 {
		t.Errorf("expected 4 highlighted characters, got %d", len(m.finder.results[0].idxs))
	}
	v := stripANSI(m.View())
	if !strings.Contains(v, "Find file · 1/5") || !strings.Contains(v, "▸ docs/TECHNICAL_SPEC.md") {
		t.Fatalf("finder overlay not rendered: %s", v)
	}
	m, cmd := press(m, "enter")
	if m.overlay != overlayNone || cmd == nil {
		t.Fatal("enter should close the finder and open the file")
	}
	if m.current.rel != "docs/TECHNICAL_SPEC.md" || !m.rendering {
		t.Fatalf("current = %v rendering = %v", m.current, m.rendering)
	}
	if m.rows[m.fileCursor] != m.current {
		t.Error("tree cursor should follow the opened file")
	}
	m = inject(t, m)
	if !strings.Contains(stripANSI(m.View()), "╭─ docs/TECHNICAL_SPEC.md") {
		t.Error("reader title should show the opened file")
	}
}

func TestFinderNavigationAndEscape(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m, _ = press(m, "1", "/")
	if m.overlay != overlayFinder {
		t.Fatal("/ from the Files pane should open the finder")
	}
	m = typeText(m, "md")
	if len(m.finder.results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(m.finder.results))
	}
	first := m.finder.selected()
	m, _ = press(m, "down", "down", "up")
	if m.finder.cursor != 1 || m.finder.selected() == first {
		t.Fatalf("cursor = %d", m.finder.cursor)
	}
	m, _ = press(m, "esc")
	if m.overlay != overlayNone || m.overlay == overlayFinder {
		t.Fatal("esc should close the finder")
	}
	if !strings.Contains(stripANSI(m.View()), "Files · 5") {
		t.Fatal("base view should be back")
	}
}

func TestInDocSearchCount(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "3", "/")
	if m.searchState != searchTyping {
		t.Fatal("/ in the reader should open the search bar")
	}
	m = typeText(m, "widget")
	if len(m.matches) != 10 {
		t.Fatalf("expected 10 case-insensitive matches, got %d: %+v", len(m.matches), m.matches)
	}
	v := stripANSI(m.View())
	if !strings.Contains(v, "/ widget") || !strings.Contains(v, "1/10") {
		t.Fatalf("search bar missing query or count: %q", v[strings.LastIndex(v, "\n"):])
	}
	assertFits(t, "search", m.View(), 120, 40)
	m = typeText(m, "zzz")
	if len(m.matches) != 0 || !strings.Contains(stripANSI(m.View()), "no matches") {
		t.Fatal("no-match state not shown")
	}
}

func TestSearchNavigationAndClear(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "3", "/")
	m = typeText(m, "filler")
	if len(m.matches) != fillerCount {
		t.Fatalf("expected %d matches, got %d", fillerCount, len(m.matches))
	}
	m, _ = press(m, "enter")
	if m.searchState != searchActive {
		t.Fatal("enter should commit the search")
	}
	m = settle(m)
	line := m.matches[0].line
	if line < m.vp.YOffset || line >= m.vp.YOffset+m.vp.Height {
		t.Fatalf("first match line %d not visible at offset %d", line, m.vp.YOffset)
	}
	m, _ = press(m, "n", "n")
	if m.matchIdx != 2 {
		t.Fatalf("matchIdx = %d, want 2", m.matchIdx)
	}
	m, _ = press(m, "p")
	if m.matchIdx != 1 {
		t.Fatalf("matchIdx = %d, want 1", m.matchIdx)
	}
	if !strings.Contains(stripANSI(m.View()), fmt.Sprintf("2/%d", fillerCount)) {
		t.Fatalf("status bar should show 2/%d", fillerCount)
	}
	if !strings.Contains(m.vp.View(), styleHighlight.Render("filler")) {
		t.Fatal("matches should be highlighted in the viewport")
	}
	m, _ = press(m, "esc")
	if m.searchState != searchOff || len(m.matches) != 0 || m.search.Value() != "" {
		t.Fatal("esc should clear the search")
	}
	if strings.Contains(m.vp.View(), styleHighlight.Render("filler")) {
		t.Fatal("highlights should be removed")
	}
}

func TestSidebarToggleRerenders(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	narrow := m.doc.width
	m, cmd := press(m, "b")
	if m.sidebarVisible || cmd == nil || !m.rendering {
		t.Fatal("b should hide the sidebar and start a re-render")
	}
	if m.focus != paneReader {
		t.Fatal("focus should move to the reader when the sidebar hides")
	}
	assertFits(t, "hidden while rendering", m.View(), 120, 40)
	m = inject(t, m)
	if m.doc.width <= narrow || m.doc.width != 120-4 {
		t.Fatalf("doc width %d after hide (was %d)", m.doc.width, narrow)
	}
	if strings.Contains(stripANSI(m.View()), "Files ·") {
		t.Fatal("sidebar should be hidden")
	}
	assertFits(t, "hidden", m.View(), 120, 40)
	m, _ = press(m, "b")
	if !m.sidebarVisible || !m.rendering {
		t.Fatal("b again should show the sidebar and re-render")
	}
	m = inject(t, m)
	if m.doc.width != narrow {
		t.Fatalf("doc width %d, want %d", m.doc.width, narrow)
	}
}

func TestFocusCycle(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	if m.focus != paneFiles {
		t.Fatal("initial focus should be Files")
	}
	m, _ = press(m, "tab")
	if m.focus != paneOutline || !m.animating() {
		t.Fatal("tab should focus Outline and start the accordion")
	}
	m = settle(m)
	l := m.layout()
	if l.filesH != collapsedPaneH || l.outlineH != l.bodyH-collapsedPaneH {
		t.Fatalf("accordion not settled: %+v", l)
	}
	m, _ = press(m, "tab")
	if m.focus != paneReader {
		t.Fatal("tab should focus Reader")
	}
	m, _ = press(m, "tab")
	if m.focus != paneFiles {
		t.Fatal("tab should wrap to Files")
	}
	m, _ = press(m, "shift+tab")
	if m.focus != paneReader {
		t.Fatal("shift+tab should go back to Reader")
	}
	m, _ = press(m, "h")
	if m.focus != paneFiles {
		t.Fatal("h should return to the last sidebar pane")
	}
	m, _ = press(m, "l")
	if m.focus != paneReader {
		t.Fatal("l should focus the reader")
	}
	m, _ = press(m, "2")
	if m.focus != paneOutline {
		t.Fatal("2 should focus Outline")
	}
	m, _ = press(m, "1")
	if m.focus != paneFiles {
		t.Fatal("1 should focus Files")
	}
	m, _ = press(m, "3")
	if m.focus != paneReader {
		t.Fatal("3 should focus Reader")
	}
	m = settle(m)
	if v := stripANSI(m.View()); !strings.Contains(v, "j/k scroll") {
		t.Fatal("status bar should show reader hints")
	}
}

func TestFilesPaneNavigation(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m, _ = press(m, "j", "j")
	if m.rows[m.fileCursor].rel != "docs/guide/cjk.md" {
		t.Fatalf("cursor on %s", m.rows[m.fileCursor].rel)
	}
	m, cmd := press(m, "enter")
	if cmd == nil || m.current.rel != "docs/guide/cjk.md" {
		t.Fatal("enter should open the file")
	}
	m, _ = press(m, "k", "enter")
	if len(m.rows) != 7 || m.rows[2].rel != "docs/crlf.md" {
		t.Fatalf("enter on a dir should collapse it: %v", rowNames(m))
	}
	m, _ = press(m, "right")
	if len(m.rows) != 8 {
		t.Fatal("→ should expand the dir")
	}
	m, _ = press(m, "left")
	if len(m.rows) != 7 {
		t.Fatal("← should collapse the dir")
	}
	m, _ = press(m, "G")
	if m.rows[m.fileCursor].rel != "examples/big.md" {
		t.Fatal("G should move to the last row")
	}
	m, _ = press(m, "left")
	if m.rows[m.fileCursor].rel != "examples" {
		t.Fatal("← on a file should jump to its parent directory")
	}
	m, _ = press(m, "g", "k")
	if m.fileCursor != 0 {
		t.Fatal("cursor should clamp at the top")
	}
	m = settle(m)
	if !strings.Contains(stripANSI(m.View()), "▸ guide/") {
		t.Fatal("collapsed dir should render with ▸")
	}
}

func TestOutlineJumpAnimates(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "2")
	m = settle(m)
	m, _ = press(m, "j", "j", "j", "j")
	target := m.doc.headings[4].line
	m, cmd := press(m, "enter")
	if !m.scrollAnimating || cmd == nil {
		t.Fatal("enter should start the scroll animation")
	}
	if m.vp.YOffset == target {
		t.Fatal("scroll should not jump instantly")
	}
	m = settle(m)
	if m.vp.YOffset != target {
		t.Fatalf("YOffset = %d, want %d", m.vp.YOffset, target)
	}
	if m.currentHeading() != 4 {
		t.Fatalf("current heading = %d", m.currentHeading())
	}
	if !strings.Contains(stripANSI(m.View()), "## API ◀") {
		t.Fatal("outline should mark the current heading")
	}
}

func TestHeadingStepKeys(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "3", "]")
	m = settle(m)
	if m.vp.YOffset != m.doc.headings[1].line {
		t.Fatalf("] should go to heading 1 (line %d), got %d", m.doc.headings[1].line, m.vp.YOffset)
	}
	m, _ = press(m, "]", "]")
	m = settle(m)
	if m.vp.YOffset != m.doc.headings[3].line {
		t.Fatalf("]] should go to heading 3, got %d", m.vp.YOffset)
	}
	m, _ = press(m, "[")
	m = settle(m)
	if m.vp.YOffset != m.doc.headings[2].line {
		t.Fatalf("[ should go to heading 2, got %d", m.vp.YOffset)
	}
	m, _ = press(m, "j", "[")
	m = settle(m)
	if m.vp.YOffset != m.doc.headings[2].line {
		t.Fatalf("[ from inside a section should go to its own heading, got %d", m.vp.YOffset)
	}
}

func TestReaderScrollKeysAndProgress(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "3")
	if m.vp.YOffset != 0 || m.progress() != 0 {
		t.Fatal("should start at the top")
	}
	m, _ = press(m, "j", "j", "k")
	if m.vp.YOffset != 1 {
		t.Fatalf("YOffset = %d after j j k", m.vp.YOffset)
	}
	m, _ = press(m, "ctrl+d")
	if m.vp.YOffset != 1+m.vp.Height/2 {
		t.Fatalf("ctrl+d YOffset = %d", m.vp.YOffset)
	}
	m, _ = press(m, "ctrl+u")
	if m.vp.YOffset != 1 {
		t.Fatalf("ctrl+u YOffset = %d", m.vp.YOffset)
	}
	m, _ = press(m, "G")
	if m.vp.YOffset != m.maxYOffset() || m.progress() != 1 {
		t.Fatalf("G: YOffset=%d max=%d progress=%v", m.vp.YOffset, m.maxYOffset(), m.progress())
	}
	if !strings.Contains(stripANSI(m.View()), "100%") {
		t.Fatal("header should show 100%")
	}
	if !strings.HasPrefix(plainLines(m.View())[38], strings.Repeat("━", 120)) {
		t.Fatal("progress bar should be full")
	}
	m, _ = press(m, "g", "space")
	if m.vp.YOffset != m.vp.Height {
		t.Fatalf("space should page down, got %d", m.vp.YOffset)
	}
	m, _ = press(m, "pgup", "pgdown", "up", "down")
	if m.vp.YOffset != m.vp.Height {
		t.Fatalf("pgup/pgdown/up/down mismatch: %d", m.vp.YOffset)
	}
}

func TestMouseClickAndWheel(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	click := func(x, y int) {
		next, _ := m.Update(tea.MouseMsg{X: x, Y: y, Action: tea.MouseActionPress, Button: tea.MouseButtonLeft})
		m = next.(model)
	}
	wheel := func(x, y int, down bool) {
		b := tea.MouseButtonWheelUp
		if down {
			b = tea.MouseButtonWheelDown
		}
		next, _ := m.Update(tea.MouseMsg{X: x, Y: y, Action: tea.MouseActionPress, Button: b})
		m = next.(model)
	}
	click(100, 10)
	if m.focus != paneReader {
		t.Fatal("click in the reader should focus it")
	}
	wheel(100, 10, true)
	wheel(100, 10, true)
	if m.vp.YOffset != 6 {
		t.Fatalf("two wheel-downs should scroll 6 lines, got %d", m.vp.YOffset)
	}
	wheel(100, 10, false)
	if m.vp.YOffset != 3 {
		t.Fatalf("wheel-up should scroll back 3 lines, got %d", m.vp.YOffset)
	}
	click(5, 4) // third row of the Files pane: docs/guide/cjk.md
	if m.focus != paneFiles || m.current.rel != "docs/guide/cjk.md" || !m.rendering {
		t.Fatalf("click should open cjk.md: focus=%v current=%v", m.focus, m.current)
	}
	m = inject(t, m)
	click(5, 2) // docs/ directory toggles
	if len(m.rows) != 3 {
		t.Fatalf("click on a dir should collapse it, rows=%v", rowNames(m))
	}
	click(5, 39) // status bar: no region
	if m.focus != paneFiles {
		t.Fatal("click outside the panes should not change focus")
	}
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "2")
	m = settle(m)
	click(5, 1+collapsedPaneH+1+2) // third outline row: "Data Model"
	m = settle(m)
	if m.vp.YOffset != m.doc.headings[2].line {
		t.Fatalf("outline click should scroll to heading 2 (line %d), got %d", m.doc.headings[2].line, m.vp.YOffset)
	}
}

func TestLargeFileRendersAsync(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	var big *node
	for _, f := range m.files {
		if f.rel == "examples/big.md" {
			big = f
		}
	}
	if info, err := os.Stat(big.path); err != nil || info.Size() < 500*1024 {
		t.Fatalf("fixture should be >= 500KB: %v %v", info, err)
	}
	cmd := m.openNode(big, false)
	if cmd == nil || !m.rendering {
		t.Fatal("opening should return a background command and enter the rendering state")
	}
	v := stripANSI(m.View())
	if !strings.Contains(v, "Rendering big.md") {
		t.Fatalf("spinner message missing: %s", v)
	}
	assertFits(t, "rendering", m.View(), 120, 40)

	stale, _ := m.Update(renderDoneMsg{seq: m.renderSeq - 1, path: big.path, doc: renderedDoc{lines: []string{"stale"}, plain: []string{"stale"}}})
	if !stale.(model).rendering {
		t.Fatal("a stale render result must be ignored")
	}

	lines := make([]string, 20000)
	plain := make([]string, 20000)
	var hs []heading
	for i := range lines {
		lines[i] = fmt.Sprintf("line %d", i)
		plain[i] = lines[i]
		if i%1000 == 0 {
			hs = append(hs, heading{level: 2, title: fmt.Sprintf("Section %d", i), line: i})
		}
	}
	w, _ := m.readerInner()
	next, _ := m.Update(renderDoneMsg{seq: m.renderSeq, path: big.path, raw: "x", width: w, doc: renderedDoc{width: w, lines: lines, plain: plain, headings: hs}})
	m = next.(model)
	if m.rendering || !m.hasDoc || len(m.doc.headings) != 20 {
		t.Fatal("injected render should be applied")
	}
	assertFits(t, "big", m.View(), 120, 40)
	m, _ = press(m, "3", "G")
	if !strings.Contains(stripANSI(m.View()), "line 19999") {
		t.Fatal("G should reach the last line")
	}
}

func TestRenderErrorIsShown(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m.openNode(m.files[0], false)
	next, _ := m.Update(renderDoneMsg{seq: m.renderSeq, path: m.current.path, err: fmt.Errorf("boom")})
	m = next.(model)
	if m.rendering || m.hasDoc {
		t.Fatal("error should end the rendering state without a document")
	}
	if !strings.Contains(stripANSI(m.View()), "boom") {
		t.Fatal("error should be displayed in the reader")
	}
	assertFits(t, "error", m.View(), 120, 40)
}

func TestToastLifecycle(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	_, cmd := press(m, "y")
	if cmd == nil {
		t.Fatal("y should issue the copy command")
	}
	next, tick := m.Update(copiedMsg{})
	m = next.(model)
	if tick == nil || m.toast == nil || m.toast.text != "copied" {
		t.Fatal("copiedMsg should show the copied toast and schedule expiry")
	}
	if !strings.Contains(stripANSI(m.View()), "copied") {
		t.Fatal("toast should be visible in the status bar")
	}
	assertFits(t, "toast", m.View(), 120, 40)
	next, _ = m.Update(toastExpireMsg{id: m.toast.id - 1})
	if next.(model).toast == nil {
		t.Fatal("an older expiry must not dismiss a newer toast")
	}
	next, _ = m.Update(toastExpireMsg{id: m.toast.id})
	if next.(model).toast != nil {
		t.Fatal("expiry should dismiss the toast")
	}
	next, _ = m.Update(copiedMsg{err: fmt.Errorf("no tty")})
	if tt := next.(model).toast; tt == nil || tt.kind != toastErr {
		t.Fatal("copy failure should show an error toast")
	}
}

func TestHelpOverlay(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m, _ = press(m, "?")
	if m.overlay != overlayHelp {
		t.Fatal("? should open help")
	}
	v := stripANSI(m.View())
	for _, s := range []string{"Keys · docscope", "fuzzy file finder", "copy path", "edit in $EDITOR", "toggle sidebar"} {
		if !strings.Contains(v, s) {
			t.Errorf("help missing %q", s)
		}
	}
	assertFits(t, "help", m.View(), 120, 40)
	m, _ = press(m, "j")
	if m.overlay != overlayHelp || m.fileCursor != 0 {
		t.Fatal("keys under the help overlay must not reach the panes")
	}
	m, _ = press(m, "?")
	if m.overlay != overlayNone {
		t.Fatal("? should close help")
	}
	m, _ = press(m, "?", "esc")
	if m.overlay != overlayNone {
		t.Fatal("esc should close help")
	}
	_, cmd := press(m, "q")
	if cmd == nil {
		t.Fatal("q should quit")
	}
}

func TestEditorReturnReloads(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "3", "G")
	_, cmd := press(m, "e")
	if cmd == nil {
		t.Fatal("e should launch the editor command")
	}
	if err := os.WriteFile(m.current.path, []byte("# Edited\n\n## Only\n\nshort now\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	next, cmd := m.Update(editorDoneMsg{})
	m = next.(model)
	if cmd == nil || !m.rendering {
		t.Fatal("returning from the editor should re-read the file")
	}
	m = inject(t, m)
	if len(m.doc.headings) != 2 || m.doc.headings[0].title != "Edited" {
		t.Fatalf("headings after edit = %+v", m.doc.headings)
	}
	if m.toast == nil || !strings.Contains(m.toast.text, "reloaded") {
		t.Fatal("expected a reloaded toast")
	}
}

func TestReloadPicksUpNewFiles(t *testing.T) {
	root := fixture(t)
	m := newTestModel(t, root, 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	if err := os.WriteFile(filepath.Join(root, "NEW.md"), []byte("# New\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, cmd := press(m, "r")
	if cmd == nil || len(m.files) != 6 {
		t.Fatalf("r should rescan: %d files", len(m.files))
	}
	if m.current == nil || m.current.rel != "docs/TECHNICAL_SPEC.md" || !m.rendering {
		t.Fatal("current file should be kept and re-read")
	}
	if m.toast == nil || !strings.Contains(m.toast.text, "6 docs") {
		t.Fatalf("toast = %+v", m.toast)
	}
	m = inject(t, m)
	if !strings.Contains(stripANSI(m.View()), "NEW.md") {
		t.Fatal("new file should appear in the tree")
	}
}

func TestHeaderShowsContext(t *testing.T) {
	root := fixture(t)
	m := newTestModel(t, root, 120, 40)
	m = openRel(t, m, "docs/crlf.md")
	head := plainLines(m.View())[0]
	for _, s := range []string{"docscope", "5 docs", "docs/crlf.md", "%"} {
		if !strings.Contains(head, s) {
			t.Errorf("header %q missing %q", head, s)
		}
	}
	if visibleWidth(head) != 120 {
		t.Errorf("header width %d", visibleWidth(head))
	}
}

func TestSearchHighlightKeepsLinesInsideReader(t *testing.T) {
	m := newTestModel(t, fixture(t), 80, 24)
	m = openRel(t, m, "docs/guide/cjk.md")
	m, _ = press(m, "3", "/")
	m = typeText(m, "日本語")
	if len(m.matches) < 2 {
		t.Fatalf("expected CJK matches, got %d", len(m.matches))
	}
	assertFits(t, "cjk search", m.View(), 80, 24)
	w, _ := m.readerInner()
	for i, l := range strings.Split(m.vp.View(), "\n") {
		if visibleWidth(l) > w {
			t.Errorf("highlighted line %d is %d wide (> %d)", i, visibleWidth(l), w)
		}
	}
}
