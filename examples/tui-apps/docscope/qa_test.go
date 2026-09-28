package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// ---- QA fixtures ---------------------------------------------------------------

func writeFile(t *testing.T, root, rel, content string) string {
	t.Helper()
	p := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func wideDoc() string {
	var b strings.Builder
	b.WriteString("# Wide\n\n## Table\n\n|")
	for i := 0; i < 30; i++ {
		fmt.Fprintf(&b, " col%02d_xxxx |", i)
	}
	b.WriteString("\n|")
	for i := 0; i < 30; i++ {
		b.WriteString("------------|")
	}
	b.WriteString("\n|")
	for i := 0; i < 30; i++ {
		fmt.Fprintf(&b, " val%02d_yyyy |", i)
	}
	b.WriteString("\n\n## Code\n\n```\n" + strings.Repeat("x", 400) + "\n```\n\n## Link\n\nSee https://example.com/" + strings.Repeat("a", 500) + " now.\n")
	return b.String()
}

func qaSizes() [][2]int {
	return [][2]int{{80, 24}, {100, 30}, {120, 40}, {200, 60}, {79, 23}, {90, 25}, {140, 35}, {60, 20}, {40, 12}}
}

func tick(m model, n int) model {
	for i := 0; i < n; i++ {
		next, _ := m.Update(animTickMsg{})
		m = next.(model)
	}
	return m
}

func mouse(m model, x, y int, b tea.MouseButton) model {
	next, _ := m.Update(tea.MouseMsg{X: x, Y: y, Action: tea.MouseActionPress, Button: b})
	return next.(model)
}

func openRelSlow(t *testing.T, m model, rel string) (model, tea.Cmd) {
	t.Helper()
	for _, f := range m.files {
		if f.rel == rel {
			return m, m.openNode(f, false)
		}
	}
	t.Fatalf("file %s not in tree", rel)
	return m, nil
}

// ---- 1. layout ------------------------------------------------------------------

func TestQALayoutEveryStateEverySize(t *testing.T) {
	root := fixture(t)
	writeFile(t, root, "wide.md", wideDoc())
	many := t.TempDir()
	for i := 0; i < 200; i++ {
		writeFile(t, many, fmt.Sprintf("f%03d.md", i), fmt.Sprintf("# F %d\n\nbody\n", i))
	}
	for _, sz := range qaSizes() {
		w, h := sz[0], sz[1]
		name := func(s string) string { return fmt.Sprintf("%dx%d %s", w, h, s) }
		m := newTestModel(t, root, w, h)
		m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
		for _, k := range []string{"1", "2", "3"} {
			f, _ := press(m, k)
			assertFits(t, name("focus "+k+" mid-anim"), tick(f, 2).View(), w, h)
			assertFits(t, name("focus "+k), settle(f).View(), w, h)
		}
		hidden, _ := press(m, "b")
		assertFits(t, name("collapsed rendering"), hidden.View(), w, h)
		hidden = inject(t, hidden)
		assertFits(t, name("collapsed"), hidden.View(), w, h)

		fz, _ := press(m, "ctrl+p")
		fz = typeText(fz, "zzqqzz")
		if len(fz.finder.results) != 0 {
			t.Fatalf("expected 0 results")
		}
		assertFits(t, name("finder 0"), fz.View(), w, h)
		f1, _ := press(m, "ctrl+p")
		f1 = typeText(f1, "spec")
		assertFits(t, name("finder 1"), f1.View(), w, h)
		mm := newTestModel(t, many, w, h)
		f200, _ := press(mm, "ctrl+p")
		if len(f200.finder.results) != 200 {
			t.Fatalf("expected 200 results, got %d", len(f200.finder.results))
		}
		assertFits(t, name("finder 200"), f200.View(), w, h)
		f200, _ = press(f200, "up") // cursor clamp
		for i := 0; i < 150; i++ {
			f200, _ = press(f200, "down")
		}
		assertFits(t, name("finder 200 scrolled"), f200.View(), w, h)

		s, _ := press(m, "3", "/")
		s = typeText(s, "widget")
		assertFits(t, name("search typing"), s.View(), w, h)
		s, _ = press(s, "enter")
		assertFits(t, name("search active mid-anim"), tick(s, 1).View(), w, h)
		s = settle(s)
		s, _ = press(s, "n", "n", "p")
		assertFits(t, name("search active"), s.View(), w, h)

		hp, _ := press(m, "?")
		assertFits(t, name("help"), hp.View(), w, h)

		tst, _ := m.Update(copiedMsg{})
		assertFits(t, name("toast"), tst.(model).View(), w, h)
		tst, _ = m.Update(copiedMsg{err: fmt.Errorf("a very long error message that goes on and on and on and on and on")})
		assertFits(t, name("toast err"), tst.(model).View(), w, h)

		j, _ := press(m, "3", "]")
		assertFits(t, name("jump mid-anim"), tick(j, 1).View(), w, h)
		j, _ = press(settle(j), "]", "]")
		assertFits(t, name("jump 2"), tick(j, 3).View(), w, h)

		wide := openRel(t, m, "wide.md")
		assertFits(t, name("wide"), wide.View(), w, h)
		wide, _ = press(wide, "3", "G")
		assertFits(t, name("wide bottom"), wide.View(), w, h)
		wide, _ = press(wide, "/")
		wide = typeText(wide, "xxxx")
		assertFits(t, name("wide search"), wide.View(), w, h)
		wide, _ = press(wide, "esc", "b")
		wide = inject(t, wide)
		assertFits(t, name("wide collapsed"), wide.View(), w, h)

		rendering, _ := openRelSlow(t, m, "docs/guide/cjk.md")
		assertFits(t, name("rendering"), rendering.View(), w, h)
		e, _ := rendering.Update(renderDoneMsg{seq: rendering.renderSeq, path: rendering.current.path, err: fmt.Errorf("boom boom boom boom boom boom boom boom boom boom boom boom")})
		assertFits(t, name("render error"), e.(model).View(), w, h)
	}
}

func TestQAAccordionFollowsResize(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m = settle(m)
	m = resize(m, 200, 60)
	m = settle(m)
	l := m.layout()
	if l.filesH != l.bodyH-collapsedPaneH {
		t.Fatalf("after growing the terminal the Files pane should fill the sidebar: filesH=%d bodyH=%d", l.filesH, l.bodyH)
	}
	m, _ = press(m, "2")
	m = settle(m)
	m = resize(m, 80, 24)
	m = settle(m)
	l = m.layout()
	if l.filesH != collapsedPaneH {
		t.Fatalf("after shrinking, Files should stay collapsed: %+v", l)
	}
	assertFits(t, "80x24 after shrink", m.View(), 80, 24)
}

// ---- 2. content ---------------------------------------------------------------------

func TestQAContentEdgeCases(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "fence.md", "```go\nfunc main() {}\n```\n")
	writeFile(t, root, "fence-heading.md", "# Real\n\n```\n# not a heading\n## also not\n```\n\n~~~\n### nope\n~~~\n\n## Real Two\n")
	writeFile(t, root, "dup.md", "# Doc\n\n## Setup\n\nfirst setup\n\n## Usage\n\n## Setup\n\nsecond setup\n\n## Usage\n\nend\n")
	writeFile(t, root, "inline.md", "# Use `docscope` now\n\n## See [the docs](https://x.y/z)\n\n## Ship 🚀 it\n\n### **bold** _em_\n")
	writeFile(t, root, "eof.md", "text\n\n# Last")
	writeFile(t, root, "setext.md", "Title\n=====\n\nbody\n\nSub\n---\n\nmore\n")
	writeFile(t, root, "cr.md", "# CR Doc\rline one\r\r## Second\rline two")
	writeFile(t, root, "empty.md", "")
	writeFile(t, root, "ws.md", "   \n\n\t\n")
	writeFile(t, root, "hashes.md", "#\n#\t\n####### seven\n#hashtag\n")
	m := newTestModel(t, root, 100, 30)

	m = openRel(t, m, "fence.md")
	assertFits(t, "fence only", m.View(), 100, 30)
	if len(m.doc.headings) != 0 {
		t.Errorf("fence only: headings %+v", m.doc.headings)
	}

	m = openRel(t, m, "fence-heading.md")
	if len(m.doc.headings) != 2 || m.doc.headings[0].title != "Real" || m.doc.headings[1].title != "Real Two" {
		t.Errorf("headings in code fences leaked into the outline: %+v", m.doc.headings)
	}

	m = openRel(t, m, "dup.md")
	hs := m.doc.headings
	if len(hs) != 5 {
		t.Fatalf("dup headings = %+v", hs)
	}
	for i := 1; i < len(hs); i++ {
		if hs[i].line <= hs[i-1].line {
			t.Errorf("duplicate heading %d (%q) mapped to line %d, not after %d", i, hs[i].title, hs[i].line, hs[i-1].line)
		}
	}
	joined := strings.Join(m.doc.plain, "\n")
	if idx := strings.Index(joined, "second setup"); idx > 0 {
		lineOfSecond := strings.Count(joined[:idx], "\n")
		if hs[3].line >= lineOfSecond || hs[3].line <= hs[1].line {
			t.Errorf("second 'Setup' should sit just before 'second setup' (line %d), got %d", lineOfSecond, hs[3].line)
		}
	}

	m = openRel(t, m, "inline.md")
	hs = m.doc.headings
	if len(hs) != 4 {
		t.Fatalf("inline headings = %+v", hs)
	}
	want := []string{"Use docscope now", "See the docs", "Ship 🚀 it", "bold em"}
	for i, h := range hs {
		if h.title != want[i] {
			t.Errorf("heading %d title %q, want %q", i, h.title, want[i])
		}
		if !strings.Contains(squash(m.doc.plain[h.line]), squash(firstCells(h.title, 4))) {
			t.Errorf("heading %q maps to line %d %q", h.title, h.line, m.doc.plain[h.line])
		}
	}

	m = openRel(t, m, "eof.md")
	if len(m.doc.headings) != 1 || m.doc.headings[0].title != "Last" {
		t.Errorf("heading at EOF without newline: %+v", m.doc.headings)
	}

	m = openRel(t, m, "setext.md")
	if len(m.doc.headings) != 2 || m.doc.headings[0].level != 1 || m.doc.headings[1].level != 2 || m.doc.headings[1].title != "Sub" {
		t.Errorf("setext headings: %+v", m.doc.headings)
	}

	m = openRel(t, m, "cr.md")
	for i, l := range m.doc.lines {
		if strings.ContainsAny(l, "\r") {
			t.Errorf("CR-only file: rendered line %d contains a carriage return: %q", i, l)
		}
	}
	if len(m.doc.headings) != 2 {
		t.Errorf("CR-only file: headings = %+v", m.doc.headings)
	}
	assertFits(t, "cr", m.View(), 100, 30)

	for _, rel := range []string{"empty.md", "ws.md", "hashes.md"} {
		m = openRel(t, m, rel)
		assertFits(t, rel, m.View(), 100, 30)
		if len(m.doc.headings) != 0 {
			t.Errorf("%s: unexpected headings %+v", rel, m.doc.headings)
		}
		m, _ = press(m, "3", "]", "[", "j", "G", "g")
		m, _ = press(m, "2", "enter", "j", "k")
		m = settle(m)
	}
}

func TestQAOneMegabyteLine(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "big.md", strings.Repeat("word ", 200*1024)+"\n")
	m := newTestModel(t, root, 120, 40)
	start := time.Now()
	raw, err := readDoc(m.current.path)
	if err != nil {
		t.Fatal(err)
	}
	w, _ := m.readerInner()
	doc, err := renderMarkdown(raw, w)
	if err != nil {
		t.Fatal(err)
	}
	d := time.Since(start)
	t.Logf("1MB single line rendered in %v (%d lines)", d, len(doc.lines))
	if d > 10*time.Second {
		t.Errorf("1MB line took %v", d)
	}
	for i, l := range doc.plain {
		if visibleWidth(l) > w {
			t.Fatalf("line %d wider than %d", i, w)
		}
	}
	next, _ := m.Update(renderDoneMsg{seq: m.renderSeq, path: m.current.path, raw: raw, width: w, doc: doc})
	m = next.(model)
	m, _ = press(m, "3", "/")
	start = time.Now()
	m = typeText(m, "word")
	t.Logf("search over 1MB in %v (%d matches)", time.Since(start), len(m.matches))
	assertFits(t, "1MB", m.View(), 120, 40)
}

func TestQABinaryGarbage(t *testing.T) {
	root := t.TempDir()
	r := rand.New(rand.NewSource(42))
	garbage := make([]byte, 64*1024)
	r.Read(garbage)
	writeFile(t, root, "bin.md", string(garbage))
	writeFile(t, root, "nul.md", "# A\x00B\n\n\x00\x00\x01\x02\x1b[31mred\x1b[0m \x08\x07\n\n| \x00 | \xff |\n|--|--|\n")
	writeFile(t, root, "bad-utf8.md", "# \xff\xfe Title\n\n\xc0\xaf text \xed\xa0\x80\n\n```\n\xff\n```\n")
	m := newTestModel(t, root, 100, 30)
	for _, f := range []string{"bin.md", "nul.md", "bad-utf8.md"} {
		m = openRel(t, m, f)
		assertFits(t, f, m.View(), 100, 30)
		m, _ = press(m, "3", "G", "]", "[", "/")
		m = typeText(m, "a")
		m, _ = press(m, "enter", "n", "p", "esc")
		assertFits(t, f+" searched", m.View(), 100, 30)
	}
}

func TestQATreeEdgeCases(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a/b/c.md", "# C\n")
	if err := os.Symlink(filepath.Join(root, "a"), filepath.Join(root, "a", "b", "loop")); err != nil {
		t.Skip("symlinks unsupported")
	}
	if err := os.Symlink(filepath.Join(root, "a"), filepath.Join(root, "dirlink.md")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(root, "missing.md"), filepath.Join(root, "dangling.md")); err != nil {
		t.Fatal(err)
	}
	done := make(chan model, 1)
	go func() { done <- newTestModel(t, root, 100, 30) }()
	var m model
	select {
	case m = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("tree walk did not terminate on a symlink loop")
	}
	if len(m.files) < 1 {
		t.Fatalf("files = %v", fileRels(m))
	}
	for _, rel := range fileRels(m) {
		if strings.Count(rel, "loop") > 1 {
			t.Fatalf("symlink loop was followed: %s", rel)
		}
	}
	for _, rel := range []string{"dirlink.md", "dangling.md"} {
		for _, f := range m.files {
			if f.rel == rel {
				cmd := m.openNode(f, false)
				msg := cmd()
				if b, ok := msg.(tea.BatchMsg); ok {
					for _, c := range b {
						if c != nil {
							c()
						}
					}
				}
			}
		}
	}

	// root that is a file
	fileRoot := writeFile(t, root, "solo.md", "# Solo\n")
	fm := newTestModel(t, fileRoot, 100, 30)
	assertFits(t, "file root", fm.View(), 100, 30)
	fm, _ = press(fm, "r", "j", "?", "ctrl+p")
	assertFits(t, "file root keys", fm.View(), 100, 30)

	// root that does not exist
	nm := newTestModel(t, filepath.Join(root, "nope", "nada"), 100, 30)
	assertFits(t, "missing root", nm.View(), 100, 30)
	nm, _ = press(nm, "r", "j", "?", "ctrl+p", "/")
	assertFits(t, "missing root keys", nm.View(), 100, 30)
	nm = resize(nm, 40, 12)
	assertFits(t, "missing root 40x12", nm.View(), 40, 12)

	// symlinked root must still find the files behind it
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(root, link); err != nil {
		t.Fatal(err)
	}
	lm := newTestModel(t, link, 100, 30)
	if len(lm.files) == 0 {
		t.Errorf("a symlinked root directory finds no files: %v", fileRels(lm))
	}

	// unreadable dir
	if os.Geteuid() != 0 {
		ur := t.TempDir()
		writeFile(t, ur, "ok.md", "# ok\n")
		writeFile(t, ur, "locked/secret.md", "# secret\n")
		if err := os.Chmod(filepath.Join(ur, "locked"), 0o000); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(filepath.Join(ur, "locked"), 0o755) })
		um := newTestModel(t, ur, 100, 30)
		assertFits(t, "unreadable subdir", um.View(), 100, 30)
		if len(um.files) != 1 {
			t.Errorf("unreadable subdir should be skipped: %v", fileRels(um))
		}
		if err := os.Chmod(ur, 0o000); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(ur, 0o755) })
		um2 := newTestModel(t, ur, 100, 30)
		assertFits(t, "unreadable root", um2.View(), 100, 30)
	}
}

func TestQAThreeThousandFilesFast(t *testing.T) {
	root := t.TempDir()
	for i := 0; i < 3000; i++ {
		writeFile(t, root, fmt.Sprintf("d%02d/file_%04d.md", i%40, i), fmt.Sprintf("# File %d\n", i))
	}
	start := time.Now()
	m := newTestModel(t, root, 120, 40)
	build := time.Since(start)
	if len(m.files) != 3000 {
		t.Fatalf("files = %d", len(m.files))
	}
	assertFits(t, "3000 files", m.View(), 120, 40)
	m, _ = press(m, "ctrl+p")
	m = typeText(m, "file_29")
	m, _ = press(m, "enter")
	m, _ = press(m, "G", "g", "1", "G")
	total := time.Since(start)
	t.Logf("build %v total %v", build, total)
	if total > 2*time.Second {
		t.Fatalf("3000 files: tree+finder took %v", total)
	}
	assertFits(t, "3000 files after finder", m.View(), 120, 40)
}

// ---- 3. search ------------------------------------------------------------------------

func TestQASearchRobustness(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "s.md", "# Search Me\n\nparen (x) bracket [y] backslash \\z and m and 38;5;178 stuff\n\nUPPER lower Mixed\n\n日本語 abc 日本語 def 中文\n\n```\nesc \\x1b[31m in code\n```\n")
	m := newTestModel(t, root, 100, 30)
	m, _ = press(m, "3", "/")
	for _, q := range []string{"(", "[", "\\", "(x)", "[y]", "\\z", "38;5;178", "m", "\x1b", "["} {
		m, _ = press(m, "esc", "/")
		m = typeText(m, q)
		assertFits(t, "query "+q, m.View(), 100, 30)
		w, _ := m.readerInner()
		for i, l := range strings.Split(m.vp.View(), "\n") {
			if visibleWidth(l) > w {
				t.Errorf("query %q: line %d overflows", q, i)
			}
		}
		if q == "m" {
			// no match may live inside an escape sequence; every match must be a
			// visible 'm' in the plain text
			for _, mp := range m.matches {
				r := []rune(m.doc.plain[mp.line])
				if mp.end > len(r) || strings.ToLower(string(r[mp.start:mp.end])) != "m" {
					t.Errorf("match %+v does not point at a visible m", mp)
				}
			}
			if len(m.matches) == 0 {
				t.Error("expected matches for m")
			}
		}
		if q == "38;5;178" && len(m.matches) != 1 {
			t.Errorf("38;5;178 should match once in plain text, got %d", len(m.matches))
		}
		if q == "\x1b" && len(m.matches) != 0 {
			t.Errorf("ESC must never match: %+v", m.matches)
		}
	}
	m, _ = press(m, "esc", "/")
	m = typeText(m, "MIXED")
	if len(m.matches) != 1 {
		t.Errorf("case-insensitive: %d", len(m.matches))
	}
	m, _ = press(m, "esc", "/")
	m = typeText(m, "日本語")
	if len(m.matches) != 2 {
		t.Fatalf("cjk matches %d", len(m.matches))
	}
	for _, mp := range m.matches {
		r := []rune(m.doc.plain[mp.line])
		if string(r[mp.start:mp.end]) != "日本語" {
			t.Errorf("CJK offsets not rune-correct: %+v -> %q", mp, string(r[mp.start:mp.end]))
		}
	}
	hl := highlightLine(m.doc.plain[m.matches[0].line], m.matches, m.matches[0])
	if visibleWidth(hl) != visibleWidth(m.doc.plain[m.matches[0].line]) {
		t.Errorf("highlight changed the visible width")
	}
	m, _ = press(m, "enter")
	if m.matchIdx != 0 {
		t.Fatal("idx")
	}
	m, _ = press(m, "p")
	if m.matchIdx != 1 {
		t.Errorf("p should wrap to the last match, got %d", m.matchIdx)
	}
	m, _ = press(m, "n")
	if m.matchIdx != 0 {
		t.Errorf("n should wrap to the first match, got %d", m.matchIdx)
	}
	// resize while a search is active
	m = resize(m, 60, 20)
	m = inject(t, m)
	if len(m.matches) != 2 {
		t.Errorf("matches after resize: %d", len(m.matches))
	}
	for _, mp := range m.matches {
		if mp.line >= len(m.doc.plain) {
			t.Fatalf("match beyond doc: %+v", mp)
		}
	}
	assertFits(t, "search after resize", m.View(), 60, 20)
	m, _ = press(m, "b")
	m = inject(t, m)
	if len(m.matches) != 2 {
		t.Errorf("matches after sidebar toggle: %d", len(m.matches))
	}
	assertFits(t, "search after toggle", m.View(), 60, 20)
	m, _ = press(m, "n", "n", "p")
	assertFits(t, "search nav after toggle", m.View(), 60, 20)
}

// ---- 4. state machine ----------------------------------------------------------------

func TestQAStaleRenderKeepsRightDoc(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, cmdA := openRelSlow(t, m, "examples/big.md")
	seqA := m.renderSeq
	pathA := m.current.path
	// keys while the render is in flight
	m, _ = press(m, "3", "j", "G", "]", "[", "/")
	m = typeText(m, "x")
	m, _ = press(m, "esc", "2", "enter", "1")
	m = settle(m)
	assertFits(t, "in flight", m.View(), 120, 40)
	if strings.Contains(stripANSI(m.View()), "Outline · 6") {
		t.Error("outline still shows the previous document's headings while the new one renders")
	}
	m, cmdB := openRelSlow(t, m, "docs/guide/cjk.md")
	seqB := m.renderSeq
	if seqB == seqA {
		t.Fatal("seq did not advance")
	}
	// resize mid-flight on top
	m = resize(m, 100, 30)
	seqC := m.renderSeq
	// A finishes late
	w, _ := m.readerInner()
	next, _ := m.Update(renderDoneMsg{seq: seqA, path: pathA, raw: "# A", width: w, doc: renderedDoc{width: w, lines: []string{"A"}, plain: []string{"A"}}})
	m = next.(model)
	if m.hasDoc || !m.rendering {
		t.Fatal("stale A result must be discarded")
	}
	next, _ = m.Update(renderDoneMsg{seq: seqB, path: m.current.path, raw: "# B", width: w, doc: renderedDoc{width: w, lines: []string{"B"}, plain: []string{"B"}}})
	m = next.(model)
	if seqC != seqB && m.hasDoc {
		t.Fatal("result for the pre-resize width must be discarded")
	}
	m = inject(t, m)
	if !m.hasDoc || m.current.rel != "docs/guide/cjk.md" || len(m.doc.headings) != 3 {
		t.Fatalf("wrong doc applied: %v %d", m.current.rel, len(m.doc.headings))
	}
	_ = cmdA
	_ = cmdB
}

func TestQAEscAndQuitInsideOverlays(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m, _ = press(m, "ctrl+p")
	m, cmd := press(m, "q")
	if cmd != nil && isQuit(cmd) {
		t.Fatal("q inside the finder quit the app")
	}
	if m.finder.input.Value() != "q" {
		t.Fatalf("q should be typed into the finder, got %q", m.finder.input.Value())
	}
	m, cmd = press(m, "esc")
	if m.overlay != overlayNone || isQuit(cmd) {
		t.Fatal("esc in finder")
	}
	m, _ = press(m, "ctrl+p", "enter")
	if m.overlay != overlayNone {
		t.Fatal("enter on the finder with results should close it")
	}
	m, _ = press(m, "ctrl+p")
	m = typeText(m, "zzzzz")
	m, cmd = press(m, "enter")
	if m.overlay != overlayNone || isQuit(cmd) {
		t.Fatal("enter on an empty finder must just close it")
	}
	m, _ = press(m, "3", "/")
	m, cmd = press(m, "q")
	if isQuit(cmd) || m.search.Value() != "q" {
		t.Fatal("q inside search quit or was not typed")
	}
	m, cmd = press(m, "esc")
	if isQuit(cmd) || m.searchState != searchOff {
		t.Fatal("esc in search")
	}
	m, _ = press(m, "?")
	m, cmd = press(m, "q")
	if isQuit(cmd) || m.overlay != overlayNone {
		t.Fatal("q in help should close help, not quit")
	}
	m, cmd = press(m, "ctrl+p", "ctrl+c")
	if !isQuit(cmd) {
		t.Fatal("ctrl+c must always quit")
	}
}

func isQuit(cmd tea.Cmd) bool {
	if cmd == nil {
		return false
	}
	_, ok := cmd().(tea.QuitMsg)
	return ok
}

func TestQAEditorAndReloadEdgeCases(t *testing.T) {
	root := fixture(t)
	m := newTestModel(t, root, 120, 40)
	m = openRel(t, m, "docs/crlf.md")
	t.Setenv("EDITOR", "")
	_, cmd := press(m, "e")
	if cmd == nil {
		t.Fatal("e with unset EDITOR should still produce a command (vi)")
	}
	t.Setenv("EDITOR", "  ")
	_, cmd = press(m, "e")
	if cmd == nil {
		t.Fatal("e with blank EDITOR")
	}
	if err := os.Remove(m.current.path); err != nil {
		t.Fatal(err)
	}
	_, cmd = press(m, "e")
	if cmd == nil {
		t.Fatal("e on a deleted file should still try")
	}
	next, _ := m.Update(editorDoneMsg{err: fmt.Errorf("exit 1")})
	m = next.(model)
	msg := cmd2msg(m, "docs/crlf.md")
	next, _ = m.Update(msg)
	m = next.(model)
	if m.renderErr == nil {
		t.Fatal("deleted file after edit should show an error")
	}
	assertFits(t, "deleted after edit", m.View(), 120, 40)
	m, _ = press(m, "3", "j", "G", "]", "[", "/")
	m = typeText(m, "x")
	m, _ = press(m, "esc")
	assertFits(t, "keys on error doc", m.View(), 120, 40)

	// r when the current file vanished
	m, cmd = press(m, "r")
	if cmd == nil {
		t.Fatal("r")
	}
	assertFits(t, "reload vanished", m.View(), 120, 40)
	if len(m.files) != 4 {
		t.Fatalf("files after vanish: %v", fileRels(m))
	}
	for _, msg := range runBatch(cmd) {
		next, _ = m.Update(msg)
		m = next.(model)
	}
	assertFits(t, "reload vanished applied", m.View(), 120, 40)
	m, _ = press(m, "1", "g", "j", "j", "enter")
	if m.current == nil || m.current.rel != "docs/guide/cjk.md" {
		t.Fatalf("expected cjk.md opened, got %v", m.current)
	}
	m = inject(t, m)
	assertFits(t, "open after vanish", m.View(), 120, 40)

	// everything vanishes
	if err := os.RemoveAll(filepath.Join(root, "docs")); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, "examples")); err != nil {
		t.Fatal(err)
	}
	m, cmd = press(m, "r")
	assertFits(t, "all vanished", m.View(), 120, 40)
	for _, msg := range runBatch(cmd) {
		next, _ = m.Update(msg)
		m = next.(model)
	}
	assertFits(t, "all vanished applied", m.View(), 120, 40)
	m, _ = press(m, "j", "enter", "?", "/", "ctrl+p", "3", "]", "y", "e")
	assertFits(t, "all vanished keys", m.View(), 120, 40)
	m = mouse(m, 10, 10, tea.MouseButtonLeft)
	m = mouse(m, 10, 10, tea.MouseButtonWheelDown)
	assertFits(t, "all vanished mouse", m.View(), 120, 40)
}

// cmd2msg renders the model's current file (which may fail) and returns the msg.
func cmd2msg(m model, rel string) tea.Msg {
	w, _ := m.readerInner()
	return openFileCmd(m.current.path, w, m.renderSeq)()
}

func runBatch(cmd tea.Cmd) []tea.Msg {
	if cmd == nil {
		return nil
	}
	msg := cmd()
	if b, ok := msg.(tea.BatchMsg); ok {
		var out []tea.Msg
		for _, c := range b {
			if c != nil {
				if r, ok := c().(renderDoneMsg); ok {
					out = append(out, r)
				}
			}
		}
		return out
	}
	if r, ok := msg.(renderDoneMsg); ok {
		return []tea.Msg{r}
	}
	return nil
}

func TestQAMouseEdgeCases(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	// wheel before any doc is loaded / while rendering
	fresh := resize(newModel(fixture(t)), 120, 40)
	fresh = mouse(fresh, 100, 10, tea.MouseButtonWheelDown)
	fresh = mouse(fresh, 100, 10, tea.MouseButtonWheelUp)
	fresh = mouse(fresh, 100, 10, tea.MouseButtonLeft)
	assertFits(t, "wheel before doc", fresh.View(), 120, 40)

	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "2")
	m = settle(m)
	l := m.layout()
	if l.filesH != collapsedPaneH {
		t.Fatal("files should be collapsed")
	}
	cur := m.current
	off := m.vp.YOffset
	// click on the collapsed Files pane's single content row
	m = mouse(m, 5, 2, tea.MouseButtonLeft)
	if m.focus != paneFiles {
		t.Fatal("click on collapsed pane should focus it")
	}
	if m.current != cur || m.rendering || m.vp.YOffset != off {
		t.Errorf("clicking the collapsed Files pane must not activate an invisible row (rendering=%v)", m.rendering)
	}
	if len(m.rows) != 8 {
		t.Errorf("clicking the collapsed Files pane toggled an invisible directory: %v", rowNames(m))
	}
	m = settle(m)
	// now the Outline is collapsed: clicking its summary line must not scroll
	m, _ = press(m, "3", "j", "j", "j")
	off = m.vp.YOffset
	m = mouse(m, 5, 1+m.layout().filesH+1, tea.MouseButtonLeft)
	if m.focus != paneOutline {
		t.Fatal("click on collapsed outline should focus it")
	}
	if m.scrollAnimating || m.vp.YOffset != off {
		t.Errorf("clicking the collapsed Outline pane must not jump to an invisible heading")
	}
	m = settle(m)
	// outside all panes: header, progress bar, status bar, beyond width
	for _, pt := range [][2]int{{5, 0}, {5, 38}, {5, 39}, {200, 10}, {-1, 10}, {5, 100}, {5, -5}} {
		before := m
		m = mouse(m, pt[0], pt[1], tea.MouseButtonLeft)
		m = mouse(m, pt[0], pt[1], tea.MouseButtonWheelDown)
		if m.focus != before.focus || m.vp.YOffset != before.vp.YOffset {
			t.Errorf("click at %v changed state", pt)
		}
	}
	// wheel over collapsed outline / right/middle buttons / release actions
	m = mouse(m, 5, 38, tea.MouseButtonRight)
	m = mouse(m, 60, 10, tea.MouseButtonMiddle)
	next, _ := m.Update(tea.MouseMsg{X: 60, Y: 10, Action: tea.MouseActionRelease, Button: tea.MouseButtonLeft})
	m = next.(model)
	next, _ = m.Update(tea.MouseMsg{X: 60, Y: 10, Action: tea.MouseActionMotion})
	m = next.(model)
	assertFits(t, "mouse misc", m.View(), 120, 40)
	// clicks at every cell must never panic
	for y := -1; y <= 41; y += 3 {
		for x := -1; x <= 121; x += 7 {
			m = mouse(m, x, y, tea.MouseButtonLeft)
			m = mouse(m, x, y, tea.MouseButtonWheelUp)
		}
	}
	m = settle(m)
	assertFits(t, "mouse sweep", m.View(), 120, 40)
	// overlays swallow the mouse
	m, _ = press(m, "ctrl+p")
	m = mouse(m, 5, 4, tea.MouseButtonLeft)
	if m.overlay != overlayFinder {
		t.Fatal("mouse should not close the finder")
	}
}

func TestQAMouseClickTicksModel(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, _ = press(m, "2")
	m = settle(m)
	m, _ = press(m, "3")
	if m.ticking {
		t.Fatal("precondition")
	}
	// Files pane is collapsed; click its header row to focus it: accordion starts
	next, cmd := m.Update(tea.MouseMsg{X: 5, Y: 1, Action: tea.MouseActionPress, Button: tea.MouseButtonLeft})
	m = next.(model)
	if !m.sideAnimating || cmd == nil {
		t.Fatalf("click should start the accordion: anim=%v cmd=%v", m.sideAnimating, cmd != nil)
	}
	if !m.ticking {
		t.Fatal("model returned from a click that started a tick chain must record ticking=true (otherwise a second chain doubles the animation speed)")
	}
	// the same via a row click
	m = settle(m)
	m, _ = press(m, "2")
	m = settle(m)
	next, cmd = m.Update(tea.MouseMsg{X: 5, Y: 2, Action: tea.MouseActionPress, Button: tea.MouseButtonLeft})
	m = next.(model)
	if cmd == nil || !m.sideAnimating {
		t.Fatal("row click should start the accordion")
	}
	if !m.ticking {
		t.Fatal("row click: ticking flag lost on the returned model")
	}
}

func TestQACopyPathDoesNotPanicWithoutTTY(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	_, cmd := press(m, "y")
	if cmd == nil {
		t.Fatal("y")
	}
	// without a tty the sequence falls back to stderr; keep test output clean
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer devnull.Close()
	stderr := os.Stderr
	os.Stderr = devnull
	defer func() { os.Stderr = stderr }()
	t.Setenv("TMUX", "")
	t.Setenv("TERM", "dumb")
	msg := cmd()
	if _, ok := msg.(copiedMsg); !ok {
		t.Fatalf("msg %T", msg)
	}
	t.Setenv("TMUX", "x")
	msg = cmd()
	if _, ok := msg.(copiedMsg); !ok {
		t.Fatalf("msg %T", msg)
	}
	t.Setenv("TMUX", "")
	t.Setenv("TERM", "screen")
	msg = cmd()
	if _, ok := msg.(copiedMsg); !ok {
		t.Fatalf("msg %T", msg)
	}
}

// ---- 5. animation ----------------------------------------------------------------------

func TestQAAnimationStopsAndYields(t *testing.T) {
	m := newTestModel(t, fixture(t), 120, 40)
	m = openRel(t, m, "docs/TECHNICAL_SPEC.md")
	m, cmd := press(m, "3", "]")
	if cmd == nil || !m.ticking {
		t.Fatal("] should start ticking")
	}
	var last tea.Cmd
	for i := 0; i < 500 && m.animating(); i++ {
		next, c := m.Update(animTickMsg{})
		m = next.(model)
		last = c
		if m.animating() && c == nil {
			t.Fatal("tick returned nil while animating")
		}
	}
	if m.animating() {
		t.Fatal("scroll animation never settled")
	}
	if last != nil {
		t.Fatal("after settling, the tick must return a nil cmd")
	}
	if m.ticking {
		t.Fatal("ticking flag stuck")
	}
	next, c := m.Update(animTickMsg{})
	if c != nil || next.(model).ticking {
		t.Fatal("a late tick after settling must not restart the loop")
	}
	m = next.(model)
	// accordion
	m, cmd = press(m, "2")
	if cmd == nil || !m.ticking {
		t.Fatal("2 should start ticking")
	}
	// pressing 2 again mid-animation must not double the chain
	m = tick(m, 2)
	_, cmd = press(m, "2")
	if cmd != nil {
		if _, ok := cmd().(animTickMsg); ok {
			t.Fatal("second key while ticking spawned a second tick chain")
		}
	}
	m = settle(m)
	if m.ticking {
		t.Fatal("ticking flag stuck after accordion")
	}

	// heading jump interrupted by the user: the user wins
	m, _ = press(m, "3", "g", "]", "]")
	m = tick(m, 1)
	if !m.scrollAnimating {
		t.Fatal("should be animating")
	}
	m, _ = press(m, "j")
	if m.scrollAnimating {
		t.Fatal("j must cancel the animation")
	}
	y := m.vp.YOffset
	m = tick(m, 5)
	if m.vp.YOffset != y {
		t.Fatalf("animation kept fighting the user: %d -> %d", y, m.vp.YOffset)
	}
	if m.ticking {
		t.Fatal("ticking after cancel")
	}
	m, _ = press(m, "]")
	m = tick(m, 1)
	m = mouse(m, 100, 10, tea.MouseButtonWheelDown)
	if m.scrollAnimating {
		t.Fatal("wheel must cancel the animation")
	}
	m, _ = press(m, "]")
	m = tick(m, 1)
	m, _ = press(m, "G")
	if m.scrollAnimating || m.vp.YOffset != m.maxYOffset() {
		t.Fatal("G must cancel the animation and land at the bottom")
	}
	m = tick(m, 5)
	if m.vp.YOffset != m.maxYOffset() {
		t.Fatal("animation moved the view after G")
	}
}

func TestQARenderRunsConcurrentlyWithView(t *testing.T) {
	root := fixture(t)
	m := newTestModel(t, root, 120, 40)
	m, cmd := openRelSlow(t, m, "docs/TECHNICAL_SPEC.md")
	done := make(chan tea.Msg, 4)
	msg := cmd()
	if b, ok := msg.(tea.BatchMsg); ok {
		for _, c := range b {
			c := c
			go func() { done <- c() }()
		}
	}
	for i := 0; i < 50; i++ {
		_ = m.View()
		m, _ = press(m, "j")
	}
	for i := 0; i < 2; i++ {
		select {
		case r := <-done:
			if rd, ok := r.(renderDoneMsg); ok {
				next, _ := m.Update(rd)
				m = next.(model)
			}
		case <-time.After(20 * time.Second):
			t.Fatal("render did not finish")
		}
	}
	if !m.hasDoc {
		t.Fatal("doc not applied")
	}
	assertFits(t, "concurrent", m.View(), 120, 40)
}
