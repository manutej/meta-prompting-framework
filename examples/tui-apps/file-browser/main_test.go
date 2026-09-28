package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ---------- fixture ----------

const (
	longASCIIName   = "long_name_" // repeated to 200+ chars below
	unicodeName     = "héllo wörld.txt"
	cjkName         = "日本語ファイル.txt"
	unreadableName  = "locked_dir"
	textContent     = "line one\nline two\n\tindented\nline four\n"
	bigFileSize     = 100 * 1024 // > maxPreviewBytes
	manyLinesCount  = 200
	manyEntriesName = "many"
)

type fixture struct {
	root        string
	longName    string
	longUnicode string
	hasLocked   bool
}

func mustWrite(t *testing.T, p string, data []byte) {
	t.Helper()
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
}

func mkFixture(t *testing.T) fixture {
	t.Helper()
	root := t.TempDir()
	fx := fixture{root: root}

	// nested dirs
	if err := os.MkdirAll(filepath.Join(root, "sub", "inner", "deep"), 0o755); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(root, "sub", "inner", "deep", "leaf.txt"), []byte("leaf\n"))
	mustWrite(t, filepath.Join(root, "sub", "note.txt"), []byte("note\n"))
	if err := os.Mkdir(filepath.Join(root, "zeta"), 0o755); err != nil {
		t.Fatal(err)
	}

	// files
	mustWrite(t, filepath.Join(root, "text.txt"), []byte(textContent))
	mustWrite(t, filepath.Join(root, "binary.bin"), []byte("PK\x03\x04\x00\x00\x01\x02\xff"))
	mustWrite(t, filepath.Join(root, "empty.txt"), nil)
	big := strings.Repeat("0123456789abcdef\n", bigFileSize/17+1)
	mustWrite(t, filepath.Join(root, "big.log"), []byte(big))
	var many strings.Builder
	for i := 0; i < manyLinesCount; i++ {
		many.WriteString("preview line ")
		many.WriteString(strings.Repeat("x", i%7))
		many.WriteString("\n")
	}
	mustWrite(t, filepath.Join(root, "manylines.txt"), []byte(many.String()))
	mustWrite(t, filepath.Join(root, "wide.txt"), []byte(strings.Repeat("w", 400)+"\n"))

	// unicode + very long names
	mustWrite(t, filepath.Join(root, unicodeName), []byte("unicode\n"))
	mustWrite(t, filepath.Join(root, cjkName), []byte("cjk\n"))
	fx.longName = strings.Repeat(longASCIIName, 22) + ".txt" // 224 chars
	mustWrite(t, filepath.Join(root, fx.longName), []byte("long\n"))
	fx.longUnicode = strings.Repeat("é", 120) + ".txt" // 240 bytes, 124 runes: > NAME_MAX if it were 200 runes
	mustWrite(t, filepath.Join(root, fx.longUnicode), []byte("long unicode\n"))

	// many entries dir for scroll tests
	manyDir := filepath.Join(root, manyEntriesName)
	if err := os.Mkdir(manyDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 60; i++ {
		mustWrite(t, filepath.Join(manyDir, "f"+padNum(i)+".txt"), []byte("x"))
	}

	// unreadable dir (root bypasses permissions)
	if os.Geteuid() != 0 {
		locked := filepath.Join(root, unreadableName)
		if err := os.Mkdir(locked, 0o755); err != nil {
			t.Fatal(err)
		}
		mustWrite(t, filepath.Join(locked, "secret.txt"), []byte("s"))
		if err := os.Chmod(locked, 0); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })
		fx.hasLocked = true
	}
	return fx
}

func padNum(i int) string {
	s := "00" + itoa(i)
	return s[len(s)-2:]
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

// ---------- drivers ----------

func newModel(t *testing.T, cwd string) model {
	t.Helper()
	m := initialModelAt(cwd)
	return m
}

// initialModelAt mirrors initialModel but pins cwd (no os.Getwd).
func initialModelAt(cwd string) model {
	m := initialModel()
	m.cwd = cwd
	m.loadDir()
	return m
}

func size(t *testing.T, m model, w, h int) model {
	t.Helper()
	mm, _ := m.Update(tea.WindowSizeMsg{Width: w, Height: h})
	return mm.(model)
}

func key(t *testing.T, m model, k string) (model, tea.Cmd) {
	t.Helper()
	var msg tea.KeyMsg
	switch k {
	case "esc":
		msg = tea.KeyMsg{Type: tea.KeyEsc}
	case "enter":
		msg = tea.KeyMsg{Type: tea.KeyEnter}
	case "tab":
		msg = tea.KeyMsg{Type: tea.KeyTab}
	case "backspace":
		msg = tea.KeyMsg{Type: tea.KeyBackspace}
	case "ctrl+d":
		msg = tea.KeyMsg{Type: tea.KeyCtrlD}
	case "ctrl+u":
		msg = tea.KeyMsg{Type: tea.KeyCtrlU}
	case "ctrl+c":
		msg = tea.KeyMsg{Type: tea.KeyCtrlC}
	case "up":
		msg = tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		msg = tea.KeyMsg{Type: tea.KeyDown}
	case "left":
		msg = tea.KeyMsg{Type: tea.KeyLeft}
	case "right":
		msg = tea.KeyMsg{Type: tea.KeyRight}
	default:
		msg = tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)}
	}
	if msg.String() != k {
		t.Fatalf("key helper: %q renders as %q", k, msg.String())
	}
	mm, cmd := m.Update(msg)
	return mm.(model), cmd
}

func keys(t *testing.T, m model, ks ...string) model {
	t.Helper()
	for _, k := range ks {
		m, _ = key(t, m, k)
	}
	return m
}

func typeText(t *testing.T, m model, s string) model {
	t.Helper()
	for _, r := range s {
		m, _ = key(t, m, string(r))
	}
	return m
}

func indexOf(m model, name string) int {
	for i, e := range m.filtered {
		if e.name == name {
			return i
		}
	}
	return -1
}

func goTo(t *testing.T, m model, name string) model {
	t.Helper()
	i := indexOf(m, name)
	if i < 0 {
		t.Fatalf("entry %q not in filtered list %v", name, names(m.filtered))
	}
	m = keys(t, m, "g")
	for m.cursor < i {
		m = keys(t, m, "j")
	}
	if m.filtered[m.cursor].name != name {
		t.Fatalf("goTo %q landed on %q", name, m.filtered[m.cursor].name)
	}
	return m
}

func names(es []entry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.name
	}
	return out
}

func stripANSI(s string) string {
	var b strings.Builder
	in := false
	for _, r := range s {
		switch {
		case r == 0x1b:
			in = true
		case in && r == 'm':
			in = false
		case !in:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// expectedLines is the layout invariant: header + bordered pane (h+2) + status + help.
func expectedLines(m model) int { return m.listHeight() + 5 }

func assertLayout(t *testing.T, m model, ctx string) string {
	t.Helper()
	v := m.View()
	if v == "" {
		t.Fatalf("%s: View() is empty", ctx)
	}
	if !utf8.ValidString(v) {
		t.Fatalf("%s: View() contains invalid UTF-8 (byte-sliced rune?)", ctx)
	}
	got := strings.Count(v, "\n") + 1
	if want := expectedLines(m); got != want {
		t.Errorf("%s: View() has %d lines, want %d (pane content wrapped and broke the layout)\n%s", ctx, got, want, v)
	}
	return v
}

// ---------- tests ----------

func TestFixtureListing(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	if len(m.filtered) == 0 || m.filtered[0].name != ".." {
		t.Fatalf("first entry should be '..', got %v", names(m.filtered))
	}
	// dirs before files, both case-insensitively sorted
	sawFile := false
	for _, e := range m.filtered[1:] {
		if e.isDir && sawFile {
			t.Fatalf("dir %q after a file: %v", e.name, names(m.filtered))
		}
		if !e.isDir {
			sawFile = true
		}
	}
	if m.err != "" {
		t.Fatalf("unexpected err %q", m.err)
	}
}

func TestViewNeverPanicsAtEverySize(t *testing.T) {
	fx := mkFixture(t)
	base := newModel(t, fx.root)
	for _, sz := range [][2]int{{100, 30}, {20, 5}, {300, 80}, {0, 0}, {1, 1}, {40, 10}} {
		m := size(t, base, sz[0], sz[1])
		ctx := "size " + itoa(sz[0]) + "x" + itoa(sz[1])
		assertLayout(t, m, ctx+" initial")
		// walk every entry, checking each render
		for i := 0; i < len(m.filtered)+2; i++ {
			assertLayout(t, m, ctx+" cursor "+itoa(m.cursor)+" on "+m.filtered[m.cursor].name)
			m = keys(t, m, "j")
		}
		m = keys(t, m, "tab")
		assertLayout(t, m, ctx+" preview focused")
		m = keys(t, m, "j", "j", "ctrl+d", "G", "g", "ctrl+u", "k")
		assertLayout(t, m, ctx+" after preview scrolling")
		m = keys(t, m, "tab", "/")
		m = typeText(t, m, "txt")
		assertLayout(t, m, ctx+" searching")
		m = keys(t, m, "esc", "G", "k", "ctrl+u", "ctrl+d", "h", "l")
		assertLayout(t, m, ctx+" after nav")
	}
}

func TestViewBeforeWindowSize(t *testing.T) {
	fx := mkFixture(t)
	m := newModel(t, filepath.Join(fx.root, "sub"))
	if v := m.View(); v == "" {
		t.Fatal("View() before ready is empty")
	}
	// keys before any WindowSizeMsg must not panic (viewport zero value)
	m = keys(t, m, "j", "k", "G", "g", "ctrl+d", "ctrl+u", "tab", "j", "k", "G", "ctrl+d", "ctrl+u", "tab", "h")
	if m.ready {
		t.Fatal("model should not be ready without WindowSizeMsg")
	}
	if m.cwd != fx.root {
		t.Fatalf("h before ready: cwd %q", m.cwd)
	}
	m = size(t, m, 100, 30)
	assertLayout(t, m, "after late WindowSizeMsg")
}

func TestNavigateIntoDirAndBackRestoresCursor(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = goTo(t, m, "sub")
	subIdx := m.cursor
	m = keys(t, m, "l")
	if m.cwd != filepath.Join(fx.root, "sub") {
		t.Fatalf("cwd = %q, want sub", m.cwd)
	}
	if m.cursor != 0 || m.offset != 0 {
		t.Fatalf("entering dir should reset cursor/offset, got %d/%d", m.cursor, m.offset)
	}
	// enter with enter key too, then deeper with right arrow
	m = goTo(t, m, "inner")
	m = keys(t, m, "enter")
	m = goTo(t, m, "deep")
	m = keys(t, m, "right")
	if !strings.HasSuffix(m.cwd, filepath.Join("sub", "inner", "deep")) {
		t.Fatalf("cwd = %q", m.cwd)
	}
	// back with h, backspace, left
	m = keys(t, m, "h")
	if m.filtered[m.cursor].name != "deep" {
		t.Fatalf("after h cursor on %q, want deep", m.filtered[m.cursor].name)
	}
	m = keys(t, m, "backspace")
	if m.filtered[m.cursor].name != "inner" {
		t.Fatalf("after backspace cursor on %q, want inner", m.filtered[m.cursor].name)
	}
	m = keys(t, m, "left")
	if m.cwd != fx.root {
		t.Fatalf("cwd = %q, want root", m.cwd)
	}
	if m.cursor != subIdx || m.filtered[m.cursor].name != "sub" {
		t.Fatalf("cursor = %d (%q), want %d (sub)", m.cursor, m.filtered[m.cursor].name, subIdx)
	}
	assertLayout(t, m, "after round trip")
}

func TestDotDotEntryGoesToParent(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, filepath.Join(fx.root, "sub")), 100, 30)
	if m.filtered[0].name != ".." || !m.filtered[0].isDir {
		t.Fatalf("first entry %+v, want '..' dir", m.filtered[0])
	}
	m = keys(t, m, "g", "enter")
	if m.cwd != fx.root {
		t.Fatalf("enter on '..' gave cwd %q, want %q", m.cwd, fx.root)
	}
	// '..' must not show a size column
	v := stripANSI(m.View())
	for _, line := range strings.Split(v, "\n") {
		if strings.Contains(line, "../") && strings.Contains(line, " B") {
			t.Fatalf("'..' row shows a size: %q", line)
		}
	}
	// root has no '..' and h at root is a no-op
	r := size(t, newModel(t, "/"), 100, 30)
	if len(r.filtered) > 0 && r.filtered[0].name == ".." {
		t.Fatal("root listing should not contain '..'")
	}
	r2 := keys(t, r, "h")
	if r2.cwd != "/" {
		t.Fatalf("h at root moved to %q", r2.cwd)
	}
}

func TestFuzzyFilterNarrowsAndClampsCursor(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	total := len(m.filtered)
	m = keys(t, m, "G")
	if m.cursor != total-1 {
		t.Fatalf("G cursor %d", m.cursor)
	}
	m = keys(t, m, "/")
	if !m.search.Focused() {
		t.Fatal("/ should focus search")
	}
	m = typeText(t, m, "bnr") // fuzzy for binary.bin
	if got := names(m.filtered); len(got) != 2 || got[0] != ".." || got[1] != "binary.bin" {
		t.Fatalf("filtered = %v, want [.. binary.bin]", got)
	}
	if m.cursor >= len(m.filtered) {
		t.Fatalf("cursor %d not clamped to %d", m.cursor, len(m.filtered))
	}
	if m.offset > m.cursor {
		t.Fatalf("offset %d > cursor %d after clamp", m.offset, m.cursor)
	}
	assertLayout(t, m, "filtered")
	// no-match query keeps only '..' and cursor clamps to 0
	m = typeText(t, m, "zzzzzz")
	if len(m.filtered) != 1 || m.cursor != 0 {
		t.Fatalf("no-match: filtered=%v cursor=%d", names(m.filtered), m.cursor)
	}
	// entries slice must be untouched by filtering (no aliasing/append corruption)
	if len(m.entries) != total {
		t.Fatalf("entries mutated by filter: %d vs %d", len(m.entries), total)
	}
	m = keys(t, m, "esc")
	if len(m.filtered) != total {
		t.Fatalf("after esc filtered=%d want %d", len(m.filtered), total)
	}
}

func TestEscClearsSearchAndBlurs(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = keys(t, m, "/")
	m = typeText(t, m, "text")
	if m.search.Value() != "text" {
		t.Fatalf("search value %q", m.search.Value())
	}
	m = keys(t, m, "esc")
	if m.search.Focused() || m.search.Value() != "" {
		t.Fatalf("esc did not clear/blur: focused=%v value=%q", m.search.Focused(), m.search.Value())
	}
	if len(m.filtered) != len(m.entries) {
		t.Fatalf("filter not reset: %d vs %d", len(m.filtered), len(m.entries))
	}
	// enter blurs but keeps the filter
	m = keys(t, m, "/")
	m = typeText(t, m, "text")
	m = keys(t, m, "enter")
	if m.search.Focused() || m.search.Value() != "text" || len(m.filtered) != 2 {
		t.Fatalf("enter: focused=%v value=%q filtered=%v", m.search.Focused(), m.search.Value(), names(m.filtered))
	}
	if !strings.Contains(stripANSI(m.View()), "/ text") {
		t.Fatal("status line should show the active search")
	}
}

func TestSearchFocusCapturesNavigationKeysAsText(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = goTo(t, m, "text.txt")
	before := m.cursor
	m = keys(t, m, "/")
	m = keys(t, m, "j", "k", "h", "l", "g", "G", "q")
	if m.search.Value() != "jkhlgGq" {
		t.Fatalf("search value %q, want jkhlgGq", m.search.Value())
	}
	if m.cwd != fx.root {
		t.Fatalf("h/l while searching changed cwd to %q", m.cwd)
	}
	_ = before
	if m.focus != 0 {
		t.Fatal("focus changed while typing")
	}
	// arrows also don't navigate while focused (cursor is clamped by the filter, not moved)
	m = keys(t, m, "esc", "/")
	m = typeText(t, m, "txt")
	c := m.cursor
	m = keys(t, m, "down", "down")
	if m.cursor != c {
		t.Fatalf("down moved cursor while search focused: %d -> %d", c, m.cursor)
	}
}

func TestCtrlCQuitsEvenWhileSearchFocused(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	_, cmd := key(t, m, "ctrl+c")
	if cmd == nil {
		t.Fatal("ctrl+c unfocused returned no cmd")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("ctrl+c unfocused did not quit")
	}
	m = keys(t, m, "/")
	_, cmd = key(t, m, "ctrl+c")
	if cmd == nil {
		t.Fatal("ctrl+c while search focused is swallowed by textinput: cannot quit")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("ctrl+c while search focused did not quit")
	}
}

func TestEnterOnFileDoesNothing(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = goTo(t, m, "text.txt")
	c := m.cursor
	for _, k := range []string{"enter", "l", "right"} {
		m2 := keys(t, m, k)
		if m2.cwd != fx.root || m2.cursor != c || len(m2.filtered) != len(m.filtered) {
			t.Fatalf("%s on file changed state: cwd=%q cursor=%d", k, m2.cwd, m2.cursor)
		}
	}
}

func TestPreviewContents(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)

	m = goTo(t, m, "binary.bin")
	if v := stripANSI(m.preview.View()); !strings.Contains(v, "binary file") {
		t.Fatalf("binary preview = %q", v)
	}

	m = goTo(t, m, "big.log")
	if v := stripANSI(m.View()); strings.Contains(v, "(truncated)") {
		t.Fatal("truncated marker is at the end of the content, it should not be visible at the top")
	}
	m = keys(t, m, "tab", "G")
	if v := stripANSI(m.preview.View()); !strings.Contains(v, "(truncated)") {
		t.Fatalf("big file preview bottom = %q, want (truncated)", v)
	}
	m = keys(t, m, "tab")

	m = goTo(t, m, "text.txt")
	v := stripANSI(m.preview.View())
	if !strings.Contains(v, "line one") || strings.Contains(v, "\t") || !strings.Contains(v, "    indented") {
		t.Fatalf("text preview = %q", v)
	}
	if strings.Contains(v, "(truncated)") {
		t.Fatal("small file marked truncated")
	}

	m = goTo(t, m, "empty.txt")
	assertLayout(t, m, "empty file")
	if v := stripANSI(m.preview.View()); strings.Contains(v, "binary") || strings.Contains(v, "truncated") {
		t.Fatalf("empty file preview = %q", v)
	}

	m = goTo(t, m, "sub")
	if v := stripANSI(m.preview.View()); !strings.Contains(v, "2 items") || !strings.Contains(v, "inner/") {
		t.Fatalf("dir preview = %q", v)
	}
}

func TestScrollOffsetKeepsCursorVisible(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, filepath.Join(fx.root, manyEntriesName)), 100, 30)
	h := m.listHeight()
	if len(m.filtered) <= h {
		t.Fatalf("fixture too small: %d entries, list height %d", len(m.filtered), h)
	}
	check := func(ctx string) {
		t.Helper()
		if m.cursor < m.offset || m.cursor >= m.offset+h {
			t.Fatalf("%s: cursor %d not visible in window [%d,%d)", ctx, m.cursor, m.offset, m.offset+h)
		}
		v := stripANSI(assertLayout(t, m, ctx))
		if !strings.Contains(v, m.filtered[m.cursor].name) {
			t.Fatalf("%s: selected %q not rendered", ctx, m.filtered[m.cursor].name)
		}
	}
	m = keys(t, m, "G")
	check("after G")
	if m.cursor != len(m.filtered)-1 {
		t.Fatal("G not at end")
	}
	m = keys(t, m, "k")
	check("after G k")
	m = keys(t, m, "ctrl+u")
	check("after ctrl+u")
	m = keys(t, m, "g")
	check("after g")
	if m.offset != 0 {
		t.Fatalf("g offset %d", m.offset)
	}
	for i := 0; i < len(m.filtered)+3; i++ {
		m = keys(t, m, "j")
		check("j walk " + itoa(i))
	}
	for i := 0; i < len(m.filtered)+3; i++ {
		m = keys(t, m, "k")
		check("k walk " + itoa(i))
	}
	for i := 0; i < len(m.filtered)/(h/2)+2; i++ {
		m = keys(t, m, "ctrl+d")
		check("ctrl+d " + itoa(i))
	}
	if m.cursor != len(m.filtered)-1 {
		t.Fatalf("ctrl+d should clamp at end, cursor %d", m.cursor)
	}
}

func TestTabTogglesFocusAndScrollsPreview(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = goTo(t, m, "manylines.txt")
	c := m.cursor
	m = keys(t, m, "tab")
	if m.focus != 1 {
		t.Fatal("tab did not focus preview")
	}
	m = keys(t, m, "j", "j", "j")
	if m.cursor != c {
		t.Fatalf("j with preview focused moved list cursor %d -> %d", c, m.cursor)
	}
	if m.preview.YOffset != 3 {
		t.Fatalf("preview YOffset %d, want 3", m.preview.YOffset)
	}
	m = keys(t, m, "k")
	if m.preview.YOffset != 2 {
		t.Fatalf("preview YOffset %d, want 2", m.preview.YOffset)
	}
	m = keys(t, m, "G")
	if !m.preview.AtBottom() {
		t.Fatal("G did not scroll preview to bottom")
	}
	m = keys(t, m, "g")
	if m.preview.YOffset != 0 {
		t.Fatal("g did not scroll preview to top")
	}
	m = keys(t, m, "ctrl+d")
	if m.preview.YOffset == 0 {
		t.Fatal("ctrl+d did not scroll preview")
	}
	m = keys(t, m, "tab")
	if m.focus != 0 {
		t.Fatal("tab did not return focus to list")
	}
	m = keys(t, m, "k")
	if m.cursor != c-1 {
		t.Fatalf("k after tab back did not move list cursor: %d", m.cursor)
	}
}

func TestLongAndUnicodeNamesAreTruncatedSafely(t *testing.T) {
	fx := mkFixture(t)
	for _, sz := range [][2]int{{100, 30}, {20, 5}, {30, 8}, {300, 80}} {
		m := size(t, newModel(t, fx.root), sz[0], sz[1])
		lw := max(20, m.width/2-4)
		for _, name := range []string{fx.longName, fx.longUnicode, unicodeName, cjkName} {
			m = goTo(t, m, name)
			v := assertLayout(t, m, "long name "+name[:8])
			plain := stripANSI(v)
			for _, line := range strings.Split(plain, "\n") {
				line = strings.TrimRight(line, " ") // JoinVertical pads rows to the widest line (the header)
				if strings.Contains(line, "│") && lipgloss.Width(line) > 2*(lw+2)+1 {
					t.Errorf("%dx%d: pane row wider than layout (%d cells): %q", sz[0], sz[1], lipgloss.Width(line), line)
				}
			}
			if name == fx.longUnicode && !strings.Contains(plain, "…") {
				t.Errorf("%dx%d: long unicode name not truncated with ellipsis", sz[0], sz[1])
			}
			if name == fx.longUnicode && strings.Contains(plain, "�") {
				t.Errorf("%dx%d: replacement char in view: rune split by byte slicing", sz[0], sz[1])
			}
		}
	}
}

func TestFuzzyMatch(t *testing.T) {
	cases := []struct {
		p, s string
		want bool
	}{
		{"", "anything", true},
		{"abc", "a-b-c", true},
		{"ABC", "xaxbxc", true},
		{"cba", "abc", false},
		{"bnr", "binary.bin", true},
		{"é", "héllo wörld.txt", true},
		{"ö", "héllo wörld.txt", true},
		{"héllo", "héllo wörld.txt", true},
		{"日本", cjkName, true},
		{"ファイル", cjkName, true},
		{"É", "héllo", true},
		{"éx", "héllo", false},
	}
	for _, c := range cases {
		if got := fuzzyMatch(c.p, c.s); got != c.want {
			t.Errorf("fuzzyMatch(%q, %q) = %v, want %v", c.p, c.s, got, c.want)
		}
	}
}

func TestSearchWithMultibytePatternMatches(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = keys(t, m, "/")
	m = typeText(t, m, "wörld")
	if got := names(m.filtered); len(got) != 2 || got[1] != unicodeName {
		t.Fatalf("multibyte search filtered = %v, want [.. %s]", got, unicodeName)
	}
	m = keys(t, m, "esc", "/")
	m = typeText(t, m, "日本")
	if got := names(m.filtered); len(got) != 2 || got[1] != cjkName {
		t.Fatalf("CJK search filtered = %v, want [.. %s]", got, cjkName)
	}
	assertLayout(t, m, "cjk search")
}

func TestEmptyDirectoryKeysDoNotPanic(t *testing.T) {
	empty := t.TempDir()
	m := size(t, newModel(t, empty), 100, 30)
	if len(m.filtered) != 1 || m.filtered[0].name != ".." {
		t.Fatalf("empty dir listing %v", names(m.filtered))
	}
	m = keys(t, m, "j", "k", "G", "g", "ctrl+d", "ctrl+u", "tab", "j", "G", "tab")
	if m.cursor != 0 {
		t.Fatalf("cursor %d in empty dir", m.cursor)
	}
	assertLayout(t, m, "empty dir")
	// truly empty listing (unreadable / nothing at all)
	m.entries = nil
	m.applyFilter()
	m = keys(t, m, "j", "k", "G", "g", "ctrl+d", "ctrl+u", "enter", "l")
	if m.cursor < 0 {
		t.Fatalf("cursor went negative: %d", m.cursor)
	}
	assertLayout(t, m, "nil entries")
	if !strings.Contains(stripANSI(m.View()), "(empty)") {
		t.Fatal("empty preview marker missing")
	}
}

// ctrl+d on an empty filtered list sets cursor to -1; the next applyFilter that
// yields entries then indexes filtered[-1] and panics.
func TestCtrlDOnEmptyFilterThenClearDoesNotPanic(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	// simulate a root-like listing (no '..' entry, which is always kept by the filter)
	m.entries = m.entries[1:]
	m.applyFilter()
	m = keys(t, m, "/")
	m = typeText(t, m, "qqqqqqqq")
	m = keys(t, m, "enter") // blur, keep no-match filter
	if len(m.filtered) != 0 {
		t.Fatalf("expected empty filter, got %v", names(m.filtered))
	}
	m = keys(t, m, "ctrl+d")
	if m.cursor < 0 {
		t.Errorf("ctrl+d on empty list set cursor to %d", m.cursor)
	}
	assertLayout(t, m, "empty filter")
	m = keys(t, m, "/", "esc") // clears filter -> applyFilter -> updatePreview
	if m.cursor < 0 || m.cursor >= len(m.filtered) {
		t.Fatalf("cursor %d out of range after clearing filter", m.cursor)
	}
	assertLayout(t, m, "after clearing filter")
}

func TestCtrlDAtRealRoot(t *testing.T) {
	m := size(t, newModel(t, "/"), 100, 30)
	if m.err != "" {
		t.Skip("cannot read /")
	}
	m = keys(t, m, "/")
	m = typeText(t, m, strings.Repeat("z", 40))
	if len(m.filtered) != 0 {
		t.Skip("something in / fuzzy-matches 40 z's; cannot force an empty filter")
	}
	m = keys(t, m, "enter", "ctrl+d", "/", "esc")
	if m.cursor < 0 || m.cursor >= len(m.filtered) {
		t.Fatalf("cursor %d out of range", m.cursor)
	}
}

func TestBackRestoresCursorAfterFilteredEntry(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = keys(t, m, "/")
	m = typeText(t, m, "zeta")
	m = keys(t, m, "enter")
	if got := names(m.filtered); len(got) != 2 || got[1] != "zeta" {
		t.Fatalf("filtered %v", got)
	}
	m = keys(t, m, "j", "l")
	if filepath.Base(m.cwd) != "zeta" {
		t.Fatalf("cwd %q", m.cwd)
	}
	if m.search.Value() != "" {
		t.Fatal("entering a dir should clear the search")
	}
	m = keys(t, m, "h")
	if m.search.Value() != "" || len(m.filtered) != len(m.entries) {
		t.Fatal("going back should show the full, unfiltered parent")
	}
	if m.filtered[m.cursor].name != "zeta" {
		t.Fatalf("cursor on %q, want zeta", m.filtered[m.cursor].name)
	}
}

func TestUnreadableDir(t *testing.T) {
	fx := mkFixture(t)
	if !fx.hasLocked {
		t.Skip("running as root; permission bits are not enforced")
	}
	m := size(t, newModel(t, fx.root), 100, 30)
	m = goTo(t, m, unreadableName)
	if v := stripANSI(m.preview.View()); !strings.Contains(v, "permission denied") {
		t.Fatalf("preview of unreadable dir = %q", v)
	}
	m = keys(t, m, "l")
	if m.err == "" {
		t.Fatal("entering unreadable dir should set err")
	}
	if len(m.filtered) != 0 {
		t.Fatalf("unreadable dir listing = %v", names(m.filtered))
	}
	assertLayout(t, m, "inside unreadable dir")
	m = keys(t, m, "j", "k", "G", "ctrl+d", "enter")
	m = keys(t, m, "h")
	if m.err != "" || m.filtered[m.cursor].name != unreadableName {
		t.Fatalf("back from unreadable: err=%q cursor on %q", m.err, m.filtered[m.cursor].name)
	}
}

func TestResizeAfterReadyKeepsState(t *testing.T) {
	fx := mkFixture(t)
	m := size(t, newModel(t, fx.root), 100, 30)
	m = goTo(t, m, "manylines.txt")
	c := m.cursor
	m = keys(t, m, "tab", "ctrl+d")
	m = size(t, m, 40, 12)
	if m.cursor != c || m.focus != 1 {
		t.Fatalf("resize changed state: cursor %d focus %d", m.cursor, m.focus)
	}
	// viewport must match the pane's inner width (Width(lw) minus 1-cell padding each side)
	if want := max(20, 40/2-4) - 2; m.preview.Width != want {
		t.Fatalf("preview width %d after resize, want %d", m.preview.Width, want)
	}
	assertLayout(t, m, "after shrink")
	m = size(t, m, 300, 80)
	assertLayout(t, m, "after grow")
}

func TestHumanSize(t *testing.T) {
	cases := map[int64]string{0: "0 B", 1023: "1023 B", 1024: "1.0 KB", 1536: "1.5 KB", 1 << 20: "1.0 MB", 1 << 40: "1.0 TB"}
	for n, want := range cases {
		if got := humanSize(n); got != want {
			t.Errorf("humanSize(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestIsBinary(t *testing.T) {
	if isBinary(nil) || isBinary([]byte("plain")) {
		t.Fatal("text detected as binary")
	}
	if !isBinary([]byte("ab\x00cd")) {
		t.Fatal("NUL not detected")
	}
	late := append([]byte(strings.Repeat("a", 600)), 0)
	if isBinary(late) {
		t.Log("note: NUL after byte 512 is not detected (documented 512-byte sniff window)")
	}
}
