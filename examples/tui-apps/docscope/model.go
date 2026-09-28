package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"time"

	osc52 "github.com/aymanbagabas/go-osc52/v2"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/charmbracelet/lipgloss"
)

type focusPane int

const (
	paneFiles focusPane = iota
	paneOutline
	paneReader
)

type overlayKind int

const (
	overlayNone overlayKind = iota
	overlayFinder
	overlayHelp
)

type searchState int

const (
	searchOff searchState = iota
	searchTyping
	searchActive
)

type toastKind int

const (
	toastOK toastKind = iota
	toastWarn
	toastErr
)

type toast struct {
	id   int
	text string
	kind toastKind
}

type (
	animTickMsg    struct{}
	toastExpireMsg struct{ id int }
	editorDoneMsg  struct{ err error }
	copiedMsg      struct{ err error }
)

const (
	collapsedPaneH = 3
	minWidth       = 40
	minHeight      = 10
	toastDuration  = 2500 * time.Millisecond
	animFPS        = 30
)

type model struct {
	root    string
	rootErr error
	tree    *node
	files   []*node
	rows    []*node

	width, height int
	ready         bool

	focus          focusPane
	sidePane       focusPane
	sidebarVisible bool

	fileCursor, fileOffset       int
	outlineCursor, outlineOffset int

	filesH, filesVel float64
	sideSpring       harmonica.Spring
	sideAnimating    bool
	sideInit         bool

	vp             viewport.Model
	current        *node
	raw            string
	doc            renderedDoc
	hasDoc         bool
	rendering      bool
	renderSeq      int
	renderWidth    int
	renderErr      error
	pendingPercent float64
	spinner        spinner.Model

	scrollPos, scrollVel float64
	scrollTarget         int
	scrollSpring         harmonica.Spring
	scrollAnimating      bool
	ticking              bool

	search      textinput.Model
	searchState searchState
	matches     []matchPos
	matchIdx    int

	overlay overlayKind
	finder  finder

	toast    *toast
	toastSeq int
}

func newModel(root string) model {
	si := textinput.New()
	si.Prompt = "/ "
	si.Placeholder = "search in document"
	si.PromptStyle = styleKey
	si.TextStyle = styleText
	si.PlaceholderStyle = styleMuted
	si.Cursor.Style = styleKey
	si.CharLimit = 128

	vp := viewport.New(0, 0)
	vp.MouseWheelEnabled = false

	m := model{
		root:           root,
		sidebarVisible: true,
		focus:          paneFiles,
		sidePane:       paneFiles,
		vp:             vp,
		spinner:        spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(styleSpinner)),
		search:         si,
		finder:         newFinder(),
		sideSpring:     harmonica.NewSpring(harmonica.FPS(animFPS), 9.0, 1.0),
		scrollSpring:   harmonica.NewSpring(harmonica.FPS(animFPS), 24.0, 1.0),
	}
	m.loadTree()
	return m
}

func (m *model) loadTree() {
	tree, files, err := buildTree(m.root)
	m.rootErr = err
	m.tree, m.files = tree, files
	if err != nil || tree == nil {
		m.tree = &node{isDir: true, expanded: true}
		m.files = nil
	}
	if m.current != nil {
		for _, f := range m.files {
			if f.path == m.current.path {
				m.current = f
				expandTo(m.tree, f)
				break
			}
		}
	}
	m.refreshRows()
}

func (m *model) refreshRows() {
	m.rows = visibleRows(m.tree)
	m.fileCursor = clamp(m.fileCursor, 0, max(len(m.rows)-1, 0))
}

func (m model) Init() tea.Cmd { return nil }

// ---- layout ----------------------------------------------------------------

type layout struct {
	sidebarW, readerW, bodyH int
	filesH, outlineH         int
}

func (m model) layout() layout {
	l := layout{bodyH: max(m.height-3, 2)}
	if m.sidebarVisible {
		l.sidebarW = clamp(m.width*30/100, 24, 48)
		if m.width-l.sidebarW < 30 {
			l.sidebarW = max(m.width-30, 0)
		}
	}
	l.readerW = m.width - l.sidebarW
	l.filesH = clamp(int(math.Round(m.filesH)), collapsedPaneH, max(l.bodyH-collapsedPaneH, collapsedPaneH))
	if l.filesH > l.bodyH {
		l.filesH = l.bodyH
	}
	l.outlineH = l.bodyH - l.filesH
	return l
}

func (m model) readerInner() (w, h int) {
	l := m.layout()
	return max(l.readerW-4, 1), max(l.bodyH-2, 1)
}

func (m model) sideTarget() float64 {
	l := m.layout()
	if m.sidePane == paneFiles {
		return float64(max(l.bodyH-collapsedPaneH, collapsedPaneH))
	}
	return collapsedPaneH
}

func (m model) animating() bool { return m.sideAnimating || m.scrollAnimating }

func animTick() tea.Cmd {
	return tea.Tick(time.Second/animFPS, func(time.Time) tea.Msg { return animTickMsg{} })
}

func (m *model) ensureTicking() tea.Cmd {
	if m.ticking || !m.animating() {
		return nil
	}
	m.ticking = true
	return animTick()
}

func (m *model) setFocus(p focusPane) {
	m.focus = p
	if p == paneReader {
		return
	}
	m.sidebarVisible = true
	if m.sidePane != p {
		m.sidePane = p
		m.sideAnimating = true
	}
	if p == paneOutline {
		m.outlineCursor = m.currentHeading()
	}
}

// ---- documents ---------------------------------------------------------------

func (m *model) openNode(n *node, keepScroll bool) tea.Cmd {
	if n == nil || n.isDir {
		return nil
	}
	if keepScroll && m.hasDoc && n == m.current {
		m.pendingPercent = m.vp.ScrollPercent()
	} else {
		m.pendingPercent = 0
	}
	if n != m.current {
		m.doc = renderedDoc{}
	}
	m.current = n
	m.raw = ""
	m.hasDoc = false
	m.renderErr = nil
	m.matches = nil
	m.rendering = true
	m.renderSeq++
	m.scrollAnimating = false
	m.vp.SetContent("")
	m.vp.GotoTop()
	w, _ := m.readerInner()
	m.renderWidth = w
	return tea.Batch(openFileCmd(n.path, w, m.renderSeq), m.spinner.Tick)
}

// syncReader fits the viewport to the reader pane and, when the wrap width
// changed, re-renders the document while remembering the scroll position.
func (m *model) syncReader() tea.Cmd {
	w, h := m.readerInner()
	if m.vp.Width == w && m.vp.Height == h {
		return m.requestRender()
	}
	if m.hasDoc {
		m.pendingPercent = m.vp.ScrollPercent()
	}
	m.vp.Width, m.vp.Height = w, h
	m.vp.SetYOffset(m.vp.YOffset)
	return m.requestRender()
}

// requestRender re-renders the current document at the current reader width
// unless a render at that width is already done or in flight.
func (m *model) requestRender() tea.Cmd {
	if m.current == nil {
		return nil
	}
	w, _ := m.readerInner()
	if m.hasDoc && m.doc.width == w || m.rendering && m.renderWidth == w {
		return nil
	}
	m.rendering = true
	m.renderWidth = w
	m.renderSeq++
	m.scrollAnimating = false
	if m.raw == "" {
		return tea.Batch(openFileCmd(m.current.path, w, m.renderSeq), m.spinner.Tick)
	}
	return tea.Batch(rerenderCmd(m.current.path, m.raw, w, m.renderSeq), m.spinner.Tick)
}

func (m *model) applyRender(msg renderDoneMsg) {
	if msg.seq != m.renderSeq {
		return
	}
	m.rendering = false
	if msg.err != nil {
		m.renderErr = msg.err
		m.hasDoc = false
		m.vp.SetContent("")
		return
	}
	m.raw = msg.raw
	m.doc = msg.doc
	m.hasDoc = true
	m.renderErr = nil
	m.refreshMatches()
	m.applyContent()
	m.vp.SetYOffset(int(math.Round(m.pendingPercent * float64(m.maxYOffset()))))
	m.syncOutline()
}

func (m model) maxYOffset() int { return max(len(m.doc.lines)-m.vp.Height, 0) }

func (m *model) applyContent() {
	if !m.hasDoc {
		m.vp.SetContent("")
		return
	}
	if len(m.matches) == 0 {
		m.vp.SetContent(strings.Join(m.doc.lines, "\n"))
		return
	}
	lines := make([]string, len(m.doc.lines))
	copy(lines, m.doc.lines)
	var cur matchPos
	if m.matchIdx >= 0 && m.matchIdx < len(m.matches) {
		cur = m.matches[m.matchIdx]
	}
	i := 0
	for i < len(m.matches) {
		j := i
		for j < len(m.matches) && m.matches[j].line == m.matches[i].line {
			j++
		}
		ln := m.matches[i].line
		lines[ln] = highlightLine(m.doc.plain[ln], m.matches[i:j], cur)
		i = j
	}
	m.vp.SetContent(strings.Join(lines, "\n"))
}

func (m *model) refreshMatches() {
	if m.searchState == searchOff || !m.hasDoc {
		m.matches = nil
		return
	}
	m.matches = findMatches(m.doc.plain, m.search.Value())
	m.matchIdx = clamp(m.matchIdx, 0, max(len(m.matches)-1, 0))
}

func (m model) currentHeading() int { return m.headingAt(m.vp.YOffset) }

// headingAt returns the index of the heading in effect at the given line: the
// last heading starting at or above it.
func (m model) headingAt(line int) int {
	cur := 0
	for i, h := range m.doc.headings {
		if h.line <= line {
			cur = i
		}
	}
	return cur
}

func (m *model) syncOutline() {
	if m.focus != paneOutline {
		m.outlineCursor = m.currentHeading()
	}
}

func (m *model) scrollTo(line int) tea.Cmd {
	m.scrollTarget = clamp(line, 0, m.maxYOffset())
	m.scrollPos = float64(m.vp.YOffset)
	m.scrollVel = 0
	m.scrollAnimating = m.scrollTarget != m.vp.YOffset
	return m.ensureTicking()
}

func (m *model) jumpToMatch(delta int) tea.Cmd {
	if len(m.matches) == 0 {
		return nil
	}
	m.matchIdx = ((m.matchIdx+delta)%len(m.matches) + len(m.matches)) % len(m.matches)
	m.applyContent()
	_, h := m.readerInner()
	return m.scrollTo(m.matches[m.matchIdx].line - h/3)
}

func (m *model) stepHeading(delta int) tea.Cmd {
	if len(m.doc.headings) == 0 {
		return nil
	}
	pos := m.vp.YOffset
	if m.scrollAnimating {
		pos = m.scrollTarget
	}
	cur := m.headingAt(pos)
	if delta < 0 && m.doc.headings[cur].line < pos {
		return m.scrollTo(m.doc.headings[cur].line)
	}
	next := clamp(cur+delta, 0, len(m.doc.headings)-1)
	return m.scrollTo(m.doc.headings[next].line)
}

// ---- toasts & commands -------------------------------------------------------

func (m *model) showToast(text string, kind toastKind) tea.Cmd {
	m.toastSeq++
	id := m.toastSeq
	m.toast = &toast{id: id, text: text, kind: kind}
	return tea.Tick(toastDuration, func(time.Time) tea.Msg { return toastExpireMsg{id: id} })
}

func copyPathCmd(path string) tea.Cmd {
	return func() tea.Msg {
		seq := osc52.New(path)
		if os.Getenv("TMUX") != "" {
			seq = seq.Tmux()
		} else if strings.HasPrefix(os.Getenv("TERM"), "screen") {
			seq = seq.Screen()
		}
		tty, err := os.OpenFile("/dev/tty", os.O_WRONLY, 0)
		if err != nil {
			_, err = seq.WriteTo(os.Stderr)
			return copiedMsg{err: err}
		}
		defer tty.Close()
		_, err = seq.WriteTo(tty)
		return copiedMsg{err: err}
	}
}

func editCmd(path string) tea.Cmd {
	editor := strings.Fields(os.Getenv("EDITOR"))
	if len(editor) == 0 {
		editor = []string{"vi"}
	}
	c := exec.Command(editor[0], append(editor[1:], path)...)
	return tea.ExecProcess(c, func(err error) tea.Msg { return editorDoneMsg{err: err} })
}

// ---- update ------------------------------------------------------------------

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := m.update(msg)
	next.settleOffsets()
	return next, cmd
}

// settleOffsets keeps the Files and Outline windows scrolled so their cursor is
// visible; View and the mouse hit-test both read the settled values.
func (m *model) settleOffsets() {
	l := m.layout()
	m.fileOffset = windowOffset(m.fileCursor, m.fileOffset, l.filesH-2, len(m.rows))
	m.outlineOffset = windowOffset(m.outlineCursor, m.outlineOffset, l.outlineH-2, len(m.doc.headings))
}

func windowOffset(cursor, offset, innerH, n int) int {
	if innerH < 1 {
		return 0
	}
	if cursor < offset {
		offset = cursor
	}
	if cursor >= offset+innerH {
		offset = cursor - innerH + 1
	}
	return clamp(offset, 0, max(n-innerH, 0))
}

func (m model) update(msg tea.Msg) (model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		return m.onResize(msg.Width, msg.Height)
	case renderDoneMsg:
		m.applyRender(msg)
		return m, nil
	case spinner.TickMsg:
		if !m.rendering {
			return m, nil
		}
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	case animTickMsg:
		return m.onTick()
	case toastExpireMsg:
		if m.toast != nil && m.toast.id == msg.id {
			m.toast = nil
		}
		return m, nil
	case copiedMsg:
		if msg.err != nil {
			return m, m.showToast("copy failed: "+msg.err.Error(), toastErr)
		}
		return m, m.showToast("copied", toastOK)
	case editorDoneMsg:
		cmds := []tea.Cmd{m.openNode(m.current, true)}
		if msg.err != nil {
			cmds = append(cmds, m.showToast("editor: "+msg.err.Error(), toastErr))
		} else {
			cmds = append(cmds, m.showToast("reloaded after edit", toastOK))
		}
		return m, tea.Batch(cmds...)
	case tea.MouseMsg:
		return m.onMouse(msg)
	case tea.KeyMsg:
		return m.onKey(msg)
	}
	return m, nil
}

func (m model) onResize(w, h int) (model, tea.Cmd) {
	m.width, m.height = w, h
	m.ready = true
	if !m.sideInit {
		m.sideInit = true
		m.filesH = m.sideTarget()
	} else if !m.sideAnimating {
		m.filesH = m.sideTarget()
	}
	var cmd tea.Cmd
	if m.current == nil && len(m.files) > 0 {
		m.vp.Width, m.vp.Height = m.readerInner()
		cmd = m.openNode(m.files[0], false)
	} else {
		cmd = m.syncReader()
	}
	m.syncOutline()
	return m, cmd
}

// focusCmd moves focus and keeps the accordion and reader in step with it.
func (m *model) focusCmd(p focusPane) tea.Cmd {
	m.setFocus(p)
	return tea.Batch(m.ensureTicking(), m.syncReader())
}

func (m model) onTick() (model, tea.Cmd) {
	if m.sideAnimating {
		target := m.sideTarget()
		m.filesH, m.filesVel = m.sideSpring.Update(m.filesH, m.filesVel, target)
		if math.Abs(m.filesH-target) < 0.5 && math.Abs(m.filesVel) < 0.5 {
			m.filesH, m.filesVel = target, 0
			m.sideAnimating = false
		}
	}
	if m.scrollAnimating {
		m.scrollPos, m.scrollVel = m.scrollSpring.Update(m.scrollPos, m.scrollVel, float64(m.scrollTarget))
		m.vp.SetYOffset(int(math.Round(m.scrollPos)))
		if math.Abs(m.scrollPos-float64(m.scrollTarget)) < 0.5 && math.Abs(m.scrollVel) < 0.5 {
			m.vp.SetYOffset(m.scrollTarget)
			m.scrollAnimating = false
		}
		m.syncOutline()
	}
	if m.animating() {
		return m, animTick()
	}
	m.ticking = false
	return m, nil
}

func (m model) onKey(msg tea.KeyMsg) (model, tea.Cmd) {
	if key.Matches(msg, keys.Quit) && msg.String() == "ctrl+c" {
		return m, tea.Quit
	}
	switch m.overlay {
	case overlayFinder:
		return m.onFinderKey(msg)
	case overlayHelp:
		if key.Matches(msg, keys.Help, keys.Escape, keys.Quit) {
			m.overlay = overlayNone
		}
		return m, nil
	}
	if m.searchState == searchTyping {
		return m.onSearchKey(msg)
	}
	if m.rootErr != nil || len(m.files) == 0 {
		switch {
		case key.Matches(msg, keys.Quit):
			return m, tea.Quit
		case key.Matches(msg, keys.Reload):
			m.loadTree()
			if len(m.files) > 0 {
				return m, tea.Batch(m.openNode(m.files[0], false), m.showToast(fmt.Sprintf("found %d docs", len(m.files)), toastOK))
			}
		}
		return m, nil
	}

	switch {
	case key.Matches(msg, keys.Quit):
		return m, tea.Quit
	case key.Matches(msg, keys.Help):
		m.overlay = overlayHelp
		return m, nil
	case key.Matches(msg, keys.Escape):
		if m.searchState != searchOff {
			m.clearSearch()
		}
		return m, nil
	case key.Matches(msg, keys.Finder):
		return m.openFinder()
	case key.Matches(msg, keys.NextPane):
		return m, m.focusCmd((m.focus + 1) % 3)
	case key.Matches(msg, keys.PrevPane):
		return m, m.focusCmd((m.focus + 2) % 3)
	case key.Matches(msg, keys.FocusFiles):
		return m, m.focusCmd(paneFiles)
	case key.Matches(msg, keys.FocusOutline):
		return m, m.focusCmd(paneOutline)
	case key.Matches(msg, keys.FocusReader):
		return m, m.focusCmd(paneReader)
	case key.Matches(msg, keys.Left) && m.focus == paneReader:
		return m, m.focusCmd(m.sidePane)
	case key.Matches(msg, keys.Right) && m.focus != paneReader:
		return m, m.focusCmd(paneReader)
	case key.Matches(msg, keys.ToggleSidebar):
		m.sidebarVisible = !m.sidebarVisible
		if !m.sidebarVisible {
			m.focus = paneReader
		}
		return m, m.syncReader()
	case key.Matches(msg, keys.Copy):
		if m.current == nil {
			return m, nil
		}
		return m, copyPathCmd(m.current.path)
	case key.Matches(msg, keys.Edit):
		if m.current == nil {
			return m, nil
		}
		return m, editCmd(m.current.path)
	case key.Matches(msg, keys.Reload):
		m.loadTree()
		cmds := []tea.Cmd{m.showToast(fmt.Sprintf("reloaded · %d docs", len(m.files)), toastOK)}
		if m.current != nil {
			cmds = append(cmds, m.openNode(m.current, true))
		} else if len(m.files) > 0 {
			cmds = append(cmds, m.openNode(m.files[0], false))
		}
		return m, tea.Batch(cmds...)
	}

	switch m.focus {
	case paneFiles:
		return m.onFilesKey(msg)
	case paneOutline:
		return m.onOutlineKey(msg)
	default:
		return m.onReaderKey(msg)
	}
}

func (m model) onFilesKey(msg tea.KeyMsg) (model, tea.Cmd) {
	switch {
	case key.Matches(msg, keys.Search):
		return m.openFinder()
	case key.Matches(msg, keys.Down):
		m.fileCursor = clamp(m.fileCursor+1, 0, max(len(m.rows)-1, 0))
	case key.Matches(msg, keys.Up):
		m.fileCursor = clamp(m.fileCursor-1, 0, max(len(m.rows)-1, 0))
	case key.Matches(msg, keys.Top):
		m.fileCursor = 0
	case key.Matches(msg, keys.Bottom):
		m.fileCursor = max(len(m.rows)-1, 0)
	case key.Matches(msg, keys.Enter):
		return m.activateRow(m.fileCursor)
	case key.Matches(msg, keys.Collapse):
		if n := m.rowAt(m.fileCursor); n != nil {
			if n.isDir && n.expanded {
				n.expanded = false
				m.refreshRows()
			} else if n.depth > 1 {
				for i := m.fileCursor - 1; i >= 0; i-- {
					if m.rows[i].isDir && m.rows[i].depth == n.depth-1 {
						m.fileCursor = i
						break
					}
				}
			}
		}
	case key.Matches(msg, keys.Expand):
		if n := m.rowAt(m.fileCursor); n != nil {
			if n.isDir {
				n.expanded = true
				m.refreshRows()
			} else {
				return m, m.openNode(n, false)
			}
		}
	}
	return m, nil
}

func (m model) rowAt(i int) *node {
	if i < 0 || i >= len(m.rows) {
		return nil
	}
	return m.rows[i]
}

func (m model) activateRow(i int) (model, tea.Cmd) {
	n := m.rowAt(i)
	if n == nil {
		return m, nil
	}
	if n.isDir {
		n.expanded = !n.expanded
		m.refreshRows()
		return m, nil
	}
	return m, m.openNode(n, false)
}

func (m model) onOutlineKey(msg tea.KeyMsg) (model, tea.Cmd) {
	n := len(m.doc.headings)
	switch {
	case key.Matches(msg, keys.Search):
		return m.openFinder()
	case key.Matches(msg, keys.Down):
		m.outlineCursor = clamp(m.outlineCursor+1, 0, max(n-1, 0))
	case key.Matches(msg, keys.Up):
		m.outlineCursor = clamp(m.outlineCursor-1, 0, max(n-1, 0))
	case key.Matches(msg, keys.Top):
		m.outlineCursor = 0
	case key.Matches(msg, keys.Bottom):
		m.outlineCursor = max(n-1, 0)
	case key.Matches(msg, keys.Enter):
		if m.outlineCursor < n {
			return m, m.scrollTo(m.doc.headings[m.outlineCursor].line)
		}
	}
	return m, nil
}

func (m model) onReaderKey(msg tea.KeyMsg) (model, tea.Cmd) {
	switch {
	case key.Matches(msg, keys.Search):
		m.searchState = searchTyping
		m.matchIdx = 0
		return m, m.search.Focus()
	case key.Matches(msg, keys.PrevHeading):
		return m, m.stepHeading(-1)
	case key.Matches(msg, keys.NextHeading):
		return m, m.stepHeading(1)
	case key.Matches(msg, keys.NextMatch):
		return m, m.jumpToMatch(1)
	case key.Matches(msg, keys.PrevMatch):
		return m, m.jumpToMatch(-1)
	}
	m.scrollAnimating = false
	switch {
	case key.Matches(msg, keys.Down):
		m.vp.LineDown(1)
	case key.Matches(msg, keys.Up):
		m.vp.LineUp(1)
	case key.Matches(msg, keys.HalfDown):
		m.vp.HalfViewDown()
	case key.Matches(msg, keys.HalfUp):
		m.vp.HalfViewUp()
	case key.Matches(msg, keys.PageDown):
		m.vp.ViewDown()
	case key.Matches(msg, keys.PageUp):
		m.vp.ViewUp()
	case key.Matches(msg, keys.Top):
		m.vp.GotoTop()
	case key.Matches(msg, keys.Bottom):
		m.vp.GotoBottom()
	}
	m.syncOutline()
	return m, nil
}

func (m model) onSearchKey(msg tea.KeyMsg) (model, tea.Cmd) {
	switch {
	case key.Matches(msg, keys.Escape):
		m.clearSearch()
		return m, nil
	case key.Matches(msg, keys.Enter):
		m.search.Blur()
		if strings.TrimSpace(m.search.Value()) == "" {
			m.clearSearch()
			return m, nil
		}
		m.searchState = searchActive
		return m, m.scrollToCurrentMatch()
	}
	before := m.search.Value()
	var cmd tea.Cmd
	m.search, cmd = m.search.Update(msg)
	if m.search.Value() != before {
		m.matchIdx = 0
		m.refreshMatches()
		m.applyContent()
		return m, tea.Batch(cmd, m.scrollToCurrentMatch())
	}
	return m, cmd
}

func (m *model) scrollToCurrentMatch() tea.Cmd {
	if len(m.matches) == 0 {
		return nil
	}
	_, h := m.readerInner()
	line := m.matches[m.matchIdx].line
	if line >= m.vp.YOffset && line < m.vp.YOffset+h {
		return nil
	}
	return m.scrollTo(line - h/3)
}

func (m *model) clearSearch() {
	m.searchState = searchOff
	m.search.Blur()
	m.search.Reset()
	m.matches = nil
	m.matchIdx = 0
	m.applyContent()
}

func (m model) openFinder() (model, tea.Cmd) {
	m.overlay = overlayFinder
	return m, m.finder.open(m.files)
}

func (m model) onFinderKey(msg tea.KeyMsg) (model, tea.Cmd) {
	switch {
	case key.Matches(msg, keys.Escape):
		m.overlay = overlayNone
		m.finder.close()
		return m, nil
	case msg.String() == "up" || msg.String() == "ctrl+k" || msg.String() == "ctrl+p":
		m.finder.move(-1)
		return m, nil
	case msg.String() == "down" || msg.String() == "ctrl+j" || msg.String() == "ctrl+n":
		m.finder.move(1)
		return m, nil
	case key.Matches(msg, keys.Enter):
		n := m.finder.selected()
		m.overlay = overlayNone
		m.finder.close()
		if n == nil {
			return m, nil
		}
		expandTo(m.tree, n)
		m.refreshRows()
		for i, r := range m.rows {
			if r == n {
				m.fileCursor = i
			}
		}
		m.setFocus(paneReader)
		return m, m.openNode(n, false)
	}
	return m, m.finder.update(msg)
}

// ---- mouse -------------------------------------------------------------------

type region int

const (
	regionNone region = iota
	regionFiles
	regionOutline
	regionReader
)

func (m model) regionAt(x, y int) (region, int) {
	l := m.layout()
	if y < 1 || y >= 1+l.bodyH || x < 0 || x >= m.width {
		return regionNone, -1
	}
	if m.sidebarVisible && x < l.sidebarW {
		if y < 1+l.filesH {
			return regionFiles, y - 2
		}
		return regionOutline, y - (1 + l.filesH) - 1
	}
	return regionReader, y - 2
}

func (m model) onMouse(msg tea.MouseMsg) (model, tea.Cmd) {
	if m.overlay != overlayNone || msg.Action != tea.MouseActionPress {
		return m, nil
	}
	reg, row := m.regionAt(msg.X, msg.Y)
	switch msg.Button {
	case tea.MouseButtonWheelUp, tea.MouseButtonWheelDown:
		delta := 1
		if msg.Button == tea.MouseButtonWheelUp {
			delta = -1
		}
		switch reg {
		case regionFiles:
			m.fileCursor = clamp(m.fileCursor+delta, 0, max(len(m.rows)-1, 0))
		case regionOutline:
			m.outlineCursor = clamp(m.outlineCursor+delta, 0, max(len(m.doc.headings)-1, 0))
		case regionReader:
			m.scrollAnimating = false
			if delta < 0 {
				m.vp.LineUp(3)
			} else {
				m.vp.LineDown(3)
			}
			m.syncOutline()
		}
		return m, nil
	case tea.MouseButtonLeft:
		switch reg {
		case regionFiles:
			m.setFocus(paneFiles)
			l := m.layout()
			tick := m.ensureTicking()
			// a collapsed pane shows a summary line, not rows
			if l.filesH-2 > 1 && row >= 0 && row < l.filesH-2 && m.fileOffset+row < len(m.rows) {
				m.fileCursor = m.fileOffset + row
				mm, cmd := m.activateRow(m.fileCursor)
				return mm, tea.Batch(cmd, tick)
			}
			return m, tick
		case regionOutline:
			m.setFocus(paneOutline)
			l := m.layout()
			if l.outlineH-2 > 1 && row >= 0 && row < l.outlineH-2 && m.outlineOffset+row < len(m.doc.headings) {
				m.outlineCursor = m.outlineOffset + row
				return m, tea.Batch(m.scrollTo(m.doc.headings[m.outlineCursor].line), m.ensureTicking())
			}
			return m, m.ensureTicking()
		case regionReader:
			m.setFocus(paneReader)
		}
	}
	return m, nil
}

// ---- view --------------------------------------------------------------------

func (m model) View() string {
	if !m.ready {
		return ""
	}
	if m.width < minWidth || m.height < minHeight {
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center,
			styleMuted.Render(fmt.Sprintf("docscope needs at least %dx%d", minWidth, minHeight)))
	}
	if m.rootErr != nil || len(m.files) == 0 {
		return m.viewEmpty()
	}

	l := m.layout()
	header := m.viewHeader()
	reader := m.viewReader(l)
	body := reader
	if m.sidebarVisible && l.sidebarW > 0 {
		sidebar := lipgloss.JoinVertical(lipgloss.Left, m.viewFiles(l), m.viewOutline(l))
		body = lipgloss.JoinHorizontal(lipgloss.Top, sidebar, reader)
	}
	base := strings.Join([]string{header, body, m.viewProgress(), m.viewStatus()}, "\n")

	switch m.overlay {
	case overlayFinder:
		return overlay(base, m.finder.view(m.width, m.height), m.width, m.height)
	case overlayHelp:
		return overlay(base, m.viewHelp(), m.width, m.height)
	}
	return base
}

func (m model) viewEmpty() string {
	var lines []string
	if m.rootErr != nil {
		lines = append(lines, styleError.Render("Cannot read "+m.root), "", styleMuted.Render(m.rootErr.Error()))
	} else {
		lines = append(lines,
			styleHeaderGold.Render("No markdown files found"),
			"",
			styleText.Render(clip(m.root, m.width-8)),
			"",
			styleMuted.Render("docscope looks for *.md up to 6 levels deep"),
			styleMuted.Render("(skipping node_modules, .git and vendor)"),
		)
	}
	lines = append(lines, "", styleKey.Render("r")+styleMuted.Render(" rescan   ")+styleKey.Render("q")+styleMuted.Render(" quit"))
	box := pane("docscope", lines, min(m.width-2, 64), len(lines)+2, true)
	return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, box)
}

func displayRoot(root string) string {
	if home, err := os.UserHomeDir(); err == nil && home != "" && strings.HasPrefix(root, home) {
		return "~" + strings.TrimPrefix(root, home)
	}
	return root
}

func (m model) progress() float64 {
	if !m.hasDoc {
		return 0
	}
	return m.vp.ScrollPercent()
}

func (m model) viewHeader() string {
	sep := styleHeaderDim.Render(" │ ")
	right := styleHeaderGold.Render(fmt.Sprintf("%3.0f%%", m.progress()*100))
	file := ""
	if m.current != nil {
		file = m.current.rel
	}
	docs := fmt.Sprintf("%d docs", len(m.files))
	if len(m.files) == 1 {
		docs = "1 doc"
	}
	badge := styleBadge.Render("docscope")
	avail := m.width - visibleWidth(badge) - visibleWidth(right) - 2
	rootStr := clip(displayRoot(m.root), max(avail/3, 8))
	fixed := visibleWidth(rootStr) + 3 + len(docs) + 3
	file = clip(file, max(avail-fixed, 0))
	left := badge + " " + styleHeaderDim.Render(rootStr) + sep + styleHeader.Render(docs) + sep + styleHeaderGold.Render(file)
	gap := m.width - visibleWidth(left) - visibleWidth(right) - 1
	if gap < 1 {
		left = fit(left, m.width-visibleWidth(right)-2)
		gap = 1
	}
	return fit(left+strings.Repeat(" ", gap)+right+" ", m.width)
}

func (m model) viewProgress() string {
	filled := int(math.Round(m.progress() * float64(m.width)))
	filled = clamp(filled, 0, m.width)
	return styleBarFill.Render(strings.Repeat("━", filled)) + styleBarEmpty.Render(strings.Repeat("─", m.width-filled))
}

func (m model) viewStatus() string {
	var left string
	if m.searchState != searchOff {
		count := styleMuted.Render("no matches")
		if len(m.matches) > 0 {
			count = styleKey.Render(fmt.Sprintf("%d/%d", m.matchIdx+1, len(m.matches)))
		}
		m.search.Width = max(min(m.width/3, 32), 8)
		hint := styleMuted.Render("enter next · n/p · esc clear")
		if m.searchState == searchTyping {
			hint = styleMuted.Render("enter jump · esc clear")
		}
		left = " " + m.search.View() + "  " + count + "  " + hint
	} else {
		parts := make([]string, 0, 10)
		for _, h := range statusHints(m.focus, m.sidebarVisible) {
			parts = append(parts, styleKey.Render(h.key)+" "+styleMuted.Render(h.desc))
		}
		left = " " + strings.Join(parts, styleHeaderDim.Render(" · "))
	}
	right := ""
	if m.toast != nil {
		st := styleToastOK
		switch m.toast.kind {
		case toastWarn:
			st = styleToastWarn
		case toastErr:
			st = styleToastErr
		}
		right = st.Render(clip(m.toast.text, max(m.width/2, 10)))
	}
	rw := visibleWidth(right)
	left = clip(left, max(m.width-rw-1, 0))
	gap := max(m.width-visibleWidth(left)-rw, 0)
	return fit(left+strings.Repeat(" ", gap)+right, m.width)
}

func (m model) viewFiles(l layout) string {
	innerH := l.filesH - 2
	innerW := l.sidebarW - 4
	title := fmt.Sprintf("Files · %d", len(m.files))
	focused := m.focus == paneFiles
	body := make([]string, 0, innerH)
	if innerH <= 1 {
		cur := styleMuted.Render("no file open")
		if m.current != nil {
			cur = styleMuted.Render(clip("▸ "+m.current.rel, innerW))
		}
		return pane(title, []string{cur}, l.sidebarW, l.filesH, focused)
	}
	offset := m.fileOffset
	for i := offset; i < len(m.rows) && i < offset+innerH; i++ {
		n := m.rows[i]
		indent := strings.Repeat("  ", max(n.depth-1, 0))
		var label string
		switch {
		case n.isDir && n.expanded:
			label = indent + "▾ " + n.name + "/"
		case n.isDir:
			label = indent + "▸ " + n.name + "/"
		default:
			label = indent + "  " + n.name
		}
		label = fit(clip(label, innerW), innerW)
		switch {
		case i == m.fileCursor && focused:
			body = append(body, styleSelected.Render(label))
		case i == m.fileCursor:
			body = append(body, styleCurrent.Render(label))
		case n == m.current:
			body = append(body, styleCurrent.Render(label))
		case n.isDir:
			body = append(body, styleDir.Render(label))
		default:
			body = append(body, styleFile.Render(label))
		}
	}
	return pane(title, body, l.sidebarW, l.filesH, focused)
}

func (m model) viewOutline(l layout) string {
	innerH := l.outlineH - 2
	innerW := l.sidebarW - 4
	hs := m.doc.headings
	title := fmt.Sprintf("Outline · %d", len(hs))
	focused := m.focus == paneOutline
	cur := m.currentHeading()
	if innerH <= 1 {
		line := styleMuted.Render("no headings")
		if len(hs) > 0 {
			line = styleMuted.Render(clip("▸ "+hs[cur].title, innerW))
		}
		return pane(title, []string{line}, l.sidebarW, l.outlineH, focused)
	}
	if len(hs) == 0 {
		msg := "no headings"
		if m.rendering {
			msg = "rendering…"
		} else if m.current == nil {
			msg = "no file open"
		}
		return pane(title, []string{styleMuted.Render(msg)}, l.sidebarW, l.outlineH, focused)
	}
	offset := m.outlineOffset
	body := make([]string, 0, innerH)
	for i := offset; i < len(hs) && i < offset+innerH; i++ {
		h := hs[i]
		label := strings.Repeat(" ", h.level-1) + strings.Repeat("#", h.level) + " " + h.title
		if i == cur {
			label = clip(label, innerW-2) + " ◀"
		}
		label = fit(clip(label, innerW), innerW)
		switch {
		case i == m.outlineCursor && focused:
			body = append(body, styleSelected.Render(label))
		case i == cur:
			body = append(body, styleCurrent.Render(label))
		default:
			body = append(body, styleText.Render(label))
		}
	}
	return pane(title, body, l.sidebarW, l.outlineH, focused)
}

func (m model) viewReader(l layout) string {
	innerW, innerH := m.readerInner()
	focused := m.focus == paneReader
	title := "Reader"
	if m.current != nil {
		title = m.current.rel
	}
	var body []string
	switch {
	case m.rendering:
		name := ""
		if m.current != nil {
			name = m.current.name
		}
		msg := m.spinner.View() + " " + styleMuted.Render("Rendering "+clip(name, max(innerW-14, 4))+"…")
		body = centered(msg, innerW, innerH)
	case m.renderErr != nil:
		body = centered(styleError.Render(clip(m.renderErr.Error(), innerW)), innerW, innerH)
	case m.current == nil:
		body = centered(styleMuted.Render("select a file"), innerW, innerH)
	default:
		body = strings.Split(m.vp.View(), "\n")
	}
	return pane(title, body, l.readerW, l.bodyH, focused)
}

func centered(msg string, w, h int) []string {
	body := make([]string, h)
	pad := max((w-visibleWidth(msg))/2, 0)
	body[h/2] = strings.Repeat(" ", pad) + msg
	return body
}

func (m model) viewHelp() string {
	boxW := min(m.width-2, 80)
	innerW := boxW - 4
	colW := innerW/2 - 1
	sections := helpSections()
	col := func(secs []helpSection) []string {
		var out []string
		for i, s := range secs {
			if i > 0 {
				out = append(out, "")
			}
			out = append(out, styleHeaderGold.Render(s.title))
			for _, r := range s.rows {
				out = append(out, styleKey.Render(fit(r.key, 15))+" "+styleText.Render(clip(r.desc, colW-16)))
			}
		}
		return out
	}
	left := col([]helpSection{sections[0], sections[1], sections[3]})
	right := col([]helpSection{sections[2], sections[4]})
	rows := max(len(left), len(right))
	body := make([]string, 0, rows)
	for i := 0; i < rows; i++ {
		var a, b string
		if i < len(left) {
			a = left[i]
		}
		if i < len(right) {
			b = right[i]
		}
		body = append(body, fit(a, colW)+"  "+b)
	}
	boxH := min(m.height-2, len(body)+2)
	return pane("Keys · docscope", body, boxW, boxH, true)
}
