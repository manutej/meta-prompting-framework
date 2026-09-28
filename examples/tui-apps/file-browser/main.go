package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	gold = lipgloss.Color("178")
	navy = lipgloss.Color("24")
	gray = lipgloss.Color("240")
	red  = lipgloss.Color("196")

	titleStyle = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true).Padding(0, 1)
	pathStyle  = lipgloss.NewStyle().Foreground(gold).Bold(true)
	selStyle   = lipgloss.NewStyle().Foreground(gold).Background(navy).Bold(true)
	dirStyle   = lipgloss.NewStyle().Foreground(gold)
	fileStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	dimStyle   = lipgloss.NewStyle().Foreground(gray)
	errStyle   = lipgloss.NewStyle().Foreground(red)
	paneStyle  = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(navy).Padding(0, 1)
	focusPane  = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(gold).Padding(0, 1)
)

const maxPreviewBytes = 64 * 1024

type entry struct {
	name  string
	isDir bool
	size  int64
}

type model struct {
	cwd      string
	entries  []entry
	filtered []entry
	cursor   int
	offset   int
	search   textinput.Model
	preview  viewport.Model
	focus    int // 0 = list, 1 = preview
	width    int
	height   int
	err      string
	ready    bool
}

func initialModel() model {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "/"
	}
	ti := textinput.New()
	ti.Placeholder = "type / to search"
	ti.Prompt = "/ "
	ti.PromptStyle = pathStyle
	ti.TextStyle = fileStyle
	ti.CharLimit = 64

	m := model{cwd: cwd, search: ti}
	m.loadDir()
	return m
}

func (m *model) loadDir() {
	m.err = ""
	dirents, err := os.ReadDir(m.cwd)
	if err != nil {
		m.err = err.Error()
		m.entries = nil
		m.applyFilter()
		return
	}
	entries := make([]entry, 0, len(dirents)+1)
	if m.cwd != "/" && filepath.Dir(m.cwd) != m.cwd {
		entries = append(entries, entry{name: "..", isDir: true})
	}
	for _, d := range dirents {
		e := entry{name: d.Name(), isDir: d.IsDir()}
		if info, err := d.Info(); err == nil {
			e.size = info.Size()
		}
		entries = append(entries, e)
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].name == ".." {
			return true
		}
		if entries[j].name == ".." {
			return false
		}
		if entries[i].isDir != entries[j].isDir {
			return entries[i].isDir
		}
		return strings.ToLower(entries[i].name) < strings.ToLower(entries[j].name)
	})
	m.entries = entries
	m.cursor = 0
	m.offset = 0
	m.applyFilter()
}

// fuzzyMatch reports whether every rune of pattern appears in s in order.
func fuzzyMatch(pattern, s string) bool {
	p := []rune(strings.ToLower(pattern))
	s = strings.ToLower(s)
	pi := 0
	for _, r := range s {
		if pi < len(p) && p[pi] == r {
			pi++
		}
	}
	return pi == len(p)
}

func (m *model) applyFilter() {
	q := m.search.Value()
	if q == "" {
		m.filtered = m.entries
	} else {
		m.filtered = m.filtered[:0:0]
		for _, e := range m.entries {
			if e.name == ".." || fuzzyMatch(q, e.name) {
				m.filtered = append(m.filtered, e)
			}
		}
	}
	if m.cursor >= len(m.filtered) {
		m.cursor = max(0, len(m.filtered)-1)
	}
	m.updatePreview()
}

func (m *model) updatePreview() {
	if len(m.filtered) == 0 {
		m.preview.SetContent(dimStyle.Render("(empty)"))
		return
	}
	e := m.filtered[m.cursor]
	full := filepath.Join(m.cwd, e.name)
	if e.isDir {
		dirents, err := os.ReadDir(full)
		if err != nil {
			m.preview.SetContent(errStyle.Render(err.Error()))
			return
		}
		var b strings.Builder
		b.WriteString(dimStyle.Render(fmt.Sprintf("%d items", len(dirents))) + "\n\n")
		for i, d := range dirents {
			if i >= 200 {
				b.WriteString(dimStyle.Render("...\n"))
				break
			}
			if d.IsDir() {
				b.WriteString(dirStyle.Render(d.Name()+"/") + "\n")
			} else {
				b.WriteString(fileStyle.Render(d.Name()) + "\n")
			}
		}
		m.preview.SetContent(b.String())
		m.preview.GotoTop()
		return
	}
	f, err := os.Open(full)
	if err != nil {
		m.preview.SetContent(errStyle.Render(err.Error()))
		return
	}
	defer f.Close()
	buf := make([]byte, maxPreviewBytes)
	n, _ := io.ReadFull(f, buf)
	data := buf[:n]
	if isBinary(data) {
		m.preview.SetContent(dimStyle.Render(fmt.Sprintf("binary file (%s)", humanSize(e.size))))
		m.preview.GotoTop()
		return
	}
	content := strings.ReplaceAll(string(data), "\t", "    ")
	if int64(n) < e.size {
		content += "\n" + dimStyle.Render("... (truncated)")
	}
	m.preview.SetContent(content)
	m.preview.GotoTop()
}

func isBinary(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	for _, c := range b[:min(len(b), 512)] {
		if c == 0 {
			return true
		}
	}
	return false
}

// truncateWidth cuts s to at most w terminal cells without splitting a rune.
func truncateWidth(s string, w int) string {
	var b strings.Builder
	cells := 0
	for _, r := range s {
		rw := lipgloss.Width(string(r))
		if cells+rw > w {
			break
		}
		b.WriteRune(r)
		cells += rw
	}
	return b.String()
}

func humanSize(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

func (m model) Init() tea.Cmd { return nil }

func (m model) listHeight() int {
	return max(3, m.height-6)
}

func (m *model) clampScroll() {
	h := m.listHeight()
	if m.cursor < m.offset {
		m.offset = m.cursor
	}
	if m.cursor >= m.offset+h {
		m.offset = m.cursor - h + 1
	}
	if m.offset < 0 {
		m.offset = 0
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		pw := max(20, m.width/2-4) - 2 // pane Width includes its 1-cell padding on each side
		ph := m.listHeight()
		if !m.ready {
			m.preview = viewport.New(pw, ph)
			m.ready = true
		} else {
			m.preview.Width, m.preview.Height = pw, ph
		}
		m.updatePreview()
		return m, nil

	case tea.KeyMsg:
		if m.search.Focused() {
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc":
				m.search.Blur()
				m.search.SetValue("")
				m.applyFilter()
				return m, nil
			case "enter":
				m.search.Blur()
				return m, nil
			}
			var cmd tea.Cmd
			m.search, cmd = m.search.Update(msg)
			m.applyFilter()
			return m, cmd
		}

		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "/":
			m.search.Focus()
			return m, textinput.Blink
		case "tab":
			m.focus = (m.focus + 1) % 2
		case "j", "down":
			if m.focus == 1 {
				m.preview.LineDown(1)
			} else if m.cursor < len(m.filtered)-1 {
				m.cursor++
				m.clampScroll()
				m.updatePreview()
			}
		case "k", "up":
			if m.focus == 1 {
				m.preview.LineUp(1)
			} else if m.cursor > 0 {
				m.cursor--
				m.clampScroll()
				m.updatePreview()
			}
		case "g", "home":
			if m.focus == 1 {
				m.preview.GotoTop()
			} else {
				m.cursor = 0
				m.clampScroll()
				m.updatePreview()
			}
		case "G", "end":
			if m.focus == 1 {
				m.preview.GotoBottom()
			} else {
				m.cursor = max(0, len(m.filtered)-1)
				m.clampScroll()
				m.updatePreview()
			}
		case "ctrl+d", "pgdown":
			if m.focus == 1 {
				m.preview.HalfViewDown()
			} else {
				m.cursor = max(0, min(len(m.filtered)-1, m.cursor+m.listHeight()/2))
				m.clampScroll()
				m.updatePreview()
			}
		case "ctrl+u", "pgup":
			if m.focus == 1 {
				m.preview.HalfViewUp()
			} else {
				m.cursor = max(0, m.cursor-m.listHeight()/2)
				m.clampScroll()
				m.updatePreview()
			}
		case "enter", "l", "right":
			if len(m.filtered) == 0 {
				break
			}
			e := m.filtered[m.cursor]
			if e.isDir {
				m.cwd = filepath.Clean(filepath.Join(m.cwd, e.name))
				m.search.SetValue("")
				m.loadDir()
			}
		case "h", "left", "backspace":
			parent := filepath.Dir(m.cwd)
			if parent != m.cwd {
				prev := filepath.Base(m.cwd)
				m.cwd = parent
				m.search.SetValue("")
				m.loadDir()
				for i, e := range m.filtered {
					if e.name == prev {
						m.cursor = i
						break
					}
				}
				m.clampScroll()
				m.updatePreview()
			}
		}
	}
	return m, nil
}

func (m model) View() string {
	if !m.ready {
		return "loading..."
	}
	h := m.listHeight()
	lw := max(20, m.width/2-4)
	iw := lw - 2 // content width inside the pane: Width(lw) minus Padding(0, 1)

	var list strings.Builder
	end := min(len(m.filtered), m.offset+h)
	for i := m.offset; i < end; i++ {
		e := m.filtered[i]
		name := e.name
		if e.isDir {
			name += "/"
		}
		if lipgloss.Width(name) > iw-12 {
			name = truncateWidth(name, iw-13) + "…"
		}
		sz := ""
		if !e.isDir && e.name != ".." {
			sz = humanSize(e.size)
		}
		pad := strings.Repeat(" ", max(0, iw-11-lipgloss.Width(name)))
		line := fmt.Sprintf("%s%s %10s", name, pad, sz)
		switch {
		case i == m.cursor:
			list.WriteString(selStyle.Render(line))
		case e.isDir:
			list.WriteString(dirStyle.Render(line))
		default:
			list.WriteString(fileStyle.Render(line))
		}
		list.WriteString("\n")
	}
	for i := end - m.offset; i < h; i++ {
		list.WriteString("\n")
	}

	leftStyle, rightStyle := focusPane, paneStyle
	if m.focus == 1 {
		leftStyle, rightStyle = paneStyle, focusPane
	}
	left := leftStyle.Width(lw).Height(h).Render(strings.TrimRight(list.String(), "\n"))
	right := rightStyle.Width(lw).Height(h).Render(m.preview.View())

	header := titleStyle.Render("📁 FILE BROWSER") + " " + pathStyle.Render(m.cwd)
	if m.err != "" {
		header += "  " + errStyle.Render(m.err)
	}
	status := fmt.Sprintf("%d/%d", m.cursor+1, len(m.filtered))
	if m.search.Focused() || m.search.Value() != "" {
		status = m.search.View() + "  " + dimStyle.Render(status)
	} else {
		status = dimStyle.Render(status)
	}
	help := dimStyle.Render("j/k move • enter/l open • h back • / search • tab focus pane • g/G top/bottom • q quit")

	return lipgloss.JoinVertical(lipgloss.Left,
		header,
		lipgloss.JoinHorizontal(lipgloss.Top, left, " ", right),
		status,
		help,
	)
}

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
