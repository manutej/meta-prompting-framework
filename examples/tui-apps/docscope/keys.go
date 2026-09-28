package main

import "github.com/charmbracelet/bubbles/key"

type keyMap struct {
	Quit, Help, Escape                    key.Binding
	NextPane, PrevPane                    key.Binding
	FocusFiles, FocusOutline, FocusReader key.Binding
	Left, Right                           key.Binding
	ToggleSidebar                         key.Binding
	Up, Down, Enter                       key.Binding
	Collapse, Expand                      key.Binding
	HalfDown, HalfUp                      key.Binding
	PageDown, PageUp                      key.Binding
	Top, Bottom                           key.Binding
	PrevHeading, NextHeading              key.Binding
	NextMatch, PrevMatch                  key.Binding
	Search, Finder                        key.Binding
	Copy, Edit, Reload                    key.Binding
}

var keys = keyMap{
	Quit:          key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit")),
	Help:          key.NewBinding(key.WithKeys("?"), key.WithHelp("?", "help")),
	Escape:        key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "close")),
	NextPane:      key.NewBinding(key.WithKeys("tab"), key.WithHelp("tab", "next pane")),
	PrevPane:      key.NewBinding(key.WithKeys("shift+tab"), key.WithHelp("shift+tab", "prev pane")),
	FocusFiles:    key.NewBinding(key.WithKeys("1"), key.WithHelp("1", "files")),
	FocusOutline:  key.NewBinding(key.WithKeys("2"), key.WithHelp("2", "outline")),
	FocusReader:   key.NewBinding(key.WithKeys("3"), key.WithHelp("3", "reader")),
	Left:          key.NewBinding(key.WithKeys("h"), key.WithHelp("h", "sidebar")),
	Right:         key.NewBinding(key.WithKeys("l"), key.WithHelp("l", "reader")),
	ToggleSidebar: key.NewBinding(key.WithKeys("b"), key.WithHelp("b", "toggle sidebar")),
	Up:            key.NewBinding(key.WithKeys("k", "up"), key.WithHelp("k/↑", "up")),
	Down:          key.NewBinding(key.WithKeys("j", "down"), key.WithHelp("j/↓", "down")),
	Enter:         key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "open")),
	Collapse:      key.NewBinding(key.WithKeys("left"), key.WithHelp("←", "collapse dir")),
	Expand:        key.NewBinding(key.WithKeys("right"), key.WithHelp("→", "expand dir")),
	HalfDown:      key.NewBinding(key.WithKeys("ctrl+d"), key.WithHelp("ctrl+d", "half page down")),
	HalfUp:        key.NewBinding(key.WithKeys("ctrl+u"), key.WithHelp("ctrl+u", "half page up")),
	PageDown:      key.NewBinding(key.WithKeys(" ", "pgdown"), key.WithHelp("space/pgdn", "page down")),
	PageUp:        key.NewBinding(key.WithKeys("pgup"), key.WithHelp("pgup", "page up")),
	Top:           key.NewBinding(key.WithKeys("g", "home"), key.WithHelp("g", "top")),
	Bottom:        key.NewBinding(key.WithKeys("G", "end"), key.WithHelp("G", "bottom")),
	PrevHeading:   key.NewBinding(key.WithKeys("["), key.WithHelp("[", "prev heading")),
	NextHeading:   key.NewBinding(key.WithKeys("]"), key.WithHelp("]", "next heading")),
	NextMatch:     key.NewBinding(key.WithKeys("n"), key.WithHelp("n", "next match")),
	PrevMatch:     key.NewBinding(key.WithKeys("p"), key.WithHelp("p", "prev match")),
	Search:        key.NewBinding(key.WithKeys("/"), key.WithHelp("/", "search")),
	Finder:        key.NewBinding(key.WithKeys("ctrl+p"), key.WithHelp("ctrl+p", "find file")),
	Copy:          key.NewBinding(key.WithKeys("y"), key.WithHelp("y", "copy path")),
	Edit:          key.NewBinding(key.WithKeys("e"), key.WithHelp("e", "edit in $EDITOR")),
	Reload:        key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "reload")),
}

type helpRow struct{ key, desc string }

type helpSection struct {
	title string
	rows  []helpRow
}

func row(b key.Binding) helpRow { h := b.Help(); return helpRow{h.Key, h.Desc} }

func helpSections() []helpSection {
	return []helpSection{
		{"Navigation", []helpRow{
			{"tab / shift+tab", "next / prev pane"},
			{"1 / 2 / 3", "files·outline·reader"},
			{"h / l", "sidebar / reader"},
			row(keys.ToggleSidebar),
		}},
		{"Files & Outline", []helpRow{
			row(keys.Down), row(keys.Up),
			{"enter", "open · toggle · jump"},
			row(keys.Collapse), row(keys.Expand),
			{"/ · ctrl+p", "fuzzy file finder"},
			{"click", "open · toggle · jump"},
		}},
		{"Reader", []helpRow{
			{"j / k / ↓ / ↑", "scroll one line"},
			row(keys.HalfDown), row(keys.HalfUp),
			row(keys.PageDown), row(keys.PageUp),
			{"g / G", "top / bottom"},
			{"[ / ]", "prev / next heading"},
			{"wheel", "scroll"},
		}},
		{"Search", []helpRow{
			{"/", "search in document"},
			{"enter", "jump to match"},
			{"n / p", "next / prev match"},
			{"esc", "clear search"},
		}},
		{"Actions", []helpRow{
			row(keys.Copy), row(keys.Edit), row(keys.Reload),
			row(keys.Help), {"esc", "close overlay"}, row(keys.Quit),
		}},
	}
}

// statusHints returns the key hints for the status bar, most useful first.
func statusHints(focus focusPane, sidebarVisible bool) []helpRow {
	switch focus {
	case paneFiles:
		return []helpRow{{"j/k", "move"}, {"enter", "open"}, {"←/→", "fold"}, {"/", "find"}, {"tab", "pane"}, {"b", "sidebar"}, {"?", "help"}, {"q", "quit"}}
	case paneOutline:
		return []helpRow{{"j/k", "move"}, {"enter", "jump"}, {"tab", "pane"}, {"ctrl+p", "find"}, {"b", "sidebar"}, {"?", "help"}, {"q", "quit"}}
	default:
		hints := []helpRow{{"j/k", "scroll"}, {"[/]", "heading"}, {"/", "search"}, {"ctrl+p", "find"}, {"y", "copy"}, {"e", "edit"}}
		if sidebarVisible {
			hints = append(hints, helpRow{"h", "sidebar"})
		} else {
			hints = append(hints, helpRow{"b", "sidebar"})
		}
		return append(hints, helpRow{"?", "help"}, helpRow{"q", "quit"})
	}
}
