package main

import (
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/ansi"
	"github.com/charmbracelet/lipgloss"
)

// Gold & Navy palette shared by the chrome (styles.go) and the markdown renderer.
const (
	ansiGold     = "#d4a017"
	ansiNavy     = "#0e1830"
	ansiDarkNavy = "#1a1407"
	ansiText     = "#eceae3"
	ansiMuted    = "#9ca3af"
	ansiSuccess  = "#22c55e"
	ansiWarn     = "#fbbf24"
	ansiError    = "#ef4444"
	ansiAccent   = "#60a5fa"
)

var (
	colGold     = lipgloss.Color(ansiGold)
	colNavy     = lipgloss.Color(ansiNavy)
	colDarkNavy = lipgloss.Color(ansiDarkNavy)
	colText     = lipgloss.Color(ansiText)
	colMuted    = lipgloss.Color(ansiMuted)
	colSuccess  = lipgloss.Color(ansiSuccess)
	colWarn     = lipgloss.Color(ansiWarn)
	colError    = lipgloss.Color(ansiError)
	colAccent   = lipgloss.Color(ansiAccent)
)

func strPtr(s string) *string { return &s }
func boolPtr(b bool) *bool    { return &b }
func uintPtr(u uint) *uint    { return &u }

// docscopeStyle is glamour's dark style re-skinned in Gold & Navy.
func docscopeStyle() ansi.StyleConfig {
	s := glamour.DarkStyleConfig

	s.Document.BlockPrefix = ""
	s.Document.BlockSuffix = "\n"
	s.Document.Color = strPtr(ansiText)
	s.Document.Margin = uintPtr(2)

	s.Heading.Color = strPtr(ansiGold)
	s.Heading.Bold = boolPtr(true)
	s.H1.Prefix = " "
	s.H1.Suffix = " "
	s.H1.Color = strPtr(ansiGold)
	s.H1.BackgroundColor = strPtr(ansiNavy)
	s.H1.Bold = boolPtr(true)
	s.H2.Prefix = "## "
	s.H2.Color = strPtr(ansiGold)
	s.H3.Prefix = "### "
	s.H3.Color = strPtr(ansiGold)
	s.H4.Color = strPtr(ansiGold)
	s.H5.Color = strPtr(ansiGold)
	s.H6.Color = strPtr(ansiGold)
	s.H6.Bold = boolPtr(true)

	s.Link.Color = strPtr(ansiAccent)
	s.Link.Underline = boolPtr(true)
	s.LinkText.Color = strPtr(ansiAccent)
	s.LinkText.Bold = boolPtr(true)
	s.LinkText.Underline = boolPtr(true)

	s.Code.Prefix = " "
	s.Code.Suffix = " "
	s.Code.Color = strPtr(ansiAccent)
	s.Code.BackgroundColor = strPtr(ansiDarkNavy)

	s.CodeBlock.Margin = uintPtr(0)
	s.CodeBlock.Color = strPtr(ansiText)

	// The indent token is rendered with the parent block's style, so the bar
	// carries its own colour and the quoted text keeps the muted italic look.
	s.BlockQuote.Indent = uintPtr(1)
	s.BlockQuote.IndentToken = strPtr(lipgloss.NewStyle().Foreground(colGold).Render("▌") + " ")
	s.BlockQuote.Color = strPtr(ansiMuted)
	s.BlockQuote.Italic = boolPtr(true)

	s.Item.BlockPrefix = ""
	s.Item.Prefix = "◆ "
	s.Item.Color = strPtr(ansiGold)
	s.Enumeration.BlockPrefix = ". "

	s.HorizontalRule.Color = strPtr(ansiNavy)
	s.HorizontalRule.Format = "\n" + strings.Repeat("─", 36) + "\n"

	s.Table.CenterSeparator = strPtr("┼")
	s.Table.ColumnSeparator = strPtr("│")
	s.Table.RowSeparator = strPtr("─")
	s.Table.Color = strPtr(ansiText)

	s.Task.Ticked = "[✓] "
	s.Task.Unticked = "[ ] "
	s.Emph.Italic = boolPtr(true)
	s.Strong.Bold = boolPtr(true)
	s.Strong.Color = strPtr(ansiGold)

	return s
}
