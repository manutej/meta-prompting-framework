package main

import (
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"
)

var (
	colGold     = lipgloss.Color("178")
	colNavy     = lipgloss.Color("24")
	colDarkNavy = lipgloss.Color("17")
	colText     = lipgloss.Color("252")
	colMuted    = lipgloss.Color("240")
	colSuccess  = lipgloss.Color("76")
	colWarn     = lipgloss.Color("220")
	colError    = lipgloss.Color("196")
	colAdded    = lipgloss.Color("76")
	colRemoved  = lipgloss.Color("196")
	colHunk     = lipgloss.Color("51")
	colBlack    = lipgloss.Color("0")
	colWhite    = lipgloss.Color("231")

	stText       = lipgloss.NewStyle().Foreground(colText)
	stMuted      = lipgloss.NewStyle().Foreground(colMuted)
	stGold       = lipgloss.NewStyle().Foreground(colGold)
	stGoldBold   = lipgloss.NewStyle().Foreground(colGold).Bold(true)
	stSuccess    = lipgloss.NewStyle().Foreground(colSuccess)
	stWarn       = lipgloss.NewStyle().Foreground(colWarn)
	stError      = lipgloss.NewStyle().Foreground(colError)
	stAdded      = lipgloss.NewStyle().Foreground(colAdded)
	stRemoved    = lipgloss.NewStyle().Foreground(colRemoved)
	stHunk       = lipgloss.NewStyle().Foreground(colHunk)
	stFileHeader = lipgloss.NewStyle().Foreground(colGold).Bold(true)

	stBorderFocused = lipgloss.NewStyle().Foreground(colGold)
	stBorder        = lipgloss.NewStyle().Foreground(colNavy)
	stSelected      = lipgloss.NewStyle().Foreground(colGold).Background(colNavy).Bold(true)
	stSectionHeader = lipgloss.NewStyle().Foreground(colGold).Bold(true)

	stBadge      = lipgloss.NewStyle().Foreground(colGold).Background(colNavy).Bold(true).Padding(0, 1)
	stBar        = lipgloss.NewStyle().Foreground(colText).Background(colDarkNavy)
	stBarBold    = lipgloss.NewStyle().Foreground(colText).Background(colDarkNavy).Bold(true)
	stBarGold    = lipgloss.NewStyle().Foreground(colGold).Background(colDarkNavy).Bold(true)
	stBarMuted   = lipgloss.NewStyle().Foreground(colMuted).Background(colDarkNavy)
	stBarSuccess = lipgloss.NewStyle().Foreground(colSuccess).Background(colDarkNavy)
	stBarWarn    = lipgloss.NewStyle().Foreground(colWarn).Background(colDarkNavy)

	stToastOK  = lipgloss.NewStyle().Foreground(colBlack).Background(colSuccess).Bold(true).Padding(0, 1)
	stToastErr = lipgloss.NewStyle().Foreground(colWhite).Background(colError).Bold(true).Padding(0, 1)

	stOverlay = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(colGold).
			Padding(0, 2)
	stDim = lipgloss.NewStyle().Foreground(colMuted)
)

func huhTheme() *huh.Theme {
	t := huh.ThemeBase()
	f := &t.Focused
	f.Base = f.Base.BorderForeground(colGold)
	f.Title = lipgloss.NewStyle().Foreground(colGold).Bold(true)
	f.Description = lipgloss.NewStyle().Foreground(colMuted)
	f.ErrorIndicator = lipgloss.NewStyle().Foreground(colError)
	f.ErrorMessage = lipgloss.NewStyle().Foreground(colError)
	f.FocusedButton = lipgloss.NewStyle().Foreground(colBlack).Background(colGold).Bold(true).Padding(0, 2).MarginRight(1)
	f.BlurredButton = lipgloss.NewStyle().Foreground(colText).Background(colNavy).Padding(0, 2).MarginRight(1)
	f.TextInput.Cursor = lipgloss.NewStyle().Foreground(colGold)
	f.TextInput.Placeholder = lipgloss.NewStyle().Foreground(colMuted)
	f.TextInput.Prompt = lipgloss.NewStyle().Foreground(colGold)
	f.TextInput.Text = lipgloss.NewStyle().Foreground(colText)

	b := &t.Blurred
	b.Base = b.Base.BorderForeground(colNavy)
	b.Title = lipgloss.NewStyle().Foreground(colMuted)
	b.Description = lipgloss.NewStyle().Foreground(colMuted)
	b.TextInput.Prompt = lipgloss.NewStyle().Foreground(colMuted)
	b.TextInput.Text = lipgloss.NewStyle().Foreground(colText)
	b.TextInput.Placeholder = lipgloss.NewStyle().Foreground(colMuted)
	b.FocusedButton = f.FocusedButton
	b.BlurredButton = f.BlurredButton

	t.Help.ShortKey = lipgloss.NewStyle().Foreground(colGold)
	t.Help.ShortDesc = lipgloss.NewStyle().Foreground(colMuted)
	t.Help.ShortSeparator = lipgloss.NewStyle().Foreground(colNavy)
	return t
}
