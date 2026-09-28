package main

import (
	"fmt"
	"os"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	task := "file browser with fuzzy search, preview pane, and vim keybindings"
	if len(os.Args) > 1 {
		task = strings.Join(os.Args[1:], " ")
	}
	p := tea.NewProgram(newModel(task), tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
