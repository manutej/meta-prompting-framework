package main

import (
	"fmt"
	"os"
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	root, err := os.Getwd()
	if err != nil {
		root = "."
	}
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	p := tea.NewProgram(newModel(root), tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "docscope:", err)
		os.Exit(1)
	}
}
