package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}
	root, err := repoRoot(cwd)
	if err != nil {
		root = ""
	}
	p := tea.NewProgram(newModel(root, cwd), tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "gitscope:", err)
		os.Exit(1)
	}
}
