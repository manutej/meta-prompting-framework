package main

import (
	"fmt"
	"os"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	padding  = 2
	maxWidth = 80
)

var (
	// Gold & Navy theme
	goldColor = lipgloss.Color("178")
	navyColor = lipgloss.Color("24")

	titleStyle = lipgloss.NewStyle().
		Foreground(goldColor).
		Background(navyColor).
		Bold(true).
		Padding(0, 2).
		Width(maxWidth - 4)

	helpStyle = lipgloss.NewStyle().
		Foreground(lipgloss.Color("240"))
)

type tickMsg time.Time

type model struct {
	progress  progress.Model
	percent   float64
	duration  time.Duration
	startTime time.Time
}

func initialModel() model {
	prog := progress.New(
		progress.WithSolidFill(string(goldColor)),
		progress.WithWidth(maxWidth-padding*2-4),
	)

	// Gold fill on Navy track (FullColor is ignored in gradient mode, so
	// the bar must be built with a solid fill for the theme to apply).
	prog.FullColor = string(goldColor)
	prog.EmptyColor = string(navyColor)

	return model{
		progress:  prog,
		percent:   0.0,
		duration:  30 * time.Second, // 30 second timer
		startTime: time.Now(),
	}
}

func (m model) Init() tea.Cmd {
	return tickCmd()
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		case "r":
			// Reset timer. Keep the bar width from the last WindowSizeMsg, and
			// don't start a second tick loop: the one from Init() is still
			// running (it only ends with tea.Quit).
			nm := initialModel()
			nm.progress.Width = m.progress.Width
			return nm, nil
		}

	case tea.WindowSizeMsg:
		m.progress.Width = msg.Width - padding*2 - 4
		if m.progress.Width > maxWidth {
			m.progress.Width = maxWidth
		}
		return m, nil

	case tickMsg:
		elapsed := time.Since(m.startTime)
		m.percent = float64(elapsed) / float64(m.duration)

		if m.percent >= 1.0 {
			m.percent = 1.0
			return m, tea.Quit
		}

		return m, tickCmd()

	case progress.FrameMsg:
		progressModel, cmd := m.progress.Update(msg)
		m.progress = progressModel.(progress.Model)
		return m, cmd
	}

	return m, nil
}

func (m model) View() string {
	elapsed := time.Since(m.startTime)
	remaining := m.duration - elapsed
	if remaining < 0 {
		remaining = 0
	}

	pad := lipgloss.NewStyle().Padding(1, 2)

	title := titleStyle.Render("⏱  PROGRESS TIMER")

	progressBar := m.progress.ViewAs(m.percent)

	timeInfo := lipgloss.NewStyle().
		Foreground(goldColor).
		Bold(true).
		Render(fmt.Sprintf("Time: %s / %s",
			elapsed.Round(time.Millisecond),
			m.duration))

	percentInfo := lipgloss.NewStyle().
		Foreground(goldColor).
		Render(fmt.Sprintf("Progress: %.1f%%", m.percent*100))

	var status string
	if m.percent >= 1.0 {
		status = lipgloss.NewStyle().
			Foreground(lipgloss.Color("76")).
			Bold(true).
			Render("✓ COMPLETE!")
	} else {
		status = lipgloss.NewStyle().
			Foreground(goldColor).
			Render(fmt.Sprintf("Remaining: %s", remaining.Round(time.Millisecond)))
	}

	help := helpStyle.Render("r: reset • q/esc: quit")

	content := lipgloss.JoinVertical(
		lipgloss.Left,
		title,
		"",
		timeInfo,
		percentInfo,
		"",
		progressBar,
		"",
		status,
		"",
		help,
	)

	return pad.Render(content)
}

func tickCmd() tea.Cmd {
	return tea.Tick(50*time.Millisecond, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func main() {
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v", err)
		os.Exit(1)
	}
}
