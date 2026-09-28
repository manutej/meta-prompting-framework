// alembic is an operator console for an agent harness: it tails the harness
// feed, writes commands to its outbox, manages git worktrees and gates
// decisions through Jev question packs.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"alembic/harness"
)

func defaultConfig() (config, error) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = "."
	}
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}
	cfg := config{
		Feed:     os.Getenv("ALEMBIC_FEED"),
		Outbox:   os.Getenv("ALEMBIC_OUTBOX"),
		Packs:    os.Getenv("ALEMBIC_PACKS"),
		Receipts: filepath.Join(home, ".alembic", "receipts"),
		Repo:     cwd,
		Poll:     500 * time.Millisecond,
	}
	if cfg.Feed == "" {
		cfg.Feed = filepath.Join(home, ".ormus", "feed.jsonl")
	}
	if cfg.Outbox == "" {
		cfg.Outbox = filepath.Join(home, ".ormus", "outbox.jsonl")
	}
	if cfg.Packs == "" {
		cfg.Packs = filepath.Join(home, ".alembic", "packs")
		candidates := []string{"packs"}
		if exe, err := os.Executable(); err == nil {
			candidates = append(candidates, filepath.Join(filepath.Dir(exe), "packs"))
		}
		for _, c := range candidates {
			if st, err := os.Stat(c); err == nil && st.IsDir() {
				cfg.Packs, _ = filepath.Abs(c)
				break
			}
		}
	}
	return cfg, nil
}

func parseFlags(args []string) (config, error) {
	cfg, err := defaultConfig()
	if err != nil {
		return cfg, err
	}
	fs := flag.NewFlagSet("alembic", flag.ContinueOnError)
	fs.StringVar(&cfg.Feed, "feed", cfg.Feed, "harness feed (JSONL, append-only) [$ALEMBIC_FEED]")
	fs.StringVar(&cfg.Outbox, "outbox", cfg.Outbox, "command outbox (JSONL) [$ALEMBIC_OUTBOX]")
	fs.StringVar(&cfg.Packs, "packs", cfg.Packs, "directory of Jev question packs [$ALEMBIC_PACKS]")
	fs.StringVar(&cfg.Receipts, "receipts", cfg.Receipts, "directory for Jev receipts")
	fs.StringVar(&cfg.Repo, "repo", cfg.Repo, "git repository (default: the repo containing the cwd)")
	fs.BoolVar(&cfg.Demo, "demo", false, "run a simulated harness under ~/.alembic/demo (Jev is MOCK unless --live)")
	fs.BoolVar(&cfg.Live, "live", false, "with --demo: call the real Jev API when TYPESAFE_API_KEY is set")
	fs.DurationVar(&cfg.Poll, "poll", cfg.Poll, "feed poll interval")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: alembic [--feed PATH] [--outbox PATH] [--packs DIR] [--receipts DIR] [--repo DIR] [--demo [--live]] [--poll 500ms]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	if cfg.Poll < 50*time.Millisecond {
		cfg.Poll = 50 * time.Millisecond
	}
	if root, err := harness.RepoRoot(cfg.Repo); err == nil {
		cfg.Repo, cfg.RepoOK = root, true
	} else if abs, err := filepath.Abs(cfg.Repo); err == nil {
		cfg.Repo = abs
	}
	return cfg, nil
}

// setupDemo points feed and outbox at a fresh demo directory and seeds it.
func setupDemo(cfg *config) error {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = "."
	}
	dir := filepath.Join(home, ".alembic", "demo")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cfg.Feed = filepath.Join(dir, "feed.jsonl")
	cfg.Outbox = filepath.Join(dir, "outbox.jsonl")
	_ = os.Remove(cfg.Feed)
	_ = os.Remove(cfg.Outbox)
	return harness.NewDemo(cfg.Feed, cfg.Outbox).Seed(cfg.Repo)
}

func main() {
	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		if err == flag.ErrHelp {
			os.Exit(0)
		}
		fmt.Fprintln(os.Stderr, "alembic:", err)
		os.Exit(2)
	}
	if cfg.Demo {
		if err := setupDemo(&cfg); err != nil {
			fmt.Fprintln(os.Stderr, "alembic: demo:", err)
			os.Exit(1)
		}
	}
	p := tea.NewProgram(newModel(cfg), tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "alembic:", err)
		os.Exit(1)
	}
}
