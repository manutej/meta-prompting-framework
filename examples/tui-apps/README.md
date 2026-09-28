# TUI Apps — Gold & Navy

Terminal applications built on the Charmbracelet stack (Bubble Tea, Bubbles, Lip Gloss,
Glamour, Huh, Harmonica). Gold `#D4AF37` (ANSI 178) for action and focus, Navy `#1B365D`
(ANSI 24) for structure. Every app is its own Go module and can be copied out standalone.

## Demo apps

Each has a `DEMO.md` with a timed talk track and an honest "known limits" section.

| App | What it is | Highlights |
|-----|------------|------------|
| [`nexus-command`](nexus-command/) | Operator console for the NEXUS multi-agent pipeline | Six agents, spring-animated progress, token-streaming logs, live quality gauge + sparklines, fuzzy command palette (`ctrl+k`), chaos injection with self-healing retry, kill/resume with confirm dialogs, tabs, mouse |
| [`gitscope`](gitscope/) | lazygit-class git dashboard on a real repository | Accordion panes (status / branches / commits / stash), syntax-highlighted diffs, stage/unstage/discard, commit & new-branch forms (Huh), fuzzy filter, auto-refresh, toasts, mouse |
| [`docscope`](docscope/) | glow-class markdown reader | Custom Gold/Navy Glamour theme, file tree + live outline with current-heading tracking, spring-animated jump-to-heading, command-palette file finder, in-document search with match highlighting, reading progress bar, `$EDITOR` round-trip |

```bash
make demo                    # builds the three demo apps
./nexus-command/nexus-command
./gitscope/gitscope          # run inside any git repo
./docscope/docscope ../../docs
```

## Basic apps

Smaller single-file apps kept as reference implementations: `progress-timer`,
`file-browser`, `system-monitor`.

## Verification

```bash
make test     # go vet + headless tests (drive Update/View directly, no TTY) for all six
make smoke    # launches every binary in a real PTY, sends keys, checks rendered output
```

Every app went through an adversarial review pass by an independent agent whose brief was to
break it; each reproduced bug was fixed and pinned by a test.

Requirements: Go 1.21+, `git` on PATH for gitscope. `system-monitor` is Linux-only.
