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
| [`alembic`](alembic/) | Operator console for an agent harness (Ormus / Hermes-style) with TypeSafe Jev | Tasks grouped by production workflow with a live status line each, open any element (file:line, PR, URL, log), quick pings to agents with acks, cancel/retry, git worktrees (list, add, remove, link to tasks), Jev question packs with visual results (noul bars, choice distributions, score ladders), gate verdicts, write-once receipts, history and compare |

```bash
make demo                    # builds the four demo apps
./nexus-command/nexus-command
./gitscope/gitscope          # run inside any git repo
./docscope/docscope ../../docs
./alembic/alembic --demo      # self-contained simulated harness; add TYPESAFE_API_KEY for live Jev
```

`alembic` plugs into a real harness through two append-only JSONL files
([contract](alembic/docs/HARNESS-CONTRACT.md), [TypeScript adapter](alembic/contrib/ormus-adapter.ts))
and runs [question packs](alembic/docs/JEV.md) against TypeSafe AI's Jev.

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
