# Demo runbook — four TUIs, ~10 minutes

Everything below runs offline. No API keys, no network.

## Before you go on stage (5 minutes, once)

```bash
cd examples/tui-apps
make demo                      # builds nexus-command, gitscope, docscope, alembic
make test                      # optional: 100+ headless tests, ~10 s
```

- Terminal: **120×40 or larger**, a 256-colour terminal (iTerm2, Ghostty, WezTerm, Windows Terminal, kitty all fine),
  a Nerd-font-free monospace font is enough (the apps use only Unicode box drawing and block glyphs).
- Bump the font size until the window is still ≥ 120 columns — the layouts reflow down to 80×24 but look best wide.
- Keep four tabs open, one per app, already `cd`'d; switching tabs is faster than relaunching.
- Dry-run each `DEMO.md` once so the key sequence is in your fingers.

## Order and story

| # | App | Minutes | The point you're making |
|---|-----|---------|------------------------|
| 1 | [`nexus-command`](nexus-command/DEMO.md) | 2.5 | The engine: agents iterate until quality ≥ 0.85, and self-heal when a test fails. Chaos button is the moment. |
| 2 | [`gitscope`](gitscope/DEMO.md) | 2 | Real utility on a real repo: stage, diff, commit — all keyboard, all animated, no lag. |
| 3 | [`docscope`](docscope/DEMO.md) | 2 | Reading experience: custom-themed markdown, live outline, fuzzy finder, search. |
| 4 | [`alembic`](alembic/DEMO.md) | 3 | The operator seat: every task's status line at a glance, ping an agent, open its PR, gate a diff with a Jev question pack and read the verdict. |

Bridge line between 1 and 2: *"That pipeline is what generated apps like the next two."*
Bridge line into 4: *"And this is the seat you run all of it from."*

## One-liners if someone asks

- **Stack:** Go, Bubble Tea (Elm architecture), Lip Gloss, Bubbles, Glamour, Huh, Harmonica springs, Chroma. Single static binary per app, no runtime deps.
- **Why terminal:** it is the one UI every developer already has open, and it's the natural surface for agent tooling — logs, diffs, docs.
- **Is it tested:** yes — every app has headless tests that drive the Elm `Update`/`View` loop directly at multiple terminal sizes, plus a real-PTY smoke script; each app was also red-teamed by an independent agent before shipping.
- **Is the pipeline real:** `nexus-command` replays a real iteration trace (scripted, no API calls) so the demo is deterministic; the Go↔Python bridge in the NEXUS repo is where the live engine plugs in.

## Recovery

- Any app: `q` quits cleanly; relaunch takes < 100 ms.
- Terminal got weird after a crash (it won't, but): `reset`.
- Colours look wrong: `export TERM=xterm-256color`.
