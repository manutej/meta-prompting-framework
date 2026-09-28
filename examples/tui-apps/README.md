# TUI Apps — Gold & Navy

Three fully functional terminal applications built on the Charmbracelet stack
(Bubble Tea, Bubbles, Lip Gloss). Gold `#D4AF37` (ANSI 178) for action/focus,
Navy `#1B365D` (ANSI 24) for structure.

| App | What it does | Keys |
|-----|--------------|------|
| `progress-timer` | 30-second animated progress timer with live elapsed/remaining | `r` reset · `q` quit |
| `file-browser` | Two-pane browser: fuzzy-filtered file list + live preview (text, dir listing, binary detection, 64KB cap) | `j/k` move · `enter`/`l` open · `h` back · `/` search · `tab` focus pane · `g/G` · `ctrl+d/u` · `q` |
| `system-monitor` | Live CPU/memory gauges + 60-sample sparklines, load average, uptime, top processes by CPU (reads `/proc`) | `p`/`space` pause · `+`/`-` sample interval · `q` |

## Build & run

```bash
make build            # builds all three
./progress-timer/progress-timer
./file-browser/file-browser
./system-monitor/system-monitor
```

Each app is its own Go module with no shared code, so any one can be copied out standalone.

## Verification

```bash
make test             # go vet + headless unit tests (drive Update/View directly, no TTY)
make smoke            # launches each binary in a real PTY, sends keys, checks rendered output
```

The unit tests were written by adversarial review agents whose brief was to break
the apps; every real bug they reproduced was fixed and is pinned by a test.

Requirements: Go 1.21+. `system-monitor` is Linux-only (reads `/proc`).
