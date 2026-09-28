# nexus-command — demo script (≈2 minutes)

**What it is:** the operator console for NEXUS. Six specialised agents run the
meta-prompting pipeline — research → plan → generate → review → test → ship —
and you watch quality climb from 0.62 to 0.88 across iterations, live.

Start in a 120×40 terminal (anything ≥ 100×30 works; the layout reflows):

```bash
cd examples/tui-apps/nexus-command && go build -o nexus-command . && ./nexus-command
```

## Talk track

| Time | Press | Say / point at |
|------|-------|----------------|
| 0:00 | — | "This is a real Bubble Tea app, not a mock-up. Gold is action and focus, navy is structure." Point at the **IDLE** badge and the task in the header. |
| 0:05 | `r` | "Run the pipeline." Researcher spins up — the progress bar is **spring-animated** (harmonica), not linear. Point at the cyan `⟩` line: "That's the LLM output streaming token by token." |
| 0:20 | — | Planner scores complexity **0.72 → iterative strategy**. "This number decides how many refinement loops we budget for." |
| 0:30 | `i` | "Now let's break it." Toast says **chaos armed**. "When the Tester runs, one test will fail." |
| 0:35 | `4` | Metrics tab: quality gauge animating up, tok/s sparkline, per-agent token split. `1` to go back. |
| 0:50 | — | Reviewer says **Quality 0.62 < 0.85** and hands back to Coder. "That's the loop — reviewer context feeds the next iteration. Watch the quality gauge on the right." |
| 1:15 | — | Iteration 3: **0.88 ≥ 0.85, threshold met**. Then the Tester **fails** (red ✖), Coder runs iteration 4 from the stack trace, Tester re-runs green. "It self-healed. Nobody touched anything." |
| 1:30 | `ctrl+k`, type `kil` | "Everything is also a command." Fuzzy palette highlights matches in gold. `esc`. |
| 1:35 | `k` → `n` | Kill dialog with confirm. Decline. (Or `y` to kill, then `ctrl+k` → **Resume after kill**.) |
| 1:45 | `?` | Help overlay dims the background. `?` to close. |
| 1:50 | mouse | Click an agent row, click a tab, wheel-scroll the logs. `e` filters to warnings & errors only. |
| 2:00 | — | Footer toast: **pipeline complete ✓** — 4 iterations, quality 0.88, ~$0.4. `q`. |

## If something goes sideways

- Pipeline finished before you got to `i`? Press `r` again — a fresh run resets everything.
- Paused by accident (`p` / space)? Press it again; the held step resumes.
- Terminal too narrow? Below 120 columns the metrics pane drops to a strip along the bottom; below 20×8 it says so.

## Known limits (honest)

- The pipeline is **scripted** — the timings, log lines and quality scores replay the real
  file-browser iteration trace (`examples/iterations/file-browser-iteration-trace.md`);
  it does not call the Claude API. Wiring the real engine is a matter of replacing
  `pipelineScript()` with events from `internal/bridge`.
- Costs use list prices per 1k tokens baked into `costFor()`.
- Log lines are truncated to the pane width rather than wrapped.
