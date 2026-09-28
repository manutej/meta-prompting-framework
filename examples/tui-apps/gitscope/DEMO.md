# gitscope — 90-second demo

A lazygit-class git TUI in Go: Bubble Tea + Lip Gloss + huh + harmonica + chroma,
Gold & Navy theme, mouse-aware, driven by real `git` porcelain output.

```sh
cd examples/tui-apps/gitscope && go build -o gitscope . && cd ../../.. && examples/tui-apps/gitscope/gitscope
```

Run it from anywhere inside a repository (a subdirectory is fine). Before the demo,
make sure the repo has one staged file, one modified file and one untracked file so
every section of the Status pane is populated.

## Talk track

| Time | Press | Say | Point at |
|------|-------|-----|----------|
| 0:00 | *(launch)* | "This is gitscope. Everything on screen comes straight from `git status --porcelain=v2`, `git log`, `git branch` and `git diff`; nothing is mocked." | Header: gold `gitscope` badge, repo name, `⎇ branch`, `↑ahead ↓behind`, live clock. |
| 0:10 | `j`, `k` | "The left column is an accordion: the focused pane expands, the others collapse to a one-line summary. Moving the cursor loads that file's diff on the right; staged entries show the index diff, unstaged the working tree, untracked files get syntax highlighting." | Status pane sections `▾ Staged` / `▾ Unstaged`; right pane title `Staged · path`; green/red lines, cyan hunk headers, gold file headers. |
| 0:25 | `space` | "Space toggles stage and unstage; that is a real `git add` / `git restore --staged`, and the toast confirms it. The view refreshes itself every three seconds, so edits from another terminal show up on their own." | Green toast bottom-right; entry moves between sections. |
| 0:35 | `2` | "Watch the spring animation as focus moves. Branches shows the last fifteen commits of whatever is selected; Enter checks it out, `n` creates a new branch through a form." | Accordion expand with harmonica spring; right pane `Branch · name`. |
| 0:45 | `3`, `j`, `Enter` | "Commits: selecting one runs `git show --stat -p`. Enter jumps focus into the diff, where j/k, ctrl+d/u, g/G and the mouse wheel all scroll. `y` copies the SHA to the system clipboard through OSC 52." | Percentage indicator in the diff title; `● sha subject age` rows. |
| 0:58 | *(click a Status row)* | "It is fully mouse-aware: clicking a row focuses that pane and selects the row; the wheel scrolls whatever is under the cursor." | Cursor jumps to the clicked row. |
| 1:05 | `1`, `c` | "Commit opens a huh form: title, optional body, confirm. If nothing is staged you get a warning toast instead of an empty commit." | Gold/navy modal over the dimmed app; after submit, toast `committed <sha>`. |
| 1:18 | `/`, type, `Esc` | "Slash is a fuzzy filter for the focused list — sahilm/fuzzy under the hood — Esc clears it." | Filter text in the status bar and in the pane title. |
| 1:25 | `?` | "Every key is on the help overlay. Discard and stash both go through a Yes/No confirm; git errors surface as red toasts and never crash the app." | Centered help box, two columns. |
| 1:30 | `?`, `q` | "That's gitscope." | |

If a toast fires that you did not expect (e.g. `checkout failed: … local changes would
be overwritten`), lean into it: that is git's own error text, shown instead of a crash.

## What is wired up

- Panes `1`–`4` / `tab` / `shift+tab`, `h`/`l` between the column and the diff.
- Status: `space`, `a`, `A`, `d` (confirm), `s` (confirm, `git stash push --include-untracked`), `c`, `enter`.
- Branches: `enter` checkout, `n` new branch. Commits: `enter`, `y` copy. Stash: `enter` pop.
- `/` fuzzy filter, `r`/`F5` refresh, 3 s auto-refresh (status + branches, paused while a form is open), `?` help, toasts, spinner while git runs.
- Handles: launched from a subdirectory, not-a-repo screen, empty repo (unborn branch), detached HEAD, terminal below 60x14.

## Known limits

- Whole-file staging only; there is no hunk or line-level staging.
- No push/pull/fetch, rebase, merge or cherry-pick; the `↑ahead ↓behind` counters reflect whatever the last fetch left behind.
- `d` on a file that is both staged and modified discards both the index and working-tree change (`git restore --staged` then `git checkout --`); on a newly added (`A`) staged file it only unstages, leaving the file untracked so nothing is deleted without a second, explicit `d`.
- Stash pop of the selected entry only; there is no stash drop or apply-without-pop.
- The diff viewer truncates long lines to the pane width rather than wrapping them, and untracked files are read up to 512 KB.
- Auto-refresh re-runs the selected file's diff every 3 s while the Status pane is active; on a very large repository with a huge diff that is the one command that can feel heavy.
- OSC 52 clipboard copy depends on the terminal supporting it (iTerm2, kitty, WezTerm, foot, recent xterm do; some tmux setups need `set -g set-clipboard on`).
- Commit body newlines are entered with `ctrl+j` inside the form (huh 0.3 binds Enter to "next field").
