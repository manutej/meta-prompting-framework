# docscope — 90-second demo

A glow-class markdown reader for the terminal: file tree, live outline,
fuzzy finder, in-document search, animated navigation, mouse support.
Gold & Navy theme throughout.

```bash
cd examples/tui-apps/docscope
go build -o docscope . && ./docscope ../../../docs
```

Run it in a terminal that is at least 120x40 for the demo; 80x24 also works.

## Talk track

| Time | Press | Say | Point at |
|------|-------|-----|----------|
| 0:00 | *(launch)* | "docscope opens the first doc immediately. Gold is focus and actions, navy is structure." | Header badge, the gold-bordered **Files** pane, the reading-progress `0%` on the right, the thin progress bar under the reader. |
| 0:08 | `j` `j` `enter` | "The sidebar is a real tree, walked to six levels, sorted dirs-first. Enter opens." | Reader title changes; outline count updates. |
| 0:15 | `tab` | "Tab moves focus. The sidebar is an accordion: the focused pane springs open, the other collapses to a one-line summary." | The spring animation between Files and Outline. |
| 0:22 | `j` `j` `enter` | "Outline is H1–H3 from the document. Enter scrolls there with a spring, not a jump." | Reader scrolls smoothly; the `◀` marker follows the heading at the top of the viewport. |
| 0:32 | `tab` then `]` `]` `[` | "In the reader, bracket keys step between headings." | Outline marker moves as you step; header percentage climbs. |
| 0:40 | `/` type `claude` `enter` `n` `n` | "Slash searches inside the document, case-insensitive, live-highlighted, with a match counter. Enter and `n`/`p` walk the matches." | Gold highlights in the reader; `3/12` in the bottom bar. |
| 0:52 | `esc` then `ctrl+p` type `spec` | "Ctrl-P is a command-palette fuzzy finder over every file path. Matched characters light up in gold." | Centered overlay with dimmed backdrop; highlighted letters. |
| 1:00 | `enter` | "Enter opens it, and the tree cursor follows." | Reader title, Files cursor on the opened file. |
| 1:05 | *(scroll the mouse wheel, click a heading in Outline, click a file in Files)* | "Everything is mouse-aware: wheel scrolls, clicks focus, open and jump." | — |
| 1:15 | `b` | "`b` hides the sidebar; the document re-wraps at the new width and keeps its scroll position." | Full-width reader. `b` again to restore. |
| 1:22 | `y` | "`y` copies the file path to the system clipboard through OSC 52 — works over SSH." | Toast "copied" bottom-right, auto-dismisses. |
| 1:26 | `?` | "Every key is on the help overlay." | Two-column key list. `?` closes. |
| 1:30 | `q` | "That's docscope." | — |

Also available if asked: `e` opens the current file in `$EDITOR` (default `vi`)
and reloads it on return; `r` rescans the tree; `1`/`2`/`3` and `h`/`l` jump
focus; `ctrl+d`/`ctrl+u`, `space`/`pgup`/`pgdn`, `g`/`G` in the reader.

## Known limits

- **Rendering speed on very large files.** glamour renders 500 KB of markdown
  in about 3.5 s and produces ~40 MB of styled text; a 5 MB file takes
  proportionally longer and uses hundreds of MB. The render runs in a
  background command with a spinner, so the UI stays responsive, but the
  document is not usable until it lands. Files above 32 MB are refused with a
  message.
- **Every resize re-renders.** Word-wrap width changes, so the document is
  re-rendered (in the background) on each terminal resize or sidebar toggle;
  stale renders are discarded by sequence number. Dragging a window edge on a
  large file queues several renders.
- **Heading and code-block mapping is text-based.** Headings and fenced code
  lines are located in glamour's output by matching their text. A heading whose
  text is exactly repeated in a paragraph directly before it, or a code line
  that wraps at an unusual place, can map one line off. Setext headings are
  detected only when the underline directly follows a paragraph line.
- **Search highlights drop inline styling on matched lines.** A line that
  contains a match is rebuilt from its plain text with gold highlights, so
  bold/links/code colour on that line disappear while the search is active.
- **Code blocks show a left bar, not a full frame.** glamour 0.7.0 hard-codes
  the code-block indent, so the navy bar is added in post-processing; long code
  lines are word-wrapped by glamour and then clipped to the pane width.
- **Table headers are recognised by their rule line.** The gold header style
  is applied to the row above a `───┼───` rule; a single-column table has no
  `┼` and keeps the default style.
- **Clipboard via OSC 52** only works in terminals that support it (kitty,
  iTerm2, WezTerm, foot, recent xterm/Windows Terminal; tmux needs
  `set -g set-clipboard on`). The toast reports success of the write, not of
  the terminal actually accepting it.
- **`$EDITOR` must be a terminal editor**; GUI editors that return
  immediately will trigger the reload before you save.
- **Minimum size 40x10.** Below that the app shows a "needs at least" notice
  instead of the panes.
