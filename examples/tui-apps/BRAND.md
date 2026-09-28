# Ormus brand chrome for the terminal

Source of truth: ormus.solutions (stylesheets and copy, read 2026-09-28) and the
Liquid Gold kit READMEs. Nothing here is invented; every value below appears on the site.

## Tokens

| token | hex | site usage | terminal usage |
|---|---|---|---|
| gold | `#d4a017` | primary accent, headings, links | focus, selection, actions, titles, key hints |
| amber | `#d29e3d` | borders, soft accent, hover | secondary highlight, warning, "review" state |
| bronze | `#8a6519` | deep gold | dim gold: inactive titles, gauge track ends |
| ink | `#1a1407` | body text on light, deep ground | text on gold badges; darkest ground |
| navy | `#0e1830` | deep panels | structure: unfocused borders, containers, gauge track |
| ivory | `#eceae3` | light ground | primary text |
| gray | `#9ca3af` | secondary text | muted text |
| rule | `#2a2a2a` | dividers | separators, dim text |
| success | `#22c55e` | | done, ok |
| warn | `#fbbf24` | | warnings, blocked |
| error | `#ef4444` | | failed, refuse |
| info | `#60a5fa` | | links, code, "review" accents |

Colors are truecolor hex in Lip Gloss; on 256-color terminals they degrade to the
nearest palette entry automatically (gold → 178, navy → 17, ivory → 255).

## Type (web only — terminals use the user's font)

Cormorant Garamond for display, Inter for body, JetBrains Mono for code — the site's
own stack. The preview page uses all three from Google Fonts.

## Chrome

- Wordmark: `ORMUS` set ink-on-gold, followed by the app name in gold on navy.
- The kintsugi idea, translated: the *focused* pane's border is the gold seam; every
  other border is navy. Focus is the crack filled with gold.
- Tagline where there is room for a quiet line (help overlay, empty states):
  *Liquid gold · empower, don't extract.*
- Vocabulary from the kits stays exactly where it already is (aurum-gate decisions
  `auto` / `escalate` / `refuse`); the UI does not rename anything to alchemy.

## What is deliberately not used

The kintsugi mark itself (`/images/kintsugi-mark.svg`) is Ormus's logo. The apps and
the preview page use the palette, type and voice, not the logo, so an internal demo
never reads as an official Ormus artifact.
