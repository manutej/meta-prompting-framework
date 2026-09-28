"""Assemble captured frame files into a single preview page.
usage: preview_page.py OUT.html nexus-frames.html gitscope-frames.html docscope-frames.html
"""
import html, re, sys

APPS = [
    ("nexus-command", "Agent operator console",
     "Six agents run the meta-prompting pipeline; quality climbs 0.62 → 0.88 across iterations, and a chaos button makes the Tester fail so the Coder self-heals on camera.",
     [("18", "headless tests"), ("~1.2k", "lines of Go"), ("2.5 min", "on stage")],
     "r run · i chaos · ctrl+k palette · k kill · 1-4 tabs · mouse"),
    ("gitscope", "Git dashboard on a live repository",
     "Accordion panes for status, branches, commits and stash; syntax-highlighted diffs; stage, discard, stash, commit and branch through forms with confirmation.",
     [("50", "headless tests"), ("~3.5k", "lines of Go"), ("4", "bugs fixed in red-team")],
     "space stage · c commit · enter view · / filter · y copy sha · ? help"),
    ("docscope", "Markdown reader with a live outline",
     "Custom Gold/Navy rendering, file tree, outline that tracks the heading under the cursor, spring-animated jumps, a fuzzy file finder and in-document search.",
     [("45", "headless tests"), ("~3.5k", "lines of Go"), ("6", "bugs fixed in red-team")],
     "tab panes · ctrl+p find · / search · [ ] headings · b sidebar · e edit"),
]

def frames_of(path):
    s = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"<h2>(.*?)</h2><div class=term>(.*?)</div>", s, re.S):
        out.append((html.unescape(m.group(1)), m.group(2)))
    return out

def main(outp, *frame_files):
    parts = []
    parts.append("""<title>Gold and Navy Demo Trio</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=JetBrains+Mono:wght@400;700&display=swap">
<style>
:root{--bg:#0b1526;--bg2:#0f1c33;--ink:#e8e4d8;--muted:#8fa0b8;--gold:#d4af37;--gold2:#f0d060;--navy:#1b365d;--term:#05080f;--rule:#1f3557;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;padding-block:32px 64px;padding-inline:clamp(16px,4vw,48px);line-height:1.5}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(30px,5vw,44px);font-weight:600;letter-spacing:-.01em;color:var(--gold)}
.lede{max-width:64ch;color:var(--muted);margin-top:8px;font-size:17px}
.lede b{color:var(--ink);font-weight:500}
.jump{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.jump a{color:var(--gold);text-decoration:none;border:1px solid var(--rule);border-radius:999px;padding:4px 12px;font-size:14px}
.jump a:hover,.jump a:focus-visible{border-color:var(--gold);outline:none}
section{margin-top:56px;padding-top:24px;border-top:1px solid var(--rule)}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px}
.name{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:26px;font-weight:700;color:var(--gold2)}
h2{font-size:20px;font-weight:500;color:var(--ink)}
.pitch{max-width:70ch;color:var(--muted);margin-top:8px}
.facts{display:flex;flex-wrap:wrap;gap:12px 28px;margin-top:14px;font-variant-numeric:tabular-nums}
.facts div{display:flex;flex-direction:column}
.facts b{font-family:"JetBrains Mono",monospace;font-size:20px;color:var(--ink);font-weight:700}
.facts span{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.keys{margin-top:12px;font-family:"JetBrains Mono",monospace;font-size:13px;color:var(--gold);background:var(--bg2);display:inline-block;padding:6px 10px;border-radius:6px}
figure{margin:24px 0 0}
figcaption{font-size:14px;color:var(--muted);margin-bottom:8px}
figcaption::before{content:"▸ ";color:var(--gold)}
.scroll{overflow-x:auto;border-radius:10px;border:1px solid var(--navy);background:var(--term);box-shadow:0 12px 40px #0009}
.term{display:block;white-space:pre;font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.25;padding:12px 14px;width:120ch;box-sizing:content-box}
.note{margin-top:48px;padding:16px 20px;border:1px solid var(--rule);border-radius:10px;color:var(--muted);max-width:72ch}
.note b{color:var(--ink);font-weight:500}
a{color:var(--gold)}
@media (prefers-reduced-motion:no-preference){.jump a{transition:border-color .15s}}
</style>
<h1>Three terminal apps, captured live</h1>
<p class="lede">Every frame below is a <b>real render</b>: each binary was run in a pseudo-terminal at 120×40, driven with the keys from its demo script, and the screen buffer was converted to HTML. Nothing is mocked or retouched.</p>
<div class="jump">""")
    for name, *_ in APPS:
        parts.append(f'<a href="#{name}">{name}</a>')
    parts.append("</div>")
    for (name, tag, pitch, facts, keys), path in zip(APPS, frame_files):
        parts.append(f'<section id="{name}"><div class="head"><span class="name">{name}</span><h2>{html.escape(tag)}</h2></div>')
        parts.append(f'<p class="pitch">{html.escape(pitch)}</p><div class="facts">')
        for v, l in facts:
            parts.append(f"<div><b>{html.escape(v)}</b><span>{html.escape(l)}</span></div>")
        parts.append(f'</div><div class="keys">{html.escape(keys)}</div>')
        for label, body in frames_of(path):
            parts.append(f'<figure><figcaption>{html.escape(label)}</figcaption><div class="scroll"><code class="term">{body}</code></div></figure>')
        parts.append("</section>")
    parts.append("""<div class="note"><b>How to run them:</b> <code>cd examples/tui-apps && make demo</code>, then launch each binary from its folder. The per-app <code>DEMO.md</code> files hold timed talk tracks; <code>DEMO.md</code> at the top level sequences all three in about seven minutes.</div>""")
    open(outp, "w", encoding="utf-8").write("\n".join(parts))
    print("wrote", outp)

if __name__ == "__main__":
    main(*sys.argv[1:])
