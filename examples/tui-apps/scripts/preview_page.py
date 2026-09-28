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
    ("alembic", "Operator console for an agent harness, with Jev",
     "Every task's live status line grouped by production workflow; open its files, PRs and logs; ping the agent and see the ack; manage git worktrees; run Jev question packs and read calibrated answers and a gate verdict. Plugs into the Ormus harness through two append-only JSONL files.",
     [("41", "headless tests"), ("~5.7k", "lines of Go"), ("live", "Jev verified, 586 ms")],
     "j/k · p ping · J jev · o open · x cancel · 1-4 tabs · ctrl+k palette"),
]

def frames_of(path):
    s = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"<h2>(.*?)</h2><div class=term>(.*?)</div>", s, re.S):
        out.append((html.unescape(m.group(1)), m.group(2)))
    return out

def main(outp, *frame_files):
    parts = []
    parts.append("""<title>Cast in Liquid Gold</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,400&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;700&display=swap">
<style>
:root{--bg:#0f0c06;--bg2:#1a1407;--ink:#eceae3;--muted:#9ca3af;--gold:#d4a017;--amber:#d29e3d;--bronze:#8a6519;--navy:#0e1830;--term:#0a0805;--rule:#2a2a2a;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif;padding-block:40px 72px;padding-inline:clamp(16px,4vw,56px);line-height:1.55}
h1,h2,h3{font-family:"Cormorant Garamond",Georgia,serif;text-wrap:balance;margin:0;font-weight:500}
.mark{display:inline-flex;align-items:center;gap:10px;font-family:Inter,sans-serif;font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--gold)}
.mark i{display:inline-block;width:28px;height:1px;background:linear-gradient(90deg,transparent,var(--gold))}
h1{font-size:clamp(38px,6vw,64px);letter-spacing:-.01em;color:var(--ink);margin-top:10px;line-height:1.05}
h1 em{font-style:italic;color:var(--gold)}
.lede{max-width:62ch;color:var(--muted);margin-top:14px;font-size:17px}
.lede b{color:var(--ink);font-weight:500}
.tag{margin-top:8px;font-family:"Cormorant Garamond",serif;font-style:italic;font-size:19px;color:var(--amber)}
.jump{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}
.jump a{color:var(--gold);text-decoration:none;border:1px solid var(--rule);border-radius:999px;padding:5px 14px;font-size:13px;font-family:"JetBrains Mono",monospace}
.jump a:hover,.jump a:focus-visible{border-color:var(--gold);outline:none}
.seam{height:1px;margin:56px 0 28px;background:linear-gradient(90deg,var(--gold) 0,var(--bronze) 18%,var(--rule) 30%,var(--rule) 62%,var(--bronze) 74%,var(--gold) 88%,transparent 100%)}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px}
.name{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:24px;font-weight:700;color:var(--gold)}
h2{font-size:26px;color:var(--ink)}
.pitch{max-width:70ch;color:var(--muted);margin-top:8px}
.facts{display:flex;flex-wrap:wrap;gap:12px 28px;margin-top:14px;font-variant-numeric:tabular-nums}
.facts div{display:flex;flex-direction:column}
.facts b{font-family:"Cormorant Garamond",serif;font-size:28px;color:var(--gold);font-weight:600;line-height:1}
.facts span{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-top:4px}
.keys{margin-top:14px;font-family:"JetBrains Mono",monospace;font-size:12.5px;color:var(--amber);background:var(--bg2);display:inline-block;padding:6px 10px;border-radius:4px;border:1px solid var(--rule)}
figure{margin:24px 0 0}
figcaption{font-size:14px;color:var(--muted);margin-bottom:8px;font-family:"Cormorant Garamond",serif;font-size:17px}
figcaption::before{content:"◆ ";color:var(--gold);font-size:11px;vertical-align:middle}
.scroll{overflow-x:auto;border-radius:6px;border:1px solid var(--navy);background:var(--term);box-shadow:0 0 0 1px #000 inset,0 16px 48px #000a}
.term{display:block;white-space:pre;font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.25;padding:12px 14px;width:120ch;box-sizing:content-box}
.note{margin-top:56px;padding:18px 22px;border:1px solid var(--rule);border-left:2px solid var(--gold);border-radius:4px;color:var(--muted);max-width:72ch}
.note b{color:var(--ink);font-weight:500}
a{color:var(--gold)}
@media (prefers-reduced-motion:no-preference){.jump a{transition:border-color .15s}}
</style>
<div class="mark"><i></i>Ormus · terminal console preview</div>
<h1>Four consoles, <em>cast in liquid gold</em></h1>
<p class="tag">Liquid gold · empower, don't extract.</p>
<p class="lede">Every frame below is a <b>real render</b>: each binary was run in a pseudo-terminal at 120×40, driven with the keys from its demo script, and the screen buffer was converted to HTML. Ormus palette and voice throughout; nothing is mocked or retouched.</p>
<div class="jump">""")
    for name, *_ in APPS:
        parts.append(f'<a href="#{name}">{name}</a>')
    parts.append("</div>")
    for (name, tag, pitch, facts, keys), path in zip(APPS, frame_files):
        parts.append(f'<div class="seam"></div><section id="{name}"><div class="head"><span class="name">{name}</span><h2>{html.escape(tag)}</h2></div>')
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
