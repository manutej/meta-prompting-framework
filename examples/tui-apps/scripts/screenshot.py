"""Capture real frames of a TUI binary in a PTY and render them as HTML.

usage: screenshot.py OUT.html [--cols 120] [--rows 40] [--cwd DIR] -- BIN [ARGS...] @ t=SECONDS:KEYS ...
  each "t=3.0:r" spec sends KEYS at that time (\\t, \\r, \\x1b escapes allowed); "t=8.0:" with no keys = snapshot only.
"""
import html, os, pty, select, signal, struct, sys, termios, fcntl, time, re
import pyte

def parse_keys(s):
    return s.encode().decode("unicode_escape").encode("latin1")

def run(binary, args, cwd, cols, rows, events):
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLORTERM"] = "truecolor"
        os.environ["COLUMNS"], os.environ["LINES"] = str(cols), str(rows)
        os.execv(binary, [binary] + args)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    frames = []
    t0 = None
    plain = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)")
    seen = b""
    started = False
    end = time.time() + max(e[0] for e in events) + 6
    pending = sorted(events)
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.03)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if b"\x1b]10;?" in chunk:
                os.write(fd, b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\")
            if b"\x1b]11;?" in chunk:
                os.write(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            if b"\x1b[6n" in chunk:
                os.write(fd, b"\x1b[1;1R")
            stream.feed(chunk)
            if not started:
                seen += plain.sub(b"", chunk)
                if len(seen.strip()) > 200:
                    started = True
                    t0 = time.time()
        if started and pending and time.time() - t0 >= pending[0][0]:
            t, keys, label = pending.pop(0)
            if keys:
                os.write(fd, keys)
            else:
                frames.append((label, snapshot(screen)))
        if started and not pending:
            break
    os.write(fd, b"q")
    time.sleep(0.3)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    os.waitpid(pid, 0)
    return frames

ANSI16 = ["#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
          "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#ffffff"]
NAMES = {"black": 0, "red": 1, "green": 2, "brown": 3, "blue": 4, "magenta": 5, "cyan": 6, "white": 7}

def color(c, default):
    if c == "default":
        return default
    if c in NAMES:
        return ANSI16[NAMES[c]]
    if c.startswith("bright"):
        return ANSI16[8 + NAMES[c[6:]]]
    if re.fullmatch(r"[0-9a-f]{6}", c):
        return "#" + c
    if c.isdigit():
        n = int(c)
        if n < 16:
            return ANSI16[n]
        if n < 232:
            n -= 16
            r, g, b = n // 36, (n // 6) % 6, n % 6
            return "#%02x%02x%02x" % tuple(0 if v == 0 else 55 + v * 40 for v in (r, g, b))
        v = 8 + (n - 232) * 10
        return "#%02x%02x%02x" % (v, v, v)
    return default

def snapshot(screen):
    rows = []
    for y in range(screen.lines):
        line = screen.buffer[y]
        spans, cur, buf = [], None, []
        for x in range(screen.columns):
            ch = line[x]
            key = (ch.fg, ch.bg, ch.bold, ch.italics, ch.underscore, ch.reverse)
            if key != cur:
                if buf:
                    spans.append((cur, "".join(buf)))
                cur, buf = key, []
            buf.append(ch.data or " ")
        if buf:
            spans.append((cur, "".join(buf)))
        rows.append(spans)
    return rows

def to_html(frames, cols, title):
    out = [f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(title)}</title><style>"
           "body{background:#0b0f1a;color:#d0d0d0;font-family:'JetBrains Mono','Fira Code',Menlo,monospace;margin:24px}"
           "h1{font:600 20px sans-serif;color:#d4af37} h2{font:500 14px sans-serif;color:#8a94a6;margin:28px 0 8px}"
           f".term{{display:inline-block;background:#000;padding:12px 14px;border-radius:8px;border:1px solid #1b365d;box-shadow:0 8px 32px #0008;white-space:pre;font-size:12.5px;line-height:1.25;width:{cols}ch;overflow:hidden}}"
           "</style></head><body>", f"<h1>{html.escape(title)}</h1>"]
    for label, rows in frames:
        out.append(f"<h2>{html.escape(label)}</h2><div class=term>")
        for spans in rows:
            for (fg, bg, bold, ital, ul, rev), txt in spans:
                f, b = color(fg, "#d0d0d0"), color(bg, "#000000")
                if rev:
                    f, b = b, f
                st = f"color:{f};background:{b}" + (";font-weight:700" if bold else "") + (";font-style:italic" if ital else "") + (";text-decoration:underline" if ul else "")
                out.append(f"<span style=\"{st}\">{html.escape(txt)}</span>")
            out.append("\n")
        out.append("</div>")
    out.append("</body></html>")
    return "".join(out)

def main(argv):
    outp = argv[1]
    cols, rows, cwd = 120, 40, os.getcwd()
    i = 2
    while argv[i].startswith("--") and argv[i] != "--":
        if argv[i] == "--cols":
            cols = int(argv[i + 1])
        elif argv[i] == "--rows":
            rows = int(argv[i + 1])
        elif argv[i] == "--cwd":
            cwd = argv[i + 1]
        i += 2
    assert argv[i] == "--"
    i += 1
    binargs = []
    while argv[i] != "@":
        binargs.append(argv[i])
        i += 1
    events = []
    for spec in argv[i + 1:]:
        t, _, rest = spec.partition(":")
        t = float(t[2:])
        keys, _, label = rest.partition("#")
        events.append((t, parse_keys(keys) if keys else b"", label or f"t={t}s"))
    frames = run(os.path.abspath(binargs[0]), binargs[1:], cwd, cols, rows, events)
    with open(outp, "w") as f:
        f.write(to_html(frames, cols, os.path.basename(binargs[0])))
    print(f"wrote {outp} with {len(frames)} frames")

if __name__ == "__main__":
    main(sys.argv)
