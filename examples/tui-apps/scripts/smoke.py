"""Real-PTY smoke tests for every app, driven by one spec table.

usage: smoke.py [app ...]     (default: all)

Each spec forks the built binary in a pseudo-terminal, answers the terminal's
capability queries the way xterm would, waits for the first frame, sends the
scripted keys with delays, then checks that the ANSI-stripped output contains
every expected marker and that the process exits 0 on `q`.
"""
import fcntl, os, pty, re, select, signal, struct, sys, termios, time

import subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# gitscope and alembic run inside the enclosing git repository, whichever repo this workspace lives in
try:
    REPO = subprocess.run(["git", "-C", ROOT, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    REPO = ROOT
# docscope reads a docs directory if the repo has one, else the workspace itself
DOCS = next((d for d in (os.path.join(REPO, "docs"), os.path.join(ROOT, "alembic", "docs")) if os.path.isdir(d)), ROOT)
ESC, TAB, ENTER, CTRL_K, CTRL_P = b"\x1b", b"\t", b"\r", b"\x0b", b"\x10"

# name: dict(bin, cwd, args, first (marker that means the first frame is up),
#            keys [(delay_s, bytes)], expect [markers], any [(m1, m2)] = at least one of)
SPECS = {
    "nexus-command": dict(
        first=b"ORMUS",
        keys=[(0.8, b"r"), (3.0, b"i"), (0.5, TAB), (0.5, b"2"), (0.8, b"3"), (0.8, b"4"), (0.8, b"1"),
              (0.5, CTRL_K), (0.4, b"kil"), (0.6, ESC), (0.4, b"k"), (0.6, b"n"), (0.4, b"?"), (0.8, b"?"), (0.4, b"p"), (0.8, b"p")],
        expect=[b"ORMUS", b"NEXUS", b"command center", b"Agents", b"Logs", b"Metrics", b"Researcher", b"Planner",
                b"pipeline started", b"RUNNING", b"chaos armed", b"keys", b"paused", b"quality"],
        any=[(b"mid-task?", b"is not running")],
    ),
    "gitscope": dict(
        cwd=REPO, first=b"gitscope",
        keys=[(0.8, b"j"), (0.6, b"2"), (0.6, b"3"), (0.6, b"j"), (0.6, ENTER), (0.8, b"?"), (0.8, b"?")],
        expect=[b"gitscope", b"Status", b"Branches", b"Commits", b"Stash"],
    ),
    "docscope": dict(
        cwd=REPO, args=[DOCS], first=b"docscope",
        keys=[(0.8, b"j"), (0.4, ENTER), (1.2, TAB), (0.4, TAB), (0.4, b"j"), (0.4, b"j"), (0.4, CTRL_P), (0.4, b"spec"),
              (0.6, ENTER), (1.0, b"?"), (0.8, b"?")],
        expect=[b"docscope", b"Files", b"Outline", b"Find file"],
    ),
    "alembic": dict(
        cwd=REPO, args=["--demo"], first=b"ORMUS",
        keys=[(1.0, b"j"), (0.4, b"j"), (0.4, TAB), (0.4, b"j"), (0.4, TAB), (0.4, b"p"), (0.4, b"status?"), (0.4, ENTER),
              (0.8, b"2"), (1.0, b"3"), (0.6, ENTER), (1.5, b"4"), (0.8, b"1"), (0.5, CTRL_K), (0.4, b"read"), (0.6, ESC), (0.4, b"?"), (0.8, b"?")],
        expect=[b"ORMUS", b"alembic", b"Tasks", b"checkout-service", b"T-1041", b"ping sent", b"Worktrees", b"Packs", b"MOCK", b"Agents", b"triage"],
    ),
    "progress-timer": dict(first=b"PROGRESS TIMER", keys=[], expect=[b"PROGRESS TIMER", b"%"], settle=2.0),
    "file-browser": dict(cwd=ROOT, first=b"FILE BROWSER", keys=[(0.5, b"j"), (0.3, b"j"), (0.3, ENTER), (0.3, b"/"), (0.3, b"go"), (0.3, ESC), (0.3, TAB)],
                         expect=[b"FILE BROWSER", b"tui-apps"]),
    "system-monitor": dict(cwd=ROOT, first=b"SYSTEM MONITOR", keys=[(0.8, b"p"), (0.3, b"+"), (0.3, b"-")],
                           expect=[b"SYSTEM MONITOR", b"CPU", b"MEMORY", b"PID"]),
}

ANSI = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[=>]")
COLS, ROWS = 120, 40


def run(name, spec):
    binary = os.path.join(ROOT, name, name)
    if not os.path.exists(binary):
        return False, f"binary missing: {binary} (run `make build`)"
    cwd = spec.get("cwd", os.path.join(ROOT, name))
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.environ.update(TERM="xterm-256color", COLORTERM="truecolor", COLUMNS=str(COLS), LINES=str(ROWS))
        os.execv(binary, [binary] + spec.get("args", []))
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    out, keys = b"", list(spec["keys"])
    next_at, started = None, False
    settle = spec.get("settle", 0.8)
    deadline = time.time() + 20 + sum(d for d, _ in keys)
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.03)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            out += chunk
            if b"\x1b]10;?" in chunk:
                os.write(fd, b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\")
            if b"\x1b]11;?" in chunk:
                os.write(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            if b"\x1b[6n" in chunk:
                os.write(fd, b"\x1b[1;1R")
            if not started and spec["first"] in ANSI.sub(b"", out):
                started = True
                next_at = time.time() + (keys[0][0] if keys else settle)
        if started and next_at is not None and time.time() >= next_at:
            if keys:
                os.write(fd, keys.pop(0)[1])
                next_at = time.time() + (keys[0][0] if keys else settle)
            else:
                break
    os.write(fd, b"q")
    time.sleep(0.4)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _, status = os.waitpid(pid, 0)
    plain = ANSI.sub(b"", out)
    missing = [m.decode() for m in spec["expect"] if m not in plain]
    for group in spec.get("any", []):
        if not any(m in plain for m in group):
            missing.append(" | ".join(m.decode() for m in group))
    if not started:
        missing.insert(0, f"first frame ({spec['first'].decode()}) never appeared")
    ok = not missing and status == 0
    detail = f"{len(out)} bytes, exit {status}" + (f", missing: {missing}" if missing else "")
    return ok, detail


def main(argv):
    names = argv[1:] or list(SPECS)
    failed = 0
    for name in names:
        ok, detail = run(name, SPECS[name])
        print(f"{'PASS' if ok else 'FAIL'}  {name:16} {detail}")
        failed += not ok
    print("SMOKE", "PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
