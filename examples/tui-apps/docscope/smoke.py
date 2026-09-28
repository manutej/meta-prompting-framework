#!/usr/bin/env python3
"""PTY smoke test for docscope.

Forks the binary in a real pseudo-terminal at 120x40, answers the terminal
colour queries the way xterm would, sends a scripted key sequence, strips ANSI
and checks that the expected markers were rendered and that the process exited
cleanly on `q`.
"""
import os, pty, sys, time, select, signal, re, fcntl, termios, struct

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "docscope")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "docs"))
COLS, ROWS = 120, 40
KEYS = [b"j", b"\r", b"\t", b"\t", b"j", b"j", b"\x10", b"spec", b"\r", b"?", b"?", b"q"]
EXPECT = [b"docscope", b"Files", b"Outline", b"Find file", b"RESEARCH_SPEC_USE_CASES.md", b"Keys \xc2\xb7 docscope"]

ansi = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[=>]")


def run(binary, args, keys, seconds=8.0, cols=COLS, rows=ROWS):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = str(cols)
        os.environ["LINES"] = str(rows)
        os.execv(binary, [binary] + args)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    out = b""
    end = time.time() + seconds
    exited = False
    # Keys are interleaved with reads: if the harness stopped draining the pty
    # while typing, the renderer would block and coalesce away the intermediate
    # frames (finder, help) that this test wants to see.
    next_key = time.time() + seconds / 4
    ki = 0
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                exited = True
                break
            if not chunk:
                exited = True
                break
            out += chunk
            if b"\x1b]10;?" in chunk:
                os.write(fd, b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\")
            if b"\x1b]11;?" in chunk:
                os.write(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            if b"\x1b[6n" in chunk:
                os.write(fd, b"\x1b[1;1R")
        if ki < len(keys) and time.time() >= next_key:
            os.write(fd, keys[ki])
            ki += 1
            next_key = time.time() + 0.25
    if not exited:
        try:
            os.write(fd, b"q")
            time.sleep(0.5)
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        status = -1
    return out, status


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found: {BINARY} (run `go build -o docscope .` first)")
        return 2
    out, status = run(BINARY, [ROOT], KEYS)
    plain = ansi.sub(b"", out)
    missing = [e for e in EXPECT if e not in plain]
    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
    print(f"== docscope: {len(out)} bytes raw, exit status {exit_code}")
    ok = True
    if missing:
        ok = False
        print("   MISSING:", missing)
    else:
        print("   all expected markers present:", [e.decode() for e in EXPECT])
    if exit_code != 0:
        ok = False
        print("   expected exit status 0")
    tail = plain.decode("utf-8", "replace").strip().splitlines()[-8:]
    for line in tail:
        print("   |", line[:110])
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
