#!/usr/bin/env python3
"""PTY smoke test for gitscope.

Forks the binary inside a pseudo-terminal at 120x40, answers the terminal
capability queries the way xterm would, drives the UI with real keystrokes and
checks that the rendered frames contain the expected markers and that the
program exits cleanly on `q`.
"""
import os
import pty
import re
import select
import signal
import struct
import sys
import time
import fcntl
import termios

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "gitscope")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
COLS, ROWS = 120, 40
KEYS = [b"j", b"2", b"3", b"j", b"\r", b"?", b"?"]
MARKERS = [b"gitscope", b"Status", b"Branches", b"Commits", b"Stash", b"keyboard reference"]

ANSI = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*(\x07|\x1b\\)|\x1b[=>]")


def run():
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(REPO)
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = str(COLS)
        os.environ["LINES"] = str(ROWS)
        os.execv(BINARY, [BINARY])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    out = b""
    frames = []
    queue = list(KEYS) + [b"q"]
    deadline = time.time() + 12.0
    next_key_at = time.time() + 2.0  # let the first refresh land
    exited = False
    while time.time() < deadline:
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
        if queue and time.time() >= next_key_at:
            k = queue.pop(0)
            frames.append(ANSI.sub(b"", out))
            os.write(fd, k)
            next_key_at = time.time() + 0.6
        if not queue and time.time() >= next_key_at + 1.0:
            break
    if not exited:
        time.sleep(0.5)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _, status = os.waitpid(pid, 0)
    return out, status, frames


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found: {BINARY} (run `go build -o gitscope .` first)")
        return 1
    out, status, frames = run()
    plain = ANSI.sub(b"", out)
    code = os.waitstatus_to_exitcode(status)
    missing = [m for m in MARKERS if m not in plain]
    print(f"== gitscope: {len(out)} bytes raw, exit status {code}, {len(frames)} frames captured")
    ok = True
    if missing:
        ok = False
        print("   MISSING markers:", missing)
    else:
        print("   all expected markers present:", [m.decode() for m in MARKERS])
    if code != 0:
        ok = False
        print("   non-zero exit status")
    if b"panic:" in plain or b"goroutine " in plain:
        ok = False
        print("   PANIC detected in output")
    tail = plain.decode("utf-8", "replace").strip().splitlines()[-12:]
    for line in tail:
        print("   |", line[:118])
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
