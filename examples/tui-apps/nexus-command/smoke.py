import os, pty, sys, time, select, signal, re, fcntl, termios, struct

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, "nexus-command")
COLS, ROWS = 120, 40

# (delay before sending, bytes)
SCRIPT = [
    (0.8, b"r"),          # run pipeline
    (3.0, b"i"),          # arm chaos failure
    (0.5, b"\t"),         # focus logs
    (0.5, b"2"),          # agents tab
    (0.8, b"3"),          # logs tab
    (0.8, b"4"),          # metrics tab
    (0.8, b"1"),          # overview
    (0.5, b"\x0b"),       # ctrl+k palette
    (0.4, b"kil"),        # fuzzy "Kill selected agent"
    (0.6, b"\x1b"),       # close palette
    (0.4, b"k"),          # kill dialog
    (0.6, b"n"),          # decline
    (0.4, b"?"),          # help
    (0.8, b"?"),          # close help
    (0.4, b"p"),          # pause
    (0.8, b"p"),          # resume
]
EXPECT = [b"NEXUS", b"command center", b"Agents", b"Logs", b"Metrics", b"Researcher", b"Planner",
          b"pipeline started", b"RUNNING", b"chaos armed", b"keys", b"paused", b"quality"]
ansi = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[=>]")

def main():
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"], os.environ["LINES"] = str(COLS), str(ROWS)
        os.execv(BIN, [BIN])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    out = b""
    deadline = time.time() + 26
    script = list(SCRIPT)
    next_at = None  # keys start only after the first frame is on screen
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.05)
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
            if next_at is None and b"NEXUS" in ansi.sub(b"", out):
                next_at = time.time() + script[0][0]
        if script and next_at is not None and time.time() >= next_at:
            os.write(fd, script[0][1])
            script.pop(0)
            if script:
                next_at = time.time() + script[0][0]
    os.write(fd, b"q")
    time.sleep(0.4)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _, status = os.waitpid(pid, 0)
    plain = ansi.sub(b"", out)
    missing = [e for e in EXPECT if e not in plain]
    # kill: either the confirm dialog opened, or the "not running" toast fired (timing-dependent)
    if b"mid-task?" not in plain and b"is not running" not in plain:
        missing.append(b"kill dialog/toast")
    print(f"nexus-command: {len(out)} bytes raw, exit status {status}")
    print("missing markers:", [m.decode() for m in missing] or "none")
    ok = not missing and status == 0
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
