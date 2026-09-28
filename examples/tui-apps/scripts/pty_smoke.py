import os, pty, sys, time, select, signal, re

def run(binary, cwd, keys, seconds=4.0, cols=100, rows=30):
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = str(cols)
        os.environ["LINES"] = str(rows)
        os.execv(binary, [binary])
    import fcntl, termios, struct
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    out = b""
    end = time.time() + seconds
    sent = False
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            out += chunk
            # answer terminal capability queries the way xterm would
            if b"\x1b]10;?" in chunk:
                os.write(fd, b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\")
            if b"\x1b]11;?" in chunk:
                os.write(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            if b"\x1b[6n" in chunk:
                os.write(fd, b"\x1b[1;1R")
        if not sent and time.time() > end - seconds / 2:
            for k in keys:
                os.write(fd, k)
                time.sleep(0.15)
            sent = True
    try:
        os.write(fd, b"q")
        time.sleep(0.3)
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        status = -1
    return out, status

ansi = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
apps = [
    (f"{ROOT}/progress-timer/progress-timer", f"{ROOT}/progress-timer", [], [b"PROGRESS TIMER", b"%"]),
    (f"{ROOT}/file-browser/file-browser", ROOT, [b"j", b"j", b"\r", b"/", b"go", b"\x1b", b"\t"], [b"FILE BROWSER", b"tui-apps"]),
    (f"{ROOT}/system-monitor/system-monitor", ROOT, [b"p", b"+", b"-"], [b"SYSTEM MONITOR", b"CPU", b"MEMORY", b"PID"]),
]
ok = True
for binary, cwd, keys, expect in apps:
    out, status = run(binary, cwd, keys)
    plain = ansi.sub(b"", out)
    missing = [e for e in expect if e not in plain]
    print(f"== {os.path.basename(binary)}: {len(out)} bytes raw, exit status {status}")
    if missing:
        ok = False
        print("   MISSING:", missing)
    else:
        print("   all expected markers present:", [e.decode() for e in expect])
    tail = plain.decode("utf-8", "replace").strip().splitlines()[-8:]
    for line in tail:
        print("   |", line[:110])
print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
