"""
chatPDB web app — chatpdb.mdeller.com
Flask + WebSocket PTY: each browser tab gets its own chat_remote.py subprocess in a
pseudo-terminal. xterm.js renders it in the browser exactly as the CLI looks locally.
"""
from __future__ import annotations

import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import uuid
from pathlib import Path

from flask import Flask, render_template, session
from flask_sock import Sock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CHAT_PY = BASE_DIR / "chat_remote.py"       # thin wrapper around chatPDB's chat.py
PYTHON_BIN = BASE_DIR / ".venv" / "bin" / "python"
HF_API_URL = os.environ.get("HF_SPACE_URL", "")   # set in systemd / .env

app = Flask(__name__)
sock = Sock(app)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(32))

# Active PTY sessions: session_id -> (master_fd, process)
_sessions: dict[str, tuple[int, subprocess.Popen]] = {}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


def _resize_pty(master_fd: int, cols: int, rows: int) -> None:
    """Apply a real terminal resize via the TIOCSWINSZ ioctl."""
    try:
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
    except OSError:
        pass


@sock.route("/ws")
def terminal(ws):
    """WebSocket endpoint: proxies keystrokes -> PTY, PTY output -> browser."""
    sid = session.get("sid", str(uuid.uuid4()))

    # Spawn a new chat_remote.py PTY if none exists for this session
    if sid not in _sessions:
        master_fd, slave_fd = pty.openpty()
        env = {**os.environ, "HF_SPACE_URL": HF_API_URL, "TERM": "xterm-256color"}
        proc = subprocess.Popen(
            [str(PYTHON_BIN), str(CHAT_PY)],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            close_fds=True, env=env,
        )
        os.close(slave_fd)
        _sessions[sid] = (master_fd, proc)

    master_fd, proc = _sessions[sid]

    import threading

    def _read_pty():
        """Forward PTY output to browser."""
        while proc.poll() is None:
            r, _, _ = select.select([master_fd], [], [], 0.05)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    ws.send(data.decode("utf-8", errors="replace"))
                except OSError:
                    break

    reader = threading.Thread(target=_read_pty, daemon=True)
    reader.start()

    try:
        while proc.poll() is None:
            msg = ws.receive(timeout=1)
            if msg is None:
                continue

            # Real fix (found + fixed 2026-07-22, chatPDB Phase 9): chem_sage's own version of
            # this handler writes every incoming WebSocket message straight to the PTY as raw
            # keystrokes -- including the frontend's own JSON resize control messages
            # ({"type": "resize", "cols": N, "rows": M}), which would appear as literal garbage
            # text typed into the terminal. Parse each message as JSON first; only messages that
            # actually decode to a resize control message are treated specially (and applied via
            # a real TIOCSWINSZ ioctl, not silently dropped); anything else -- including plain
            # keystrokes, which are never valid JSON -- falls through to the original raw-write
            # behaviour unchanged.
            is_resize = False
            try:
                parsed = json.loads(msg)
                if isinstance(parsed, dict) and parsed.get("type") == "resize":
                    is_resize = True
                    cols = int(parsed.get("cols", 80))
                    rows = int(parsed.get("rows", 24))
                    _resize_pty(master_fd, cols, rows)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

            if not is_resize:
                try:
                    os.write(master_fd, msg.encode("utf-8"))
                except OSError:
                    break
    except Exception:
        pass
    finally:
        _sessions.pop(sid, None)
        try:
            proc.terminate()
            os.close(master_fd)
        except OSError:
            pass
