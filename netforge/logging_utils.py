"""
Thread-safe logging.

`shutdown_requested` and `TUI_ACTIVE` are module attributes that are *mutated*
rather than imported by value. `from logging_utils import shutdown_requested`
would freeze a copy and the training loop would never see the flag flip - this
is the one Python gotcha that genuinely bites in a threaded app.
"""

import logging
import os
import sys
import threading

shutdown_requested = False
TUI_ACTIVE = False

_print_lock = threading.Lock()
_logger = None


def setup_log_file(path):
    """Attach a file handler. Returns the path, or None if it could not be made."""
    global _logger
    if not path:
        return None
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        _logger = logging.getLogger("netforge")
        _logger.setLevel(logging.INFO)
        for h in list(_logger.handlers):
            _logger.removeHandler(h)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(fh)
        return path
    except Exception:
        return None


def append_log(msg, level="info"):
    """Write to the file and stdout. Never raises - a dead log must not stop a run."""
    try:
        with _print_lock:
            ts = time_str()
            tag = {"warn": "WARN ", "error": "ERROR", "debug": "debug"}.get(level, "     ")
            if level == "error":
                print(f"[{ts}] {tag} {msg}", file=sys.stderr, flush=True)
            elif level != "debug":
                print(f"[{ts}] {tag} {msg}", flush=True)
    except Exception:
        pass
    try:
        if _logger:
            _logger.info("%s", msg)
    except Exception:
        pass


def append_debug(msg):
    append_log(msg, "debug")


def append_error(msg):
    append_log(msg, "error")


def time_str():
    import time
    return time.strftime("%H:%M:%S")


def request_shutdown():
    global shutdown_requested
    shutdown_requested = True
    append_log("shutdown requested")


def clear_shutdown():
    global shutdown_requested
    shutdown_requested = False