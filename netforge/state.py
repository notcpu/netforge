"""
SharedState - the single channel between the training thread and the GUI.

One module-level instance. The training worker writes every iteration; the GUI
reads on a timer. Everything goes through `state.lock`, which is why the GUI
can render a consistent snapshot without tearing.

If this were a library instead of an app you would pass a state object around.
It is a singleton here because the training loop runs in a daemon thread
spawned by the GUI and threading a handle through subprocess boundaries is
pointless.
"""

import threading
import time
from collections import deque


class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        # scalars the GUI renders every frame
        self.status = "idle"
        self.stage = ""
        self.global_step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.loss_smoothed = 0.0
        self.best_loss = float("inf")
        self.grad_norm = 0.0
        self.lr = 0.0
        self.sps = 0.0
        self.tps = 0.0
        self.total_tokens = 0
        self.elapsed = 0.0
        self.eta = 0.0
        self.device = "cpu"
        self.model_name = ""
        self.n_params = 0
        self.mem_estimated = 0
        self.mem_measured = 0
        self.mem_detail = ""
        self.eval_loss = None
        self.error_msg = ""

        # history buffers. Bounded: an 8-hour run at 10 logs/sec would
        # otherwise allocate ~2M floats for the chart and nothing else.
        self.loss_history = deque(maxlen=20000)
        self.eval_history = deque(maxlen=2000)
        self.lr_history = deque(maxlen=20000)
        self.grad_history = deque(maxlen=2000)
        self.mem_history = deque(maxlen=2000)
        self.tok_history = deque(maxlen=2000)

        self.log_lines = deque(maxlen=2000)
        self.samples = deque(maxlen=8)
        self.param_stats = []

    # -- logging ------------------------------------------------------------

    def log(self, msg, level="info"):
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.log_lines.append((ts, level, str(msg)))

    def fail(self, msg):
        """Record a fatal error. Named `fail` not `error` because `error` is
        also a data field - an attribute and a method with the same name means
        whichever was defined last silently wins."""
        self.log(msg, "error")
        with self.lock:
            self.error_msg = str(msg)

    def clear_log(self):
        with self.lock:
            self.log_lines.clear()

    # -- metric publishing --------------------------------------------------

    def publish(self, **kw):
        """Atomically update a batch of metrics in one lock acquisition."""
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def push_metrics(self, step, loss, lr, grad_norm, tokens, mem_bytes):
        with self.lock:
            self.loss_history.append((step, loss))
            self.lr_history.append((step, lr))
            self.grad_history.append((step, grad_norm))
            self.tok_history.append((step, tokens))
            self.mem_history.append((step, mem_bytes))

    def push_eval(self, step, loss):
        with self.lock:
            self.eval_history.append((step, loss))
            self.eval_loss = loss

    def push_sample(self, prompt, text):
        with self.lock:
            self.samples.append({"prompt": prompt, "text": text})

    # -- GUI snapshots ------------------------------------------------------

    def snapshot(self):
        """A consistent copy for rendering. Never blocks for more than the
        duration of a few attribute copies."""
        with self.lock:
            return {
                "status": self.status,
                "global_step": self.global_step,
                "total_steps": self.total_steps,
                "loss": self.loss,
                "loss_smoothed": self.loss_smoothed,
                "best_loss": self.best_loss,
                "grad_norm": self.grad_norm,
                "lr": self.lr,
                "sps": self.sps,
                "tps": self.tps,
                "total_tokens": self.total_tokens,
                "elapsed": self.elapsed,
                "eta": self.eta,
                "device": self.device,
                "n_params": self.n_params,
                "mem_estimated": self.mem_estimated,
                "mem_measured": self.mem_measured,
                "eval_loss": self.eval_loss,
                "error": self.error_msg,
                "loss_history": list(self.loss_history),
                "eval_history": list(self.eval_history),
                "grad_history": list(self.grad_history),
                "mem_history": list(self.mem_history),
                "samples": list(self.samples),
                "log_lines": list(self.log_lines),
            }


state = SharedState()
