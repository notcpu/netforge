"""
Charts.

Matplotlib canvases embedded in tkinter widgets. All dark-styled once here so
the monitor tab does not have to care about colours.

The canvas draw is the single most expensive thing in the GUI at 6 Hz, so
`Charts.update` throttles to `min_interval` and skips the redraw entirely when
the underlying series has not changed.
"""


import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

BG = "#141922"
PANEL = "#1c2230"
GRID = "#2b3446"
TEXT = "#e8e3d9"
MUTED = "#8b94a5"
ACCENT = "#7fd4a8"
ACCENT2 = "#8fb8ff"
WARN = "#e5a36b"
ERR = "#e57373"


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, pad=6, loc="left")


def _empty(ax, msg="waiting for data"):
    ax.clear()
    _style_ax(ax)
    ax.text(0.5, 0.5, msg, color=MUTED, fontsize=9, ha="center", va="center",
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


class Charts:
    """Owns four matplotlib figures and repaints them from SharedState."""

    def __init__(self, master, hz=6):
        self.hz = max(1, int(hz))
        self._last = {}
        self.widgets = {}

        self.fig_loss = Figure(figsize=(6, 3.2), dpi=100, facecolor=BG)
        self.fig_perf = Figure(figsize=(4, 3.2), dpi=100, facecolor=BG)
        self.fig_mem = Figure(figsize=(4, 3.2), dpi=100, facecolor=BG)
        self.fig_params = Figure(figsize=(4, 3.2), dpi=100, facecolor=BG)

    def attach(self, master, key, figure, parent, row=0, column=0, **kw):
        """Create the canvas and return the tk widget to grid it into."""
        canvas = FigureCanvasTkAgg(figure, master)
        widget = canvas.get_tk_widget()
        widget.configure(highlightthickness=0, bg=BG)
        self.widgets[key] = (canvas, widget)
        return widget

    # -- painting -----------------------------------------------------------

    def paint_all(self, snap, training=True):
        self.paint_loss(snap)
        self.paint_perf(snap)
        self.paint_mem(snap)
        self.paint_params(snap)

    def paint_loss(self, snap):
        key = "loss"
        lh, eh = snap["loss_history"], snap["eval_history"]
        sig = (len(lh), len(eh), lh[-1][0] if lh else 0)
        if self._last.get(key) == sig:
            return
        self._last[key] = sig
        if not self.widgets.get(key):
            return
        canvas, _ = self.widgets[key]
        self.fig_loss.clear()
        ax = self.fig_loss.add_subplot(111)
        if not lh:
            _empty(ax, "loss  -  start training to see curves")
        else:
            xs = [p[0] for p in lh]
            ys = [p[1] for p in lh]
            ax.plot(xs, ys, color=ACCENT2, linewidth=0.7, alpha=0.55,
                    label="train (per step)")
            # rolling average - a raw per-step curve at 6 Hz is unreadable
            w = max(1, min(200, len(ys) // 60 or 1))
            if len(ys) >= w:
                sm = [sum(ys[max(0, i - w + 1):i + 1]) /
                      len(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]
                ax.plot(xs, sm, color=ACCENT, linewidth=1.8, label=f"smoothed ({w})")
            if eh:
                ax.plot([p[0] for p in eh], [p[1] for p in eh], color=WARN,
                        linewidth=1.4, marker="o", markersize=3, label="eval")
            ax.set_xlabel("step", color=MUTED, fontsize=8)
            ax.set_ylabel("loss", color=MUTED, fontsize=8)
            leg = ax.legend(fontsize=7, loc="upper right", framealpha=0.15)
            for t in leg.get_texts():
                t.set_color(TEXT)
            _style_ax(ax, "loss")
            if eh and eh[-1][1] is not None:
                ax.set_yscale("log")
        canvas.draw_idle()

    def paint_perf(self, snap):
        key = "perf"
        lh, gh = snap["loss_history"], snap["grad_history"]
        sig = (len(lh), len(gh))
        if self._last.get(key) == sig:
            return
        self._last[key] = sig
        if not self.widgets.get(key):
            return
        canvas, _ = self.widgets[key]
        self.fig_perf.clear()
        ax = self.fig_perf.add_subplot(111)
        if not lh:
            _empty(ax, "gradient norm  -  start training")
        else:
            xs = [p[0] for p in gh] or [p[0] for p in lh]
            ys = [p[1] for p in gh]
            if ys:
                ax.plot(xs, ys, color=ACCENT2, linewidth=0.9, label="grad norm")
                ax.set_yscale("log")
            ax2 = ax.twinx()
            ax2.plot([p[0] for p in lh], [p[1] for p in lh], color=ACCENT,
                     linewidth=0.8, alpha=0.6)
            ax2.set_yscale("log")
            ax2.tick_params(colors=MUTED, labelsize=8)
            ax2.spines["right"].set_color(GRID)
            ax.set_xlabel("step", color=MUTED, fontsize=8)
            _style_ax(ax, f"gradient norm   |   {snap['sps']:.1f} it/s · "
                          f"{snap['tps'] / 1000:.1f}k tok/s")
        canvas.draw_idle()

    def paint_mem(self, snap, capacity=None, estimated=None):
        key = "mem"
        mh = snap["mem_history"]
        sig = (len(mh), capacity)
        if self._last.get(key) == sig:
            return
        self._last[key] = sig
        if not self.widgets.get(key):
            return
        canvas, _ = self.widgets[key]
        self.fig_mem.clear()
        ax = self.fig_mem.add_subplot(111)
        if not mh:
            _empty(ax, "memory  -  reserved VRAM over time")
        else:
            # mem_history entries are (step, bytes) - see state.push_metrics
            xs = [p[0] for p in mh]
            ys = [p[1] / 1024 ** 3 for p in mh]
            ax.plot(xs, ys, color=WARN, linewidth=1.2, label="reserved")
            if capacity:
                ax.axhline(capacity / 1024 ** 3, color=ERR, linestyle="--",
                           linewidth=1, label="capacity")
                ax.set_ylim(0, max(capacity / 1024 ** 3, max(ys) * 1.25))
            ax.set_ylabel("GiB", color=MUTED, fontsize=8)
            ax.set_xlabel("step", color=MUTED, fontsize=8)
            leg = ax.legend(fontsize=7, loc="upper left", framealpha=0.15)
            for t in leg.get_texts():
                t.set_color(TEXT)
            _style_ax(ax, f"VRAM  now {max(ys) if ys else 0:.2f} GiB   "
                          f"est {estimated / 1024 ** 3:.2f} GiB"
                          if estimated else "VRAM")
        canvas.draw_idle()

    def paint_params(self, snap, breakdown=None):
        key = "params"
        # n_params belongs in the signature: it is published by the training
        # thread some time after the first paint, and without it the title
        # stays stuck on the 0 it saw at startup and never repaints.
        sig = (snap.get("n_params", 0), tuple(sorted((n, c) for n, c in (breakdown or []))))
        if self._last.get(key) == sig:
            return
        self._last[key] = sig
        if not self.widgets.get(key):
            return
        canvas, _ = self.widgets[key]
        self.fig_params.clear()
        ax = self.fig_params.add_subplot(111)
        if not breakdown:
            _empty(ax, "parameter budget")
        else:
            items = breakdown[:12]
            names = [i[0] for i in items][::-1]
            vals = [i[1] for i in items][::-1]
            total = sum(v for _, v in items) or 1
            ax.barh(names, [100 * v / total for v in vals], color=ACCENT,
                    height=0.62)
            for i, (n, v) in enumerate(items):
                ax.text(100 * v / total + 1, len(items) - 1 - i,
                        f"{v / 1e6:.2f}M", color=MUTED, fontsize=7, va="center")
            ax.set_xlabel("% of parameters", color=MUTED, fontsize=8)
            ax.set_xlim(0, 118)
            _style_ax(ax, f"parameter budget  ·  "
                          f"{snap.get('n_params', 0):,} total")
        canvas.draw_idle()

    def force(self, *keys):
        for k in (keys or tuple(self._last.keys())):
            self._last[k] = None
