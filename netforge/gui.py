"""
The application.

Five tabs, one config dict, one training thread.

  Model     family, sizes, and a param-target auto-fit that back-solves layers
            /width /heads from "7b" as you type
  Data      synthetic toy tasks (no download), local .txt, CSV, or HuggingFace
            with live subset/split discovery
  Train     optimizer, schedule, steps, sampling
  Memory    the analytic report, the per-device bar, the fit plan, and a
            "measure for real" button that allocates the model and compares
  Monitor   live charts, samples, and the log

The GUI never trains and the training loop never draws. They meet only at
SharedState. That split is why the Monitor tab is safe to open mid-run and why
closing the window kills training cleanly.

Performance notes (why this file is shaped the way it is):

  * Typing in a field only *schedules* a refresh (debounced). The refresh
    collects the widgets once and fans the result out; nothing re-collects.
  * Charts are matplotlib. They are only repainted while the Monitor tab is
    actually visible, at most ~2 Hz, and only when the snapshot changed.
  * The tick does nothing at all while no training is running.
  * Anything slow (HF lookups, "measure for real") runs in a worker thread
    and hands its result back through a queue that the tick drains. Worker
    threads never touch Tk directly.
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
from tkinter import filedialog, messagebox, ttk

from . import arch as arch_mod
from . import checkpoint as ckpt_mod
from . import config as cfg_mod
from . import dataset as ds_mod
from . import hfhub as hfhub_mod
from . import logging_utils
from . import memory as mem_mod
from . import tokenizer as tok_mod
from .charts import (ACCENT, ACCENT2, BG, ERR, GRID, MUTED, PANEL, TEXT,
                     WARN, Charts)
from .state import state

BG_PANEL = PANEL

# -- tunables ---------------------------------------------------------------

CHART_MIN_INTERVAL = 0.5      # seconds between chart repaints while training
FORM_MAX_WIDTH = 1240         # form tabs stop stretching past this and centre
LOG_KEEP_LINES = 1500         # when the log trims, it trims down to this
LOG_TRIM_AT = 2200            # ...once it grows past this
LOG_BURST_MAX = 400           # never insert more than this many lines per tick
LOG_CHARTS = (("loss", "loss"), ("perf", "perf"))   # (chart key, label)

# ints that legitimately take fractions ("0.5" GB must not truncate to 0,
# which means "uncapped")
_FRACTIONAL_INT_KEYS = {"data.max_gb"}

# errors a half-typed field is allowed to cause; anything else gets printed
_HALF_TYPED = (ValueError, TypeError, KeyError, ZeroDivisionError,
               OverflowError, IndexError)

# one extra point of font on tall screens (set once, before the theme is built)
_BUMP = 0


def ui(size, *flags):
    return ("Segoe UI", size + _BUMP) + tuple(flags)


def mono(size, *flags):
    return ("Consolas", size + _BUMP) + tuple(flags)


def _to_bool(raw):
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _n(v):
    """Format a maybe-number for display without ever raising."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:,}"
    return "?" if v is None else str(v)


def build_theme(root):
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=BG, foreground=TEXT,
                    fieldbackground="#232c3d", bordercolor=GRID,
                    focuscolor=ACCENT, font=ui(10))
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=PANEL, relief="flat")
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Card.TLabel", background=PANEL, foreground=TEXT)
    style.configure("Muted.TLabel", background=PANEL, foreground=MUTED,
                    font=ui(9))
    style.configure("Accent.TLabel", background=PANEL, foreground=ACCENT,
                    font=ui(10, "bold"))
    style.configure("Head.TLabel", background=BG, foreground=TEXT,
                    font=ui(13, "bold"))
    style.configure("Mono.TLabel", background=PANEL, foreground=MUTED,
                    font=mono(9))
    style.configure("TButton", background="#2a3448", foreground=TEXT,
                    padding=(10, 6), relief="flat")
    style.map("TButton",
              background=[("active", "#374560"), ("disabled", "#222a38")])
    style.configure("Small.TButton", padding=(8, 3))
    style.configure("Go.TButton", background=ACCENT, foreground="#10241b",
                    padding=(14, 8), font=ui(10, "bold"))
    style.map("Go.TButton", background=[("active", "#96e6bd"),
                                        ("disabled", "#3c4a45")])
    style.configure("Stop.TButton", background=ERR, foreground="#2a0f0f",
                    padding=(14, 8), font=ui(10, "bold"))
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background="#1a2130", foreground=MUTED,
                    padding=(16 + 4 * _BUMP, 8 + 2 * _BUMP))
    style.map("TNotebook.Tab",
              background=[("selected", PANEL)], foreground=[("selected", TEXT)])
    style.configure("TEntry", fieldbackground="#232c3d", foreground=TEXT,
                    insertcolor=TEXT, borderwidth=1, relief="flat", padding=4)
    style.configure("TCombobox", fieldbackground="#232c3d", background="#232c3d",
                    foreground=TEXT, arrowcolor=MUTED, borderwidth=1,
                    relief="flat", padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", "#232c3d")],
              foreground=[("readonly", TEXT)])
    style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
    style.map("TCheckbutton", background=[("active", PANEL)])
    style.configure("Bar.TCheckbutton", background=BG, foreground=TEXT)
    style.map("Bar.TCheckbutton", background=[("active", BG)])
    style.configure("TProgressbar", background=ACCENT, troughcolor="#232c3d",
                    borderwidth=0)
    # Treeview ships as a white listbox under clam, so without this the
    # dataset browser's status text (tinted for a dark bg) is unreadable.
    style.configure("Treeview", background="#161c28", fieldbackground="#161c28",
                    foreground=TEXT, rowheight=24, borderwidth=0,
                    font=mono(9))
    style.map("Treeview",
              background=[("selected", "#2a3448")],
              foreground=[("selected", TEXT)])
    style.configure("Treeview.Heading", background="#232c3d", foreground=MUTED,
                    relief="flat", padding=(6, 4), font=ui(9, "bold"))
    style.map("Treeview.Heading",
              background=[("active", "#2f3b52")], foreground=[("active", TEXT)])
    style.configure("TScrollbar", background="#2a3448", troughcolor=BG,
                    borderwidth=0, arrowcolor=MUTED)
    return style


def make_scrollable(tab, style, max_width=FORM_MAX_WIDTH):
    """Wrap a tab in a canvas + scrollbar so short windows do not clip fields.

    Returns the inner frame to put widgets into. The container is packed here
    rather than by the caller - forgetting to pack it yields a tab that builds
    every widget correctly and then renders nothing at all.

    The inner frame stops growing at `max_width` and is centred, so on a wide
    monitor the form stays readable instead of stretching every entry across
    half the screen. The mouse wheel works while the pointer is anywhere over
    the tab (including over entries and labels), and does nothing when the
    content already fits.
    """
    outer = tk.Frame(tab, bg=BG)
    outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=BG)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=sb.set)

    def relayout(_e=None):
        cw = max(1, canvas.winfo_width())
        w = min(cw, max_width)
        canvas.itemconfigure(win, width=w)
        canvas.coords(win, max(0, (cw - w) // 2), 0)
        canvas.configure(scrollregion=(0, 0, cw, max(1, inner.winfo_reqheight())))

    def wheel(e):
        if inner.winfo_reqheight() <= canvas.winfo_height():
            return
        if getattr(e, "num", None) == 4:
            step = -3
        elif getattr(e, "num", None) == 5:
            step = 3
        elif e.delta:
            # Windows sends multiples of 120, macOS sends small deltas
            step = -3 if e.delta > 0 else 3
        else:
            return
        canvas.yview_scroll(step, "units")

    def pointer_inside():
        try:
            x, y = outer.winfo_pointerxy()
            w = outer.winfo_containing(x, y)
        except (tk.TclError, KeyError):
            return False
        return w is not None and str(w).startswith(str(outer))

    def on_enter(_e):
        canvas.bind_all("<MouseWheel>", wheel)
        canvas.bind_all("<Button-4>", wheel)
        canvas.bind_all("<Button-5>", wheel)

    def on_leave(_e):
        # Leave also fires when the pointer moves onto a child widget; only
        # unbind when it really left the tab.
        if pointer_inside():
            return
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas._nf_wheel = wheel          # comboboxes forward their wheel here
    inner.bind("<Configure>", relayout)
    canvas.bind("<Configure>", relayout)
    outer.bind("<Enter>", on_enter)
    outer.bind("<Leave>", on_leave)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    return inner


class Launcher:
    def __init__(self, root, run_cfg=None):
        global _BUMP
        self.root = root
        _BUMP = 1 if root.winfo_screenheight() >= 1100 else 0

        self.run = cfg_mod._backfill(run_cfg or cfg_mod.default_run_cfg())
        self.style = build_theme(root)
        self.widgets = {}          # dotted key -> tk variable (shared by dupes)
        self.combos = {}           # dotted key -> Combobox
        self._types = {}           # dotted key -> bool/int/float/str
        self.strings = {}
        self.exc_q = queue.Queue()
        self._ui_q = queue.Queue()  # callables handed back from worker threads
        self.train_thread = None
        self.train_start = 0.0
        self._train_finished = False
        self._need_final_sync = False
        self._suspend = False
        self._refresh_job = None
        self._oops_seen = set()
        self._last_family = None
        self._last_task = None
        self._log_last_line = None
        self._log_indent_set = False
        self._last_sample = None
        self._last_chart_t = 0.0
        self._chart_sig = None
        self._breakdown = None
        self._monitor_ready = False
        self._dirty = {"memory": True, "monitor": True}
        self.log_vars = {}
        self._scale_figs = {}
        self._scale_checks = {}
        self._hf_browser = None
        self.device_total = 0
        self.device_name = "cpu"
        self._probe_device()

        root.title("netforge  ·  neural network trainer")
        root.report_callback_exception = self._on_cb_exc
        # a wheel over a combobox should scroll the page, not change the value
        root.bind_class("TCombobox", "<MouseWheel>", self._combo_wheel)
        self._fit_window()

        self._build()
        self.refresh_preview()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self._tick)

    # -- setup --------------------------------------------------------------

    def _fit_window(self):
        """Open sized for the screen instead of a fixed 1220x840.

        On a 1920x1200 panel this is 1880x1110 (clear of the title bar and
        taskbar), and on Windows it then maximises so it fills the work area
        exactly. Everything below is weight-driven, so it adapts to whatever
        the final size ends up being.
        """
        r = self.root
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h = min(sw - 40, 1880), min(sh - 90, 1110)
        x, y = max(0, (sw - w) // 2), max(0, (sh - h) // 2 - 20)
        r.geometry(f"{w}x{h}+{x}+{y}")
        r.minsize(min(1100, w), min(720, h))
        if sys.platform.startswith("win") and sh >= 1000:
            try:
                r.state("zoomed")
            except tk.TclError:
                pass

    def _probe_device(self):
        inv = mem_mod.device_inventory()
        if inv["devices"]:
            d = inv["devices"][0]
            self.device_total = d["total"]
            self.device_name = d["name"]
        elif inv["cpu_ram"]:
            self.device_total = inv["cpu_ram"] * 0.6
            self.device_name = "CPU (system RAM)"

    def _build(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=14, pady=(12, 6))
        ttk.Label(top, text="netforge", style="Head.TLabel").pack(side="left")
        self.summary_lbl = ttk.Label(top, text="", style="Muted.TLabel")
        self.summary_lbl.pack(side="left", padx=(14, 0))

        self.btn_train = ttk.Button(top, text="▶  Train", style="Go.TButton",
                                    command=self.on_train)
        self.btn_train.pack(side="right")
        self.btn_stop = ttk.Button(top, text="■  Stop", style="Stop.TButton",
                                   command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="right", padx=(0, 8))

        # The status bar is packed BEFORE the notebook. Pack gives space in
        # packing order, so packing it last meant a short window clipped the
        # status bar away first.
        status = tk.Frame(self.root, bg="#111620")
        status.pack(fill="x", side="bottom")
        self.status_lbl = tk.Label(status, text="ready", bg="#111620", fg=MUTED,
                                   font=mono(9), anchor="w", padx=12, pady=5)
        self.status_lbl.pack(fill="x")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.nb = nb
        self.tabs = {}
        for name in ("Model", "Data", "Train", "Memory", "Monitor"):
            f = tk.Frame(nb, bg=BG)
            nb.add(f, text=f"  {name}  ")
            self.tabs[name] = f
        self._tab_order = ["Model", "Data", "Train", "Memory", "Monitor"]

        self._tab_model()
        self._tab_data()
        self._tab_train()
        self._tab_memory()
        self._tab_monitor()
        self._monitor_ready = True
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _current_tab(self):
        try:
            return self._tab_order[self.nb.index(self.nb.select())]
        except (tk.TclError, IndexError, AttributeError):
            return None

    def _on_tab_changed(self, _e=None):
        if self._current_tab() == "Monitor":
            self._last_chart_t = 0.0
        self._flush_visible()

    def _combo_wheel(self, e):
        w = e.widget
        while w is not None:
            fn = getattr(w, "_nf_wheel", None)
            if fn:
                fn(e)
                break
            w = getattr(w, "master", None)
        return "break"

    def _on_cb_exc(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        try:
            self.status(f"error: {exc.__name__}: {val}")
        except Exception:
            pass

    def _oops(self, where):
        """Report an unexpected exception once, instead of swallowing it."""
        exc = sys.exc_info()[1]
        key = (where, type(exc).__name__, str(exc)[:80])
        if key in self._oops_seen:
            return
        self._oops_seen.add(key)
        print(f"[netforge gui] unexpected error in {where}:", file=sys.stderr)
        traceback.print_exc()

    # -- dotted-key widget registry -----------------------------------------

    def _get(self, dotted, default=""):
        o = self.run
        for p in dotted.split("."):
            if isinstance(o, list):
                try:
                    o = o[int(p)]
                except (ValueError, IndexError):
                    return default
            else:
                if not isinstance(o, dict) or p not in o:
                    return default
                o = o[p]
        return o if o is not None else default

    def _set(self, dotted, value):
        parts = dotted.split(".")
        o = self.run
        for p in parts[:-1]:
            if isinstance(o, list):
                o = o[int(p)]
            else:
                o = o.setdefault(p, {})
        o[parts[-1]] = value

    def _pane(self, parent, column=0):
        """A half-width container that lays out one label/entry pair.

        Fields used to be placed straight into a 4-column grid (label, entry,
        label, entry) with both entry columns weighted. That makes each entry
        absorb half the window, shoving the second column of labels off the
        right edge. One frame per pair fixes it.

        The two panes get equal *weights* but deliberately not `uniform`:
        uniform sizes every pane to the largest request in the group, so a
        long descriptive label in one pane widens both and overflows the
        window. Plain weights honour each pane's own request first and split
        only the leftover.
        """
        f = tk.Frame(parent, bg=parent.cget("bg"))
        f.columnconfigure(0, weight=0, minsize=110)
        f.columnconfigure(1, weight=1)
        try:
            parent.columnconfigure(column, weight=1)
        except tk.TclError:
            pass
        f.grid(row=self._prow, column=column, sticky="ew", padx=6, pady=2)
        return f

    def _subframe(self, parent):
        """Switch the row counter to a nested frame (a LabelFrame group).

        Returns a restore callable, so nesting never loses the outer counter.
        """
        saved = self._prow
        self._prow = 0

        def restore():
            self._prow = saved
        return restore

    def _pair(self, parent, left, right=None):
        """Add one row of up to two fields in side-by-side panes.

        Each spec is (label, key) or (label, key, kwargs). A spec with key=None
        renders the label as plain descriptive text instead of an input.
        Returns the grid row the panes were placed on.
        """
        row = self._prow
        for spec, col in ((left, 0), (right, 1)):
            if not spec:
                continue
            label, key = spec[0], spec[1]
            kw = dict(spec[2]) if len(spec) > 2 else {}
            pane = self._pane(parent, column=col)
            if key is None:
                ttk.Label(pane, text=label, style="Muted.TLabel",
                          wraplength=460, justify="left").grid(
                    row=0, column=0, columnspan=3, sticky="w", padx=10, pady=3)
                continue
            self._field(pane, 0, 0, label, key, **kw)
        self._prow += 1
        return row

    def _field(self, parent, row, col, label, key, width=18, values=None,
               vartype=str, hint=None, colspan=1, style="Card.TLabel",
               button=None, editable=False):
        """One labelled input bound to a dotted config key.

        A key can appear in more than one place (dropout, activation and the
        byte cap each live in two frames). Every widget for a key shares ONE
        tk variable. Previously the second registration silently replaced the
        first, so edits in the transformer frame's dropout/activation boxes
        were ignored.
        """
        cur = self._get(key, "")
        if isinstance(cur, bool):
            kind = bool
        elif isinstance(cur, int):
            kind = int
        elif isinstance(cur, float):
            kind = float
        else:
            kind = str
        self._types.setdefault(key, kind)
        kind = self._types[key]

        var = self.widgets.get(key)
        if var is None:
            if kind is bool:
                var = tk.BooleanVar(value=_to_bool(cur))
            else:
                var = tk.StringVar(value=str(cur))
            var.trace_add("write", self._on_var_write)
            self.widgets[key] = var

        ttk.Label(parent, text=label, style=style).grid(
            row=row, column=col, sticky="w", padx=(10, 6), pady=3)

        if kind is bool and values is None:
            ttk.Checkbutton(parent, variable=var).grid(
                row=row, column=col + 1, sticky="w", padx=(0, 10), pady=3)
        elif values is not None:
            cb = ttk.Combobox(parent, textvariable=var, width=width,
                              state="normal" if editable else "readonly",
                              values=list(values))
            cb.grid(row=row, column=col + 1, sticky="ew", padx=(0, 10),
                    pady=3, columnspan=colspan)
            self.combos[key] = cb
        else:
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 10), pady=3,
                columnspan=colspan)
        if button:
            ttk.Button(parent, text=button[0], command=button[1],
                       style="Small.TButton").grid(
                row=row, column=col + 2, sticky="w", padx=(0, 10))
        if hint:
            ttk.Label(parent, text=hint, style="Muted.TLabel").grid(
                row=row + 1, column=col + 1, sticky="w", padx=(0, 10),
                columnspan=colspan + 1)
            return 2
        return 1

    def _collect(self):
        """Pull widget values back into the run dict, with light coercion.

        Coercion is driven by the type each field had when it was built, not
        by whatever happens to be in the run dict right now. Before, one
        half-typed value (an emptied int field) was stored as a string, which
        changed the field's apparent type and turned off coercion for it
        forever after. Now an unparseable value just leaves the last good one
        in place.
        """
        for key, var in self.widgets.items():
            raw = var.get()
            kind = self._types.get(key, str)
            try:
                if kind is bool:
                    val = _to_bool(raw)
                elif kind is int:
                    f = float(raw)
                    val = f if (key in _FRACTIONAL_INT_KEYS
                                and not f.is_integer()) else int(f)
                elif kind is float:
                    val = float(raw)
                else:
                    val = raw
            except (TypeError, ValueError):
                continue
            self._set(key, val)

    # -- debounced refresh --------------------------------------------------

    def _on_var_write(self, *_):
        if not self._suspend:
            self._schedule_refresh()

    def _schedule_refresh(self, delay=180):
        if self._refresh_job is not None:
            try:
                self.root.after_cancel(self._refresh_job)
            except tk.TclError:
                pass
        self._refresh_job = self.root.after(delay, self._run_refresh)

    def _run_refresh(self):
        self._refresh_job = None
        self.refresh_preview()

    # -- Model tab ----------------------------------------------------------

    def _tab_model(self):
        tab = make_scrollable(self.tabs["Model"], self.style)
        self._prow = 0

        ttk.Label(tab, text="FAMILY", style="Card.TLabel").grid(
            row=self._prow, column=0, columnspan=2, sticky="w",
            padx=10, pady=(10, 2))
        self._prow += 1
        self._pair(tab, ("family", "model.family", dict(values=("gpt", "mlp"))),
                   ("gpt = decoder-only transformer\nmlp = plain feedforward net",
                    None))
        self._prow += 1

        ttk.Label(tab, text="AUTO-FIT BY PARAMETER COUNT",
                  style="Card.TLabel").grid(
            row=self._prow, column=0, columnspan=2, sticky="w",
            padx=10, pady=(14, 2))
        self._prow += 1
        row = tk.Frame(tab, bg=BG)
        row.grid(row=self._prow, column=0, columnspan=2, sticky="ew", padx=10)
        self.target_var = tk.StringVar(value="5m")
        ttk.Label(row, text="target", style="Muted.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.target_var, width=10).pack(side="left", padx=6)
        ttk.Button(row, text="solve", style="TButton",
                   command=self.on_solve_arch).pack(side="left", padx=2)
        self.solve_lbl = tk.Label(row, text="", bg=BG, fg=ACCENT2,
                                  font=mono(9), anchor="w")
        self.solve_lbl.pack(side="left", padx=10, fill="x", expand=True)
        self._prow += 1

        presets = tk.Frame(tab, bg=BG)
        presets.grid(row=self._prow, column=0, columnspan=2, sticky="w",
                     padx=10, pady=(2, 8))
        for label, fam, params, override in arch_mod.PRESETS:
            b = ttk.Button(presets, text=label, width=9,
                           command=lambda f=fam, p=params, o=override:
                           self.on_preset(f, p, o))
            b.pack(side="left", padx=2)
        self._prow += 2

        # --- GPT frame
        self.gpt_frame = tk.LabelFrame(tab, text="  transformer  ", bg=BG,
                                       fg=TEXT, font=ui(9, "bold"),
                                       bd=1, relief="flat")
        self.gpt_frame.grid(row=self._prow, column=0, columnspan=2,
                            sticky="nsew", padx=10, pady=4)
        restore = self._subframe(self.gpt_frame)
        self._pair(self.gpt_frame, ("n_layer", "model.n_layer"),
                   ("embed_dim", "model.embed_dim"))
        self._pair(self.gpt_frame, ("n_head", "model.n_head"),
                   ("block_size", "model.block_size"))
        self._pair(self.gpt_frame,
                   ("ff_mult", "model.ff_mult", dict(hint="ff_dim = embed_dim × ff_mult")),
                   ("vocab_size", "model.vocab_size", dict(hint="must equal tokenizer vocab")))
        self._pair(self.gpt_frame, ("dropout", "model.dropout"),
                   ("pos_encoding", "model.pos_encoding", dict(values=cfg_mod.POS_ENCODINGS)))
        self._pair(self.gpt_frame,
                   ("norm_type", "model.norm_type", dict(values=cfg_mod.NORMS)),
                   ("activation", "model.activation", dict(values=cfg_mod.ACTIVATIONS)))
        self._pair(self.gpt_frame, ("tie weights", "model.tie_weights"),
                   ("dtype", "model.dtype", dict(values=("fp32", "fp16", "bf16"))))
        restore()
        self._prow += 1

        # --- MLP frame
        self.mlp_frame = tk.LabelFrame(tab, text="  feedforward  ", bg=BG,
                                       fg=TEXT, font=ui(9, "bold"),
                                       bd=1, relief="flat")
        self.mlp_frame.grid(row=self._prow, column=0, columnspan=2,
                            sticky="nsew", padx=10, pady=4)
        restore = self._subframe(self.mlp_frame)
        self._pair(self.mlp_frame,
                   ("n_features", "model.n_features",
                    dict(hint="input width (set from data if mismatched)")),
                   ("n_outputs", "model.n_outputs"))
        self._pair(self.mlp_frame,
                   ("hidden", "model.hidden",
                    dict(width=20, hint='e.g. "64 64" or "64,128,64"')),
                   ("dropout", "model.dropout"))
        self._pair(self.mlp_frame,
                   ("activation", "model.activation", dict(values=cfg_mod.ACTIVATIONS)),
                   ("out_activation", "model.out_activation",
                    dict(values=("none", "logit", "sigmoid", "relu"),
                         hint="'none' = regression (MSE)")))
        restore()
        self._prow += 1

        ttk.Label(tab, text="TOKENIZER", style="Card.TLabel").grid(
            row=self._prow, column=0, columnspan=2, sticky="w",
            padx=10, pady=(14, 2))
        self._prow += 1
        self._pair(tab, ("type", "tokenizer.type", dict(values=("char", "bpe"))),
                   ("vocab_size", "tokenizer.vocab_size",
                    dict(hint="char≈101 · bpe trained from corpus")))
        self._pair(tab, ("train from corpus", "tokenizer.train_from_corpus"),
                   ("tokenizer file", "tokenizer.tokenizer_file", dict(width=22)))
        self._apply_family()

    def _apply_family(self):
        fam = str(self._get("model.family", "gpt"))
        if fam == "mlp":
            self.gpt_frame.grid_remove()
            self.mlp_frame.grid()
        else:
            self.mlp_frame.grid_remove()
            self.gpt_frame.grid()

    # -- Data tab -----------------------------------------------------------

    def _tab_data(self):
        tab = make_scrollable(self.tabs["Data"], self.style)
        self._prow = 0
        self._pair(tab, ("task", "data.task", dict(values=cfg_mod.TASKS)))

        # Fetching a corpus is not tied to which task you happen to have
        # selected, and this row sits OUTSIDE the task frames on purpose:
        # inside the huggingface frame it was invisible until you had already
        # set task=text AND typed a dataset id, which is exactly the state you
        # are not in when you go looking for the download button.
        get = tk.Frame(tab, bg=BG)
        get.grid(row=self._prow, column=0, columnspan=2, sticky="ew",
                 padx=10, pady=(10, 2))
        self.btn_hf = ttk.Button(
            get, text="Browse HF datasets & download with a size limit",
            command=self.on_browse_hf)
        self.btn_hf.pack(side="left")
        ttk.Label(get, style="Muted.TLabel",
                  text="file sizes first · all-or-nothing per file").pack(
            side="left", padx=8)
        self._prow += 1

        self.data_frames = {}
        self._prow += 1
        for key, title in (("synthetic", "  synthetic toy task  "),
                           ("local", "  local .txt directory  "),
                           ("csv", "  csv / tabular  "),
                           ("hf", "  huggingface dataset (streamed)  ")):
            f = tk.LabelFrame(tab, text=title, bg=BG, fg=TEXT,
                              font=ui(9, "bold"), bd=1, relief="flat")
            f.grid(row=self._prow, column=0, columnspan=2, sticky="nsew",
                   padx=10, pady=4)
            self.data_frames[key] = f
            self._prow += 1

        synth_tasks = ds_mod.SYNTH_TASKS
        f = self.data_frames["synthetic"]
        restore = self._subframe(f)
        self.synth_desc = tk.StringVar(
            value=synth_tasks.get(str(self._get("data.synth", "")), ""))
        self._pair(f, ("task", "data.synth", dict(values=tuple(synth_tasks.keys()))))
        ttk.Label(f, textvariable=self.synth_desc, style="Muted.TLabel",
                  wraplength=560, justify="left").grid(
            row=self._prow, column=0, columnspan=2, sticky="w", padx=10, pady=2)
        self._prow += 1
        self._pair(f, ("rows", "data.synth_n"), ("label noise", "data.synth_noise"))
        restore()

        f = self.data_frames["local"]
        restore = self._subframe(f)
        self._pair(f, ("directory", "data.data_dir",
                       dict(width=24, button=("browse", self.on_browse_dir))))
        self._pair(f, ("byte cap GB", "data.max_gb", dict(hint="0 = uncapped")))
        restore()

        f = self.data_frames["csv"]
        restore = self._subframe(f)
        self._pair(f, ("csv path", "data.csv_path",
                       dict(width=24, button=("browse", self.on_browse_csv))))
        self._pair(f, ("target column", "data.csv_target"),
                   ("label-encode target", "data.csv_index"))
        restore()

        f = self.data_frames["hf"]
        restore = self._subframe(f)
        self._pair(f, ("dataset id", "data.hf_dataset",
                       dict(width=24, hint="e.g. roneneldan/TinyStories",
                            button=("fetch subsets", self.on_fetch_subsets))))
        self._pair(f, ("split", "data.hf_split",
                       dict(values=("train", "validation"), editable=True)),
                   ("subset", "data.hf_subset", dict(values=(), editable=True)))
        self._pair(f, ("text field", "data.hf_field", dict(hint="blank = auto-detect")),
                   ("max rows", "data.max_rows", dict(hint="0 = no row cap")))
        ttk.Label(f, style="Muted.TLabel",
                  text="streaming straight from the Hub, nothing written to disk "
                       "— use the button above to fetch a copy you keep").grid(
            row=self._prow, column=0, columnspan=2, sticky="w", padx=10,
            pady=(2, 2))
        self._prow += 1

        # --- budget: the knobs that stop this from eating the machine
        ttk.Label(f, text="BUDGET  (how much gets pulled — keep these small)",
                  style="Card.TLabel").grid(row=self._prow, column=0, columnspan=2,
                                             sticky="w", padx=10, pady=(12, 2))
        self._prow += 1
        self._pair(f, ("max tokens", "data.max_tokens",
                       dict(hint="0 = auto from params × ratio")),
                   ("tokens/param", "data.tokens_per_param",
                    dict(hint="20 = Chinchilla; lower for small models")))
        self._pair(f, ("cache to disk", "data.cache_text",
                       dict(hint="off by default — writes GBs otherwise")),
                   ("bytes/token", "data.bytes_per_token",
                    dict(hint="0 = measured table (parquet 0.62 / char)")))
        self.budget_lbl = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.budget_lbl, style="Muted.TLabel",
                  wraplength=700, justify="left").grid(
            row=self._prow, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 4))
        self._prow += 1
        self.hf_status = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.hf_status, style="Muted.TLabel",
                  wraplength=700).grid(row=self._prow, column=0, columnspan=2,
                                       sticky="w", padx=10)
        restore()

        self._apply_task()

    def _apply_task(self):
        task = str(self._get("data.task", "synthetic"))
        want = {"synthetic": "synthetic", "text": "hf", "tabular": "csv"}.get(task)
        # 'text' can mean local dir or HF; show both when hf id is set
        if task == "text":
            want = "hf" if self._get("data.hf_dataset", "") else "local"
        for key, frame in self.data_frames.items():
            if key == want:
                frame.grid()
            else:
                frame.grid_remove()

    def _refresh_budget(self, run):
        """Show what the data stage will actually pull, before it does.

        This line exists because the first version of this app would silently
        stream a 2 GB corpus, write 1.8 GB to disk and build a 12 GB int64
        tensor for a 40k-parameter model, with nothing on screen saying so.
        """
        try:
            d, m, t = run["data"], run["model"], run["train"]
            if d.get("task") == "synthetic":
                self.budget_lbl.set(
                    f"synthetic task: {int(d.get('synth_n', 0)):,} rows — "
                    f"generated locally, nothing downloaded")
                return
            if d.get("task") == "tabular":
                self.budget_lbl.set(f"reads one CSV: {d.get('csv_path', '(none set)')}")
                return

            cap, src = ds_mod.resolve_token_budget(d, m, t)
            plan = ds_mod.token_plan(run)
            ram_mb = cap * 5 / 1024 ** 2     # 4B/token int32 + ~1B on the wire
            ratio = plan["ratio"]
            txt = (f"{plan['params_label']} × {ratio:g} tok/param = "
                   f"{cap:,} tokens ({src})  ·  ~{ram_mb:,.0f} MB RAM"
                   f"  ·  ~{plan['gb']:.2f} GB on disk"
                   f"  ({plan['bpt']:.2f} B/token, {plan['kind']})")
            if not d.get("hf_dataset", ""):
                txt += "  ·  from local files"
            elif src == "auto":
                txt += ("  ·  set max_tokens to override, or raise "
                        "tokens/param")
            kind = self._local_kind(str(d.get("data_dir", "")))
            if kind:
                txt = txt.replace("from local files", f"from local {kind}")
            if bool(d.get("cache_text")):
                txt += "\n⚠ disk cache is ON — it will write the corpus to disk"
            self.budget_lbl.set(txt)
        except _HALF_TYPED:
            pass
        except Exception:
            self._oops("budget")

    # -- Train tab ----------------------------------------------------------

    def _tab_train(self):
        tab = make_scrollable(self.tabs["Train"], self.style)
        self._prow = 0

        def section(title):
            ttk.Label(tab, text=title, style="Card.TLabel").grid(
                row=self._prow, column=0, columnspan=2, sticky="w",
                padx=10, pady=(14, 3))
            self._prow += 1

        section("OPTIMIZER")
        self._pair(tab, ("name", "optimizer.name",
                         dict(values=cfg_mod.OPTIMIZERS,
                              hint="sgd has no optimizer state → far less memory")),
                   ("weight decay", "optimizer.weight_decay"))
        self._pair(tab, ("grad clip", "optimizer.max_grad_norm"),
                   ("accum steps", "optimizer.accum",
                    dict(hint="simulates a larger batch")))

        section("SCHEDULE")
        self._pair(tab, ("steps", "train.steps"), ("batch size", "train.batch_size"))
        self._pair(tab, ("learning rate", "train.lr"),
                   ("schedule", "train.scheduler", dict(values=cfg_mod.SCHEDULERS)))
        self._pair(tab, ("warmup frac", "train.warmup_frac"),
                   ("min lr frac", "train.min_lr_frac"))

        section("CADENCE & SAMPLING")
        self._pair(tab, ("log every", "train.log_every"),
                   ("eval every", "train.eval_every"))
        self._pair(tab, ("eval batches", "train.eval_batches"),
                   ("sample every", "train.sample_every"))
        self._pair(tab, ("gen length", "train.gen_len"),
                   ("temperature", "train.sample_temp"))
        self._pair(tab, ("top-k", "train.sample_top_k"), ("seed", "train.seed"))

        section("RUNTIME")
        self._pair(tab, ("checkpoint", "runtime.checkpoint",
                         dict(width=26, button=("browse", self.on_browse_ckpt))))
        self._pair(tab, ("mixed precision", "runtime.amp",
                         dict(hint="auto-off below 1M params")),
                   ("resume if exists", "runtime.resume"))
        self._pair(tab, ("torch.compile", "runtime.compile"),
                   ("pin data to gpu", "runtime.gpu_data"))
        self._pair(tab, ("max pin MB", "runtime.gpu_data_max_mb"))
        self._prow += 1

        row = tk.Frame(tab, bg=BG)
        row.grid(row=self._prow, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        ttk.Button(row, text="validate", command=self.on_validate).pack(
            side="left", padx=2)
        ttk.Button(row, text="inspect checkpoint", command=self.on_ckpt_info).pack(
            side="left", padx=2)
        ttk.Button(row, text="save config", command=self.on_save_cfg).pack(
            side="left", padx=2)
        ttk.Button(row, text="load config", command=self.on_load_cfg).pack(
            side="left", padx=2)

    # -- Memory tab ---------------------------------------------------------

    def _tab_memory(self):
        tab = self.tabs["Memory"]
        top = tk.Frame(tab, bg=BG)
        top.pack(fill="x", padx=10, pady=(10, 4))
        top.columnconfigure(0, weight=1, uniform="ctrl")
        top.columnconfigure(1, weight=1, uniform="ctrl")

        # Two independent panes rather than one 4-column grid: a single grid
        # with two expanding entry columns pushes the right-hand column of
        # labels off the window edge.
        def pane(parent):
            f = tk.Frame(parent, bg=BG)
            f.columnconfigure(0, weight=0)
            f.columnconfigure(1, weight=1)
            return f

        left, right = pane(top), pane(top)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        self._field(left, 0, 0, "param dtype", "memory.dtype",
                    values=("fp32", "fp16", "bf16", "int8"))
        self._field(left, 1, 0, "compute dtype", "memory.compute_dtype",
                    values=("fp32", "fp16", "bf16"))
        self._field(left, 2, 0, "gradient ckpt", "memory.gradient_checkpointing",
                    hint="recompute activations")
        self._field(left, 3, 0, "optimizer offload", "memory.optimizer_offload",
                    hint="keep AdamW moments in CPU RAM")

        self._field(right, 0, 0, "sharding", "memory.sharding",
                    values=mem_mod.SHARDING_MODES)
        self._field(right, 1, 0, "num devices", "memory.num_devices")
        self._field(right, 2, 0, "chunked logits", "memory.max_logit_batch",
                    hint="max B×T per loss chunk (0 = off)")
        self._field(right, 3, 0, "param offload", "memory.param_offload")

        bar = tk.Frame(tab, bg=BG)
        bar.pack(fill="x", padx=10, pady=8)
        self.mode_var = tk.StringVar(value="training")
        ttk.Radiobutton(bar, text="training", value="training",
                        variable=self.mode_var, command=self.refresh_memory).pack(
            side="left", padx=4)
        ttk.Radiobutton(bar, text="inference", value="inference",
                        variable=self.mode_var, command=self.refresh_memory).pack(
            side="left", padx=4)
        ttk.Button(bar, text="what actually fits here?",
                   command=self.on_fit_plan).pack(side="left", padx=8)
        self.btn_measure = ttk.Button(bar, text="measure for real",
                                      command=self.on_measure)
        self.btn_measure.pack(side="left", padx=2)

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body.columnconfigure(0, weight=3, uniform="body")
        body.columnconfigure(1, weight=2, uniform="body")
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.mem_text = tk.Text(left, bg=PANEL, fg=TEXT, bd=0, height=12,
                                width=40, font=mono(9), wrap="none", padx=12,
                                pady=10, highlightthickness=0)
        sy = ttk.Scrollbar(left, orient="vertical", command=self.mem_text.yview)
        sx = ttk.Scrollbar(left, orient="horizontal", command=self.mem_text.xview)
        self.mem_text.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.mem_text.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.mem_text.tag_configure("label", foreground=TEXT)
        self.mem_text.tag_configure("note", foreground=MUTED)
        self.mem_text.tag_configure("total", foreground=ACCENT,
                                    font=mono(10, "bold"))
        self.mem_text.tag_configure("ok", foreground=ACCENT)
        self.mem_text.tag_configure("bad", foreground=ERR)
        self.mem_text.tag_configure("warn", foreground=WARN)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        self.fit_text = tk.Text(right, bg=PANEL, fg=MUTED, bd=0,
                                font=mono(9), wrap="word", padx=12,
                                pady=10, height=9, width=30, highlightthickness=0)
        self.fit_text.pack(fill="both", expand=True)
        self.fit_text.tag_configure("h", foreground=TEXT, font=ui(10, "bold"))
        self.fit_text.tag_configure("ok", foreground=ACCENT)
        self.fit_text.tag_configure("bad", foreground=ERR)
        self.fit_text.tag_configure("warn", foreground=WARN)
        self.fit_text.tag_configure("note", foreground=MUTED)

        self.charts = Charts(tab)
        self.mcharts = self.charts

    # -- Monitor tab --------------------------------------------------------

    def _tab_monitor(self):
        tab = self.tabs["Monitor"]
        head = tk.Frame(tab, bg=BG)
        head.pack(fill="x", padx=10, pady=(10, 4))

        self.m_status = tk.StringVar(value="idle")
        self.m_loss = tk.StringVar(value="—")
        self.m_step = tk.StringVar(value="0 / 0")
        self.m_speed = tk.StringVar(value="—")
        self.m_mem = tk.StringVar(value="—")
        self.m_params = tk.StringVar(value="—")

        cells = [("STATUS", self.m_status, ACCENT2), ("LOSS", self.m_loss, ACCENT),
                 ("STEP", self.m_step, TEXT), ("SPEED", self.m_speed, TEXT),
                 ("VRAM", self.m_mem, WARN), ("PARAMS", self.m_params, TEXT)]
        for label, var, colour in cells:
            f = tk.Frame(head, bg=PANEL, width=0)
            f.pack(side="left", fill="x", expand=True, padx=3)
            # note: pack() takes a scalar pady - the (top, bottom) tuple form
            # is grid-only and raises "bad screen distance" here.
            tk.Label(f, text=label, bg=PANEL, fg=MUTED,
                     font=ui(8), anchor="w", padx=10,
                     pady=6).pack(fill="x")
            tk.Label(f, textvariable=var, bg=PANEL, fg=colour,
                     font=mono(11, "bold"), anchor="w",
                     padx=10, pady=2).pack(fill="x")

        # -- control strip: log-scale toggles + log tools
        strip = tk.Frame(tab, bg=BG)
        strip.pack(fill="x", padx=10, pady=(4, 0))
        tk.Label(strip, text="y-axis", bg=BG, fg=MUTED, font=ui(9)).pack(
            side="left", padx=(4, 6))
        for key, title in LOG_CHARTS:
            var = tk.BooleanVar(value=False)
            self.log_vars[key] = var
            cb = ttk.Checkbutton(strip, text=f"{title}: log scale",
                                 variable=var, style="Bar.TCheckbutton",
                                 command=lambda k=key: self._on_scale_toggle(k))
            cb.pack(side="left", padx=8)
            self._scale_checks[key] = cb
        ttk.Button(strip, text="clear log", style="Small.TButton",
                   command=self._clear_log).pack(side="right", padx=4)
        self.follow_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(strip, text="follow log", variable=self.follow_log,
                        style="Bar.TCheckbutton").pack(side="right", padx=8)

        # -- body: charts (2x2) on the left, samples + log stacked on the right.
        # Both sides are grid_propagate(False): their children's *requested*
        # sizes (a matplotlib canvas asks for its figure size, a Text asks for
        # 80x24 characters) must not decide the layout, or a tall request
        # pushes the log off the bottom of the window. Weights alone decide.
        mid = tk.Frame(tab, bg=BG)
        mid.pack(fill="both", expand=True, padx=10, pady=4)
        mid.columnconfigure(0, weight=5)
        mid.columnconfigure(1, weight=3)
        mid.rowconfigure(0, weight=1)

        charts_box = tk.Frame(mid, bg=BG)
        charts_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        charts_box.grid_propagate(False)
        for i in (0, 1):
            charts_box.columnconfigure(i, weight=1, uniform="cc")
            charts_box.rowconfigure(i, weight=1, uniform="cr")

        self.charts.attach(charts_box, "loss", self.charts.fig_loss, charts_box)
        self.charts.widgets["loss"][1].grid(row=0, column=0, sticky="nsew",
                                            padx=(0, 4), pady=(0, 4))
        self.charts.attach(charts_box, "perf", self.charts.fig_perf, charts_box)
        self.charts.widgets["perf"][1].grid(row=0, column=1, sticky="nsew",
                                            padx=(0, 0), pady=(0, 4))
        self.charts.attach(charts_box, "mem", self.charts.fig_mem, charts_box)
        self.charts.widgets["mem"][1].grid(row=1, column=0, sticky="nsew",
                                           padx=(0, 4), pady=(0, 0))
        self.charts.attach(charts_box, "params", self.charts.fig_params, charts_box)
        self.charts.widgets["params"][1].grid(row=1, column=1, sticky="nsew",
                                              padx=(0, 0), pady=(0, 0))

        side = tk.Frame(mid, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(0, weight=2, uniform="side")
        side.rowconfigure(1, weight=3, uniform="side")

        samples = tk.LabelFrame(side, text="  samples  ", bg=BG, fg=TEXT,
                                font=ui(9, "bold"), bd=1, relief="flat")
        samples.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.sample_text = tk.Text(samples, bg=PANEL, fg=TEXT, bd=0, height=4,
                                   width=30, font=mono(9), wrap="word", padx=10,
                                   pady=8, highlightthickness=0)
        sy = ttk.Scrollbar(samples, orient="vertical",
                           command=self.sample_text.yview)
        self.sample_text.configure(yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        self.sample_text.pack(side="left", fill="both", expand=True)
        self.sample_text.tag_configure("prompt", foreground=ACCENT2)
        self.sample_text.tag_configure("gen", foreground=TEXT)

        logs = tk.LabelFrame(side, text="  log  ", bg=BG, fg=TEXT,
                             font=ui(9, "bold"), bd=1, relief="flat")
        logs.grid(row=1, column=0, sticky="nsew")
        self.log_text = tk.Text(logs, bg="#111620", fg=TEXT, bd=0, height=4,
                                width=30, font=mono(9), wrap="word", padx=10,
                                pady=8, highlightthickness=0, state="disabled",
                                cursor="arrow")
        sy2 = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sy2.set)
        sy2.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.tag_configure("ts", foreground=MUTED)
        self.log_text.tag_configure("error", foreground=ERR)
        self.log_text.tag_configure("warn", foreground=WARN)
        # a disabled Text still lets you select/copy once it has focus
        self.log_text.bind("<1>", lambda e: self.log_text.focus_set())

        foot = tk.Frame(tab, bg=BG)
        foot.pack(fill="x", padx=10, pady=(4, 10))
        self.progress = ttk.Progressbar(foot, mode="determinate", maximum=1000)
        self.progress.pack(fill="x")
        self.err_lbl = tk.Label(foot, text="", bg=BG, fg=ERR,
                                font=mono(9), anchor="w")
        self.err_lbl.pack(fill="x")

        # -- wire the log-scale toggles into the charts' draw path
        for key, fig in (("loss", self.charts.fig_loss),
                         ("perf", self.charts.fig_perf)):
            if self._install_scale_hook(key, fig):
                self._scale_figs[key] = fig
            else:
                self._scale_checks[key].state(["disabled"])

    # -- log scale ------------------------------------------------------------

    def _install_scale_hook(self, key, fig):
        """Make a figure honour its log toggle on *every* draw.

        The charts repaint themselves from scratch, which can reset the axis
        scale back to linear. Flipping the scale once from here would be undone
        by the next paint, so instead the figure's own draw() is wrapped: right
        before matplotlib renders, the y-scale is forced to whatever the
        checkbox says. That also means the toggle needs no knowledge of how
        charts.py plots.
        """
        if getattr(fig, "_nf_scale_hooked", False):
            return True
        if not (hasattr(fig, "draw") and hasattr(fig, "axes")):
            return False
        orig = fig.draw

        def draw(renderer, *a, **kw):
            try:
                self._enforce_scale(key, fig)
            except Exception:
                self._oops(f"log-scale:{key}")
            return orig(renderer, *a, **kw)

        fig.draw = draw
        fig._nf_scale_hooked = True
        return True

    def _enforce_scale(self, key, fig):
        want = "log" if self.log_vars[key].get() else "linear"
        for ax in fig.axes:
            if ax.get_yscale() == want:
                continue
            if want == "log":
                import numpy as np
                ax.set_yscale("log", nonpositive="clip")
                lo, hi = ax.get_ylim()
                if lo <= 0:
                    # a linear chart usually pins the bottom at 0, which a
                    # log axis cannot show: reseat it on the smallest positive
                    # value actually plotted
                    ys = [np.asarray(ln.get_ydata(), dtype=float).ravel()
                          for ln in ax.lines]
                    ys = np.concatenate(ys) if ys else np.array([])
                    ys = ys[np.isfinite(ys) & (ys > 0)]
                    if ys.size:
                        ax.set_ylim(ys.min() / 1.5, max(hi, ys.max() * 1.5))
                    else:
                        ax.set_ylim(1e-3, max(hi, 1.0))
            else:
                ax.set_yscale("linear")
                ax.relim()
                ax.autoscale_view()

    def _on_scale_toggle(self, key):
        fig = self._scale_figs.get(key)
        canvas = getattr(fig, "canvas", None)
        if canvas is not None:
            canvas.draw_idle()

    # -- live refresh -------------------------------------------------------

    def refresh_preview(self):
        """Collect the widgets ONCE, then fan the result out.

        Runs (debounced) after edits. It used to re-collect three times per
        keystroke and repaint every matplotlib chart each time; now the heavy
        stuff only happens for the tab you are actually looking at.
        Must never raise.
        """
        if self._suspend:
            return
        try:
            self._collect()
            run = cfg_mod._backfill(self.run)
            m, t = run["model"], run["train"]
            fam = str(m["family"])
            if fam != self._last_family:
                self._last_family = fam
                self._apply_family()
            task = str(run["data"]["task"])
            if task != self._last_task:
                self._last_task = task
                self._apply_task()
            self.synth_desc.set(
                ds_mod.SYNTH_TASKS.get(str(run["data"].get("synth", "")), ""))

            n = arch_mod.estimate_params(m)
            bs = int(t["batch_size"])
            seq = int(m["block_size"]) if fam == "gpt" else 1
            mem = mem_mod.estimate(m, bs, seq, training=True,
                                   memcfg=run["memory"],
                                   optimizer=run["optimizer"]["name"],
                                   mixed_precision=bool(run["runtime"]["amp"]))
            self.summary_lbl.configure(
                text=f"{arch_mod.fmt_params(n)} params   ·   "
                     f"{mem_mod.fmt_bytes(mem['total'])} @ B{bs}×T{seq}   ·   "
                     f"{self.device_name}")
            self._breakdown = arch_mod.param_breakdown(m)[0]
            self._refresh_budget(run)
            self._dirty["memory"] = True
            self._dirty["monitor"] = True
            self._flush_visible()
        except _HALF_TYPED:
            pass      # half-typed values are normal; never let them kill the UI
        except Exception:
            self._oops("refresh_preview")

    def _flush_visible(self):
        """Repaint whatever the current tab shows, if it is out of date."""
        tab = self._current_tab()
        try:
            if tab == "Memory" and self._dirty["memory"]:
                self._render_memory()
            elif tab == "Monitor" and self._dirty["monitor"]:
                self._paint_charts(force=True)
        except _HALF_TYPED:
            pass
        except Exception:
            self._oops("flush_visible")

    def refresh_memory(self):
        """Memory report for the current mode (training / inference)."""
        self._dirty["memory"] = True
        try:
            self._render_memory()
        except _HALF_TYPED:
            pass
        except Exception:
            self._oops("refresh_memory")

    def _render_memory(self):
        self._collect()
        run = cfg_mod._backfill(self.run)
        m, t, mem = run["model"], run["train"], run["memory"]
        training = self.mode_var.get() == "training"
        bs = int(t["batch_size"])
        seq = int(m["block_size"]) if m["family"] == "gpt" else 1
        est = mem_mod.estimate(m, bs, seq, training=training, memcfg=mem,
                               optimizer=run["optimizer"]["name"],
                               mixed_precision=bool(run["runtime"]["amp"]))
        self._render_mem_text(est, training, seq)
        self._dirty["memory"] = False

    def _paint_charts(self, snap=None, force=False):
        if not self._monitor_ready:
            return
        snap = snap or state.snapshot()
        est = snap["mem_estimated"] or 0
        if force:
            self.mcharts.force()
        steps = [
            ("loss", lambda: self.mcharts.paint_loss(snap)),
            ("perf", lambda: self.mcharts.paint_perf(snap)),
            ("mem", lambda: self.mcharts.paint_mem(
                snap, capacity=self.device_total or None,
                estimated=est or None)),
        ]
        if self._breakdown is not None:
            steps.append(("params", lambda: self.mcharts.paint_params(
                snap, self._breakdown)))
        for name, fn in steps:
            try:
                fn()
            except Exception:
                self._oops(f"chart:{name}")
        self._dirty["monitor"] = False
        self._last_chart_t = time.monotonic()

    def _render_mem_text(self, est, training, seq):
        txt = self.mem_text
        txt.delete("1.0", "end")
        mode = "training" if training else "inference"
        txt.insert("end", f"{est['n_params']:,} parameters  ({mode})\n\n", "label")
        width = 22
        for _, label, nbytes, note in est["items"]:
            frac = nbytes / max(1e-9, est["total"])
            bar = "█" * int(round(frac * 24))
            txt.insert("end", f"{label:<{width}} {mem_mod.fmt_bytes(nbytes):>11}  "
                              f"{bar:<24} {note}\n", "note")
        txt.insert("end", "\n" + "─" * 74 + "\n")
        txt.insert("end", f"{'TOTAL':<{width}} "
                          f"{mem_mod.fmt_bytes(est['total']):>11}\n", "total")
        if est.get("peak", 0) > est["total"] * 1.001:
            txt.insert("end", f"{'PEAK (with transients)':<{width}} "
                              f"{mem_mod.fmt_bytes(est['peak']):>11}\n", "label")
        if est.get("offloaded"):
            txt.insert("end", f"{'held in CPU RAM':<{width}} "
                              f"{mem_mod.fmt_bytes(est['offloaded']):>11}\n",
                       "note")
        if est.get("per_device_note") and est["per_device_note"] != "single device":
            txt.insert("end", f"{'per device':<{width}} "
                              f"{mem_mod.fmt_bytes(est['per_device']):>11}   "
                              f"{est['per_device_note']}\n", "label")
        if self.device_total:
            txt.insert("end", "\n")
            pct = 100.0 * est["per_device"] / self.device_total
            filled = int(round(min(1.0, pct / 100) * 30))
            bar = "█" * filled + "░" * (30 - filled)
            tag = "ok" if pct <= 100 else "bad"
            txt.insert("end", f"{self.device_name}  "
                              f"{mem_mod.fmt_bytes(self.device_total)}\n", "note")
            txt.insert("end", f"{bar}  {pct:.0f}%   ", "note")
            txt.insert("end", "FITS\n" if pct <= 100 else "DOES NOT FIT\n", tag)

    # -- actions ------------------------------------------------------------

    def on_preset(self, family, params, override):
        try:
            self._collect()
            run = cfg_mod._backfill(self.run)
            target = arch_mod.parse_params(params)

            # Apply the preset's constraints BEFORE solving. Solving first and
            # overriding after gives you a shape tuned for the old vocab and
            # context, which is how "Llama-7B" ends up at 17 million params.
            for k, v in (override or {}).items():
                self._set(f"model.{k}", v)
                run["model"][k] = v
            run["model"]["family"] = family

            solved = arch_mod.suggest_arch(
                target, family,
                block_size=int(run["model"]["block_size"]),
                vocab_size=int(run["model"]["vocab_size"]),
                ff_mult=float(run["model"]["ff_mult"]),
                n_features=int(run["model"].get("n_features", 8)),
                n_outputs=int(run["model"].get("n_outputs", 1)))
            for k, v in solved.items():
                if k == "warning":
                    continue
                self._set(f"model.{k}", v)

            self._set("model.family", family)
            if family == "mlp":
                self._set("data.task", "synthetic")
                self._set("tokenizer.type", "char")
            else:
                self._set("data.task", "text")
                self._set("tokenizer.type", self._pick_tok_type(target, family))
                self._set("tokenizer.vocab_size", str(solved.get("vocab_size", 512)))
            self._refresh_widgets()
            self.target_var.set(params)
            n = arch_mod.estimate_params(cfg_mod._backfill(self.run)["model"])
            err = 100.0 * (n - target) / max(1, target)
            note = solved.get("warning", "")
            self.solve_lbl.configure(
                text=f"{arch_mod.fmt_params(n)} ({err:+.1f}%)" +
                     (f"   {note}" if note else ""))
        except Exception as exc:
            messagebox.showerror("preset failed", str(exc))

    def _effective_vocab(self, run):
        """The vocab the dataset will actually tokenize with.

        Same precedence as training: a saved tokenizer file wins (its real
        size is read off disk), char is fixed at ~101, and BPE trains up to
        whatever size is configured. Falls back to the model field if none
        of that resolves, so solve never crashes on a half-typed config.
        """
        t = run.get("tokenizer", {}) or {}
        try:
            path = (t.get("tokenizer_file") or "").strip()
            if path and os.path.exists(path):
                return int(tok_mod.build_tokenizer(t).vocab_size)
            if str(t.get("type", "char")) == "char":
                return int(tok_mod.CharTokenizer().vocab_size)
            return int(t.get("vocab_size", 512))
        except Exception:
            pass
        try:
            return int(run["model"]["vocab_size"])
        except Exception:
            return 512

    def _pick_tok_type(self, target, family):
        """Tokenizer family by scale: BPE earns its keep at 10k+ params.

        Below that the embedding dominates the budget and a 101-symbol char
        set is the honest choice; above it char's 101 context-symbols starve
        the model (a 75M net on vocab 101 is all embedding table and no
        brain). MLP never tokenizes, so it always gets char.
        """
        if family != "gpt":
            return "char"
        return "bpe" if int(target) >= 10_000 else "char"

    def on_solve_arch(self):
        try:
            self._collect()
            run = cfg_mod._backfill(self.run)
            target = arch_mod.parse_params(self.target_var.get())
            fam = str(run["model"]["family"])
            if fam == "gpt":
                want = self._pick_tok_type(target, fam)
                cur = str(run["tokenizer"].get("type", "char"))
                if cur != want:
                    self._set("tokenizer.type", want)
                    run["tokenizer"]["type"] = want
                    try:
                        cur_v = int(run["tokenizer"].get("vocab_size", 0))
                    except Exception:
                        cur_v = 0
                    if want == "bpe" and cur_v <= 512:
                        # flipping up from char: a leftover 101 would make
                        # a nonsense BPE/101, so seed the standard default.
                        # A deliberate BPE size above char range is left alone.
                        self._set("tokenizer.vocab_size", "8000")
                        run["tokenizer"]["vocab_size"] = 8000
                    elif want == "char":
                        self._set("tokenizer.vocab_size", "101")
                        run["tokenizer"]["vocab_size"] = 101
            vocab = (self._effective_vocab(run) if fam == "gpt"
                     else int(run["model"]["vocab_size"]))
            solved = arch_mod.suggest_arch(
                target, fam,
                block_size=int(run["model"]["block_size"]),
                vocab_size=vocab,
                ff_mult=float(run["model"]["ff_mult"]),
                n_features=int(run["model"].get("n_features", 8)),
                n_outputs=int(run["model"].get("n_outputs", 1)))
            for k, v in solved.items():
                if k == "warning":
                    continue
                self._set(f"model.{k}", v)
            self._refresh_widgets()
            n = arch_mod.estimate_params(solved)
            err = 100.0 * (n - target) / max(1, target)
            self.solve_lbl.configure(
                text=f"{arch_mod.fmt_params(n)} ({err:+.1f}%)   "
                     f"{solved.get('n_layer','-')}L × {solved.get('embed_dim','-')}d"
                     f" × {solved.get('n_head','-')}h"
                     if fam == "gpt" else
                     f"{arch_mod.fmt_params(n)} ({err:+.1f}%)   "
                     f"hidden {solved.get('hidden')}")
            if solved.get("warning"):
                self.solve_lbl.configure(text=solved["warning"])
        except Exception as exc:
            self.solve_lbl.configure(text=str(exc))

    def _refresh_widgets(self):
        """Push run-dict values into every widget.

        The traces are suspended while this runs. Setting vars one at a time
        otherwise re-enters _collect() on the first write, which reads the
        not-yet-updated widgets and writes their stale values back over the
        ones just applied - which silently turns any preset or solved
        architecture back into whatever the last field happened to hold.
        """
        self._suspend = True
        try:
            for key, var in self.widgets.items():
                try:
                    val = self._get(key, "")
                    if isinstance(var, tk.BooleanVar):
                        var.set(_to_bool(val))
                    else:
                        var.set(str(val))
                except Exception:
                    pass
        finally:
            self._suspend = False
        self.refresh_preview()

    def on_fetch_subsets(self):
        ds = str(self._get("data.hf_dataset", "")).strip()
        if not ds:
            self.hf_status.set("enter a dataset id first, e.g. roneneldan/TinyStories")
            return
        self.hf_status.set(f"fetching {ds} ...")
        subset_now = str(self._get("data.hf_subset", ""))

        def work():
            try:
                subsets, e1 = ds_mod.list_hf_subsets(ds)
                splits, e2 = ds_mod.list_hf_splits(ds, subset_now)
            except Exception as exc:
                subsets, splits, e1 = [], [], f"{type(exc).__name__}: {exc}"
            # results go back through the queue; the tick applies them on the
            # Tk thread (calling root.after from here is not safe)
            self._ui_q.put(lambda: apply(subsets, splits, e1))

        def apply(subsets, splits, e1):
            cb = self.combos.get("data.hf_subset")
            if cb is not None:
                cb.configure(values=list(subsets))
            sp = self.combos.get("data.hf_split")
            if sp is not None and splits:
                sp.configure(values=list(splits))
            if splits:
                self.widgets["data.hf_split"].set(splits[0])
            msg = (f"{len(subsets)} subset(s): {', '.join(subsets[:8])}"
                   f"{' …' if len(subsets) > 8 else ''}   "
                   f"splits: {', '.join(splits) or '-'}"
                   if subsets else
                   f"no subsets listed   splits: {', '.join(splits) or '-'}")
            if e1:
                msg += f"   [{e1}]"
            self.hf_status.set(msg)

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _local_kind(data_dir):
        """What is actually sitting in data_dir - parquet, csv, jsonl or .txt.

        Walks a few levels because the Hub layout nests the shards
        (`<dest>/data/train-*.parquet`), so a one-level probe finds nothing.
        """
        if not data_dir or not os.path.isdir(data_dir):
            return ""
        found = set()
        base_depth = data_dir.rstrip(os.sep).count(os.sep)
        for root, dirs, files in os.walk(data_dir):
            if root.count(os.sep) - base_depth >= 3:
                dirs[:] = []              # deep enough; stop descending
            for f in files:
                ext = hfhub_mod.strip_gz(f)
                if ext == ".parquet":
                    found.add("parquet")
                elif ext in (".csv", ".tsv"):
                    found.add("csv")
                elif ext in (".jsonl", ".json"):
                    found.add("jsonl")
                elif ext == ".txt":
                    found.add(".txt")
        for kind in ("parquet", "csv", "jsonl", ".txt"):
            if kind in found:
                return kind
        return ""

    def on_browse_hf(self):
        """Open the dataset browser (inspect sizes, download under a cap)."""
        if self._hf_browser is not None and self._hf_browser.win.winfo_exists():
            self._hf_browser.win.lift()
            self._hf_browser.win.focus_force()
            return self._hf_browser
        try:
            from . import hfbrowser
            self._hf_browser = hfbrowser.open_browser(self)
        except Exception as exc:
            self._oops("hf_browser")
            messagebox.showerror("Dataset browser failed to open",
                                 f"{type(exc).__name__}: {exc}")
            self._hf_browser = None
        return self._hf_browser

    def on_browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.widgets["data.data_dir"].set(d)

    def on_browse_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if p:
            self.widgets["data.csv_path"].set(p)

    def on_browse_ckpt(self):
        p = filedialog.askopenfilename(filetypes=[("checkpoint", "*.pt"), ("all", "*.*")])
        if p:
            self.widgets["runtime.checkpoint"].set(p)

    def on_validate(self):
        self._collect()
        run = cfg_mod._backfill(self.run)
        errs = cfg_mod.validate_run_cfg(run)
        if errs:
            messagebox.showwarning("Fix these first", "\n\n".join(errs))
        else:
            messagebox.showinfo("Config OK",
                                "Everything validates. Estimated "
                                + mem_mod.one_line(run["model"],
                                                   int(run["train"]["batch_size"]),
                                                   int(run["model"]["block_size"]),
                                                   memcfg=run["memory"],
                                                   optimizer=run["optimizer"]["name"]))

    def on_save_cfg(self):
        self._collect()
        run = cfg_mod._backfill(self.run)
        p = filedialog.asksaveasfilename(defaultextension=".run.json",
                                        filetypes=[("json", "*.json")])
        if p:
            cfg_mod.save_run_cfg(run, p)
            self.status(f"saved config -> {p}")

    def on_load_cfg(self):
        p = filedialog.askopenfilename(filetypes=[("json", "*.json")])
        if not p:
            return
        try:
            run = cfg_mod.load_run_cfg(p)
        except Exception as exc:
            messagebox.showerror("Could not load config", f"{p}\n\n{exc}")
            return
        self.run = run
        self._refresh_widgets()
        self.status(f"loaded config <- {p}")

    def on_ckpt_info(self):
        path = str(self._get("runtime.checkpoint", ""))
        info = ckpt_mod.ckpt_info(path)
        if not info["ok"] and not info["exists"]:
            messagebox.showinfo("Checkpoint", f"{path}\n\n{info['error']}")
            return
        cfg = info.get("model_cfg") or {}
        lines = [
            f"path      {info['path']}",
            f"size      {mem_mod.fmt_bytes(info['size'])}",
            f"step      {_n(info['step'])}",
            f"params    {_n(info['n_params'])}" if info["n_params"] else "params    -",
        ]
        if cfg:
            lines.append(f"config    {cfg.get('family','gpt')} "
                         f"{cfg.get('n_layer','?')}L × {cfg.get('embed_dim','?')}d "
                         f"ctx {cfg.get('block_size','?')} "
                         f"vocab {_n(cfg.get('vocab_size'))}")
        if info["error"]:
            lines.append(f"note      {info['error']}")
        messagebox.showinfo("Checkpoint", "\n".join(lines))

    def on_fit_plan(self):
        self._collect()
        run = cfg_mod._backfill(self.run)
        m = run["model"]
        training = self.mode_var.get() == "training"
        plan = mem_mod.fit_plan(m, training=training, memcfg=run["memory"],
                                optimizer=run["optimizer"]["name"], batch=1, seq=1)
        txt = self.fit_text
        txt.delete("1.0", "end")
        txt.insert("end", "WHAT FITS ON THIS MACHINE\n", "h")
        txt.insert("end", f"\nbudget: {mem_mod.fmt_bytes(plan['budget'])} of "
                          f"{mem_mod.fmt_bytes(self.device_total)}\n\n")
        if plan["fits"]:
            txt.insert("end", f"  batch {plan['batch']}  ×  context "
                              f"{plan['seq']}\n", "ok")
            txt.insert("end", f"  {plan['note']}\n\n")
        else:
            txt.insert("end", f"  {plan['reason']}\n", "bad")
            n = plan.get("suggest_devices")
            if n:
                txt.insert("end", f"\n  would need ~{n} of these GPUs "
                                  f"(or a smaller model / more offload)\n", "warn")
        est = plan["estimate"]
        txt.insert("end", "  at batch=1, seq=1:\n", "h")
        for _, label, nb, _note in est["items"]:
            txt.insert("end", f"    {label:<24} {mem_mod.fmt_bytes(nb):>10}\n", "note")
        txt.insert("end", f"    {'TOTAL':<24} {mem_mod.fmt_bytes(est['total']):>10}\n",
                   "ok" if plan["fits"] else "bad")

    def on_measure(self):
        """Allocate the model for real - in a worker thread.

        This used to run on the Tk thread, so the whole window froze (and
        Windows marked it "not responding") while the model was built and a
        forward/backward ran.
        """
        if self.train_thread and self.train_thread.is_alive():
            self.status("stop training before measuring - they would fight over memory")
            return
        self._collect()
        run = cfg_mod._backfill(self.run)
        training = self.mode_var.get() == "training"
        args = (run["model"], int(run["train"]["batch_size"]),
                int(run["model"]["block_size"]))
        kwargs = dict(training=training, memcfg=run["memory"],
                      optimizer=run["optimizer"]["name"])
        self.btn_measure.configure(state="disabled")
        self.status("allocating the model for real ...")

        def work():
            try:
                res = mem_mod.measure(*args, **kwargs)
            except Exception as exc:
                res = {"error": f"{type(exc).__name__}: {exc}"}
            self._ui_q.put(lambda: self._show_measure(res))

        threading.Thread(target=work, daemon=True).start()

    def _show_measure(self, res):
        self.btn_measure.configure(state="normal")
        txt = self.fit_text
        txt.delete("1.0", "end")
        txt.insert("end", "MEASURED vs ESTIMATED\n", "h")
        if res.get("error"):
            txt.insert("end", f"\n  {res['error']}\n", "bad")
            self.status(res["error"])
            return
        pred = res["predicted"]["total"]
        peak = res.get("peak") or res.get("allocated") or 0
        txt.insert("end", f"\n  device        {res['device']}\n", "note")
        txt.insert("end", f"  params        {res['n_params']:,}\n", "note")
        txt.insert("end", f"  weights       {mem_mod.fmt_bytes(res['measured_weights'])}\n",
                   "note")
        if res.get("measured_states"):
            txt.insert("end", f"  opt states    {mem_mod.fmt_bytes(res['measured_states'])}\n",
                       "note")
        if res.get("allocated"):
            txt.insert("end", f"  allocated     {mem_mod.fmt_bytes(res['allocated'])}\n",
                       "note")
            txt.insert("end", f"  reserved      {mem_mod.fmt_bytes(res['reserved'])}\n",
                       "note")
            txt.insert("end", f"  peak          {mem_mod.fmt_bytes(res['peak'])}\n",
                       "note")
        if res.get("step_error"):
            txt.insert("end", f"\n  forward/backward: {res['step_error']}\n", "warn")
        if peak and pred:
            txt.insert("end", "\n  one real forward+backward was run, so this\n"
                              "  number includes activations and logits.\n", "note")
            d = 100.0 * (peak - pred) / pred
            tag = "ok" if abs(d) < 20 else ("warn" if abs(d) < 50 else "bad")
            verdict = "estimate holds" if abs(d) < 20 else \
                ("estimate is optimistic" if d < 0 else "estimate is pessimistic")
            txt.insert("end", f"\n  analytic  {mem_mod.fmt_bytes(pred)}\n", "note")
            txt.insert("end", f"  measured  {mem_mod.fmt_bytes(peak)}   "
                              f"{d:+.1f}%   {verdict}\n", tag)
        self.status("measured")

    # -- training -----------------------------------------------------------

    def on_train(self):
        if self.train_thread and self.train_thread.is_alive():
            return
        self._collect()
        run = cfg_mod._backfill(self.run)
        errs = cfg_mod.validate_run_cfg(run)
        if errs:
            messagebox.showwarning("Fix these first", "\n\n".join(errs))
            return

        # Preflight: refuse to start if the data stage looks dangerous. The
        # caps are already sane, but a pasted 500000000 max_tokens should be
        # confirmed, not obeyed.
        if not self._preflight_ok(run):
            return

        logging_utils.clear_shutdown()
        state.reset()
        self._wipe_log()
        self.sample_text.delete("1.0", "end")
        self.err_lbl.configure(text="")
        self._last_sample = None
        self._chart_sig = None
        logging_utils.setup_log_file(run["runtime"].get("log_file"))

        self.btn_train.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.train_start = time.time()
        self._train_finished = False
        self._need_final_sync = True
        self.mcharts.force()

        # No on_done callback: Tk's root.after() is not safe to call from a
        # worker thread, and it raises "main thread is not in main loop" when
        # the app is driven any way other than mainloop(). _tick() already
        # watches for the thread dying, so that is where completion is handled.
        from .training import start_thread
        self.train_thread = start_thread(run, self.exc_q)
        self.status("training started")
        self.nb_select("Monitor")

    def nb_select(self, name):
        """Switch tabs. Keeping the notebook as self.nb beats walking the
        widget tree looking for it."""
        try:
            idx = self._tab_order.index(name)
            self.nb.select(idx)
        except (ValueError, tk.TclError):
            pass

    def _preflight_ok(self, run):
        """Last chance to stop before a multi-gigabyte download. Returns False
        to abort the run."""
        d, m = run.get("data", {}), run.get("model", {})
        if d.get("task") not in ("text",):
            return True
        cap, src = ds_mod.resolve_token_budget(d, m, run.get("train", {}))
        ram_gb = cap * 5 / 1024 ** 3
        try:
            import psutil
            have_gb = psutil.virtual_memory().available / 1024 ** 3
        except Exception:
            have_gb = 0.0

        warn = []
        if src == "explicit" and cap > 50_000_000:
            warn.append(f"max_tokens is {cap:,} — that is about "
                        f"{cap * 5 / 1024 ** 3:.1f} GB just to hold the tokens")
        if ram_gb and have_gb and ram_gb > have_gb * 0.5:
            warn.append(f"needs ~{ram_gb:.1f} GB of RAM but only "
                        f"{have_gb:.1f} GB is free")
        if bool(d.get("cache_text")):
            warn.append(f"disk cache will write roughly "
                        f"{cap * 1.2 / 1024 ** 3:.1f} GB into {d.get('data_dir', 'data')}")
        if warn:
            if not messagebox.askyesno(
                    "That is a lot of data", "\n\n".join(warn) +
                    "\n\nContinue anyway?"):
                return False
        return True

    def on_train_done(self, summary=None):
        self.btn_train.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        snap = state.snapshot()
        if summary and summary.get("ok"):
            self.status(f"done in {summary['seconds']:.1f}s")
        elif snap["status"] == "done":
            self.status(f"finished {snap['global_step']:,} steps in "
                        f"{snap['elapsed']:.0f}s")

    def on_stop(self):
        logging_utils.request_shutdown()
        self.status("stop requested — finishing the current step and saving...")
        self.btn_stop.configure(state="disabled")

    # -- tick ---------------------------------------------------------------

    def _drain_ui_q(self):
        for _ in range(32):
            try:
                fn = self._ui_q.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                self._oops("ui_q")

    def _tick(self):
        try:
            self._drain_ui_q()
            busy = self.train_thread is not None and self.train_thread.is_alive()
            # Idle means idle: no snapshot, no repaint, no work.
            if busy or self._need_final_sync:
                self._tick_training(busy)
        except Exception:
            self._oops("tick")
        finally:
            try:
                self.root.after(200, self._tick)
            except tk.TclError:
                pass      # window already destroyed

    def _tick_training(self, busy):
        snap = state.snapshot()
        status = snap["status"]

        self.m_status.set(status)
        if snap["loss"]:
            self.m_loss.set(f"{snap['loss']:.4f}")
        total = max(1, snap["total_steps"])
        self.m_step.set(f"{snap['global_step']:,} / {snap['total_steps']:,}")
        if snap["sps"]:
            self.m_speed.set(f"{snap['sps']:.1f}/s")
        if snap["mem_measured"]:
            self.m_mem.set(mem_mod.fmt_bytes(snap["mem_measured"]))
        elif snap["mem_estimated"]:
            self.m_mem.set(f"~{mem_mod.fmt_bytes(snap['mem_estimated'])}")
        if snap["n_params"]:
            self.m_params.set(arch_mod.fmt_params(snap["n_params"]))
        self.progress["value"] = 1000.0 * min(1.0, snap["global_step"] / total)
        if snap["error"]:
            self.err_lbl.configure(text=snap["error"][:200])

        # Charts: only while the Monitor tab is showing, at most ~2 Hz, and
        # only when something actually changed. On the final tick, force one
        # last repaint so the end state is what's on screen.
        sig = (snap["global_step"], status, snap["loss"],
               snap["mem_measured"], snap["n_params"])
        if self._current_tab() == "Monitor":
            due = time.monotonic() - self._last_chart_t >= CHART_MIN_INTERVAL
            if (not busy) or (sig != self._chart_sig and due):
                self._paint_charts(snap, force=not busy)
                self._chart_sig = sig
        else:
            self._dirty["monitor"] = True

        self._sync_log(snap["log_lines"])
        self._sync_samples(snap["samples"])

        if busy:
            el = time.time() - self.train_start
            eta = snap.get("eta") or 0
            self.status(f"{status}  ·  {snap['global_step']:,}/{total:,}  ·  "
                        f"{el/60:.1f}m elapsed  ·  ETA {eta/60:.1f}m  ·  "
                        f"lr {(snap['lr'] or 0):.2e}  ·  "
                        f"gnorm {(snap['grad_norm'] or 0):.2f}")
        else:
            if str(self.btn_train["state"]) == "disabled":
                if not self._train_finished:
                    self._train_finished = True
                    try:
                        tb = self.exc_q.get_nowait()
                        self.err_lbl.configure(text=tb.strip().splitlines()[-1][:200])
                    except queue.Empty:
                        pass
                self.on_train_done()
            self._need_final_sync = False

    # -- log ----------------------------------------------------------------

    @staticmethod
    def _level_tag(lvl):
        l = str(lvl).lower()
        if l.startswith("err") or l in ("critical", "fatal"):
            return "error"
        if l.startswith("warn"):
            return "warn"
        return None

    def _wipe_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._log_last_line = None

    def _clear_log(self):
        """Clear the view but remember where we were, so already-shown lines
        do not come straight back on the next tick."""
        self._wipe_log()
        try:
            lines = state.snapshot()["log_lines"]
            self._log_last_line = lines[-1] if lines else None
        except Exception:
            pass

    def _sync_log(self, lines):
        """Append only the lines we have not rendered yet.

        The log buffer in state is a bounded deque. The old version tracked
        progress as a *count* of lines rendered; once the deque filled, its
        length stopped changing, so `lines[count:]` was empty forever and the
        log froze mid-run. Here progress is tracked by remembering the last
        line shown and finding it again, so it keeps flowing no matter how
        far the buffer has scrolled. (The old version also compared
        timestamp strings to detect a reset, which misfires at midnight.)
        """
        if not lines:
            return
        last = self._log_last_line
        start = 0
        if last is not None:
            for i in range(len(lines) - 1, -1, -1):
                if lines[i] is last or lines[i] == last:
                    start = i + 1
                    break
            # not found at all: the whole buffer rolled over, everything is new
        fresh = lines[start:]
        if not fresh:
            return
        self._log_last_line = lines[-1]

        skipped = 0
        if len(fresh) > LOG_BURST_MAX:
            skipped = len(fresh) - LOG_BURST_MAX
            fresh = fresh[-LOG_BURST_MAX:]

        log = self.log_text
        at_bottom = log.yview()[1] >= 0.995
        log.configure(state="normal")
        if not self._log_indent_set:
            # hanging indent: wrapped lines line up under the message, not
            # under the timestamp
            ts = str(fresh[0][0])
            indent = tkfont.Font(font=mono(9)).measure(f"{ts}  ")
            log.tag_configure("wrap", lmargin1=0, lmargin2=indent)
            self._log_indent_set = True
        if skipped:
            log.insert("end", f"… {skipped} lines skipped …\n", ("ts", "wrap"))
        for t, lvl, m in fresh:
            tag = self._level_tag(lvl)
            log.insert("end", f"{t}  ", ("ts", "wrap"))
            log.insert("end", f"{m}\n", (tag, "wrap") if tag else ("wrap",))
        nlines = int(log.index("end-1c").split(".")[0])
        if nlines > LOG_TRIM_AT:
            log.delete("1.0", f"{nlines - LOG_KEEP_LINES}.0")
        log.configure(state="disabled")
        # follow only if asked to AND the reader has not scrolled up
        if self.follow_log.get() and at_bottom:
            log.see("end")

    def _sync_samples(self, samples):
        if not samples:
            return
        latest = samples[-1]
        if self._last_sample == latest["text"]:
            return
        self._last_sample = latest["text"]
        self.sample_text.delete("1.0", "end")
        for s in samples[-4:]:
            self.sample_text.insert("end", "▸ " + s["prompt"], "prompt")
            self.sample_text.insert("end", "\n" + s["text"] + "\n\n", "gen")

    def status(self, msg):
        self.status_lbl.configure(text=msg)

    # -- lifecycle ----------------------------------------------------------

    def on_close(self):
        if self._refresh_job is not None:
            try:
                self.root.after_cancel(self._refresh_job)
            except tk.TclError:
                pass
        if self.train_thread and self.train_thread.is_alive():
            if not messagebox.askokcancel(
                    "Training is running",
                    "Stop training and quit? A checkpoint is saved on the way out."):
                return
            logging_utils.request_shutdown()
            self.status("saving checkpoint and shutting down ...")
            self.btn_stop.configure(state="disabled")
            self._close_deadline = time.monotonic() + 6.0
            self.root.after(150, self._finish_close)
            return
        self.root.destroy()

    def _finish_close(self):
        # Poll instead of blocking: the old loop slept in 200 ms slices on the
        # Tk thread for up to 6 s, freezing the window while it "closed".
        if (self.train_thread and self.train_thread.is_alive()
                and time.monotonic() < self._close_deadline):
            self.root.after(150, self._finish_close)
            return
        self.root.destroy()


def launch(run_cfg=None):
    root = tk.Tk()
    Launcher(root, run_cfg)
    root.mainloop()