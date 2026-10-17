"""
The HuggingFace dataset browser.

A modal window that lists a repo's files WITH SIZES and downloads exactly
what you tick - nothing more. The byte budget is a warning threshold, not
a selector: it never ticks, unticks, or otherwise touches your selection
(the old auto mode did, including past the row cap, which is how phantom
ticks happened). `auto-select` opts back into budget-driven picking.

Subset/split boxes scope both the server-side listing (prefix probes, so a
huge repo only walks the subtree you asked about) and the plan. A `split`
column shows which split each file belongs to, and ticked rows are always
visible no matter what the filter says.

Opened from the Data tab. It owns its own tk widgets and its own worker thread;
the parent Launcher hands it a `_ui_q` queue to post results through, so nothing
here ever touches Tk from off-thread.

`gui` is imported lazily inside `open_browser` rather than at module level:
gui.py imports this module back, so a top-level import here would be a cycle.
Only `ui`/`mono` are needed, and only once the parent exists.

Why a separate window rather than more fields on the Data tab: the file list is
a Treeview with per-row status, it needs real width, and the whole point is
that you look at the sizes before committing. A row of entry boxes on a form
tab cannot make that legible.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import hfhub
from . import config as cfg_mod
from .charts import ACCENT, BG, ERR, MUTED, WARN


# ASCII box glyphs. ttk has no checkbox widget for Treeview rows and ☑/☐ depend
# on a symbol font being present, which is not something to bet a download on.
PICKED = "[x]"
UNTICKED = "[ ]"


def _fmt_gb(x):
    """A budget entry that round-trips: enough digits to be meaningful,
    few enough that typing it is easy."""
    x = float(max(0.0, x or 0.0))
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _fmt_tokens(n):
    n = int(n or 0)
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def open_browser(launcher):
    """Factory used by the Data tab's button. Keeps the import cycle local."""
    from .hfbrowser import DatasetBrowser
    return DatasetBrowser(launcher.root, launcher,
                          dataset_id=str(launcher._get("data.hf_dataset", "")))


def ui(size, *flags):
    """Forward to gui.ui, honouring its _BUMP screen-height tweak."""
    from .gui import ui as _ui
    return _ui(size, *flags)


def mono(size, *flags):
    from .gui import mono as _mono
    return _mono(size, *flags)


class DatasetBrowser:
    """Inspect and download a HuggingFace dataset within a byte budget."""

    def __init__(self, parent, launcher, dataset_id=""):
        self.lw = launcher
        self.ui_q = launcher._ui_q
        self.files = []
        self.meta = {}
        self.plan = []
        self.skipped = []
        self.planned_bytes = 0
        self.downloader = None
        # Manual from the start: nothing is ever ticked unless you tick it.
        # The byte budget is a warning threshold, not a selector - the old
        # auto mode pre-ticked files you never asked for (sometimes past the
        # 400-row display cap, so they were ticked AND unfindable).
        # `auto-select` below opts back into budget-driven selection.
        self.picks = set()
        # set on close so worker-thread callbacks stop touching dead widgets;
        # a download thread outlives the window by however long one file takes
        self._closed = False

        self.win = tk.Toplevel(parent)
        self.win.title("HuggingFace datasets  ·  netforge")
        self.win.configure(bg=BG)
        self.win.transient(parent)
        sw = parent.winfo_screenwidth()
        self.win.geometry(f"{min(sw - 80, 1120)}x740+60+50")
        self.win.minsize(900, 560)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.v_id = tk.StringVar(value=dataset_id or
                                   str(launcher._get("data.hf_dataset", "")))
        self.v_subset = tk.StringVar(value=str(launcher._get("data.hf_subset", "")))
        self.v_split = tk.StringVar(value="")
        self.v_filter = tk.StringVar(value="")
        self.v_ticked_only = tk.BooleanVar(value=False)
        self._scopes_token = 0
        self.v_dest = tk.StringVar(value=self._default_dest())
        self.v_status = tk.StringVar(value="enter a dataset id and press Inspect")
        self.v_summary = tk.StringVar(value="")
        self.v_need = tk.StringVar(value="")
        self.v_largest = tk.BooleanVar(value=False)
        self.bpt = 2.38
        # kept as numbers, not parsed back out of a label - the label gets
        # rewritten on every recalc and parsing it was a loop waiting to happen
        self.need_tokens = 0
        self.need_gb = 0.0
        # the budget starts at whatever the current model wants
        self.v_budget = tk.StringVar(value=_fmt_gb(self._plan_from_launcher()))

        self._build()
        if self.v_id.get().strip():
            self.inspect()

    def _plan_from_launcher(self):
        """The training budget, in GB, straight from the launcher's config.

        This is the whole point of the dialog: "35M model, 20 tok/param" is
        700M tokens, which is N GB of parquet, and the budget box below should
        already be set to N. The user never has to do that arithmetic.

        Returns GB. Also refreshes need_tokens / need_gb / bpt as a side
        effect, because everything downstream needs them.
        """
        try:
            from . import dataset as ds_mod
            self.lw._collect()          # widgets, not last-run's config
            plan = ds_mod.token_plan(cfg_mod._backfill(self.lw.run))
            self.bpt = plan["bpt"]
            self.need_tokens = plan["tokens"]
            self.need_gb = plan["gb"]
            return plan["gb"]
        except Exception:
            self.need_tokens = 0
            self.need_gb = 0.0
            return 0.5

    def _default_dest(self):
        d = str(self.lw._get("data.data_dir", "data")) or "data"
        return os.path.join(d, "hf")

    # -- layout -------------------------------------------------------------

    def _build(self):
        w = self.win
        w.rowconfigure(2, weight=1)
        w.columnconfigure(0, weight=1)

        # --- row 0: the query
        top = tk.Frame(w, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        ttk.Label(top, text="dataset", style="Muted.TLabel").pack(side="left")
        e = ttk.Entry(top, textvariable=self.v_id, width=30)
        e.pack(side="left", padx=6)
        e.bind("<Return>", lambda _e: self.inspect())
        ttk.Label(top, text="subset", style="Muted.TLabel").pack(side="left")
        self.cb_subset = ttk.Combobox(top, textvariable=self.v_subset,
                                      width=16)
        self.cb_subset.pack(side="left", padx=4)
        self.cb_subset.bind("<<ComboboxSelected>>",
                            lambda _e: self._on_subset_picked())
        ttk.Label(top, text="split", style="Muted.TLabel").pack(side="left")
        self.cb_split = ttk.Combobox(top, textvariable=self.v_split, width=12)
        self.cb_split.pack(side="left", padx=4)
        ttk.Button(top, text="Inspect", command=self.inspect).pack(side="left", padx=2)
        ttk.Button(top, text="Use", width=6, command=self.on_use).pack(side="left", padx=2)

        # --- row 1: the budget controls
        ctl = tk.Frame(w, bg=BG)
        ctl.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 8))
        ctl.columnconfigure(5, weight=1)

        ttk.Label(ctl, text="budget GB", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        bb = ttk.Entry(ctl, textvariable=self.v_budget, width=8)
        bb.grid(row=0, column=1, sticky="w")
        for v in (self.v_budget, self.v_filter):
            v.trace_add("write", lambda *_: self.recalc())
        ttk.Label(ctl, text="filter", style="Muted.TLabel").grid(
            row=0, column=2, sticky="w", padx=(14, 4))
        fb = ttk.Entry(ctl, textvariable=self.v_filter, width=16)
        fb.grid(row=0, column=3, sticky="w")
        fb.bind("<Return>", lambda _e: None)
        ttk.Checkbutton(ctl, text="largest first", variable=self.v_largest).grid(
            row=0, column=4, sticky="w", padx=4)
        ttk.Checkbutton(ctl, text="ticked only", variable=self.v_ticked_only).grid(
            row=0, column=5, sticky="w", padx=4)
        self.v_largest.trace_add("write", lambda *_: self.recalc())
        self.v_ticked_only.trace_add("write", lambda *_: self.recalc())
        ttk.Button(ctl, text="use it", style="Small.TButton",
                    command=self.use_training_budget).grid(row=0, column=6, padx=2)
        self.btn_auto = ttk.Button(ctl, text="auto-select", style="Small.TButton",
                                    command=self.use_auto_selection)
        self.btn_auto.grid(row=0, column=7, padx=2)
        # Readouts live on their own row with wrapping: cramming them next
        # to the controls is what pushed the row past the window edge and
        # squeezed the summary to zero width.
        ttk.Label(ctl, textvariable=self.v_summary, style="Muted.TLabel",
                  wraplength=680, justify="left").grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))
        ttk.Label(ctl, text="need", style="Muted.TLabel").grid(
            row=1, column=6, sticky="e", padx=(4, 4), pady=(4, 0))
        ttk.Label(ctl, textvariable=self.v_need, style="Muted.TLabel",
                  wraplength=360, justify="left").grid(
            row=1, column=7, columnspan=2, sticky="w", pady=(4, 0))

        # --- row 2: the file table
        mid = tk.Frame(w, bg=BG)
        mid.grid(row=2, column=0, sticky="nsew", padx=14)
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        cols = ("pick", "size", "path", "split", "tokens", "note")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings",
                                 selectmode="extended")
        # widths live on column(), not heading() - heading() only takes text,
        # anchor and command.
        self.tree.heading("pick", text="")
        self.tree.heading("size", text="size", anchor="e")
        self.tree.heading("path", text="file")
        self.tree.heading("split", text="split")
        self.tree.heading("tokens", text="≈ tokens", anchor="e")
        self.tree.heading("note", text="note")
        self.tree.column("pick", width=52, anchor="center", stretch=False)
        self.tree.column("size", width=90, anchor="e", stretch=False)
        self.tree.column("path", width=260, stretch=True)
        self.tree.column("split", width=80, anchor="center", stretch=False)
        self.tree.column("tokens", width=90, anchor="e", stretch=False)
        self.tree.column("note", width=200, stretch=False)
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<space>", lambda _e: self._toggle_selected())
        self.tree.bind("<Button-3>", lambda e: self._toggle_selected())
        sy = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        sx = ttk.Scrollbar(mid, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        self.tree.tag_configure("skip", foreground=MUTED,
                                background="#161c28")
        self.tree.tag_configure("take", foreground=ACCENT,
                                background="#1b2a24")
        self.tree.tag_configure("over", foreground=WARN,
                                background="#161c28")
        self.tree.tag_configure("big", foreground=ERR,
                                background="#241a1c")

        # --- row 3: destination + actions
        bot = tk.Frame(w, bg=BG)
        bot.grid(row=3, column=0, sticky="ew", padx=14, pady=(10, 6))
        bot.columnconfigure(1, weight=1)
        ttk.Label(bot, text="save to", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Entry(bot, textvariable=self.v_dest).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(bot, text="browse", command=self.on_browse).grid(
            row=0, column=2, padx=2)
        self.btn_dl = ttk.Button(bot, text="Download", style="Go.TButton",
                                 command=self.start)
        self.btn_dl.grid(row=0, column=3, padx=(8, 2))
        self.btn_cancel = ttk.Button(bot, text="Cancel", command=self.cancel,
                                     state="disabled")
        self.btn_cancel.grid(row=0, column=4, padx=2)

        self.bar = ttk.Progressbar(bot, mode="determinate", maximum=1000)
        self.bar.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(8, 2))
        ttk.Label(bot, textvariable=self.v_status, style="Muted.TLabel",
                  wraplength=1000, justify="left").grid(
            row=2, column=0, columnspan=5, sticky="w")

    # -- inspection ---------------------------------------------------------

    def inspect(self):
        ds = self.v_id.get().strip()
        if not ds:
            self.v_status.set("enter a dataset id, e.g. roneneldan/TinyStories")
            return
        sub, sp = self.v_subset.get().strip(), self.v_split.get().strip()
        scope = f" [{sub}/{sp}]" if (sub or sp) else ""
        self.v_status.set(f"inspecting {ds}{scope} ...")
        self.btn_dl.configure(state="disabled")

        def work_files():
            files, meta, err = hfhub.list_dataset_files(
                ds, subset=sub, split=sp)
            self.ui_q.put(lambda: self._show(files, meta, err, ds))

        def work_scopes():
            subsets, splits = self._fetch_scopes(ds, sub)
            tok = self._scopes_token
            self.ui_q.put(lambda: self._apply_scopes(subsets, splits, tok))

        self._scopes_token += 1
        threading.Thread(target=work_files, daemon=True).start()
        threading.Thread(target=work_scopes, daemon=True).start()

    def _fetch_scopes(self, ds, subset):
        """Available subsets and splits for the dropdowns.

        Hub metadata first (no file walk); the already-listed paths as
        fallback, so the boxes still fill in when metadata is missing.
        Never raises - worst case the boxes stay editable text fields.
        """
        subsets, splits = [], []
        try:
            from datasets import (get_dataset_config_names,
                                  get_dataset_split_names)
            try:
                subsets = list(get_dataset_config_names(ds) or [])
            except Exception:
                subsets = []
            try:
                splits = list(get_dataset_split_names(
                    ds, config_name=subset or None) or [])
            except Exception:
                try:
                    splits = list(get_dataset_split_names(ds) or [])
                except Exception:
                    splits = []
        except ImportError:
            pass
        if not subsets:
            subsets = self._configs_from_paths()
        if not splits and self.files:
            seen = []
            for f in self.files:
                s = hfhub.split_of(f["path"])
                if s not in seen:
                    seen.append(s)
            splits = seen
        return subsets[:50], splits

    def _configs_from_paths(self):
        """Top-level folders holding data files - usually the configs."""
        seen = []
        for f in self.files or []:
            p = f["path"]
            if "/" in p:
                top = p.split("/")[0]
                if (top not in ("data", ".git", ".cache")
                        and hfhub.is_data_file(p)
                        and top not in seen):
                    seen.append(top)
            if len(seen) >= 50:
                break
        return seen

    def _apply_scopes(self, subsets, splits, tok):
        """Fill the dropdown values. Never overwrites what you typed."""
        if not self._alive() or tok != self._scopes_token:
            return
        if subsets:
            self.cb_subset.configure(values=subsets)
        if splits:
            self.cb_split.configure(values=splits)

    def _on_subset_picked(self):
        """A subset choice refreshes the split list for that subset."""
        ds = self.v_id.get().strip()
        sub = self.v_subset.get().strip()
        if not ds or not self._alive():
            return
        self._scopes_token += 1
        tok = self._scopes_token
        self.v_status.set(f"loading splits for {sub or 'default'} ...")

        def work():
            _subs, splits = self._fetch_scopes(ds, sub)
            self.ui_q.put(lambda: self._apply_scopes([], splits, tok))

        threading.Thread(target=work, daemon=True).start()

    def _alive(self):
        """False once the window is gone. Every UI callback checks this: the
        download thread keeps posting after the window closes, and touching a
        destroyed widget raises TclError from inside the parent's tick."""
        if self._closed:
            return False
        try:
            return bool(self.win.winfo_exists())
        except (tk.TclError, KeyError):
            self._closed = True
            return False

    def _show(self, files, meta, err, ds):
        if not self._alive():
            return
        self.files, self.meta = files, meta
        # Fresh listing, fresh (empty) selection. Old ticks pointed at a
        # different repo's paths; carrying them over is how phantom ticks
        # happen. (Auto mode, picks=None, is only ever entered deliberately
        # via auto-select, so it survives a re-inspect.)
        if self.picks is not None:
            self.picks = set()
        if err:
            self.v_status.set(err)
            self.v_summary.set("")
            self.tree.delete(*self.tree.get_children())
            return
        if not files:
            self.v_status.set(f"{ds}: no files found")
            return

        total = hfhub.total_size(files)
        bits = [f"{len(files)} files", hfhub.fmt_size(total) + " total"]
        if meta.get("scoped_to"):
            bits.append(f"scoped to {meta['scoped_to']}/")
        if meta.get("partial"):
            bits.append(f"partial ({meta.get('listed', len(files))} of "
                        f"{meta.get('total_files', '?')})")
        if meta.get("downloads"):
            bits.append(f"{meta['downloads']:,} downloads")
        if meta.get("likes"):
            bits.append(f"{meta['likes']} likes")
        if meta.get("modified"):
            bits.append(f"updated {meta['modified']}")
        if meta.get("gated"):
            bits.append("GATED")
        self.v_status.set(f"{ds}  ·  " + "  ·  ".join(bits))
        self.recalc()
        # The scopes worker can finish before the listing does, leaving the
        # subset box empty. Top-level data folders are usually the configs,
        # so derive them from the paths now that files exist.
        try:
            if not self.cb_subset.cget("values"):
                cfgs = self._configs_from_paths()
                if cfgs:
                    self.cb_subset.configure(values=cfgs)
        except tk.TclError:
            pass

    def _budget_bytes(self):
        try:
            g = float(self.v_budget.get())
        except (TypeError, ValueError):
            return 0
        return int(max(0.0, g) * 1024 ** 3)

    def _on_click(self, event):
        """Clicking the pick cell (or anywhere on the row) toggles it.

        Returning None lets the normal selection happen too, so shift-click
        still extends the selection and the space bar still works on a range.
        """
        if self.tree.identify_region(event.x, event.y) != "cell":
            return None
        row = self.tree.identify_row(event.y)
        if row:
            self._toggle_path(self.tree.item(row, "values")[2])
            return "break"        # toggling is the whole point of the click
        return None

    def _toggle_selected(self):
        for row in self.tree.selection():
            vals = self.tree.item(row, "values")
            if vals:
                self._toggle_path(vals[2])

    def _toggle_path(self, path):
        if not path:
            return
        if self.picks is None:
            # first manual click freezes whatever the budget had chosen
            self.picks = {f["path"] for f in self.plan}
        if path not in self._eligible_paths():
            return  # stale row or typo: never tick what isn't listed
        if path in self.picks:
            self.picks.discard(path)
        else:
            self.picks.add(path)
        self.recalc()

    def use_auto_selection(self):
        """Hand control back to the byte budget."""
        self.picks = None
        self.recalc()
        self.v_status.set("selection driven by the byte budget again")

    def _scope_terms(self):
        """Subset + split + filter words: everything that narrows eligibility."""
        terms = [w.strip() for w in self.v_filter.get().split(",") if w.strip()]
        for v in (self.v_subset.get().strip(), self.v_split.get().strip()):
            if v and v not in terms:
                terms.append(v)
        return terms

    def _eligible_paths(self):
        """Every downloadable data file, regardless of the filter text.

        The plan is built from this, so typing in the filter box can hide
        rows but never silently add or drop files from the download.
        """
        data, _meta = hfhub.classify(self.files)
        return {f["path"]: f for f in data}

    def _matches_scope(self, path):
        terms = self._scope_terms()
        return (not terms) or any(t in path for t in terms)

    def recalc(self):
        """Re-plan against the current budget and repaint the table."""
        if not self.files or not self._alive():
            return
        # a repo can hold thousands of shards; rebuilding that many rows on
        # every keystroke is the kind of thing that makes an app feel broken
        MAX_ROWS = 400
        cap = self._budget_bytes()
        by_path = self._eligible_paths()

        if self.picks is None:
            # auto (explicit opt-in): budget + scope decide
            self.plan, self.skipped, self.planned_bytes = hfhub.plan_download(
                self.files, cap, want=self._scope_terms(),
                largest_first=bool(self.v_largest.get()))
            chosen = {f["path"] for f in self.plan}
            auto = True
        else:
            # manual (the default): exactly what is ticked, nothing more.
            # Built from ALL eligible files, not the filtered view, so a
            # filter can never add or drop downloads behind your back.
            ordered = sorted(by_path.values(),
                             key=lambda f: -f["size"],
                             reverse=bool(self.v_largest.get()))
            self.plan = [f for f in ordered if f["path"] in self.picks]
            chosen = set(self.picks)
            self.skipped = [dict(f, reason="not ticked") for f in ordered
                            if f["path"] not in chosen]
            self.planned_bytes = sum(f["size"] for f in self.plan)
            auto = False

        ok, why = hfhub.preflight(self.plan, self.v_dest.get())
        over = self.planned_bytes > cap
        if not self.plan:
            why = ("nothing ticked - tick some files, or press auto-select"
                   if not auto else why)
        held = sum(1 for f in self.plan
                   if hfhub.split_of(f["path"]) == "validation")
        self.v_summary.set(
            f"{'auto' if auto else 'ticked'} {len(self.plan)} / {len(self.files)}"
            f"  ·  {hfhub.fmt_size(self.planned_bytes)}"
            f"  ·  {len(self.skipped)} not ticked"
            + (f"  ·  holds out {held} validation" if held else "")
            + ("  ·  OVER your budget" if over and self.plan else "")
            + (f"  ·  {why}" if not ok or not self.plan else ""))

        taken = chosen
        reasons = {f["path"]: f["reason"] for f in self.skipped}
        self._plan_from_launcher()          # refresh need_tokens / bpt
        bpt = max(1e-6, float(self.bpt))
        ticked_only = bool(self.v_ticked_only.get())
        allrows = sorted(self.plan + [f for f in self.skipped
                                      if f["path"] in by_path],
                         key=lambda f: -f["size"])
        self.tree.delete(*self.tree.get_children())
        shown = 0
        for f in allrows:
            p = f["path"]
            if p not in by_path:            # not downloadable at all
                continue
            if ticked_only and p not in taken:
                continue
            if p not in taken and not self._matches_scope(p):
                # filtered out of view - UNLESS ticked: a ticked file must
                # always be findable, or you get phantom ticks again.
                continue
            tok = int(f["size"] / bpt)
            if p in taken:
                mark, tag, note = PICKED, "take", ("ticked" if not auto
                                                    else "within budget")
            elif reasons.get(p, "").startswith("needs"):
                mark, tag, note = UNTICKED, "over", reasons[p]
            else:
                mark, tag, note = UNTICKED, "skip", "not ticked"
            if p in taken and not self._matches_scope(p):
                note += " (outside filter)"
            self.tree.insert("", "end", values=(
                mark, hfhub.fmt_size(f["size"]), p,
                hfhub.split_of(p), _fmt_tokens(tok), note),
                tags=(tag,))
            shown += 1
            if shown >= MAX_ROWS:
                self.tree.insert("", "end", values=(
                    "", "", f"… {len(allrows) - shown} more file(s) "
                             f"(narrow the filter or ticked-only)", "",
                    "", ""),
                    tags=("skip",))
                break
        if not ok:
            self.v_status.set(("⚠ " if self.plan else "") + why)

        # how far short of the model's target this selection lands
        got = int(self.planned_bytes / bpt)
        if self.need_tokens > 0:
            pct = 100.0 * got / self.need_tokens
            if got >= self.need_tokens:
                self.v_need.set(
                    f"model needs {self.need_tokens:,} tok "
                    f"({self.need_gb:.2f} GB)  ·  this covers it")
            else:
                self.v_need.set(
                    f"model needs {self.need_tokens:,} tok "
                    f"({self.need_gb:.2f} GB)  ·  this is {pct:.0f}% "
                    f"— raise the budget")
        else:
            self.v_need.set("")
        self.btn_dl.configure(
            state="normal" if (ok and self.plan and
                               not (self.downloader and not self.downloader.done))
            else "disabled")

    def use_training_budget(self):
        """Set the budget box to exactly what the current model wants.

        Only the number: the budget is a warning threshold, never a
        selector, so the ticked set is left exactly as it is.
        """
        gb = self._plan_from_launcher()
        if not gb:
            self.v_status.set("could not work out a budget from the model config")
            return
        self.v_budget.set(_fmt_gb(gb))
        self.recalc()
        self.v_status.set(f"budget set to {gb:.2f} GB for the current model")

    # -- download -----------------------------------------------------------

    def on_browse(self):
        d = filedialog.askdirectory()
        if d:
            self.v_dest.set(d)
            self.recalc()

    def start(self):
        if not self.plan:
            return
        ok, why = hfhub.preflight(self.plan, self.v_dest.get())
        cap = self._budget_bytes()
        over = self.planned_bytes > cap
        if not ok or over:
            msgs = []
            if not ok:
                msgs.append(why)
            if over:
                msgs.append(f"this selection is {hfhub.fmt_size(self.planned_bytes)} "
                            f"but the budget is {hfhub.fmt_size(cap)}")
            if not messagebox.askyesno(
                    "Bigger than planned", "\n\n".join(msgs) +
                    "\n\nDownload it anyway?"):
                return
        ds = self.v_id.get().strip()
        self.downloader = hfhub.Downloader(ds, self.plan, self.v_dest.get())
        self.btn_dl.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.bar["value"] = 0
        self.downloader.start(self._progress)

    def cancel(self):
        if self.downloader and not self.downloader.done:
            self.downloader.cancel()
            self.v_status.set("cancelling after the current file ...")

    def _progress(self, done_files, done_bytes, total_files, msg, dest, *_):
        """Called from the worker thread. Posts to the Tk thread."""
        self.ui_q.put(lambda: self._apply_progress(
            done_files, done_bytes, total_files, msg, dest))

    def _apply_progress(self, done_files, done_bytes, total_files, msg, dest):
        if not self._alive():
            return
        if total_files:
            self.bar["value"] = 1000.0 * done_files / total_files
        self.v_status.set(msg)
        if self.downloader and self.downloader.done:
            self._finish(dest)

    def _finish(self, dest):
        if not self._alive():
            return
        d = self.downloader
        self.btn_cancel.configure(state="disabled")
        self.btn_dl.configure(state="normal")
        if d.failed:
            self.btn_dl.configure(state="normal")
        local = hfhub.local_parquet_files(dest)
        nbytes = sum(f["size"] for f in local)
        extra = ""
        if d.failed:
            extra = (f"\n\n{len(d.failed)} file(s) failed:\n" +
                     "\n".join(f"  {p}: {e[:70]}" for p, e in d.failed[:4]))
        extra += (f"\n\n{dest} now holds {len(local)} data file(s), "
                  f"{hfhub.fmt_size(nbytes)}")
        if d.cancel_evt.is_set():
            extra += "\n(cancelled - the rest was not fetched)"
        self.v_status.set(
            f"{'done' if not d.failed else 'finished with errors'}  ·  "
            f"{hfhub.fmt_size(d.bytes)}  ·  {dest}")
        if messagebox.askyesno(
                "Download finished",
                f"{'Finished' if not d.failed else 'Finished with errors'}: "
                f"{len(d.plan) - len(d.failed)}/{len(d.plan)} files, "
                f"{hfhub.fmt_size(d.bytes)} into {dest}.{extra}\n\n"
                f"Use this folder as the training corpus now?"):
            self.on_use()

    def on_use(self):
        """Point the trainer at what we just downloaded."""
        dest = self.v_dest.get()
        try:
            self.lw._suspend = True
            self.lw.widgets["data.data_dir"].set(dest)
            self.lw.widgets["data.hf_dataset"].set("")
            self.lw.widgets["data.task"].set("text")
            self.lw.widgets["data.cache_text"].set(True)
            self.lw._suspend = False
            self.lw._collect()
            self.lw.refresh_preview()
            held = [f for f in hfhub.local_parquet_files(dest)
                    if hfhub.split_of(f["path"]) == "validation"]
            extra = (f" (+{len(held)} validation file(s) held out for eval)"
                     if held else "")
            self.lw.status(f"corpus set to {dest}{extra}")
        finally:
            self.lw._suspend = False
        self.on_close(force=True)

    def on_close(self, force=False):
        self._closed = True
        if self.downloader and not self.downloader.done and not force:
            if not messagebox.askokcancel(
                    "Download running",
                    "Close the window and cancel the download?"):
                self._closed = False
                return
            self.downloader.cancel()
        try:
            self.win.destroy()
        except tk.TclError:
            pass