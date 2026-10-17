"""
HuggingFace dataset inspection and budgeted download.

Covers:
  - listing a dataset repo's files WITH SIZES, before anything downloads
  - choosing which files fit a byte budget (and saying what was left out)
  - downloading them to a directory you chose, with progress and cancel
  - checking the download will actually fit on the disk first

The design rule that everything else follows: **never download a file you did
not intend to.** TinyStories is 2.7 GB on the Hub, 1.9 GB of which is a single
`TinyStories-train.txt`. A loader that says "downloading dataset..." and then
writes two gigabytes is indistinguishable from a bug, so every path here goes
through `list_dataset_files` and `plan_download` first and prints the numbers.

Two rules in `plan_download` that are not obvious:

  * **A file is all-or-nothing.** A half-downloaded parquet or a truncated
    .jsonl is not a smaller dataset, it is a corrupt one. So a file that does
    not fit the remaining budget is *skipped*, never truncated, and the plan
    says so.

  * **Smallest first, not largest first.** Filling the budget with the biggest
    files looks efficient and is not: it buys you one shard and a dataset you
    cannot shuffle. Smallest-first buys a partial-but-complete picture of the
    whole dataset, which is what you want at 200 MB out of 2.7 GB.
"""

import os
import shutil
import threading
import time

from .logging_utils import append_debug

# Files that can hold data worth reading. Note the .gz variants: C4 and most
# older corpora ship `*.jsonl.gz`, and missing them made the planner offer a
# 900 GB repo with nothing selectable under any sane budget.
DATA_EXTS = (".parquet", ".jsonl", ".json", ".csv", ".tsv", ".txt", ".arrow",
             ".bin", ".npy", ".jsonl.gz", ".json.gz", ".txt.gz", ".csv.gz",
             ".tsv.gz")


def strip_gz(path):
    """The real extension, with any .gz peeled off first."""
    p = str(path).lower()
    if p.endswith(".gz"):
        p = p[:-3]
    return os.path.splitext(p)[1]


def is_data_file(path):
    return strip_gz(path) in DATA_EXTS


# Path-segment vocabulary for "which split is this file in". Whole-token
# matches only: a substring test would file `latest/` under test and
# `pretraining/` under train.
_SPLIT_TOKENS = {
    "train": "train", "training": "train",
    "validation": "validation", "valid": "validation", "val": "validation",
    "eval": "validation", "evaluate": "validation",
    "evaluation": "validation", "dev": "validation",
    "development": "validation",
    "test": "test", "testing": "test",
    "holdout": "test", "heldout": "test",
}

# Canonical order: validation first, so a mixed path like
# train/validation-mix still counts as held out rather than training fuel.
_SPLIT_ORDER = ("validation", "test", "train")


def split_of(path):
    """Which split a repo path belongs to: train | validation | test.

    Unmarked files (data/train-00000.parquet has its marker; a lone
    corpus.txt has none) default to train, which preserves the old
    everything-trains behaviour. Test files are deliberately NOT treated
    as held-out eval here: burning the test set as a validation signal is
    a choice, and silently making it is worse than asking.
    """
    import re
    toks = [t for t in re.split(r"[/\\_.:\- ]+", str(path).lower()) if t]
    kinds = {_SPLIT_TOKENS[t] for t in toks if t in _SPLIT_TOKENS}
    for want in _SPLIT_ORDER:
        if want in kinds:
            return want
    return "train"


# Files that are repo furniture, not corpus. Downloaded only on request.
META_EXTS = (".py", ".md", ".rst", ".yaml", ".yml", ".toml", ".cfg", ".ini",
             ".gitattributes", ".gitignore", ".png", ".jpg", ".svg", ".lock")

MANIFEST_NAMES = ("dataset_infos.json", "README.md", ".gitattributes",
                  "state.json")


def fmt_size(n):
    n = float(n or 0)
    for unit, div in (("TB", 1024 ** 4), ("GB", 1024 ** 3),
                      ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{int(n)} B"


def human_gb(n_bytes):
    return f"{n_bytes / 1024 ** 3:.2f} GB"


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def _scope_prefixes(subset, split):
    """Candidate repo subtrees for a subset/split pair, most specific first.

    Hub layouts vary (`data/<subset>/<split>`, `<subset>/<split>`,
    `data/<split>`, ...), so instead of guessing one layout and being
    wrong, every plausible prefix is probed and the first that resolves
    wins. Each probe is a single cheap request, not a walk.
    """
    subset, split = (subset or "").strip(), (split or "").strip()
    cands = []
    if subset and split:
        cands += [f"{subset}/{split}", f"data/{subset}/{split}"]
    if split:
        cands += [f"data/{split}", split]
    if subset:
        cands += [f"data/{subset}", subset]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _probe_prefix(api, dataset_id, repo_type, prefixes):
    """First candidate prefix that actually exists in the repo, else ''."""
    for p in prefixes:
        try:
            kw = {"path_in_repo": p, "recursive": False}
            probe = api.list_repo_tree(dataset_id, repo_type=repo_type, **kw)
            for _e in probe:
                return p  # anything at all under it counts
        except TypeError:
            return ""  # hub too old for scoping; caller walks everything
        except Exception as exc:
            append_debug(f"prefix probe {p!r}: {type(exc).__name__}")
    return ""

def list_dataset_files(dataset_id, repo_type="dataset", timeout=30,
                       max_entries=4000, budget_s=45.0, subset="", split=""):
    """Every file in a repo, with sizes.

    Returns (files, info, error). Never raises: the GUI shows the error string.

    Two sources, because neither alone is sufficient:

    * `dataset_info().siblings` - one request, and on most repos it carries the
      file sizes. For C4 it returns 69,221 entries in ~1s with *no sizes at
      all*, which is why it cannot be the only source.
    * `list_repo_tree(expand=True)` - has sizes, but paginates. On a 69k-file
      repo walking all of it takes minutes and can drop connections, so it is
      capped by entry count AND wall-clock. When the cap bites we say so
      (`info["partial"]`) rather than quietly showing you 4,000 of 69,221 and
      letting the budget math be wrong.

    `subset`/`split` scope the slow tree-walk to a subtree: a handful of
    cheap prefix probes (`data/<subset>/<split>`, `<subset>/<split>`,
    `data/<split>`, ...) runs first, and the first prefix that resolves is
    the only thing walked. Repos whose siblings already carry sizes don't
    need the walk at all, so the subset/split there act purely as
    client-side filters downstream.

    Smallest-first everywhere: that is what the planner wants anyway.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return [], {}, ("huggingface_hub is not installed "
                        "(pip install huggingface_hub)")

    if not dataset_id or "/" not in str(dataset_id):
        return [], {}, "that does not look like a dataset id (need owner/name)"

    api = HfApi()
    try:
        info = api.dataset_info(dataset_id, timeout=timeout)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "gated" in msg.lower():
            return [], {}, ("this dataset is gated - accept the licence on "
                            "the Hub page and log in with `huggingface-cli "
                            "login` first")
        return [], {}, f"{type(exc).__name__}: {msg}"[:200]

    meta = {
        "id": info.id,
        "downloads": getattr(info, "downloads", 0),
        "likes": getattr(info, "likes", 0),
        "gated": bool(getattr(info, "gated", False) or
                      getattr(info, "private", False)),
        "tags": list(getattr(info, "tags", []) or [])[:8],
        "total_files": 0,
        "listed": 0,
        "partial": False,
        "source": "",
        "modified": str(getattr(info, "lastModified", "") or "")[:10],
    }

    try:
        from huggingface_hub import RepoFolder
    except ImportError:
        RepoFolder = ()

    files = []
    try:
        for s in (getattr(info, "siblings", None) or []):
            if RepoFolder and isinstance(s, RepoFolder):
                continue
            size = getattr(s, "size", None)
            if size is None:
                lfs = getattr(s, "lfs", None)
                size = lfs.get("size", 0) if isinstance(lfs, dict) else 0
            name = getattr(s, "rfilename", "")
            if name:
                files.append({"path": name, "size": int(size or 0)})
        meta["total_files"] = len(files)
    except Exception as exc:
        append_debug(f"siblings read failed: {exc}")

    if not any(f["size"] for f in files):
        # sizes are missing (common on very large repos) - walk the tree
        files = []
        meta["source"] = "list_repo_tree"
        prefixes = _scope_prefixes(subset, split)
        scope = _probe_prefix(api, dataset_id, repo_type, prefixes)
        if scope:
            meta["scoped_to"] = scope
        t0 = time.time()
        try:
            # Argument binding raises TypeError at call time on a hub too
            # old for path_in_repo, so the walk below stays lazy either way.
            kw = {"path_in_repo": scope} if scope else {}
            try:
                iterator = api.list_repo_tree(
                    dataset_id, repo_type=repo_type,
                    recursive=True, expand=True, **kw)
            except TypeError:
                append_debug("list_repo_tree has no path_in_repo; full walk")
                iterator = api.list_repo_tree(
                    dataset_id, repo_type=repo_type,
                    recursive=True, expand=True)
            for entry in iterator:
                name = getattr(entry, "path", None)
                if not name or name.endswith("/"):
                    continue
                if RepoFolder and isinstance(entry, RepoFolder):
                    continue
                files.append({"path": name,
                              "size": int(getattr(entry, "size", 0) or 0)})
                meta["total_files"] += 1
                if len(files) >= max_entries:
                    meta["partial"] = True
                    break
                if time.time() - t0 > budget_s:
                    meta["partial"] = True
                    break
        except Exception as exc:
            append_debug(f"list_repo_tree stopped: {type(exc).__name__}: {exc}")
            if not files:
                return [], meta, f"could not list files: {type(exc).__name__}: {exc}"[:160]
            meta["partial"] = True
        meta["listed"] = len(files)
    else:
        meta["source"] = "siblings"
        meta["listed"] = len(files)

    files = [f for f in files if f["path"]]
    files.sort(key=lambda f: f["size"])
    return files, meta, ""


def classify(files):
    """Split a file list into data / meta / manifest."""
    data, meta = [], []
    for f in files:
        name = os.path.basename(f["path"]).lower()
        ext = strip_gz(name)
        if name in MANIFEST_NAMES or name.startswith("."):
            meta.append(f)
        elif ext in DATA_EXTS:
            data.append(f)
        elif ext in META_EXTS:
            meta.append(f)
        else:
            meta.append(f)
    return data, meta


def total_size(files):
    return sum(f["size"] for f in files)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_download(files, max_bytes, want=("",), include_meta=False,
                  largest_first=False):
    """Choose files that fit `max_bytes`. Returns (plan, skipped, total).

    `want` is a list of substrings; only files whose path contains one of them
    are eligible (so you can target `data/train-`). An empty string matches
    everything.

    Files are taken smallest-first by default so the budget buys coverage of
    the whole dataset rather than one shard. Anything skipped is reported with
    its size, because "why did it only grab 200 MB of 2.7 GB" is the question
    this exists to answer.
    """
    data, meta = classify(files)
    if include_meta:
        data = data + meta

    needle = [w for w in (want or []) if w]
    if needle:
        data = [f for f in data if any(w in f["path"] for w in needle)]
        meta = [f for f in meta if any(w in f["path"] for w in needle)]

    ordered = sorted(data, key=lambda f: f["size"], reverse=largest_first)

    plan, skipped, used = [], [], 0
    for f in ordered:
        size = f["size"]
        if used + size <= max_bytes:
            plan.append(dict(f, reason="fits"))
            used += size
        else:
            # all-or-nothing: a truncated parquet is corrupt, not smaller
            skipped.append(dict(f, reason=(
                f"needs {fmt_size(size)}, only {fmt_size(max_bytes - used)} left")))
    return plan, skipped, used


def disk_free(path):
    """Free bytes on the volume holding `path`, creating it if need be."""
    try:
        p = os.path.abspath(path)
        while p and not os.path.isdir(p):
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return shutil.disk_usage(p).free if p else shutil.disk_usage(".").free
    except Exception:
        return 0


def preflight(plan, dest, headroom=0.08):
    """Refuse a download that will not fit. Returns (ok, message).

    The destination is actually created here, not just measured. A path on a
    drive that does not exist reports "unknown free space", which without this
    check would sail through the size test and then fail three minutes into a
    download.
    """
    if not plan:
        return False, "nothing to download under that limit"
    dest = os.path.abspath(dest)
    try:
        os.makedirs(dest, exist_ok=True)
    except Exception as exc:
        return False, f"cannot write to {dest}: {type(exc).__name__}: {exc}"
    if not os.access(dest, os.W_OK):
        return False, f"{dest} is not writable"

    need = sum(f["size"] for f in plan)
    free = disk_free(dest)
    if free and need > free * (1 - headroom):
        return False, (f"needs {fmt_size(need)} but only {fmt_size(free)} is "
                       f"free on that drive")
    return True, f"{fmt_size(need)} into {dest} ({fmt_size(free)} free)"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class Downloader:
    """Runs a download on a worker thread, reporting through a callback.

    cancel is checked between files. A file already in flight is not aborted
    part-way (that would leave a truncated file on disk that the planner
    believed to be whole), so cancelling takes effect at the next file
    boundary.
    """

    def __init__(self, dataset_id, plan, dest, repo_type="dataset"):
        self.dataset_id = dataset_id
        self.plan = list(plan)
        self.dest = dest
        self.repo_type = repo_type
        self.cancel_evt = threading.Event()
        self.done = False
        self.failed = []
        self.paths = []
        self.bytes = 0
        self.started = 0.0
        self.thread = None

    def cancel(self):
        self.cancel_evt.set()

    def start(self, progress_cb):
        self.thread = threading.Thread(target=self._run, args=(progress_cb,),
                                       daemon=True)
        self.thread.start()
        return self.thread

    def _run(self, cb):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            cb(0, 0, 0, "huggingface_hub is not installed", self.dest)
            self.done = True
            return

        self.started = time.time()
        n = len(self.plan)
        os.makedirs(self.dest, exist_ok=True)

        for i, f in enumerate(self.plan):
            if self.cancel_evt.is_set():
                cb(i, self.bytes, n, "cancelled", self.dest)
                break
            cb(i, self.bytes, n, f"downloading {os.path.basename(f['path'])}",
               self.dest, f)
            try:
                p = hf_hub_download(repo_id=self.dataset_id,
                                    filename=f["path"],
                                    repo_type=self.repo_type,
                                    local_dir=self.dest)
                self.paths.append(p)
                self.bytes += f["size"]
                cb(i + 1, self.bytes, n,
                   f"got {os.path.basename(f['path'])} ({fmt_size(f['size'])})",
                   self.dest)
            except Exception as exc:
                self.failed.append((f["path"], f"{type(exc).__name__}: {exc}"))
                cb(i + 1, self.bytes, n,
                   f"FAILED {os.path.basename(f['path'])}: {exc}"[:110],
                   self.dest)

        self.done = True
        elapsed = max(1e-6, time.time() - self.started)
        rate = self.bytes / elapsed
        tail = (f"{len(self.plan) - len(self.failed)}/{n} files, "
                f"{fmt_size(self.bytes)} at {fmt_size(rate)}/s"
                + (", cancelled" if self.cancel_evt.is_set() else ""))
        cb(n, self.bytes, n, tail, self.dest)


# ---------------------------------------------------------------------------
# Using what we downloaded
# ---------------------------------------------------------------------------

def local_parquet_files(root):
    """Data files sitting in a downloaded tree, smallest first."""
    out = []
    for base, _dirs, names in os.walk(root):
        for n in sorted(names):
            if is_data_file(n):
                p = os.path.join(base, n)
                try:
                    out.append({"path": p, "size": os.path.getsize(p)})
                except OSError:
                    pass
    out.sort(key=lambda f: f["size"])
    return out


def estimate_rows(path, sample_rows=2048):
    """Row count for a data file, read from the footer when possible.

    Exact for parquet (the footer carries it, and reading it does not touch
    the data), sampled for csv/jsonl. Returns None when the answer would cost
    more than it is worth.
    """
    ext = strip_gz(path)
    try:
        if ext == ".parquet":
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        if ext in (".csv", ".tsv"):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(8192)
            if not head:
                return None
            avg = len(head) / max(1, head.count("\n") + 1)
            if avg <= 0:
                return None
            return int(os.path.getsize(path) / avg)
    except Exception as exc:
        append_debug(f"row estimate failed for {path}: {exc}")
    return None
