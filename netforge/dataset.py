"""
Data.

Four sources behind one budget implementation, so the trainer never has to
know where its tensors came from:

  synthetic  procedurally generated tasks (sin/cos, xor, moons, spiral, blobs)
              - lets you train a 40-parameter MLP with zero downloads, which is
                the whole point of supporting very tiny nets
  text       plain .txt files from a directory, or a HuggingFace dataset
  tabular    a CSV, for the mlp family
  (the first two feed the gpt family; the last two feed the mlp family)

Every tabular source - Hub, local .parquet, .csv, .jsonl[.gz], .txt - goes
through one `stream_rows_to_tokens`. The caps therefore behave identically
everywhere and there is exactly one copy of the logic to get right.

The budget system is the part that matters. See `auto_token_budget`,
`BYTES_PER_TOKEN` and `token_plan` for how "this model wants N tokens, which
is X GB" is worked out, and `stream_rows_to_tokens` for how the pull is kept
inside it.
"""

import csv
import math
import os
import re
import time

import numpy as np
import torch

from . import arch as arch_mod
from .logging_utils import append_debug, append_error

_TEXT_EXT = (".txt", ".text", ".md", ".rst")

# Field names checked, in order, when auto-detecting a text column.
TEXT_FIELDS = ["text", "content", "document", "passage", "article", "body",
               "story", "sentence", "abstract", "prompt", "question", "output",
               "response", "completion", "code", "summary", "title"]

# (instruction, input, output) triples that mark a dataset as instruction-shaped.
INSTRUCT_FIELDS = [
    ("instruction", "input", "output"),
    ("prompt", None, "response"),
    ("question", None, "answer"),
    ("human", None, "assistant"),
    ("user", None, "bot"),
]

SKIP_HINTS = ("debug", "error", "log", "checkpoint", "ckpt", "cache",
              "tokenizer", "__pycache__", ".git", ".ipynb_checkpoints", "wandb")

CODE_HINTS = ("import ", "def ", "class ", "torch.", "#include", "<?php",
              "<script", "function(", "public static", "</")


# ---------------------------------------------------------------------------
# Synthetic tasks - no downloads, tiny nets, instant feedback
# ---------------------------------------------------------------------------

SYNTH_TASKS = {
    "sin": "y = sin(3x) on [-pi, pi] — regression",
    "cos": "y = cos(2x) — regression",
    "poly": "y = 3x^3 - 2x^2 + x — regression",
    "xor": "2-feature XOR — binary classification",
    "circle": "inside/outside a circle — binary classification",
    "moons": "two interleaved half moons — binary classification",
    "spiral": "2-arm spiral — binary classification",
    "blobs": "4 gaussian blobs — 4-class classification",
}


def make_synth(kind, n, n_features, noise=0.02, seed=1337):
    """Returns (X, y) numpy arrays. y is float for regression, long for classes."""
    rng = np.random.default_rng(seed)
    k = kind.lower()

    if k == "sin" or k == "cos":
        x = rng.uniform(-math.pi, math.pi, (n, n_features)).astype(np.float32)
        base = x[:, 0]
        y = (np.sin(3 * base) if k == "sin" else np.cos(2 * base)).astype(np.float32)
        y += rng.normal(0, noise, n).astype(np.float32)
        return x, y.reshape(-1, 1)

    if k == "poly":
        x = rng.uniform(-2, 2, (n, n_features)).astype(np.float32)
        t = x[:, 0]
        y = (3 * t ** 3 - 2 * t ** 2 + t + rng.normal(0, noise, n)).astype(np.float32)
        return x, y.reshape(-1, 1)

    if k == "xor":
        x = rng.integers(0, 2, (n, max(2, n_features))).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)
        for f in range(0, x.shape[1] - 1, 2):
            y ^= (x[:, f] > 0.5).astype(np.int64) & (x[:, f + 1] > 0.5).astype(np.int64)
        y = (y ^ (rng.random(n) < noise)).astype(np.int64)
        return x, y

    if k == "circle":
        x = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        y = ((x[:, 0] ** 2 + x[:, 1] ** 2) < 0.5).astype(np.int64)
        flip = rng.random(n) < noise
        y[flip] ^= 1
        return x, y

    if k == "moons":
        n1, n2 = n // 2, n - n // 2
        t1 = np.pi * rng.random(n1)
        t2 = np.pi * rng.random(n2)
        a = np.stack([np.cos(t1), np.sin(t1)], 1)
        b = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
        x = np.concatenate([a, b]).astype(np.float32)
        x += rng.normal(0, noise, x.shape).astype(np.float32)
        y = np.concatenate([np.zeros(n1), np.ones(n2)]).astype(np.int64)
        p = rng.permutation(n)
        return x[p], y[p]

    if k == "spiral":
        t = rng.uniform(0, 1, n) * 4 * math.pi + rng.normal(0, 0.2, n)
        lab = (t > 2 * math.pi).astype(np.int64)
        x = np.stack([t * np.cos(t - lab * math.pi), t * np.sin(t - lab * math.pi)], 1)
        x = (x * 0.3 + rng.normal(0, noise, (n, 2))).astype(np.float32)
        return x, lab

    if k == "blobs":
        centres = rng.uniform(-2, 2, (4, 2)).astype(np.float32)
        lab = rng.integers(0, 4, n)
        x = (centres[lab] + rng.normal(0, 0.6 + noise, (n, 2))).astype(np.float32)
        return x, lab.astype(np.int64)

    raise ValueError(f"unknown synthetic task {kind!r} "
                     f"(choose from {sorted(SYNTH_TASKS)})")


# ---------------------------------------------------------------------------
# Budgets: params x tokens_per_param -> tokens -> bytes
# ---------------------------------------------------------------------------

def auto_token_budget(n_params, ratio=20, steps=0, batch_size=0, block_size=0,
                      cap=50_000_000_000):
    """How many tokens a model of this size should be trained on.

    `ratio` is tokens per parameter. 20 is the Chinchilla figure, which is the
    right order for a real pretrain but wildly over-generous for a small model
    on a laptop, so it is a knob rather than a constant.

    The run-length term is the real limiter: a run of `steps` steps at
    `batch x block` consumes steps*batch*block tokens per epoch, so asking for
    more than that is asking for bytes nobody will train on. Taking the min of
    the two means the budget grows with your model AND with your run, and only
    a deliberate override goes past it.

    The clamp is a runaway guard, not the main mechanism. It is deliberately
    huge: at the measured 0.62 bytes/token, the old 20M-token ceiling was 12 MB
    of corpus, which is not a safety limit so much as a bug. 50B tokens is
    ~119 GB of parquet, which the download preflight will refuse long before
    anyone gets there.
    """
    n = max(1, int(n_params))
    ratio = max(0.01, float(ratio or 20))
    target = int(n * ratio)
    if steps and batch_size and block_size:
        epochs = 4                       # a corpus is usually walked a few times
        target = min(target, int(steps) * int(batch_size) * int(block_size) * epochs)
    return int(max(50_000, min(int(cap), target)))


# Bytes on disk per token, measured rather than guessed:
#   - 0.62 B/char is the C4 (English web text) snappy-parquet ratio, sampled
#     off the row-group footers; TinyStories measures 0.52, so treat this as
#     "English prose in parquet", not a universal constant.
#   - 3.86 chars/token is byte-level BPE at vocab 8000 on the same text.
#   - plain .txt is uncompressed, so it is ~1 B/char.
#
# Code, tables and non-English text all differ. `data.bytes_per_token` lets
# you override rather than argue with this table.
BYTES_PER_TOKEN = {
    ("parquet", "char"): 0.62,
    ("parquet", "bpe"): 2.38,
    ("text", "char"): 1.00,
    ("text", "bpe"): 3.86,
}
DEFAULT_BPT = 2.38           # parquet + bpe: the common real-world case


def bytes_per_token(tokenizer_type="bpe", kind="parquet", override=None):
    """Bytes on disk per token, for budget display only."""
    try:
        o = float(override)
        if o > 0:
            return o
    except (TypeError, ValueError):
        pass
    kind = "text" if kind in ("text", "txt", "raw") else "parquet"
    tt = "char" if str(tokenizer_type).lower() == "char" else "bpe"
    return BYTES_PER_TOKEN.get((kind, tt), DEFAULT_BPT)


def tokens_to_bytes(tokens, tokenizer_type="bpe", kind="parquet", override=None):
    bpt = bytes_per_token(tokenizer_type, kind, override)
    return int(tokens * bpt)


def token_plan(run):
    """The whole budget story for a config, ready to display.

    Returns {params, ratio, tokens, source, bytes, gb, bpt, tokenizer, kind}.
    """
    m, t, d = run.get("model", {}), run.get("train", {}), run.get("data", {})
    n_params = arch_mod.estimate_params(m)
    ratio = float(d.get("tokens_per_param", 20) or 20)
    explicit = int(d.get("max_tokens", 0) or 0)
    seq = int(m.get("block_size", 128)) if m.get("family") == "gpt" else 1
    if explicit > 0:
        tokens, source = explicit, "explicit"
    else:
        tokens, source = auto_token_budget(
            n_params, ratio, steps=int(t.get("steps", 0) or 0),
            batch_size=int(t.get("batch_size", 0) or 0), block_size=seq), "auto"
    # a remote dataset arrives as parquet; a local one is whatever is on disk
    has_remote = bool(d.get("hf_dataset", ""))
    kind = "parquet" if has_remote else "text"
    tt = str((run.get("tokenizer") or {}).get("type", "bpe")).lower()
    tt = "char" if tt == "char" else "bpe"
    override = d.get("bytes_per_token")
    bpt = bytes_per_token(tt, kind, override)
    nbytes = tokens_to_bytes(tokens, tt, kind, override)
    return {"params": n_params, "ratio": ratio, "tokens": tokens,
            "source": source, "bytes": nbytes, "gb": nbytes / 1024 ** 3,
            "bpt": bpt, "tokenizer": tt, "kind": kind,
            "params_label": arch_mod.fmt_params(n_params)}


def resolve_token_budget(data_cfg, mcfg, trcfg=None):
    """The token cap actually in force, plus where it came from."""
    cap = int(data_cfg.get("max_tokens", 0) or 0)
    if cap > 0:
        return cap, "explicit"
    trcfg = trcfg or {}
    return auto_token_budget(
        arch_mod.estimate_params(mcfg),
        ratio=float(data_cfg.get("tokens_per_param", 20) or 20),
        steps=trcfg.get("steps", 0),
        batch_size=trcfg.get("batch_size", 0),
        block_size=mcfg.get("block_size", 128) if mcfg.get("family") == "gpt" else 1,
    ), "auto"


# ---------------------------------------------------------------------------
# Local text corpora
# ---------------------------------------------------------------------------

def looks_clean(text):
    """Reject logs, source files and junk scraped into a text corpus.

    A corpus that is 40% Python source trains a model that is 40% Python
    source, and you will not notice until you read the samples.
    """
    t = text.strip()
    if len(t) < 32:
        return False
    visible = sum(1 for c in t if c.isprintable() or c in "\n\t") / max(1, len(t))
    if visible < 0.95:
        return False
    code_hits = sum(1 for h in CODE_HINTS if h in t)
    if code_hits >= 3:
        return False
    return True


def read_texts_from_dir(data_dir, strict=False, max_gb=0.0, progress_cb=None):
    """Recursively read .txt files, skipping junk paths. Returns one big string.

    Only used by the plain-text fallback in `stream_local_tokens`; every other
    source streams row by row.
    """
    if not os.path.isdir(data_dir):
        if strict:
            raise FileNotFoundError(f"no such data dir: {data_dir}")
        return ""
    paths = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_HINTS]
        for f in files:
            if not f.lower().endswith(_TEXT_EXT):
                continue
            low = f.lower()
            if any(h in low for h in SKIP_HINTS):
                continue
            paths.append(os.path.join(root, f))
    paths.sort()
    cap = int(float(max_gb or 0) * 1024 ** 3)
    chunks, total, skipped = [], 0, 0
    for p in paths:
        if cap and total >= cap:
            break
        text = _decode_file(p)
        if not text:
            continue
        if not looks_clean(text):
            skipped += 1
            continue
        chunks.append(text)
        total += len(text.encode("utf-8", errors="ignore"))
        if progress_cb:
            progress_cb(len(paths), total, skipped)
    if skipped:
        append_debug(f"skipped {skipped} file(s) that did not look like prose")
    return "\n\n".join(chunks)


def _decode_file(path):
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HuggingFace metadata
# ---------------------------------------------------------------------------

def list_hf_subsets(dataset_id):
    """(subsets, error) - never raises, so the GUI can populate a dropdown."""
    try:
        from datasets import get_dataset_config_names
        names = list(get_dataset_config_names(dataset_id))
        return names, ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"[:200]


def list_hf_splits(dataset_id, subset=None):
    """(splits, error). Tries three routes before giving up and guessing."""
    try:
        from datasets import get_dataset_split_names
        names = list(get_dataset_split_names(dataset_id, config=subset or None))
        return names, ""
    except Exception:
        pass
    try:
        from datasets import load_dataset_builder
        b = load_dataset_builder(dataset_id, subset or None)
        return list(b.info.splits), ""
    except Exception as exc:
        return ["train"], f"could not enumerate splits ({type(exc).__name__}); defaulting to train"


def detect_text_field(cols, requested=""):
    if requested and requested in cols:
        return requested
    low = {c.lower(): c for c in cols}
    for f in TEXT_FIELDS:
        if f in low:
            return low[f]
    return ""


def detect_instruct_format(sample):
    """Return a format string if the row looks like an instruction pair."""
    keys = {k.lower() for k in sample.keys()}
    for inst, inp, out in INSTRUCT_FIELDS:
        if inst in keys and out in keys:
            if inp:
                return (f"### Instruction:\n{{{inst}}}\n\n"
                        f"### Input:\n{{{inp}}}\n\n### Response:\n{{{out}}}")
            return f"### Instruction:\n{{{inst}}}\n\n### Response:\n{{{out}}}"
    return ""


def hf_cache_path(data_dir, dataset_id, subset, split, field):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{dataset_id}__{subset}__{split}")
    if field:
        safe += "__" + re.sub(r"[^A-Za-z0-9_.-]+", "_", field)
    return os.path.join(data_dir, safe + ".txt")


# ---------------------------------------------------------------------------
# CSV / tabular
# ---------------------------------------------------------------------------

def load_csv(path, target="", as_index=False, max_rows=0, has_header=True):
    """Returns (X, y) numpy arrays. Target may be a column name or an index."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such CSV: {path}")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except Exception:
            dialect = csv.excel
        rows = []
        if has_header:
            reader = csv.DictReader(fh, dialect=dialect)
            for r in reader:
                rows.append(r)
                if max_rows and len(rows) >= max_rows:
                    break
        else:
            reader = csv.reader(fh, dialect=dialect)
            for r in reader:
                rows.append(dict(enumerate(r)))
                if max_rows and len(rows) >= max_rows:
                    break
    if not rows:
        raise ValueError(f"CSV {path} has no rows")

    cols = list(rows[0].keys())
    if target == "" or target is None:
        target = cols[-1]
    if target not in cols:
        try:
            target = cols[int(target)]
        except (ValueError, IndexError):
            raise ValueError(f"target {target!r} not in CSV columns {cols}")

    feat_cols = [c for c in cols if c != target]
    numeric = []
    for c in feat_cols:
        try:
            numeric.append([float(r.get(c) or 0) for r in rows])
        except (TypeError, ValueError):
            # non-numeric feature: label-encode it
            vals = [str(r.get(c) or "") for r in rows]
            uniq = sorted(set(vals))
            idx = {v: i for i, v in enumerate(uniq)}
            numeric.append([float(idx[v]) for v in vals])
    X = np.asarray(numeric, dtype=np.float32).T

    if as_index:
        uniq = sorted({str(r.get(target)) for r in rows})
        lut = {v: i for i, v in enumerate(uniq)}
        y = np.asarray([lut[str(r.get(target))] for r in rows], dtype=np.int64)
    else:
        try:
            y = np.asarray([float(r.get(target) or 0) for r in rows],
                           dtype=np.float32)
        except (TypeError, ValueError):
            uniq = sorted({str(r.get(target)) for r in rows})
            lut = {v: i for i, v in enumerate(uniq)}
            y = np.asarray([lut[str(r.get(target))] for r in rows], dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Streaming: the one budget implementation
# ---------------------------------------------------------------------------

def stream_rows_to_tokens(rows, tok, *, field="", cap_tokens=0, cap_bytes=0,
                          row_cap=0, progress_cb=None, keep_text=False,
                          cache_txt_path=None, label="rows"):
    """Encode an iterable of row dicts, stopping at whichever cap hits first.

    The shared core of every tabular source. HuggingFace, local .parquet, local
    .csv and local .txt all funnel through here so the budget behaviour is
    identical and there is exactly one copy of it to get right.

    The important properties:
      * one row in, one row encoded, row dropped - the corpus is never
        materialised, so peak RAM is the token tensor and nothing else
      * int32 output (4 bytes/token); callers cast per-batch, not per-corpus
      * `cache_txt_path` streams to disk as it goes, so writing a cache costs
        disk instead of RAM
    """
    buf, n_tok = [], 0
    text_chunks = [] if keep_text else None
    # `rows` is the incoming iterable; the counter is n_rows. Shadowing the
    # parameter here makes `for row in rows` iterate an int.
    nbytes, n_rows, skipped, fmt, cols = 0, 0, 0, "", []
    stop_reason = ""
    t0 = time.time()

    sink = None
    if cache_txt_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_txt_path)),
                        exist_ok=True)
            sink = open(cache_txt_path, "w", encoding="utf-8")
        except Exception as exc:
            append_debug(f"text cache open failed: {exc}")

    try:
        for row in rows:
            if not cols:
                cols = list(row.keys())
                if not field:
                    field = detect_text_field(cols)
                    if field:
                        append_debug(f"auto-detected text field {field!r} in {label}")
                fmt = detect_instruct_format({k: (v if isinstance(v, str) else "")
                                              for k, v in row.items()})

            if fmt:
                try:
                    text = fmt.format(**{k: str(v) for k, v in row.items()})
                except KeyError:
                    text = str(row.get(field) or row)
            else:
                text = row.get(field) or ""
            text = str(text)
            if not text.strip():
                skipped += 1
                continue

            ids = tok.encode(text)
            n_rows += 1
            nbytes += len(text.encode("utf-8", errors="ignore"))
            if text_chunks is not None:
                text_chunks.append(text)
            if sink is not None:
                sink.write(text)
                sink.write("\n\n")

            room = (cap_tokens - n_tok) if cap_tokens else None
            if room is not None and room <= 0:
                stop_reason = "token cap"
                break
            if room is not None and len(ids) >= room:
                buf.extend(ids[:room])       # truncate this row, keep the rest
                n_tok += room
                stop_reason = "token cap"
                break
            buf.extend(ids)
            n_tok += len(ids)

            if row_cap and n_rows >= row_cap:
                stop_reason = "row cap"
                break
            if cap_bytes and nbytes >= cap_bytes:
                stop_reason = "byte cap"
                break
            if progress_cb and n_rows % 20000 == 0:
                progress_cb(n_rows, nbytes, n_tok, time.time() - t0)
    finally:
        if sink is not None:
            try:
                sink.close()
            except Exception:
                pass

    if not stop_reason:
        stop_reason = "exhausted"

    data = torch.tensor(buf, dtype=torch.int32)
    del buf

    meta = {"source": label, "field": field, "rows": n_rows, "bytes": nbytes,
            "tokens": int(data.numel()), "skipped": skipped,
            "stopped_because": stop_reason, "instruct_format": bool(fmt),
            "cols": cols,
            "cached_to": cache_txt_path if sink is not None else None}
    text = "\n\n".join(text_chunks) if text_chunks is not None else None
    return data, meta, text


def stream_hf_tokens(dataset_id, tok, subset="", split="train", field="",
                     max_rows=0, max_gb=0.0, max_tokens=0, progress_cb=None,
                     keep_text=False, cache_txt_path=None):
    """Stream a HF dataset into a token tensor under the same caps.

    Returns (tensor, meta, text_or_None).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("HuggingFace datasets needs: pip install datasets")

    kw = {"name": subset} if subset else {}
    ds = load_dataset(dataset_id, split=split, streaming=True, **kw)

    data, meta, text = stream_rows_to_tokens(
        ds, tok, field=field, cap_tokens=max_tokens,
        cap_bytes=int(float(max_gb or 0) * 1024 ** 3), row_cap=max_rows,
        progress_cb=progress_cb, keep_text=keep_text,
        cache_txt_path=cache_txt_path,
        label=f"{dataset_id}[{subset or '-'}/{split}]")

    if meta["rows"] == 0:
        raise ValueError(f"no text rows in {dataset_id} "
                         f"(columns were {meta['cols']})")
    meta.update({"dataset": dataset_id, "subset": subset, "split": split})
    return data, meta, text


def _data_files_under(data_dir, exts):
    """Find data files under data_dir, skipping hidden directories.

    `datasets` drops a `.cache/` tree next to whatever you load, and a naive
    recursive glob walks straight into it.
    """
    import glob
    out = []
    for ext in exts:
        for p in glob.glob(os.path.join(data_dir, "**", "*" + ext),
                           recursive=True):
            rel = os.path.relpath(p, data_dir)
            if any(part.startswith(".") for part in rel.split(os.sep)):
                continue
            if os.path.isfile(p):
                out.append(p)
    return sorted(set(out))


def stream_local_tokens(data_dir, tok, field="", max_rows=0, max_gb=0.0,
                        max_tokens=0, progress_cb=None, keep_text=False,
                        cache_txt_path=None, split_filter=None):
    """Same budget, but from files already on disk.

    Prefers .parquet (what most modern Hub datasets ship, and what the dataset
    browser downloads), then .csv, then .jsonl/.jsonl.gz, then concatenated
    .txt. Parquet is read streaming=True so a 2 GB shard still costs one row of
    RAM.

    `split_filter` is a set like {"validation"}: only files whose path falls
    in those splits (see hfhub.split_of) are read. That is how the trainer
    builds a genuinely held-out eval set from validation shards instead of
    sampling the training corpus and calling it eval.
    """
    parquet = _data_files_under(data_dir, (".parquet",))
    csvs = _data_files_under(data_dir, (".csv", ".tsv"))
    jsonls = _data_files_under(data_dir, (".jsonl", ".jsonl.gz", ".json.gz"))

    if split_filter is not None:
        from .hfhub import split_of
        keep = [p for lst in (parquet, csvs, jsonls) for p in lst
                if split_of(p) in split_filter]
        if not keep:
            raise FileNotFoundError(
                f"no {sorted(split_filter)} files under {data_dir}")
        have = {os.path.abspath(p) for p in keep}
        parquet = [p for p in parquet if os.path.abspath(p) in have]
        csvs = [p for p in csvs if os.path.abspath(p) in have]
        jsonls = [p for p in jsonls if os.path.abspath(p) in have]

    if parquet or csvs or jsonls:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("reading .parquet/.csv/.jsonl needs: pip install datasets")
        paths = (parquet or csvs or jsonls)
        kind = "parquet" if parquet else ("csv" if csvs else "json")
        ds = load_dataset(kind, data_files=paths, split="train", streaming=True)
        return stream_rows_to_tokens(
            ds, tok, field=field, cap_tokens=max_tokens,
            cap_bytes=int(float(max_gb or 0) * 1024 ** 3), row_cap=max_rows,
            progress_cb=progress_cb, keep_text=keep_text,
            cache_txt_path=cache_txt_path,
            label=f"{os.path.basename(os.path.abspath(data_dir))} ({kind})")

    # plain text: batch into pseudo-rows so the caps still apply
    def rows():
        buf = []
        for text in read_texts_from_dir(
                data_dir, max_gb=float(max_gb or 0)).split("\n\n"):
            t = text.strip()
            if t:
                buf.append(t)
                if len(buf) >= 256:
                    yield {"text": "\n\n".join(buf)}
                    buf = []
        if buf:
            yield {"text": "\n\n".join(buf)}

    return stream_rows_to_tokens(
        rows(), tok, field=field, cap_tokens=max_tokens,
        cap_bytes=int(float(max_gb or 0) * 1024 ** 3), row_cap=max_rows,
        progress_cb=progress_cb, keep_text=keep_text,
        cache_txt_path=cache_txt_path, label=os.path.basename(data_dir))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def get_batch(data, block_size, batch_size, device):
    """Vectorised next-token batch - no Python loop over the batch.

    The corpus is stored as int32 to halve its footprint (and int64 tensors
    were a third of the RAM blowup), but the batch is cast back to int64
    because embedding lookups and cross_entropy both require it. The cast is
    batch-sized, so the saving is kept where it matters.
    """
    n = data.size(0)
    if n < 2:
        raise ValueError("corpus has fewer than 2 tokens")
    bs = min(int(block_size), n - 1)
    starts = torch.randint(0, n - bs, (int(batch_size),), device=data.device)
    offs = torch.arange(bs, device=data.device)
    ix = starts[:, None] + offs[None, :]
    x, y = data[ix], data[ix + 1]
    if x.dtype != torch.long:
        x = x.long()
        y = y.long()
    if data.device != torch.device(device):
        nb = torch.device(device).type == "cuda"
        x = x.to(device, non_blocking=nb)
        y = y.to(device, non_blocking=nb)
    return x, y


def get_tab_batch(X, y, batch_size, device, replace=True):
    """Sample rows with replacement.

    A 20k-row toy set would otherwise run dry after 625 steps of batch 32 and
    the "epoch" would end long before the step budget did.
    """
    n = X.shape[0]
    if n == 0:
        raise ValueError("no rows in the tabular dataset")
    idx = torch.randint(0, n, (int(batch_size),), device=X.device)
    return X[idx].to(device), y[idx].to(device)


def standardize(X, y):
    """Zero-mean unit-variance features. Returns (Xn, yn, stats)."""
    xm, xs = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    Xn = ((X - xm) / xs).astype(np.float32)
    if y.ndim == 2 and y.shape[1] == 1 and y.dtype.kind == "f":
        ym, ys = y.mean(), y.std() + 1e-8
        yn = ((y - ym) / ys).astype(np.float32)
        return Xn, yn, {"x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys}
    return Xn, y, {"x_mean": xm, "x_std": xs}


# ---------------------------------------------------------------------------
# The orchestrating entry point
# ---------------------------------------------------------------------------

def tok_cfg_type(run):
    return str((run.get("tokenizer") or {}).get("type", "char")).lower()


def eval_token_cap(run, block_size):
    """How many held-out tokens eval gets: enough for a full eval pass."""
    tcfg = run.get("train", {}) or {}
    bs = max(1, int(tcfg.get("batch_size", 32)))
    eb = max(1, int(tcfg.get("eval_batches", 20)))
    return max(4096, eb * bs * max(1, int(block_size)) * 2)


def _local_validation_eval(data_dir, tok, meta, data_cfg, eval_cap, log):
    """Eval tokens from validation shards on disk, or (None, 'none').

    Tiny result (<2k tokens) also counts as none: a postage-stamp eval set
    is noise, and the tail-carve fallback below will do better.
    """
    try:
        field = meta.get("field", "") or data_cfg.get("hf_field", "")
        ev, emeta, _ = stream_local_tokens(
            data_dir, tok, field=field,
            max_rows=int(data_cfg.get("max_rows", 0) or 0),
            max_gb=float(data_cfg.get("max_gb", 0) or 0),
            max_tokens=eval_cap, split_filter={"validation"})
    except Exception as exc:
        append_debug(f"no validation files for eval: {exc}")
        log("  eval: no validation files on disk - a holdout will be "
            "carved from the corpus tail")
        return None, "none"
    if ev.numel() < 2048:
        return None, "none"
    log(f"  eval: {ev.numel():,} held-out tokens from "
        f"{emeta.get('rows', 0):,} validation rows")
    return ev, "files"


def _hf_validation_eval(hf_id, tok, data_cfg, eval_cap, log):
    """Eval tokens from the repo's validation split, or (None, 'none').

    Only validation-family splits qualify: burning the test split as a
    validation signal is a decision, not a default.
    """
    from .hfhub import split_of
    try:
        splits, _err = list_hf_splits(
            hf_id, (data_cfg.get("hf_subset", "") or None))
    except Exception as exc:
        append_debug(f"split list failed: {exc}")
        return None, "none"
    train_split = data_cfg.get("hf_split", "train")
    cands = [s for s in splits
             if split_of(s) == "validation" and s != train_split]
    if not cands:
        log("  eval: no validation split in this repo - a holdout will be "
            "carved from the corpus tail")
        return None, "none"
    try:
        ev, emeta, _ = stream_hf_tokens(
            hf_id, tok, data_cfg.get("hf_subset", ""), cands[0],
            field=data_cfg.get("hf_field", ""),
            max_rows=int(data_cfg.get("max_rows", 0) or 0),
            max_gb=float(data_cfg.get("max_gb", 0) or 0),
            max_tokens=eval_cap)
    except Exception as exc:
        append_debug(f"validation split stream failed: {exc}")
        return None, "none"
    if ev.numel() < 2048:
        return None, "none"
    log(f"  eval: {ev.numel():,} held-out tokens from "
        f"{hf_id}[{cands[0]}]")
    return ev, "split"


def _tab_holdout(X, y):
    """Last 10% of rows held out, or (X, y, None, None, 'none') when small."""
    n = int(X.shape[0])
    if n >= 200:
        k = max(20, n // 10)
        return X[:n - k], y[:n - k], X[n - k:], y[n - k:], "rows"
    return X, y, None, None, "none"


def prepare(run, log=append_error, progress_cb=None, tokenizer=None):
    """Build everything training needs. Returns a dict; never raises.

    {
      kind:  'text' | 'tabular',
      data / X / y, tokenizer (text only), n_tokens, meta
    }

    `tokenizer`, when given, is used as-is instead of training a fresh one -
    a resumed run must keep the checkpoint's vocabulary, or every embedding
    row lands on the wrong token.
    """
    data_cfg = run.get("data", {})
    task = data_cfg.get("task", "synthetic")
    seed = int(run.get("train", {}).get("seed", 1337))
    mcfg = dict(run.get("model", {}))

    # ---- synthetic -------------------------------------------------------
    if task == "synthetic":
        n = int(data_cfg.get("synth_n", 20000))
        nf = int(mcfg.get("n_features", 8))
        X, y = make_synth(data_cfg.get("synth", "sin"), n, nf,
                          float(data_cfg.get("synth_noise", 0.02)), seed)
        Xn, yn, stats = standardize(X, y)
        Xtr, ytr, Xev, yev, esrc = _tab_holdout(Xn, yn)
        log(f"synthetic task {data_cfg.get('synth', 'sin')}: "
            f"{Xtr.shape[0]:,} rows x {Xtr.shape[1]} features"
            + (f" (+{Xev.shape[0]:,} held out)" if Xev is not None else ""))
        return {"kind": "tabular", "X": torch.tensor(Xtr),
                "y": torch.tensor(ytr),
                "X_eval": torch.tensor(Xev) if Xev is not None else None,
                "y_eval": torch.tensor(yev) if yev is not None else None,
                "eval_source": esrc, "stats": stats,
                "n_tokens": int(Xtr.shape[0]), "meta": {"rows": int(Xtr.shape[0])}}

    # ---- csv -------------------------------------------------------------
    if task == "tabular":
        X, y = load_csv(data_cfg.get("csv_path", ""),
                        target=data_cfg.get("csv_target", ""),
                        as_index=bool(data_cfg.get("csv_index", False)),
                        max_rows=int(data_cfg.get("max_rows", 0) or 0))
        Xn, yn, stats = standardize(X, y)
        n_out = int(mcfg.get("n_outputs", 1))
        if n_out > 1 and yn.ndim == 1:
            yn = yn.reshape(-1, 1)
        Xtr, ytr, Xev, yev, esrc = _tab_holdout(Xn, yn)
        log(f"csv {data_cfg.get('csv_path')}: {Xtr.shape[0]:,} rows x "
            f"{Xtr.shape[1]} features, "
            f"target {data_cfg.get('csv_target', '(last col)')!r}"
            + (f" (+{Xev.shape[0]:,} held out)" if Xev is not None else ""))
        return {"kind": "tabular", "X": torch.tensor(Xtr),
                "y": torch.tensor(ytr),
                "X_eval": torch.tensor(Xev) if Xev is not None else None,
                "y_eval": torch.tensor(yev) if yev is not None else None,
                "eval_source": esrc, "stats": stats,
                "n_tokens": int(Xtr.shape[0]),
                "meta": {"rows": int(Xtr.shape[0]), "features": int(Xtr.shape[1])}}

    # ---- text ------------------------------------------------------------
    from .tokenizer import CharTokenizer, build_tokenizer

    cap_tokens, cap_src = resolve_token_budget(data_cfg, mcfg, run.get("train", {}))
    n_params = arch_mod.estimate_params(mcfg)
    log(f"token budget: {cap_tokens:,} ({cap_src}) for a "
        f"{arch_mod.fmt_params(n_params)} model")
    eval_cap = eval_token_cap(run, mcfg.get("block_size", 128))
    eval_data, eval_source = None, "none"

    hf_id = (data_cfg.get("hf_dataset") or "").strip()
    want_bpe = tok_cfg_type(run) == "bpe"
    cache_txt = None
    if hf_id and bool(data_cfg.get("cache_text", False)):
        cache_txt = hf_cache_path(data_cfg.get("data_dir", "data"), hf_id,
                                  data_cfg.get("hf_subset", ""),
                                  data_cfg.get("hf_split", "train"),
                                  data_cfg.get("hf_field", ""))

    def do_hf(tok, max_tok, keep):
        return stream_hf_tokens(
            hf_id, tok, data_cfg.get("hf_subset", ""),
            data_cfg.get("hf_split", "train"),
            field=data_cfg.get("hf_field", ""),
            max_rows=int(data_cfg.get("max_rows", 0) or 0),
            max_gb=float(data_cfg.get("max_gb", 0) or 0),
            max_tokens=max_tok, progress_cb=progress_cb,
            keep_text=keep, cache_txt_path=cache_txt)

    if hf_id:
        if want_bpe:
            tcfg = run.get("tokenizer", {})
            if tokenizer is not None:
                tok = tokenizer
            else:
                # BPE needs the text, so this is the one branch that keeps a
                # corpus in memory - capped the same way, and only for as long
                # as it takes to learn the merges
                probe = CharTokenizer()
                _, meta, sample = do_hf(probe,
                                        int(tcfg.get("max_corpus_chars",
                                                     256000)),
                                        True)
                tok = build_tokenizer(tcfg, sample)
                req = int(tcfg.get("vocab_size", 8000) or 8000)
                if not tok.merges:
                    log(f"  bpe: corpus produced no merges - "
                        f"falling back to a char tokenizer")
                    tok = CharTokenizer()
                else:
                    log(f"  bpe: trained vocab {tok.vocab_size:,} "
                        f"(requested {req:,})")
                del sample
            data, meta, _ = do_hf(tok, cap_tokens, False)
        else:
            tok = CharTokenizer()
            data, meta, _ = do_hf(tok, cap_tokens, False)
        eval_data, eval_source = _hf_validation_eval(
            hf_id, tok, data_cfg, eval_cap, log)
    else:
        data_dir = data_cfg.get("data_dir", "data")
        meta = {}
        if want_bpe:
            tcfg = run.get("tokenizer", {})
            if tokenizer is not None:
                tok = tokenizer
            else:
                probe = CharTokenizer()
                _, meta, sample = stream_local_tokens(
                    data_dir, probe,
                    field=data_cfg.get("hf_field", ""),
                    max_rows=int(data_cfg.get("max_rows", 0) or 0),
                    max_gb=float(data_cfg.get("max_gb", 0) or 0),
                    max_tokens=int(tcfg.get("max_corpus_chars", 256000)),
                    progress_cb=progress_cb, keep_text=True,
                    cache_txt_path=None)
                tok = build_tokenizer(tcfg, sample)
                req = int(tcfg.get("vocab_size", 8000) or 8000)
                if not tok.merges:
                    log(f"  bpe: corpus produced no merges - "
                        f"falling back to a char tokenizer")
                    tok = CharTokenizer()
                else:
                    log(f"  bpe: trained vocab {tok.vocab_size:,} "
                        f"(requested {req:,})")
                del sample
        else:
            tok = CharTokenizer()
        data, meta, _ = stream_local_tokens(
            data_dir, tok,
            field=meta.get("field", "") or data_cfg.get("hf_field", ""),
            max_rows=int(data_cfg.get("max_rows", 0) or 0),
            max_gb=float(data_cfg.get("max_gb", 0) or 0),
            max_tokens=cap_tokens, progress_cb=progress_cb,
            keep_text=False, cache_txt_path=cache_txt)
        log(f"  {meta.get('source', '?')}: {meta.get('rows', 0):,} rows, "
            f"{meta.get('bytes', 0) / 1024 ** 2:.1f} MB "
            f"({meta.get('stopped_because', '?')})")
        eval_data, eval_source = _local_validation_eval(
            data_dir, tok, meta, data_cfg, eval_cap, log)

    log(f"tokenized: {data.numel():,} tokens, vocab {tok.vocab_size:,}")
    return _text_bundle(data, tok, mcfg, meta, cap_tokens,
                        eval_cap=eval_cap,
                        eval_data=eval_data, eval_source=eval_source,
                        log=log)


def _text_bundle(data, tok, mcfg, meta, cap_tokens, eval_cap=0,
                 eval_data=None, eval_source="none", log=None):
    vocab = tok.vocab_size
    mcfg = dict(mcfg)
    if int(mcfg.get("vocab_size", 0)) != vocab:
        msg = (f"model vocab {mcfg.get('vocab_size')} != tokenizer "
               f"{vocab}; using {vocab}")
        (log or append_debug)(msg)
        mcfg["vocab_size"] = vocab
    n = int(data.numel())
    if cap_tokens and n > cap_tokens:
        data = data[:cap_tokens]
        n = cap_tokens
    if eval_data is None and eval_cap and n > eval_cap + 4096:
        # No validation files/split: hold out the tail. It never trains -
        # get_batch only ever samples the head - so eval finally measures
        # generalisation instead of memorisation. Small corpora skip this
        # and eval falls back to train batches (old behaviour).
        eval_data, data = data[n - eval_cap:], data[:n - eval_cap]
        eval_source = "tail"
        if log:
            log(f"  eval: {eval_cap:,} tokens carved from the corpus tail "
                f"as a held-out set")
        n = int(data.numel())
    elif eval_data is not None:
        eval_source = eval_source if eval_source != "none" else "files"
    return {"kind": "text", "data": data, "tokenizer": tok,
            "model_cfg_override": mcfg, "n_tokens": n,
            "eval_data": eval_data, "eval_source": eval_source,
            "meta": meta, "budget": cap_tokens}