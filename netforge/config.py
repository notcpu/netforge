"""
The run configuration: one JSON-serializable dict shared by GUI, CLI and trainer.

`run_cfg` is the single source of truth. The GUI edits it, saves it to
`<checkpoint>.run.json`, and spawns the trainer with that path. The trainer
never reads GUI state and the GUI never reads trainer state.

`validate_run_cfg` returns a list of human-readable error strings instead of
raising, so the GUI can show them in a dialog without a try/except.
"""

import copy
import json
import os

# ---------------------------------------------------------------------------
# Vocabulary of valid choices. Model/arch.py and memory.py import these so a
# new activation or norm type is a one-line change in exactly one place.
# ---------------------------------------------------------------------------

MODEL_FAMILIES = ("mlp", "gpt")
TASKS = ("synthetic", "text", "tabular")

DTYPES = {
    "fp32": 4, "float32": 4,
    "fp16": 2, "float16": 2, "half": 2,
    "bf16": 2, "bfloat16": 2,
    "fp8": 1, "int8": 1,
}

OPTIMIZERS = ("adamw", "adam", "sgd", "rmsprop", "adafactor", "lion")
SCHEDULERS = ("cosine", "linear", "constant", "wsd")
ACTIVATIONS = ("gelu", "gelu_new", "relu", "silu", "tanh", "sigmoid")
NORMS = ("layernorm", "rmsnorm")
POS_ENCODINGS = ("learned", "sinusoidal", "rope", "none")

MIN_CTX, MAX_CTX = 16, 131072
MIN_DIM = 2


def default_model_cfg():
    """A deliberately tiny GPT - the smallest thing that can still generate text."""
    return {
        "family": "gpt",
        "vocab_size": 512,
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "embed_dim": 128,
        "ff_mult": 4,
        "ff_dim": 0,            # 0 => derive as embed_dim * ff_mult
        "dropout": 0.1,
        "pos_encoding": "rope",
        "norm_type": "rmsnorm",
        "activation": "silu",
        "tie_weights": True,
        "dtype": "fp32",
        # MLP family only
        "n_features": 8,
        "n_outputs": 1,
        "hidden": [64, 64],
        "out_activation": "none",
        "residual": False,
    }


def default_tokenizer_cfg():
    return {
        "type": "char",
        "vocab_size": 512,
        "specials": ["<unk>", "<bos>", "<eos>", "<pad>"],
        "tokenizer_file": "",
        "train_from_corpus": True,
        "max_corpus_chars": 256000,
    }


def default_data_cfg():
    return {
        "task": "synthetic",
        # synthetic toy tasks - let you train a 200-param net with no downloads
        "synth": "sin",
        "synth_n": 20000,
        "synth_noise": 0.02,
        # local corpora
        "data_dir": "data",
        # Hard limits on how much data is pulled. Both are *defaults to sane
        # values you must raise deliberately*, not suggestions:
        #   max_tokens - stops the pull mid-stream. 0 means "auto", which is
        #                params x tokens_per_param, also capped by what the
        #                run can actually consume over 4 epochs.
        #   max_gb     - secondary cap on raw bytes, for very wide tokenizers.
        "max_tokens": 0,
        # tokens per parameter. Chinchilla says 20 for a real pretrain; a small
        # model on a laptop wants far less, so this is the knob that decides
        # how much data a model "should" see.
        "tokens_per_param": 20,
        # measured bytes/token used to turn a token budget into a download
        # size. 0 = use the measured table in dataset.BYTES_PER_TOKEN.
        "bytes_per_token": 0.0,
        "max_gb": 1.0,
        "max_rows": 0,
        # Disk cache of pulled corpora. OFF by default: streaming a dataset
        # writes nothing, and enabling this can put gigabytes next to your
        # checkpoint without warning.
        "cache_text": False,
        # tabular (CSV) for the mlp family
        "csv_path": "",
        "csv_target": "",
        "csv_index": False,
        # HuggingFace
        "hf_dataset": "",
        "hf_subset": "",
        "hf_split": "train",
        "hf_field": "",
    }


def default_optimizer_cfg():
    return {
        "name": "adamw",
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "max_grad_norm": 1.0,
        "accum": 1,
        "fused": True,
    }


def default_train_cfg():
    return {
        "steps": 2000,
        "batch_size": 32,
        "lr": 3e-3,
        "scheduler": "cosine",
        "warmup_frac": 0.05,
        "min_lr_frac": 0.05,
        "eval_every": 100,
        "eval_batches": 20,
        "sample_every": 250,
        "gen_len": 120,
        "sample_temp": 1.0,
        "sample_top_k": 40,
        "seed": 1337,
        "log_every": 10,
        "ckpt_every": 500,
        "ema": 0.0,
    }


def default_memory_cfg():
    """Knobs that only affect the *estimate*, never the training itself."""
    return {
        "dtype": "fp32",              # master weight dtype used in the report
        "compute_dtype": "bf16",      # what the forward pass actually runs in
        "gradient_checkpointing": False,
        "sharding": "none",           # none | ddp | fsdp | zero2 | zero3
        "num_devices": 1,
        "optimizer_offload": False,   # keep AdamW moments on CPU
        "param_offload": False,       # keep fp32 master weights on CPU
        "max_logit_batch": 0,         # cap B*T on the logits tensor (chunked CE)
    }


def default_runtime_cfg():
    return {
        "device": "auto",
        "amp": True,
        "compile": False,
        "gpu_data": False,
        "gpu_data_max_mb": 512.0,
        "resume": True,
        "checkpoint": "netforge_run.pt",
        "log_file": "netforge.log",
        "max_threads": 0,
    }


def default_run_cfg():
    return {
        "model": default_model_cfg(),
        "tokenizer": default_tokenizer_cfg(),
        "data": default_data_cfg(),
        "optimizer": default_optimizer_cfg(),
        "train": default_train_cfg(),
        "memory": default_memory_cfg(),
        "runtime": default_runtime_cfg(),
        "ui": {"theme": "dark", "refresh_hz": 6},
    }


# ---------------------------------------------------------------------------
# Validation. Returns strings, never raises.
# ---------------------------------------------------------------------------

def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def validate_model_cfg(cfg):
    """Check one model cfg in place. Returns a list of error strings."""
    errs = []
    fam = cfg.get("family", "gpt")
    if fam not in MODEL_FAMILIES:
        errs.append(f"unknown model family {fam!r} (choose from {MODEL_FAMILIES})")
        return errs

    dim = _as_int(cfg.get("embed_dim"), 0)
    if dim < MIN_DIM:
        errs.append(f"embed_dim must be >= {MIN_DIM}, got {dim}")

    if fam == "gpt":
        ctx = _as_int(cfg.get("block_size"), 0)
        if not (MIN_CTX <= ctx <= MAX_CTX):
            errs.append(f"block_size must be {MIN_CTX}..{MAX_CTX}, got {ctx}")
        heads = _as_int(cfg.get("n_head"), 0)
        if heads < 1:
            errs.append(f"n_head must be >= 1, got {heads}")
        elif dim % heads != 0:
            errs.append(f"embed_dim ({dim}) must be divisible by n_head ({heads}) "
                        f"- got head_dim {dim // heads}")
        elif dim // heads < 4:
            errs.append(f"head_dim would be {dim // heads}; too small, use >= 8")
        elif (dim // heads) % 2 and cfg.get("pos_encoding", "rope") == "rope":
            errs.append(f"head_dim {dim // heads} is odd - rotary embeddings "
                        f"work best on an even head_dim, so use n_head="
                        f"{heads - 1 if dim % (heads - 1) == 0 else heads + 1} "
                        f"or a different embed_dim")
        if _as_int(cfg.get("n_layer"), 0) < 1:
            errs.append("n_layer must be >= 1")
        if _as_int(cfg.get("vocab_size"), 0) < 2:
            errs.append("vocab_size must be >= 2")
        if cfg.get("norm_type") not in NORMS:
            errs.append(f"unknown norm_type {cfg.get('norm_type')!r} (choose from {NORMS})")
        if cfg.get("activation") not in ACTIVATIONS:
            errs.append(f"unknown activation {cfg.get('activation')!r} "
                        f"(choose from {ACTIVATIONS})")
        if cfg.get("pos_encoding") not in POS_ENCODINGS:
            errs.append(f"unknown pos_encoding {cfg.get('pos_encoding')!r} "
                        f"(choose from {POS_ENCODINGS})")
    else:
        if _as_int(cfg.get("n_features"), 0) < 1:
            errs.append("n_features must be >= 1")
        if _as_int(cfg.get("n_outputs"), 1) < 1:
            errs.append("n_outputs must be >= 1")
        hidden = cfg.get("hidden") or []
        if isinstance(hidden, str):
            hidden = [h for h in hidden.replace(",", " ").split() if h]
        if not isinstance(hidden, list):
            errs.append("hidden must be a list of widths")
        else:
            for w in hidden:
                if _as_int(w, 0) < 1:
                    errs.append(f"hidden width {w!r} must be >= 1")
        if cfg.get("out_activation") not in ("none",) + ACTIVATIONS:
            errs.append(f"unknown out_activation {cfg.get('out_activation')!r}")
    if cfg.get("dtype") not in DTYPES:
        errs.append(f"unknown dtype {cfg.get('dtype')!r} (choose from {sorted(set(DTYPES))})")
    return errs


def validate_run_cfg(run):
    """Full validation. Returns a list of human-readable error strings."""
    run = _backfill(run)
    errs = []
    errs += validate_model_cfg(run["model"])

    tok = run["tokenizer"]
    if tok.get("type") not in ("char", "bpe"):
        errs.append(f"unknown tokenizer type {tok.get('type')!r} (char | bpe)")
    tv = _as_int(tok.get("vocab_size"), 0)
    if tv < 2:
        errs.append(f"tokenizer vocab_size must be >= 2, got {tv}")
    if run["model"]["family"] == "gpt" and tv != _as_int(run["model"].get("vocab_size"), 0):
        errs.append(f"tokenizer vocab_size ({tv}) != model vocab_size "
                    f"({run['model'].get('vocab_size')}) - retrain the tokenizer "
                    f"or set both to the same value")

    data = run["data"]
    if data.get("task") not in TASKS:
        errs.append(f"unknown task {data.get('task')!r} (choose from {TASKS})")
    if data.get("task") == "text" and not data.get("hf_dataset") \
            and not data.get("data_dir"):
        errs.append("text task needs either hf_dataset or data_dir")
    if data.get("task") == "tabular" and not data.get("csv_path"):
        errs.append("tabular task needs csv_path")
    gb = _as_float(data.get("max_gb"), 0)
    if gb < 0:
        errs.append("max_gb must be >= 0 (0 = unlimited)")

    opt = run["optimizer"]
    if opt.get("name") not in OPTIMIZERS:
        errs.append(f"unknown optimizer {opt.get('name')!r} (choose from {OPTIMIZERS})")
    if _as_float(opt.get("weight_decay"), -1) < 0:
        errs.append("weight_decay must be >= 0")
    if _as_int(opt.get("accum"), 0) < 1:
        errs.append("accum must be >= 1")

    tr = run["train"]
    if _as_int(tr.get("steps"), 0) < 1:
        errs.append("steps must be >= 1")
    if _as_int(tr.get("batch_size"), 0) < 1:
        errs.append("batch_size must be >= 1")
    lr = _as_float(tr.get("lr"), -1)
    if not (0 < lr < 10):
        errs.append(f"lr must be in (0, 10), got {lr}")
    if tr.get("scheduler") not in SCHEDULERS:
        errs.append(f"unknown scheduler {tr.get('scheduler')!r} (choose from {SCHEDULERS})")
    wf = _as_float(tr.get("warmup_frac"), -1)
    if not (0 <= wf < 1):
        errs.append(f"warmup_frac must be in [0, 1), got {wf}")

    mem = run["memory"]
    if mem.get("sharding") not in ("none", "ddp", "fsdp", "zero2", "zero3"):
        errs.append(f"unknown sharding {mem.get('sharding')!r}")
    nd = _as_int(mem.get("num_devices"), 0)
    if nd < 1:
        errs.append("memory.num_devices must be >= 1")
    return errs


def _backfill(run):
    """Recursively add any missing default keys so old configs still run.

    Never mutates the caller's dict; returns a filled copy.
    """
    base = default_run_cfg()
    if not isinstance(run, dict):
        return base

    def fill(dst, src):
        for k, v in src.items():
            if k not in dst or dst[k] is None:
                dst[k] = copy.deepcopy(v)
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                fill(dst[k], v)
        return dst

    out = fill(copy.deepcopy(run), base)

    # coerce the stringly-typed values the GUI produces back into real types
    m = out["model"]
    for k in ("block_size", "n_layer", "n_head", "embed_dim", "ff_dim",
              "vocab_size", "n_features", "n_outputs"):
        m[k] = _as_int(m.get(k), base["model"][k])
    m["dropout"] = min(0.999, max(0.0, _as_float(m.get("dropout"), 0.1)))
    m["ff_mult"] = _as_float(m.get("ff_mult"), 4.0)
    m["tie_weights"] = bool(m.get("tie_weights", True))

    # `hidden` is a list in memory but the GUI shows it as one Entry, so it
    # comes back as "[4]" or "64, 64". Normalise here rather than in every
    # consumer - the GUI, the estimator and the model builder all need ints.
    h = m.get("hidden")
    if isinstance(h, str):
        h = h.strip().strip("[]")
        h = [p for p in h.replace(",", " ").split() if p]
    elif not isinstance(h, (list, tuple)):
        h = []
    m["hidden"] = [i for i in (_as_int(x, 0) for x in h) if i > 0]

    t = out["tokenizer"]
    t["vocab_size"] = _as_int(t.get("vocab_size"), 512)
    t["max_corpus_chars"] = _as_int(t.get("max_corpus_chars"), 256000)

    o = out["optimizer"]
    betas = o.get("betas")
    if not (isinstance(betas, (list, tuple)) and len(betas) == 2):
        o["betas"] = [0.9, 0.95]
    else:
        o["betas"] = [_as_float(betas[0], 0.9), _as_float(betas[1], 0.95)]
    o["accum"] = max(1, _as_int(o.get("accum"), 1))
    o["weight_decay"] = _as_float(o.get("weight_decay"), 0.01)
    o["max_grad_norm"] = _as_float(o.get("max_grad_norm"), 1.0)

    tr = out["train"]
    for k in ("steps", "batch_size", "eval_every", "eval_batches", "sample_every",
              "gen_len", "sample_top_k", "seed", "log_every", "ckpt_every"):
        tr[k] = _as_int(tr.get(k), base["train"][k])
    for k in ("lr", "warmup_frac", "min_lr_frac", "sample_temp", "ema"):
        tr[k] = _as_float(tr.get(k), base["train"][k])

    mem = out["memory"]
    mem["num_devices"] = max(1, _as_int(mem.get("num_devices"), 1))
    mem["max_logit_batch"] = max(0, _as_int(mem.get("max_logit_batch"), 0))

    return out


def load_run_cfg(path):
    """Read a run cfg from disk, backfilling any missing keys."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _backfill(json.load(fh))
    except FileNotFoundError:
        return default_run_cfg()


def save_run_cfg(run, path):
    """Write a run cfg next to its checkpoint, atomically."""
    tmp = path + ".tmp"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)
    os.replace(tmp, path)
    return path