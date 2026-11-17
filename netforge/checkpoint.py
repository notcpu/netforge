"""
Checkpointing: atomic, self-describing, resumable.

Writes go to a temp file, are fsynced, copied to `<path>.bak`, and only then
os.replace'd into position. A power cut mid-save leaves you with either the old
checkpoint or the new one, never a truncated third option - which matters when
the alternative is losing a 70B run's optimizer state.

Back-compat: `resolve_model_cfg` maps old key names (n_embd -> embed_dim,
num_layers -> n_layer) and derives n_head if it is missing, so checkpoints
written by earlier versions still load.
"""

import os
import shutil
import tempfile

import torch

LEGACY_KEYS = {
    "n_embd": "embed_dim",
    "num_layers": "n_layer",
    "num_heads": "n_head",
    "context": "block_size",
    "d_model": "embed_dim",
    "norm": "norm_type",
    "act": "activation",
    "ff": "ff_dim",
}


def resolve_model_cfg(cfg):
    """Normalise a stored model cfg to the current key names."""
    if not isinstance(cfg, dict):
        return cfg
    out = dict(cfg)
    for old, new in LEGACY_KEYS.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    if "embed_dim" in out and "n_head" not in out:
        dim = int(out["embed_dim"])
        for h in (16, 8, 4, 2, 1):
            if dim % h == 0 and dim // h >= 8:
                out["n_head"] = h
                break
    out.setdefault("family", "gpt")
    out.setdefault("dtype", "fp32")
    return out


def _atomic_save(payload, path):
    """Write a torch save atomically, keeping one generation of backup.

    The fsync must open the temp file r+b, not rb: on Windows FlushFileBuffers
    fails on a read-only handle, and that failure would abort every save while
    looking like nothing at all happened.
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "r+b") as fh:      # flush to disk before the rename
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return False


def save_ckpt(path, model, optimizer=None, scheduler=None, step=0, run_cfg=None,
              tokenizer=None, loss_history=None, extra=None):
    """Save everything needed to resume. Never raises.

    Also writes a `<path>.meta.json` sidecar. Reading the metadata should never
    require opening the tensor file: `torch.load(weights_only=True)` rejects
    the config dicts and RNG tuples this payload contains, and `weights_only
    =False` on a file from an untrusted source executes code. A plain JSON
    sidecar is both safe and instant.
    """
    from .model import unwrap
    try:
        m = unwrap(model)
        payload = {
            "format": 2,
            "step": int(step),
            "model": m.state_dict(),
            "model_cfg": getattr(m, "cfg", None),
            "run_cfg": run_cfg,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "tokenizer": tokenizer.to_config() if tokenizer is not None else None,
            "loss_history": list(loss_history or [])[-2000:],
            "rng": capture_rng(),
            "extra": extra or {},
        }
        ok = _atomic_save(payload, path)
        if ok:
            seen, total = set(), 0
            for t in (payload["model"] or {}).values():
                if torch.is_tensor(t):
                    key = (t.data_ptr(), t.numel())
                    if key in seen:
                        continue
                    seen.add(key)
                    total += t.numel()
            _write_sidecar(path, {
                "step": int(step),
                "n_params": int(total),
                "model_cfg": payload["model_cfg"],
                "tokenizer": payload["tokenizer"],
                "format": 2,
                "bytes": os.path.getsize(path),
            })
        return ok
    except Exception:
        return False


def _sidecar_path(path):
    return os.path.splitext(path)[0] + ".meta.json"


def _write_sidecar(path, meta):
    import json
    try:
        tmp = _sidecar_path(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, default=str)
        os.replace(tmp, _sidecar_path(path))
    except Exception:
        pass


def peek_tokenizer(path):
    """The tokenizer config stored in a checkpoint, without loading tensors.

    Reads the `.meta.json` sidecar when present (new checkpoints); falls back
    to a `weights_only=True` mmap load for older files whose sidecar predates
    the tokenizer entry. Returns None when there is nothing usable - never
    raises.
    """
    side = _sidecar_path(path)
    if os.path.exists(side):
        try:
            import json
            with open(side, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            tok = meta.get("tokenizer")
            if isinstance(tok, dict):
                return tok
        except Exception:
            pass
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True,
                             mmap=True)
        tok = (payload or {}).get("tokenizer")
        return tok if isinstance(tok, dict) else None
    except Exception:
        return None


def load_ckpt(path, map_location="cpu"):
    """Load a checkpoint. Tries `<path>` then `<path>.bak`.

    Returns a dict, or None if nothing loadable was found - never raises, so a
    corrupt file can be reported in the GUI instead of killing the process.
    """
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            obj = torch.load(candidate, map_location=map_location, weights_only=False)
            if isinstance(obj, dict):
                obj["_path"] = candidate
                return obj
        except Exception:
            continue
    return None


def restore_rng(ckpt):
    """Put the RNG back where it was. Best-effort."""
    rng = (ckpt or {}).get("rng")
    if not isinstance(rng, dict):
        return
    try:
        torch.set_rng_state(rng["torch"])
    except Exception:
        pass
    try:
        import random
        random.setstate(rng["python"])
    except Exception:
        pass
    try:
        import numpy as np
        np.random.set_state(rng["numpy"])
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass


def capture_rng():
    out = {}
    try:
        import random
        out["python"] = random.getstate()
    except Exception:
        pass
    try:
        out["torch"] = torch.get_rng_state()
    except Exception:
        pass
    try:
        import numpy as np
        out["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return out


def ckpt_info(path):
    """Cheap summary for the GUI's "what's in this file" panel.

    Reads the JSON sidecar when present - no tensor deserialisation, no code
    execution risk, and instant even for a 70B checkpoint. Falls back to a
    `weights_only=True` load of the tensor file when the sidecar is missing
    (a checkpoint written by an older version).
    """
    import json
    info = {"exists": os.path.exists(path), "path": path, "size": 0,
            "step": 0, "n_params": None, "model_cfg": None, "ok": False,
            "error": "", "source": "none"}
    if not info["exists"]:
        info["error"] = "no such file"
        return info
    info["size"] = os.path.getsize(path)

    side = _sidecar_path(path)
    if os.path.exists(side):
        try:
            with open(side, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            info["step"] = int(meta.get("step", 0) or 0)
            info["n_params"] = int(meta.get("n_params", 0) or 0) or None
            info["model_cfg"] = resolve_model_cfg(meta.get("model_cfg"))
            info["size"] = int(meta.get("bytes", info["size"]) or info["size"])
            info["source"] = "sidecar"
            info["ok"] = True
            return info
        except Exception as exc:
            info["error"] = f"sidecar unreadable ({type(exc).__name__})"

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True,
                             mmap=True)
        info["step"] = int(payload.get("step", 0) or 0)
        info["model_cfg"] = resolve_model_cfg(payload.get("model_cfg"))
        sd = payload.get("model")
        if isinstance(sd, dict):
            seen, total = set(), 0
            for t in sd.values():
                if torch.is_tensor(t):
                    key = (t.data_ptr(), t.numel())
                    if key in seen:
                        continue
                    seen.add(key)
                    total += t.numel()
            info["n_params"] = total
        info["source"] = "payload"
        info["ok"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:160]
    return info


def resume_into(model, optimizer, ckpt, strict=True):
    """Load weights (and optimizer state if present) into a live model.

    If the vocabulary changed, the embedding and head are dropped and
    re-initialised while every other tensor still resumes - a token-level
    mismatch should not cost you the whole run.
    """
    if not ckpt:
        return {"ok": False, "dropped": []}
    dropped = []
    try:
        sd = ckpt.get("model") or {}
        new_cfg = getattr(model, "cfg", {})
        old_cfg = resolve_model_cfg(ckpt.get("model_cfg") or {})
        vocab_changed = ("vocab_size" in new_cfg and "vocab_size" in old_cfg and
                         int(new_cfg["vocab_size"]) != int(old_cfg["vocab_size"]))
        if vocab_changed:
            for k in list(sd.keys()):
                if any(t in k for t in ("tok_emb", "pos_emb", "head")):
                    dropped.append(k)
                    sd.pop(k)
        model.load_state_dict(sd, strict=strict and not dropped)
        if optimizer is not None and ckpt.get("optimizer"):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        return {"ok": True, "dropped": dropped}
    except Exception as exc:
        return {"ok": False, "dropped": dropped, "error": str(exc)[:200]}