"""
Device / AMP / data-placement helpers.

Kept separate from training.py so the CLI, the GUI and any standalone script
resolve the device the same way. Every function swallows exceptions: a trainer
that dies because a log line failed is worse than one with a missing stat.
"""

import os


def pick_device(name="auto"):
    """Resolve a device string to something torch.cuda.is_available() agrees with."""
    try:
        import torch
    except Exception:
        return "cpu"
    if name and name != "auto":
        if name.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and \
            torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(dev):
    try:
        import torch
        if str(dev).startswith("cuda") and torch.cuda.is_available():
            i = torch.cuda.current_device()
            return f"{torch.cuda.get_device_name(i)}"
        return str(dev)
    except Exception:
        return str(dev)


def amp_dtype_for(dev):
    """bf16 on Ampere+, fp16 everywhere else. bf16 has fp32's exponent range,
    which removes the need for loss scaling - but it does not exist pre-A100."""
    try:
        import torch
        if not str(dev).startswith("cuda"):
            return None
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        return None


def autocast_ctx(dev, enabled=True, dtype=None):
    """Return (context manager, dtype, scaler).

    `scaler` is None when there is nothing to scale - callers must test it for
    truthiness, because `if scaler is not None` is True for False and would
    call .scale() on a bool.
    """
    import contextlib
    try:
        import torch
    except Exception:
        return contextlib.nullcontext(), None, None
    if not enabled or not str(dev).startswith("cuda"):
        return contextlib.nullcontext(), None, None
    dt = dtype or amp_dtype_for(dev)
    # a GradScaler is only needed for fp16; bf16 has fp32's exponent range
    scaler = torch.amp.GradScaler("cuda") if dt == torch.float16 else None
    return torch.amp.autocast("cuda", dtype=dt), dt, scaler


def set_threads(n):
    """n=0 means 'leave it alone'. Raises nothing if torch is unhappy."""
    if not n or int(n) <= 0:
        return
    n = int(n)
    try:
        import torch
        torch.set_num_threads(min(n, os.cpu_count() or n))
    except Exception:
        pass


def gpu_mem(dev):
    """(allocated, reserved, peak, total) in bytes. Zeros when not on CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0, 0, 0, 0
        return (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                torch.cuda.max_memory_allocated(),
                torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0, 0, 0, 0


def maybe_pin_to_device(data, dev, enabled=True, max_mb=512.0):
    """Move a token tensor to GPU memory if it fits under `max_mb`.

    A 1 GB corpus pinned to VRAM is a good trade on a 24 GB card and a way to
    get an instant OOM on an 8 GB one, hence the cap.
    """
    if not enabled or not str(dev).startswith("cuda") or data is None:
        return data
    try:
        nbytes = data.numel() * data.element_size()
        if nbytes > max_mb * 1024 ** 2:
            return data
        return data.to(dev)
    except Exception:
        return data


def sync(dev):
    try:
        import torch
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass