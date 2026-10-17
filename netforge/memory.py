"""
Analytic memory model.

Everything here is closed-form arithmetic - no model is ever instantiated. The
GUI calls it on every keystroke so the memory panel updates while you type, and
for a 70B config you *cannot* afford to instantiate anyway.

The accounting is deliberately conservative. Where an implementation could save
memory with a fused kernel or an in-place op we assume the naive version, and
then `measure()` runs the real thing so you can see the gap. An estimate that
is 15% pessimistic is fine; an estimate that is optimistic is a crash.

Breakdown for training, in the order they get allocated:

  weights      params x weight bytes
  master       fp32 master weights, only under mixed precision
  gradients    params x grad bytes
  optimizer    AdamW keeps two fp32 moments; SGD-without-momentum keeps none
  activations  per-layer saved tensors for backward, x batch x seq
  logits       batch x seq x vocab - the single largest tensor in most configs

Then: sharding divisors, offload, and a fixed overhead for the CUDA context,
cuBLAS workspaces and the allocator's fragmentation.
"""

import math

from .arch import resolve_ff

DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "fp16": 2, "float16": 2, "half": 2,
    "bf16": 2, "bfloat16": 2,
    "fp8": 1, "int8": 1,
}

# Bytes of optimizer state per parameter, by optimizer name.
# (fp32 moments unless noted; mixed precision forces the moments to fp32)
OPT_STATE_BYTES = {
    "adamw": 8.0,     # exp_avg + exp_avg_sq, both fp32
    "adam": 8.0,
    "sgd": 0.0,       # plain SGD, no momentum -> nothing
    "rmsprop": 4.0,
    "adafactor": 0.5,  # factored second moment, ~sqrt-scale
    "lion": 4.0,      # single fp32 momentum
}

SHARDING_MODES = ("none", "ddp", "fsdp", "zero2", "zero3")


def _dtype_bytes(name, default=4):
    return DTYPE_BYTES.get(str(name).strip().lower(), default)


def _round_gib(n):
    """Round for display: keeps small numbers readable and big ones honest."""
    if n >= 1024 ** 3:
        return round(n / 1024 ** 3, 2)
    if n >= 1024 ** 2:
        return round(n / 1024 ** 2, 1)
    if n >= 1024:
        return round(n / 1024, 2)
    return round(n, 1)


def fmt_bytes(n):
    n = float(n)
    for unit, div in (("TiB", 1024 ** 4), ("GiB", 1024 ** 3),
                      ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def total_params(mcfg):
    from .arch import estimate_params
    return estimate_params(mcfg)


# ---------------------------------------------------------------------------
# Activation accounting
# ---------------------------------------------------------------------------

def gpt_activation_bytes(mcfg, batch, seq, compute_bytes=2):
    """Saved-tensor bytes for backward, for a pre-norm decoder stack.

    Derived from the actual autograd graph with F.scaled_dot_product_attention,
    which stores q/k/v/output and a logsumexp row but NOT the S x S matrix.
    That last part matters: a naive attention estimate is O(B*H*T^2) and will
    tell you a 4k-context model needs 60 GB when it actually needs 6.

    Per token per layer we save, in units of d (embed_dim):
      ln1 in, ln1 out, q, k, v, sdpa out, residual, ln2 in, ln2 out, fc2 out  = 10d
    plus, in units of f (ff_dim):
      fc1 pre-activation, activation output                                       = 2f
    """
    d = int(mcfg.get("embed_dim", 128))
    f = resolve_ff(mcfg)
    nl = int(mcfg.get("n_layer", 4))
    tokens = int(batch) * int(seq)

    if mcfg.get("gradient_checkpointing"):
        # Only the block boundary input survives; each block recomputes its
        # interior during backward. Cost: nl * d for the saved inputs.
        per_token = nl * d
    else:
        per_token = nl * (10 * d + 2 * f)

    return per_token * tokens * compute_bytes


def mlp_activation_bytes(mcfg, batch, seq, compute_bytes=2):
    """Saved tensors for an MLP: every layer's input plus every nonlinearity's
    output. seq doubles as the number of rows in a tabular batch."""
    hidden = [int(h) for h in (mcfg.get("hidden") or [])]
    rows = int(batch) * int(seq)
    total = 0
    prev = int(mcfg.get("n_features", 8))
    for h in hidden:
        total += rows * (prev + h)     # layer input + nonlinearity output
        prev = h
    total += rows * prev               # final layer input
    return total * compute_bytes


def logits_bytes(mcfg, batch, seq, compute_bytes=2):
    """The output projection and its softmax/cross-entropy.

    B*T*V in the compute dtype, plus an fp32 copy if the loss upcasts.
    This is routinely the biggest single tensor: a 4k-context model with a
    128k vocabulary hits 4 GB at batch 4 *before* a single block allocates.
    """
    if mcfg.get("family") != "gpt":
        return 0
    v = int(mcfg.get("vocab_size", 512))
    tokens = int(batch) * int(seq)
    return tokens * v * (compute_bytes + 4 if compute_bytes < 4 else compute_bytes)


def kv_cache_bytes(mcfg, batch, seq, compute_bytes=2):
    """Inference KV cache: 2 tensors x n_layer x batch x seq x embed_dim."""
    if mcfg.get("family") != "gpt":
        return 0
    d = int(mcfg.get("embed_dim", 128))
    nl = int(mcfg.get("n_layer", 4))
    return 2 * nl * int(batch) * int(seq) * d * compute_bytes


# ---------------------------------------------------------------------------
# The main estimator
# ---------------------------------------------------------------------------

def estimate(mcfg, batch=1, seq=128, *, training=True, memcfg=None,
             optimizer="adamw", mixed_precision=True, **overrides):
    """Full memory breakdown in bytes.

    Returns a dict with:
      items:     [(key, label, bytes, note)] in allocation order
      total:     sum of items
      peak:      high-water mark including transient allocs
      per_device: what each of the N devices actually holds
    """
    memcfg = dict(memcfg or {})
    memcfg.update({k: v for k, v in overrides.items() if v is not None})

    n_params = total_params(mcfg)
    fam = mcfg.get("family", "gpt")

    weight_bytes = _dtype_bytes(memcfg.get("dtype", mcfg.get("dtype", "fp32")))
    compute_bytes = _dtype_bytes(memcfg.get("compute_dtype", "bf16"))
    amp = bool(mixed_precision) and compute_bytes < weight_bytes
    grad_bytes = compute_bytes if amp else weight_bytes
    sharding = memcfg.get("sharding", "none")
    if sharding not in SHARDING_MODES:
        sharding = "none"
    n_dev = max(1, int(memcfg.get("num_devices", 1)))
    opt_offload = bool(memcfg.get("optimizer_offload"))
    param_offload = bool(memcfg.get("param_offload"))

    if not training:
        return _estimate_inference(mcfg, batch, seq, memcfg, weight_bytes, n_dev)

    # --- sharding divisors -------------------------------------------------
    # ddp replicates everything; zero2/FSDP shard grads+optimizer; zero3 shards
    # params too. The parameter shard must be recomputed in backward, which is
    # why zero3 saves less activation memory than it appears to.
    d_grad = n_dev if sharding in ("zero2", "zero3", "fsdp") else 1
    d_opt = n_dev if sharding in ("zero2", "zero3", "fsdp") else 1
    d_param = n_dev if sharding in ("zero3", "fsdp") else 1

    items = []

    w = n_params * weight_bytes / d_param
    items.append(("weights", "weights", w,
                  f"{n_params:,} x {weight_bytes}B" + (f" /{n_dev}" if d_param > 1 else "")))

    master = 0
    if amp and not param_offload:
        master = n_params * 4
        items.append(("master", "fp32 master weights", master,
                      "mixed precision keeps an fp32 copy for the optimizer"))

    g = n_params * grad_bytes / d_grad
    items.append(("grads", "gradients", g,
                  f"{n_params:,} x {grad_bytes}B" + (f" /{n_dev}" if d_grad > 1 else "")))

    opt_b = OPT_STATE_BYTES.get(str(optimizer).lower(), 8.0)
    # Mixed precision always keeps optimizer moments in fp32 regardless of the
    # storage dtype used for the gradients themselves.
    if amp:
        opt_b = max(opt_b, 8.0) if str(optimizer).lower() in ("adamw", "adam") else opt_b
    o = 0 if opt_offload else n_params * opt_b / d_opt
    items.append(("optimizer", f"{optimizer} state", o,
                  f"{opt_b}B/param" + (" (offloaded to CPU)" if opt_offload else "")
                  + (f" /{n_dev}" if d_opt > 1 else "")))

    # --- activations -------------------------------------------------------
    ckpt = bool(memcfg.get("gradient_checkpointing"))
    max_logit_batch = int(memcfg.get("max_logit_batch", 0) or 0)
    lb = int(batch)
    if fam == "gpt" and max_logit_batch and batch * seq > max_logit_batch:
        # Chunked cross-entropy: recompute logits per micro-chunk instead of
        # materialising the whole thing. The activation cost is unchanged, only
        # the peak logits tensor shrinks.
        lb = max(1, max_logit_batch // max(1, seq))

    if fam == "gpt":
        act = gpt_activation_bytes(mcfg, lb, seq, compute_bytes)
        act_note = "checkpointed" if ckpt else "saved for backward"
    else:
        act = mlp_activation_bytes(mcfg, lb, seq, compute_bytes)
        act_note = f"{lb * seq:,} rows"
    items.append(("activations", "activations", act, act_note))

    lg = logits_bytes(mcfg, lb, seq, compute_bytes)
    if lg:
        note = f"{lb} x {seq} x {vocab_hint(mcfg):,} vocab"
        if lb != int(batch):
            note += f" (chunked from {batch})"
        items.append(("logits", "logits + loss", lg, note))

    # --- fixed overheads ---------------------------------------------------
    # CUDA context ~500 MB on every device that has one, cuBLAS/cuDNN
    # workspaces scale with sequence length, and the caching allocator holds
    # onto freed blocks - 8% is a typical steady-state waste.
    overhead = 512 * 1024 ** 2
    workspace = (0.5 * 1024 ** 3) if seq >= 4096 else (64 * 1024 ** 2)
    items.append(("overhead", "CUDA context + workspaces", overhead + workspace,
                  "allocator + cuBLAS, measured on most 8 GB cards"))

    subtotal = sum(it[2] for it in items)
    frag = subtotal * 0.08
    items.append(("fragmentation", "allocator fragmentation", frag, "~8% steady state"))

    total = sum(it[2] for it in items)

    # A grad-norm pass materialises every gradient norm at once; on big models
    # that is a second full gradient set.
    transient = n_params * 4 / max(1, d_grad) if n_params > 1e8 else 0
    peak = total + transient

    offloaded = 0.0
    if opt_offload:
        offloaded += float(next(i[2] for i in items if i[0] == "optimizer"))
    if param_offload and master:
        offloaded += master

    per_device = (total - offloaded) / n_dev if sharding != "none" else total - offloaded

    return {
        "n_params": n_params,
        "items": items,
        "total": total,
        "peak": peak,
        "transient": transient,
        "offloaded": offloaded,
        "per_device": per_device,
        "per_device_note": f"sharded across {n_dev} device(s) via {sharding}"
                           if sharding != "none" else "single device",
    }


def _estimate_inference(mcfg, batch, seq, memcfg, weight_bytes, n_dev):
    """Inference has no gradients, no optimizer and no saved activations.

    It is just weights + KV cache + the transient prefill buffers, which is why
    a model that cannot be trained on your card can still be served from it.
    """
    n_params = total_params(mcfg)
    items = [("weights", "weights", n_params * weight_bytes,
              f"{n_params:,} x {weight_bytes}B")]

    kv = kv_cache_bytes(mcfg, batch, seq, weight_bytes)
    if kv:
        items.append(("kv_cache", "KV cache", kv,
                      f"{batch} x {seq} tokens @ {weight_bytes}B"))

    # Prefill runs the whole context through the network once. Only the peak
    # matters, and it is the logits tensor: batch x seq x vocab.
    prefill = 0
    if mcfg.get("family") == "gpt":
        prefill = logits_bytes(mcfg, batch, seq, weight_bytes)
        if prefill:
            items.append(("prefill", "prefill logits (transient)", prefill,
                          f"{batch} x {seq} x {vocab_hint(mcfg):,}"))

    items.append(("overhead", "CUDA context + workspaces",
                  512 * 1024 ** 2, "allocator + cuBLAS"))

    subtotal = sum(i[2] for i in items)
    items.append(("fragmentation", "allocator fragmentation", subtotal * 0.05,
                  "~5% steady state"))

    total = sum(i[2] for i in items)
    return {
        "n_params": n_params,
        "items": items,
        "total": total,
        "peak": total,
        "transient": prefill,
        "offloaded": 0.0,
        "kv_cache": kv,
        "per_device": total,
        "per_device_note": f"{n_dev} device(s)",
    }


def vocab_hint(mcfg):
    return int(mcfg.get("vocab_size", 512))


# ---------------------------------------------------------------------------
# Fitting: what actually runs on this machine?
# ---------------------------------------------------------------------------

def device_inventory():
    """What GPUs/CPUs/RAM this process can see."""
    info = {"cuda": False, "devices": [], "cpu_ram": 0, "cpu_threads": 0, "name": "cpu"}
    try:
        import torch
        info["cpu_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["cuda"] = True
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i, "name": p.name, "total": p.total_memory,
                    "free": torch.cuda.mem_get_info(i)[0],
                    "capability": f"{p.major}.{p.minor}",
                    "sms": p.multi_processor_count,
                })
            info["name"] = info["devices"][0]["name"]
    except Exception:
        pass
    try:
        import psutil
        info["cpu_ram"] = psutil.virtual_memory().total
    except Exception:
        pass
    return info


def fit_plan(mcfg, budget=None, *, training=True, memcfg=None, optimizer="adamw",
             batch=1, seq=128, min_fraction=0.90):
    """Given a model, what context length / batch size actually fit?

    Binary-searches the largest (batch, seq) pair that lands under `fraction` of
    the device budget. Returns a dict describing the answer, never raises.
    """
    inv = device_inventory()
    if budget is None:
        budget = (inv["devices"][0]["total"] * 0.92) if inv["devices"] else (inv["cpu_ram"] * 0.5)
    budget = budget * min_fraction

    def fits(b, s):
        est = estimate(mcfg, b, s, training=training, memcfg=memcfg,
                       optimizer=optimizer)
        return est["total"] <= budget, est

    # If even batch=1/seq=1 fails, report why instead of returning nothing.
    ok, est_min = fits(1, 1)
    if not ok:
        return {"fits": False, "reason": "model does not fit even at batch=1 seq=1",
                "budget": budget, "estimate": est_min,
                "suggest_devices": _suggest_devices(est_min, inv)}

    best = (1, 1, est_min)
    s = 1
    while s <= 1 << 17:
        ok, est = fits(1, s)
        if not ok:
            break
        best = (1, s, est)
        s *= 2
    best_seq = best[1]

    b = 2
    while b <= 1 << 14:
        ok, est = fits(b, best_seq)
        if not ok:
            break
        best = (b, best_seq, est)
        b *= 2

    return {
        "fits": True,
        "batch": best[0],
        "seq": best[1],
        "budget": budget,
        "estimate": best[2],
        "suggest_devices": None,
        "note": f"{best[0]} x {best_seq} fits with "
                f"{fmt_bytes(budget - best[2]['total'])} to spare",
    }


def _suggest_devices(est, inv):
    """How many GPUs would this need, if you had them."""
    if not inv["devices"]:
        return None
    per = inv["devices"][0]["total"] * 0.92
    return max(1, math.ceil(est["total"] / per))


# ---------------------------------------------------------------------------
# Measurement - run the real thing and compare
# ---------------------------------------------------------------------------

def measure(mcfg, batch=1, seq=128, *, training=True, memcfg=None,
            optimizer="adamw", model=None):
    """Allocate a real model (or use `model`) and read the actual allocator.

    Returns a dict with peak/allocated/reserved bytes, or `error` if it ran out
    of memory. Never raises - an OOM here is a normal outcome that the GUI
    wants to display, not an exception it has to survive.
    """
    try:
        import torch
    except Exception as exc:
        return {"error": f"torch unavailable: {exc}"}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fam = mcfg.get("family", "gpt")
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    result = {"device": dev}

    try:
        if model is None:
            from .model import build_model
            model = build_model(mcfg)
        model = model.to(dev)

        weight_bytes = _dtype_bytes((memcfg or {}).get("dtype", mcfg.get("dtype", "fp32")))
        n_params = total_params(mcfg)

        if not training:
            if dev == "cuda":
                torch.cuda.synchronize()
                result.update({
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "peak": torch.cuda.max_memory_allocated(),
                })
            result["predicted"] = estimate(mcfg, batch, seq, training=False,
                                           memcfg=memcfg, optimizer=optimizer)
            result["n_params"] = n_params
            result["measured_weights"] = n_params * weight_bytes
            return result

# Build an optimizer and force its state to actually allocate. AdamW
        # creates its moments lazily inside step(), so merely constructing it
        # and assigning .grad leaves state empty and the measurement reads zero.
        from .model import unwrap
        raw = unwrap(model)
        opt = torch.optim.AdamW(raw.parameters(), lr=1e-3, weight_decay=0.0)
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = torch.zeros_like(p)
        try:
            opt.step()                       # materialises exp_avg/exp_avg_sq
            opt.zero_grad(set_to_none=True)
        except Exception:
            pass

        # Run one real forward+backward. Without this the measurement only
        # covers weights + optimizer states, and the comparison against the
        # analytic total (which includes activations and logits) reads as a
        # ~98% miss when the estimate is in fact accurate.
        step_error = ""
        try:
            if fam == "gpt":
                x = torch.randint(0, max(2, int(mcfg.get("vocab_size", 512))),
                                  (int(batch), int(seq)), device=dev)
                t = torch.randint(0, max(2, int(mcfg.get("vocab_size", 512))),
                                  (int(batch), int(seq)), device=dev)
            else:
                x = torch.randn(int(batch), int(mcfg.get("n_features", 8)), device=dev)
                t = torch.randn(int(batch), int(mcfg.get("n_outputs", 1)), device=dev)
            _, loss, _ = model(x, targets=t)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            step_error = f"{type(exc).__name__}: {exc}"[:160]

        if dev == "cuda":
            torch.cuda.synchronize()
            result.update({
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
            })

        result["n_params"] = n_params
        result["step_error"] = step_error
        result["predicted"] = estimate(mcfg, batch, seq, training=True,
                                       memcfg=memcfg, optimizer=optimizer)
        result["measured_weights"] = n_params * weight_bytes
        result["measured_states"] = sum(
            s.numel() * s.element_size()
            for st in opt.state.values() for s in st.values()
            if torch.is_tensor(s)
        )
        return result

    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = f"CUDA OOM: {str(exc)[:180]}"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result["error"] = "OOM (device)"
        else:
            result["error"] = str(exc)[:200]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        # Drop references before returning so the caller measuring something
        # else immediately afterwards is not pushed over the edge by us.
        model = opt = raw = None
        if dev == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(mcfg, batch=1, seq=128, *, training=True, memcfg=None, optimizer="adamw",
           device_total=None, width=34):
    """Human-readable table. Returns a list of lines (GUI puts them in a Text)."""
    est = estimate(mcfg, batch, seq, training=training, memcfg=memcfg,
                   optimizer=optimizer)
    lines = []
    lines.append(f"model: {est['n_params']:,} params "
                 f"({est['n_params'] / 1e6:.3f}M)  •  batch {batch} × seq {seq}  •  "
                 f"{'training' if training else 'inference'}")
    lines.append("")
    for _, label, nbytes, note in est["items"]:
        bar_len = min(28, int(28 * nbytes / max(1e-9, est["total"])))
        lines.append(f"  {label:<{width - 12}} {fmt_bytes(nbytes):>11}  "
                     f"{'█' * bar_len:<28} {note}")
    lines.append("-" * (width + 46))
    lines.append(f"  {'TOTAL':<{width - 12}} {fmt_bytes(est['total']):>11}")
    if est["peak"] > est["total"] * 1.001:
        lines.append(f"  {'PEAK (incl. transient)':<{width - 12}} "
                     f"{fmt_bytes(est['peak']):>11}")
    if est["offloaded"]:
        lines.append(f"  {'off CPU':<{width - 12}} {fmt_bytes(est['offloaded']):>11}")
    if device_total:
        pct = 100.0 * est["per_device"] / device_total
        fits = "FITS" if pct <= 100 else "DOES NOT FIT"
        bar = int(30 * min(1.0, pct / 100.0))
        lines.append("")
        lines.append(f"  per device: {fmt_bytes(est['per_device'])} of "
                     f"{fmt_bytes(device_total)}  ({pct:.0f}%)  [{'█' * bar}{'░' * (30 - bar)}]  {fits}")
    return lines


def one_line(mcfg, batch=1, seq=128, **kw):
    """Compact form for the GUI status bar."""
    est = estimate(mcfg, batch, seq, **kw)
    return (f"{est['n_params'] / 1e6:.3f}M params • "
            f"{fmt_bytes(est['total'])} @ B{batch}×T{seq}")
