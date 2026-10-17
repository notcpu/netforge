"""
The training loop.

One implementation, one entry point: `train_run(run)`. The CLI calls it in a
subprocess, the GUI calls it in a daemon thread, and both get identical
behaviour because everything it needs arrives in the `run` dict.

Every iteration publishes to `SharedState` under one lock acquisition. That is
the entire interface to the GUI - the loop never imports tkinter and the GUI
never imports this module's internals.

The loop branches in exactly one place: `get_batch` vs `get_tab_batch`. From
the optimizer step onward the two families are indistinguishable, which is why
a 40-parameter MLP and a 7B transformer can share a trainer.
"""

import os
import time
import traceback

import torch

from . import arch as arch_mod
from . import dataset as ds_mod
from . import memory as mem_mod
from .checkpoint import load_ckpt, restore_rng, resume_into, save_ckpt
from .config import validate_run_cfg
from .device_utils import (autocast_ctx, device_name, gpu_mem, maybe_pin_to_device,
                           pick_device, set_threads, sync)
from .logging_utils import append_debug, append_error, append_log
from .lr_schedule import SchedulerState
from .model import build_model, unwrap
from .state import state

SAMPLE_PROMPTS = ["The", "Once upon", "Hello", "In the", "def ",
                  "### Instruction:\n", "Once", "She"]


# ---------------------------------------------------------------------------

def build_optimizer(model, ocfg):
    name = str(ocfg.get("name", "adamw")).lower()
    wd = float(ocfg.get("weight_decay", 0.01))
    betas = tuple(ocfg.get("betas", [0.9, 0.95]))
    eps = float(ocfg.get("eps", 1e-8))
    fused = bool(ocfg.get("fused", True))

    # weight decay should not apply to norm weights and biases
    decay, no_decay = [], []
    for n, p in unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]

    try:
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=1e-3, betas=betas, eps=eps,
                                     fused=fused and torch.cuda.is_available())
        if name == "adam":
            return torch.optim.Adam(groups, lr=1e-3, betas=betas, eps=eps)
        if name == "sgd":
            return torch.optim.SGD(groups, lr=1e-3, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(groups, lr=1e-3, alpha=0.95)
        if name == "lion":
            try:
                return torch.optim.Lion(groups, lr=1e-3, betas=betas)
            except AttributeError:
                append_error("lion not available in this torch; using adamw")
                return torch.optim.AdamW(groups, lr=1e-3, betas=betas, eps=eps)
        if name == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(groups, lr=1e-3, relative_step=False,
                                 scale_parameter=False, warmup_init=False)
            except Exception:
                append_error("adafactor needs transformers; using adamw")
                return torch.optim.AdamW(groups, lr=1e-3, betas=betas, eps=eps)
    except Exception as exc:
        append_error(f"optimizer {name!r} unavailable ({exc}); falling back to adamw")
    return torch.optim.AdamW(groups, lr=1e-3)


def grad_norm_of(model):
    total = 0.0
    for p in unwrap(model).parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm(2) ** 2)
    return total ** 0.5


# ---------------------------------------------------------------------------

def train_run(run, exception_queue=None, on_done=None):
    """Run training to completion. Publishes to SharedState throughout.

    Returns a summary dict. Puts the traceback on `exception_queue` and in the
    shared state's `error` field rather than raising, so a crashed run shows
    up in the GUI instead of vanishing into a thread.
    """
    from . import logging_utils

    try:
        return _train_impl(run, on_done=on_done)
    except Exception:
        tb = traceback.format_exc()
        append_error(tb.strip().splitlines()[-1] if tb else "unknown error")
        if exception_queue is not None:
            exception_queue.put(tb)
        state.fail(tb.strip().splitlines()[-1])
        state.publish(status="error")
        with open("netforge_error.log", "w", encoding="utf-8") as fh:
            fh.write(tb)
        return {"ok": False, "error": tb}
    finally:
        logging_utils.TUI_ACTIVE = False


def _train_impl(run, on_done=None):
    run = dict(run)
    errs = validate_run_cfg(run)
    if errs:
        raise ValueError("; ".join(errs))

    tcfg = run.get("train", {})
    ocfg = run.get("optimizer", {})
    mcfg = dict(run.get("model", {}))
    memcfg = run.get("memory", {})
    rcfg = run.get("runtime", {})

    torch.manual_seed(int(tcfg.get("seed", 1337)))
    dev = pick_device(rcfg.get("device", "auto"))
    set_threads(rcfg.get("max_threads", 0))

    state.publish(status="loading data", device=device_name(dev),
                  total_steps=int(tcfg.get("steps", 2000)))

    # Progress callbacks in dataset.py pass different arg tuples depending on the
# source (rows/bytes/tokens/elapsed for HF streams, counts for local files).
# One permissive signature here means adding a source never breaks the caller.
    def progress(*a):
        if len(a) >= 3:
            state.log(f"[data] {int(a[0]):,} rows · "
                      f"{a[1] / 1024 ** 2:,.0f} MB · {int(a[2]):,} tokens")
        elif a:
            state.log(f"[data] {a[0]}")

    bundle = ds_mod.prepare(run, log=state.log, progress_cb=progress)

    # The corpus may have trained a tokenizer that changed the vocab size.
    mcfg.update(bundle.get("model_cfg_override") or {})
    is_text = bundle["kind"] == "text"

    # ---- model ------------------------------------------------------------
    state.publish(status="building model")
    model = build_model(mcfg).to(dev)
    raw_params = arch_mod.estimate_params(mcfg)
    actual = model.n_params
    state.publish(n_params=actual, model_name=describe_model(mcfg, actual))

    if abs(actual - raw_params) > max(8, 0.001 * max(1, actual)):
        append_debug(f"analytic estimate {raw_params:,} != actual {actual:,} "
                     f"({100 * (actual - raw_params) / max(1, actual):+.2f}%)")

    if not is_text:
        # an MLP's input width must match the data
        nf = int(bundle["X"].shape[1])
        if int(mcfg.get("n_features", nf)) != nf:
            append_debug(f"n_features {mcfg.get('n_features')} -> {nf} (from dataset)")
            mcfg["n_features"] = nf
            model = build_model(mcfg).to(dev)
            state.publish(n_params=model.n_params)

    if rcfg.get("compile"):
        try:
            model = torch.compile(model)
            append_log("torch.compile enabled")
        except Exception as exc:
            append_error(f"torch.compile failed: {exc}")

    # ---- memory report ----------------------------------------------------
    bs = int(tcfg.get("batch_size", 32))
    seq = int(mcfg.get("block_size", 128)) if is_text else 1
    est = mem_mod.estimate(mcfg, bs, seq, training=True, memcfg=memcfg,
                           optimizer=ocfg.get("name", "adamw"),
                           mixed_precision=bool(rcfg.get("amp", True)))
    state.publish(mem_estimated=int(est["total"]),
                  mem_detail=mem_mod.fmt_bytes(est["total"]))

    # ---- optimizer / schedule --------------------------------------------
    opt = build_optimizer(model, ocfg)
    steps = int(tcfg.get("steps", 2000))
    lr = float(tcfg.get("lr", 3e-3))
    sched = SchedulerState(scheduler=tcfg.get("scheduler", "cosine"), lr_max=lr,
                           total_steps=steps,
                           warmup_frac=float(tcfg.get("warmup_frac", 0.05)),
                           min_lr_frac=float(tcfg.get("min_lr_frac", 0.05)))

    start_step = 0
    ckpt_path = rcfg.get("checkpoint", "netforge_run.pt")
    if rcfg.get("resume", True) and os.path.exists(ckpt_path):
        ck = load_ckpt(ckpt_path)
        if ck:
            r = resume_into(model, opt, ck)
            if r.get("ok"):
                start_step = int(ck.get("step", 0))
                if ck.get("scheduler_state"):
                    sched = SchedulerState.from_state(ck["scheduler_state"])
                restore_rng(ck)
                state.log(f"resumed from {ck.get('_path', ckpt_path)} @ step {start_step}")
                for k in r.get("dropped", []):
                    state.log(f"  re-initialised {k} (vocab changed)")
            else:
                state.log(f"resume failed ({r.get('error','?')}); starting fresh")

    if not is_text:
        X, y = bundle["X"], bundle["y"]
        X = maybe_pin_to_device(X, dev,
                                enabled=bool(rcfg.get("gpu_data", False)),
                                max_mb=float(rcfg.get("gpu_data_max_mb", 512)))
        y = maybe_pin_to_device(y, dev,
                                enabled=bool(rcfg.get("gpu_data", False)),
                                max_mb=float(rcfg.get("gpu_data_max_mb", 512)))
        X_eval = bundle.get("X_eval")
        y_eval = bundle.get("y_eval")
        if X_eval is not None:
            X_eval = maybe_pin_to_device(X_eval, dev,
                                         enabled=bool(rcfg.get("gpu_data", False)),
                                         max_mb=float(rcfg.get("gpu_data_max_mb", 512)))
            y_eval = maybe_pin_to_device(y_eval, dev,
                                         enabled=bool(rcfg.get("gpu_data", False)),
                                         max_mb=float(rcfg.get("gpu_data_max_mb", 512)))
            state.log(f"eval: {len(X_eval):,} held-out rows "
                      f"({bundle.get('eval_source', 'rows')})")
        toks = None
        eval_toks = None
    else:
        tok = bundle["tokenizer"]
        toks = maybe_pin_to_device(bundle["data"], dev,
                                   enabled=bool(rcfg.get("gpu_data", False)),
                                   max_mb=float(rcfg.get("gpu_data_max_mb", 512)))
        X = y = None
        eval_toks = bundle.get("eval_data")
        if eval_toks is not None:
            eval_toks = maybe_pin_to_device(eval_toks, dev,
                                            enabled=bool(rcfg.get("gpu_data", False)),
                                            max_mb=float(rcfg.get("gpu_data_max_mb", 512)))
            state.log(f"eval: {int(eval_toks.numel()):,} held-out tokens "
                      f"({bundle.get('eval_source', '?')})")
        else:
            state.log("eval: no holdout available - "
                      "eval samples training batches")
        X_eval = y_eval = None

    # ---- losses -----------------------------------------------------------
    # Mixed precision is a poor trade below a few million parameters: bf16 has
    # ~3 significant decimal digits, so a 200-weight model loses more to
    # rounding than it gains from the wider memory budget, and a 37-param MLP
    # can sit at a flat loss for the whole run. Auto-disable and say so.
    n_params_est = arch_mod.estimate_params(mcfg)
    use_amp = bool(rcfg.get("amp", True))
    if use_amp and n_params_est < 1_000_000:
        append_log(f"AMP off: {n_params_est:,} params is small enough that "
                   f"reduced precision costs more than it saves")
        use_amp = False

    amp_ctx, amp_dtype, scaler = autocast_ctx(dev, use_amp)
    max_gn = float(ocfg.get("max_grad_norm", 1.0))
    accum = max(1, int(ocfg.get("accum", 1)))
    log_every = max(1, int(tcfg.get("log_every", 10)))
    eval_every = int(tcfg.get("eval_every", 100))
    sample_every = int(tcfg.get("sample_every", 250))
    ckpt_every = int(tcfg.get("ckpt_every", 500))
    ema_alpha = 0.05

    def get_batch():
        if is_text:
            return ds_mod.get_batch(toks, int(mcfg.get("block_size", 128)), bs, dev)
        return ds_mod.get_tab_batch(X, y, bs, dev)

    def get_eval_batch():
        """Batches from the held-out set. Falls back to training batches
        when there is no holdout (tiny corpora), which is the old behaviour -
        the log line above says which one this run got."""
        if is_text:
            if eval_toks is not None and int(eval_toks.numel()) > bs + 1:
                return ds_mod.get_batch(eval_toks,
                                        int(mcfg.get("block_size", 128)),
                                        bs, dev)
        elif (X_eval is not None and y_eval is not None
                and len(X_eval) > 0):
            return ds_mod.get_tab_batch(X_eval, y_eval,
                                        min(bs, len(X_eval)), dev)
        return get_batch()

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for _ in range(max(1, int(tcfg.get("eval_batches", 20)))):
            x, yy = get_eval_batch()
            with amp_ctx:
                _, loss, _ = model(x, targets=yy)
            if loss is not None:
                tot += float(loss)
                n += 1
        model.train()
        return tot / max(1, n) if n else None

    def do_sample():
        if not is_text or not hasattr(unwrap(model), "generate"):
            return
        prompt = SAMPLE_PROMPTS[int(torch.randint(0, len(SAMPLE_PROMPTS), (1,)))]
        try:
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)
            out = unwrap(model).generate(
                ids, int(tcfg.get("gen_len", 120)),
                temperature=float(tcfg.get("sample_temp", 1.0)),
                top_k=int(tcfg.get("sample_top_k", 40)))
            state.push_sample(prompt, tok.decode(out[0].tolist()))
        except Exception as exc:
            append_debug(f"sample failed: {exc}")

    # ---- main loop --------------------------------------------------------
    state.publish(status="training", total_steps=steps)
    model.train()
    wall0 = time.time()
    ema = None
    best = float("inf")
    last_lr = lr
    tokens_seen = 0
    gn = 0.0
    # `step` is the loop variable, but the teardown below reads it. If resume
    # starts at or past `steps` the loop body never runs and it would be
    # unbound - which is exactly what happens on a re-run of a finished config.
    step = start_step

    if start_step >= steps:
        append_log(f"checkpoint is already at step {start_step} >= {steps} "
                   f"requested; nothing to do (raise train.steps to continue)")
        state.publish(status="done", global_step=start_step, total_steps=steps)
        return {"ok": True, "step": start_step, "tokens": 0, "seconds": 0.0,
                "best_loss": state.best_loss, "params": state.n_params,
                "device": dev, "already_done": True}

    for step in range(start_step, steps):
        if _should_stop():
            state.log("stop requested; saving and exiting")
            save_now(model, opt, sched, step, run, tok if is_text else None)
            state.publish(status="stopped")
            return {"ok": True, "stopped": True, "step": step}

        try:
            with amp_ctx:
                x, yy = get_batch()
                _, loss, _ = model(x, targets=yy)
                loss = loss / accum
            if loss is None:
                continue
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        except torch.cuda.OutOfMemoryError as exc:
            append_error(f"CUDA OOM at step {step}: {str(exc)[:140]}")
            state.publish(status="oom")
            break

        update_due = ((step + 1) % accum == 0) or (step + 1 == steps)
        if update_due:
            last_lr = sched.get_lr()
            for pg in opt.param_groups:
                pg["lr"] = last_lr
            # Correct AMP ordering: unscale -> clip -> step. Clipping scaled
            # gradients would compare inf-scaled norms against max_grad_norm
            # and silently skip the clip on any step where the scaler grew.
            if scaler:
                scaler.unscale_(opt)
            gn = grad_norm_of(model)
            if max_gn > 0:
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_gn)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step += 1

        # ---- publish ------------------------------------------------------
        loss_val = float(loss.detach()) * accum
        ema = loss_val if ema is None else (1 - ema_alpha) * ema + ema_alpha * loss_val
        tokens_seen += bs * (seq if is_text else 1)
        elapsed = time.time() - wall0
        done = step + 1 - start_step
        sps = done / max(1e-8, elapsed)
        a, r, p, m = gpu_mem(dev)
        state.publish(
            global_step=step + 1,
            loss=loss_val,
            loss_smoothed=ema or 0.0,
            best_loss=min(best, loss_val),
            grad_norm=gn if update_due else state.grad_norm,
            lr=last_lr,
            sps=sps,
            tps=sps * bs * (seq if is_text else 1),
            total_tokens=tokens_seen,
            elapsed=elapsed,
            eta=(steps - step - 1) / max(1e-8, sps),
            mem_measured=int(r),
        )
        state.push_metrics(step + 1, loss_val, last_lr, gn if update_due else 0.0,
                           tokens_seen, int(r))
        best = min(best, loss_val)

        if (step + 1) % log_every == 0:
            vram = f"{r / 1024 ** 3:.2f}/{m / 1024 ** 3:.0f}GB" if m else "-"
            state.log(f"step {step + 1}/{steps} loss {loss_val:.4f} "
                      f"(ema {ema:.4f}) lr {last_lr:.2e} "
                      f"gnorm {gn:.2f} {sps:.1f} it/s vram {vram}")

        if eval_every and (step + 1) % eval_every == 0:
            ev = evaluate()
            if ev is not None:
                state.push_eval(step + 1, ev)
                state.log(f"  eval @{step + 1}: {ev:.4f}")

        if sample_every and (step + 1) % sample_every == 0 and is_text:
            do_sample()

        if ckpt_every and (step + 1) % ckpt_every == 0:
            save_now(model, opt, sched, step + 1, run, tok if is_text else None)

    # ---- teardown ---------------------------------------------------------
    final_step = min(max(step + 1, start_step), steps)
    if eval_every:
        ev = evaluate()
        if ev is not None:
            state.push_eval(final_step, ev)
    do_sample()
    save_now(model, opt, sched, final_step, run, tok if is_text else None)
    sync(dev)
    state.publish(status="done", global_step=final_step)
    state.log(f"finished {final_step} steps in {time.time() - wall0:.1f}s "
              f"({tokens_seen:,} tokens)")
    summary = {"ok": True, "step": final_step, "tokens": tokens_seen,
               "seconds": time.time() - wall0, "best_loss": best,
               "params": state.n_params, "device": dev}
    if on_done:
        on_done(summary)
    return summary


def save_now(model, opt, sched, step, run, tok=None):
    path = run.get("runtime", {}).get("checkpoint", "netforge_run.pt")
    if save_ckpt(path, model, opt, sched, step, run_cfg=run, tokenizer=tok,
                 loss_history=list(state.loss_history)[-2000:]):
        state.log(f"saved {path} ({os.path.getsize(path) / 1024 ** 2:.1f} MB)")
    else:
        state.log(f"WARNING: could not write {path}")


def _should_stop():
    from . import logging_utils
    return logging_utils.shutdown_requested


def describe_model(mcfg, n_params):
    if mcfg.get("family") == "mlp":
        hid = mcfg.get("hidden") or []
        shape = "-".join(str(h) for h in hid) or "linear"
        return f"MLP {mcfg.get('n_features')}→{shape}→{mcfg.get('n_outputs')} " \
               f"({n_params:,} params)"
    return (f"GPT {mcfg.get('n_layer')}L × {mcfg.get('embed_dim')}d × "
            f"{mcfg.get('n_head')}h, ctx {mcfg.get('block_size')}, "
            f"vocab {mcfg.get('vocab_size'):,} ({n_params:,} params)")


# ---------------------------------------------------------------------------
# Threading entry point for the GUI
# ---------------------------------------------------------------------------

def start_thread(run, exception_queue, on_done=None):
    """Run training in a daemon thread. Returns the thread (already started)."""
    import threading

    def worker():
        try:
            train_run(run, exception_queue, on_done)
        except Exception:
            exception_queue.put(traceback.format_exc())

    t = threading.Thread(target=worker, daemon=True, name="netforge-train")
    t.start()
    return t