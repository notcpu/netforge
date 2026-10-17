"""
Command line front end.

The GUI and the CLI are the same program with different front doors: the GUI
writes a run cfg and spawns `netforge_cli.py train --config ...`, the CLI can
also do it directly. There is one training implementation (training.py) and one
config (config.py); neither front end has its own copy of the loop.

Subcommands:
  train        run training from a config file
  gui          open the graphical trainer
  solve        print the architecture for a parameter target
  memory       print a memory report for a config
  fit          what batch/context fits on this machine
  device       what hardware was detected
  ckpt         inspect a checkpoint
"""

import argparse
import json
import multiprocessing
import sys

from . import arch as arch_mod
from . import checkpoint as ckpt_mod
from . import config as cfg_mod
from . import memory as mem_mod
from . import logging_utils


def _load(path):
    if not path:
        return cfg_mod.default_run_cfg()
    return cfg_mod.load_run_cfg(path)


def _apply_overrides(run, pairs):
    """--set model.n_layer=32 --set train.lr=1e-4"""
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        o = run
        parts = k.split(".")
        for q in parts[:-1]:
            o = o.setdefault(q, {})
        cur = o.get(parts[-1])
        try:
            if isinstance(cur, bool):
                o[parts[-1]] = v.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                o[parts[-1]] = int(float(v))
            elif isinstance(cur, float):
                o[parts[-1]] = float(v)
            else:
                o[parts[-1]] = v
        except ValueError:
            o[parts[-1]] = v
    return cfg_mod._backfill(run)


# ---------------------------------------------------------------------------

def cmd_train(args):
    run = _apply_overrides(_load(args.config), args.set)
    if args.solve:
        fam = run["model"]["family"]
        try:
            target = arch_mod.parse_params(args.solve)
        except ValueError as exc:
            raise SystemExit(str(exc))
        solved = arch_mod.suggest_arch(
            target, fam, block_size=int(run["model"]["block_size"]),
            vocab_size=int(run["model"]["vocab_size"]),
            ff_mult=float(run["model"]["ff_mult"]),
            n_features=int(run["model"].get("n_features", 8)))
        run["model"].update({k: v for k, v in solved.items() if k != "warning"})
        print(f"solved for {target:,} params:", json.dumps(
            {k: v for k, v in solved.items()}, indent=2))
        if solved.get("warning"):
            print(f"note: {solved['warning']}")

    errs = cfg_mod.validate_run_cfg(run)
    if errs:
        for e in errs:
            print(f"config error: {e}", file=sys.stderr)
        raise SystemExit(1)

    logging_utils.setup_log_file(run["runtime"].get("log_file"))
    seq = int(run["model"]["block_size"]) if run["model"]["family"] == "gpt" else 1
    print(f"model     {arch_mod.fmt_params(arch_mod.estimate_params(run['model']))} params")
    print("memory    " + "\n          ".join(
        memory_one_line(run, seq)))
    print(f"checkpoint {run['runtime']['checkpoint']}")

    if args.dry_run:
        return 0

    import queue
    from .training import train_run
    exc_q = queue.Queue()
    run["runtime"].setdefault("resume", not args.no_resume)
    summary = train_run(run, exc_q)
    if args.save_config:
        cfg_mod.save_run_cfg(run, args.save_config)
    if not summary.get("ok"):
        return 1
    print(f"\ndone in {summary['seconds']:.1f}s   "
          f"best loss {summary['best_loss']:.4f}")
    return 0


def memory_one_line(run, seq):
    est = mem_mod.estimate(run["model"], int(run["train"]["batch_size"]), seq,
                           training=True, memcfg=run["memory"],
                           optimizer=run["optimizer"]["name"],
                           mixed_precision=bool(run["runtime"]["amp"]))
    return [f"{lab:<22}{mem_mod.fmt_bytes(nb):>11}   {note}"
            for _k, lab, nb, note in est["items"]] + \
           [f"{'TOTAL':<22}{mem_mod.fmt_bytes(est['total']):>11}"]


def cmd_gui(args):
    from .gui import launch
    launch(_load(args.config) if args.config else None)
    return 0


def cmd_solve(args):
    target = arch_mod.parse_params(args.params)
    for fam in ([args.family] if args.family else ["gpt", "mlp"]):
        solved = arch_mod.suggest_arch(
            target, fam, block_size=args.ctx, vocab_size=args.vocab,
            ff_mult=args.ff_mult, n_features=args.features)
        n = arch_mod.estimate_params({**{"family": fam}, **solved})
        err = 100.0 * (n - target) / max(1, target)
        print(f"{fam}: {n:,} params ({err:+.1f}%)  " +
              (f"{solved['n_layer']}L x {solved['embed_dim']}d x {solved['n_head']}h, "
               f"ctx {solved['block_size']}, vocab {solved['vocab_size']:,}"
               if fam == "gpt" else f"hidden {solved['hidden']}"))
        if solved.get("warning"):
            print(f"  note: {solved['warning']}")
    return 0


def cmd_memory(args):
    run = _apply_overrides(_load(args.config), args.set)
    m = run["model"]
    inv = mem_mod.device_inventory()
    budget = inv["devices"][0]["total"] if inv["devices"] else inv["cpu_ram"] * 0.5
    seq = int(m["block_size"]) if m["family"] == "gpt" else 1
    for line in mem_mod.report(m, int(args.batch or run["train"]["batch_size"]),
                               int(args.seq or seq),
                               training=not args.inference,
                               memcfg=run["memory"],
                               optimizer=run["optimizer"]["name"],
                               device_total=budget):
        print(line)
    return 0


def cmd_fit(args):
    run = _apply_overrides(_load(args.config), args.set)
    plan = mem_mod.fit_plan(run["model"], training=not args.inference,
                            memcfg=run["memory"],
                            optimizer=run["optimizer"]["name"], batch=1, seq=1)
    inv = mem_mod.device_inventory()
    print(f"device:   {inv['name']}  "
          f"{mem_mod.fmt_bytes(inv['devices'][0]['total'] if inv['devices'] else inv['cpu_ram'])}")
    if plan["fits"]:
        print(f"fits:     batch {plan['batch']} x context {plan['seq']}")
        print(f"          {plan['note']}")
    else:
        print(f"fits:     no — {plan['reason']}")
        if plan.get("suggest_devices"):
            print(f"          would need ~{plan['suggest_devices']} such GPUs")
    return 0


def cmd_device(args):
    inv = mem_mod.device_inventory()
    print(f"cuda       {inv['cuda']}")
    print(f"threads    {inv['cpu_threads']}")
    if inv["cpu_ram"]:
        print(f"system ram {mem_mod.fmt_bytes(inv['cpu_ram'])}")
    for d in inv["devices"]:
        print(f"gpu {d['index']}      {d['name']}  "
              f"{mem_mod.fmt_bytes(d['free'])} free / "
              f"{mem_mod.fmt_bytes(d['total'])}  sm_{d['capability']}  "
              f"{d['sms']} SMs")
    return 0


def cmd_ckpt(args):
    info = ckpt_mod.ckpt_info(args.path)
    print(json.dumps({k: v for k, v in info.items() if k != "model_cfg"},
                     indent=2, default=str))
    if info.get("model_cfg"):
        print("model_cfg:", json.dumps(info["model_cfg"], indent=2, default=str))
    return 0 if info["ok"] else 1


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="netforge",
        description="Neural network trainer: 40 params to 70B, with a memory model.")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="train from a config file")
    t.add_argument("--config", help="run cfg json (see `netforge solve` output)")
    t.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config key, repeatable")
    t.add_argument("--solve", metavar="PARAMS",
                   help="auto-size the architecture first (e.g. 7b)")
    t.add_argument("--no-resume", action="store_true",
                   help="ignore an existing checkpoint and start over")
    t.add_argument("--dry-run", action="store_true",
                   help="print the memory report and exit")
    t.add_argument("--save-config", metavar="PATH",
                   help="write the resolved config out")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("gui", help="open the graphical trainer")
    g.add_argument("--config", help="load a run cfg on start")
    g.set_defaults(func=cmd_gui)

    s = sub.add_parser("solve", help="architecture for a parameter target")
    s.add_argument("params", help="e.g. 500, 50k, 7m, 1.5b")
    s.add_argument("--family", choices=("gpt", "mlp"))
    s.add_argument("--ctx", type=int, default=512)
    s.add_argument("--vocab", type=int, default=51200)
    s.add_argument("--ff-mult", type=float, default=4, dest="ff_mult")
    s.add_argument("--features", type=int, default=8)
    s.set_defaults(func=cmd_solve)

    m = sub.add_parser("memory", help="memory report for a config")
    m.add_argument("--config")
    m.add_argument("--set", action="append")
    m.add_argument("--batch", type=int)
    m.add_argument("--seq", type=int)
    m.add_argument("--inference", action="store_true")
    m.set_defaults(func=cmd_memory)

    f = sub.add_parser("fit", help="largest batch/context that fits here")
    f.add_argument("--config")
    f.add_argument("--set", action="append")
    f.add_argument("--inference", action="store_true")
    f.set_defaults(func=cmd_fit)

    d = sub.add_parser("device", help="what hardware was detected")
    d.set_defaults(func=cmd_device)

    c = sub.add_parser("ckpt", help="inspect a checkpoint")
    c.add_argument("path")
    c.set_defaults(func=cmd_ckpt)

    return p


def main(argv=None):
    # Required on Windows: prevents recursive subprocess spawning when this
    # module is run through a console entry point that re-invokes python.
    multiprocessing.freeze_support()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args((argv or []) + ["gui"])
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        logging_utils.request_shutdown()
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())