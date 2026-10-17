"""
netforge - a graphical neural network trainer that spans 200 params to 70B.

Module map (flat, no subpackages, mirroring the tinyllm layout):

  config.py         run_cfg: the single JSON-serializable source of truth
  arch.py           model families + param-target architecture search
  memory.py         analytic memory model (weights/grads/optimizer/activations/KV)
  model.py          TinyGPT (decoder-only transformer) + MLPNet (plain feedforward)
  tokenizer.py      CharTokenizer + byte-level BPETokenizer, zero dependencies
  dataset.py        synthetic tasks + local .txt + CSV + HuggingFace streaming
  training.py       the training loop; publishes to SharedState every iteration
  checkpoint.py     atomic save/load with optimizer + scheduler + RNG state
  state.py          SharedState singleton - the only channel between threads
  lr_schedule.py    warmup + cosine/linear/constant schedules
  device_utils.py   device / autocast / pinning helpers
  logging_utils.py  thread-safe log buffer + shutdown flag
  gui.py            the tkinter application (config + live charts + memory panel)
  cli.py            argparse front end; GUI and CLI share one code path

Design notes worth knowing before editing:

  * `run_cfg` is the single source of truth. The GUI writes it, the trainer
    reads it, the checkpoint stores it. Nothing else is passed between them.
  * `validate_run_cfg` returns a LIST OF STRINGS, never raises. The GUI shows
    them in a dialog; the CLI prints them. Same code, two front ends.
  * The GUI never trains and the training loop never draws. They meet only at
    `SharedState`, which is lock-guarded.
  * Every estimate is analytic first (`arch.estimate_params`,
    `memory.estimate`) and measured second (`memory.measure`). The GUI shows
    both so you can see where the analytic model is wrong.
"""