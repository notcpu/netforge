"""
Learning-rate schedules.

One class, `SchedulerState`, rather than four separate functions, because the
training loop needs the current LR every step regardless of which schedule is
selected, and a `None` return from each variant is a bug waiting to happen.
"""

import math


class SchedulerState:
    """Warmup followed by cosine / linear / constant / WSD decay.

    WSD (warmup-stable-decay) holds the LR flat at `lr_max` for the middle of
    training then decays hard over the last `decay_frac`. It is the schedule
    behind most recent open pretraining runs and behaves noticeably better
    than pure cosine when you are doing a final anneal.
    """

    def __init__(self, scheduler="cosine", lr_max=3e-4, total_steps=1000,
                 warmup_frac=0.05, min_lr_frac=0.05, decay_frac=0.2):
        self.scheduler = scheduler
        self.lr_max = float(lr_max)
        self.total_steps = max(1, int(total_steps))
        self.warmup_frac = max(0.0, min(0.999, float(warmup_frac)))
        self.min_lr_frac = max(0.0, min(1.0, float(min_lr_frac)))
        self.decay_frac = max(0.0, min(0.9, float(decay_frac)))
        self.warmup_steps = int(self.warmup_frac * self.total_steps)
        self.step = 0

    def get_lr(self):
        s, warm = self.step, max(1, self.warmup_steps)
        peak = self.lr_max
        floor = peak * self.min_lr_frac

        if s < warm:
            return peak * (s + 1) / warm

        if self.scheduler == "constant":
            return peak

        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        if self.scheduler == "linear":
            return floor + (peak - floor) * (1.0 - progress)

        if self.scheduler == "wsd":
            decay_start = 1.0 - self.decay_frac
            if progress < decay_start:
                return peak
            d = (progress - decay_start) / max(1e-9, self.decay_frac)
            return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * d))

        # cosine (default)
        return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step_once(self):
        self.step += 1

    # tinyllm used `sched.step += 1` from the loop; keep that working.
    def __setattr__(self, k, v):
        if k == "step" and not isinstance(v, int):
            v = int(v)
        object.__setattr__(self, k, v)

    def state_dict(self):
        return {"scheduler": self.scheduler, "lr_max": self.lr_max,
                "total_steps": self.total_steps, "warmup_frac": self.warmup_frac,
                "min_lr_frac": self.min_lr_frac, "decay_frac": self.decay_frac,
                "step": self.step}

    @classmethod
    def from_state(cls, d):
        s = cls(scheduler=d.get("scheduler", "cosine"),
                lr_max=d.get("lr_max", 3e-4),
                total_steps=d.get("total_steps", 1000),
                warmup_frac=d.get("warmup_frac", 0.05),
                min_lr_frac=d.get("min_lr_frac", 0.05),
                decay_frac=d.get("decay_frac", 0.2))
        s.step = int(d.get("step", 0))
        return s


def optimal_lr(n_params, batch_size, block_size, base_lr=6e-4):
    """Rough Chinchilla-ish scaling law: bigger models want lower LRs, and
    bigger batches can absorb a higher one. Good enough to fill in a default
    you have not tuned."""
    ref_batch = 512 * 128
    eff_batch = max(1, int(batch_size) * int(block_size))
    scale_n = (1e9 / max(int(n_params), 1_000)) ** 0.3
    scale_b = math.sqrt(eff_batch / ref_batch)
    return float(max(1e-5, min(1e-2, base_lr * scale_n * scale_b)))
