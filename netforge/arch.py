"""
Architecture search: turn "I want a 3M parameter model" into actual layer sizes.

Two families:

  gpt - decoder-only transformer. Param count is dominated by 12 * n_layer * d^2
        (4 attention matrices + 2 MLP matrices), so the search is a 2-D grid
        over (n_layer, embed_dim) with n_head and ff_dim derived.
  mlp - plain feedforward. Cheap to search exhaustively, so we do.

The interesting part is `suggest_arch`, which does NOT simply pick the closest
param count. Among candidates within a tolerance band it prefers the one whose
depth/width balance looks like a model that trains well: head_dim of 32 or 64,
and roughly n_layer ~= embed_dim / 64. A 3M model expressed as 1 layer of width
2746 has the right param count and trains terribly.
"""


from .config import MODEL_FAMILIES

# ---------------------------------------------------------------------------
# Param-count parsing
# ---------------------------------------------------------------------------

_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_params(s):
    """'500' -> 500, '50k' -> 50000, '1.5m' -> 1500000, '7B' -> 7e9.

    Raises ValueError with actionable text - the GUI catches it and shows it
    in the entry, the CLI prints it.
    """
    if isinstance(s, (int, float)):
        return int(s)
    t = str(s).strip().lower().replace("_", "").replace(",", "")
    if not t:
        raise ValueError("empty parameter count")
    mult = 1.0
    if t[-1] in _SUFFIX:
        mult = _SUFFIX[t[-1]]
        t = t[:-1]
    try:
        return int(float(t) * mult)
    except ValueError:
        raise ValueError(
            f"can't parse param count: {s!r}  (use e.g. 500, 50k, 7m, 1.5b)"
        )


def fmt_params(n):
    """1234567 -> '1.23M'"""
    n = int(n)
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Analytic param counts. These never instantiate a model - the GUI needs them
# on every keystroke to live-preview a config that hasn't been built yet.
# ---------------------------------------------------------------------------

def resolve_ff(mcfg):
    """ff_dim if set, else embed_dim * ff_mult (rounded to a multiple of 8)."""
    ff = int(mcfg.get("ff_dim") or 0)
    if ff > 0:
        return ff
    dim = int(mcfg.get("embed_dim", 128))
    mult = float(mcfg.get("ff_mult", 4) or 4)
    return max(8, int(round(dim * mult / 8.0)) * 8)


def _norm_params(dim, norm_type):
    """Parameters per norm layer.

    RMSNorm here is weight-only (no bias), LayerNorm has both. The estimator
    has to know this or it silently over-counts by `dim` per norm - which at
    32 layers of width 4096 is 262k parameters, enough to make a "124M" model
    report as something else.
    """
    return dim if norm_type == "rmsnorm" else 2 * dim


def estimate_gpt_params(vocab_size, block_size, n_layer, n_head, embed_dim,
                        ff_dim=None, ff_mult=4, pos_encoding="learned",
                        tie_weights=True, qk_norm=False, mlp_gated=False,
                        norm_type="rmsnorm"):
    """Parameter count for a decoder-only transformer, without building it."""
    d = int(embed_dim)
    f = int(ff_dim) if ff_dim else max(8, int(round(d * ff_mult)))
    tok = int(vocab_size) * d
    tok_total = tok if tie_weights else 2 * tok

    pos = int(block_size) * d if pos_encoding == "learned" else 0

    # attention: qkv (3d^2) + out (d^2); optional q/k RMSNorm (2d) for QK-norm
    attn = 4 * d * d + (2 * d if qk_norm else 0)
    # mlp: standard is 2*d*f; SwiGLU has 3 matrices (gate, up, down)
    mlp = (3 if mlp_gated else 2) * d * f
    norms = 2 * _norm_params(d, norm_type)          # ln1 + ln2 per block
    per_layer = attn + mlp + norms

    final_norm = _norm_params(d, norm_type)
    return int(tok_total + pos + n_layer * per_layer + final_norm)


def _as_hidden(hidden):
    """Accept [4, 8], "4 8", "64, 64" or "[4]". Returns a list of ints."""
    if isinstance(hidden, str):
        hidden = hidden.strip().strip("[]").replace(",", " ").split()
    if not isinstance(hidden, (list, tuple)):
        return []
    out = []
    for h in hidden:
        try:
            v = int(float(h))
            if v > 0:
                out.append(v)
        except (TypeError, ValueError):
            continue
    return out


def estimate_mlp_params(n_features, hidden, n_outputs):
    """Parameter count for a plain MLP: weights + biases everywhere."""
    widths = [int(n_features)] + _as_hidden(hidden) + [int(n_outputs)]
    total = 0
    for a, b in zip(widths, widths[1:]):
        total += a * b + b          # weight matrix + bias vector
    return int(total)


def estimate_params(mcfg):
    """Dispatch to the right estimator for a model cfg dict."""
    if mcfg.get("family") == "mlp":
        return estimate_mlp_params(mcfg.get("n_features", 8),
                                   mcfg.get("hidden") or [], mcfg.get("n_outputs", 1))
    return estimate_gpt_params(
        vocab_size=mcfg.get("vocab_size", 512),
        block_size=mcfg.get("block_size", 128),
        n_layer=mcfg.get("n_layer", 4),
        n_head=mcfg.get("n_head", 4),
        embed_dim=mcfg.get("embed_dim", 128),
        ff_dim=resolve_ff(mcfg),
        pos_encoding=mcfg.get("pos_encoding", "learned"),
        tie_weights=bool(mcfg.get("tie_weights", True)),
        norm_type=mcfg.get("norm_type", "rmsnorm"),
    )


def param_breakdown(mcfg):
    """Per-component param counts so the GUI can show where the memory goes."""
    if mcfg.get("family") == "mlp":
        widths = [int(mcfg.get("n_features", 8))] + \
                 _as_hidden(mcfg.get("hidden") or []) + \
                 [int(mcfg.get("n_outputs", 1))]
        layers, total = [], 0
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            n = a * b + b
            total += n
            layers.append((f"fc{i} ({a}->{b})", n))
        return layers, total

    d = int(mcfg.get("embed_dim", 128))
    f = resolve_ff(mcfg)
    nl = int(mcfg.get("n_layer", 4))
    v = int(mcfg.get("vocab_size", 512))
    bs = int(mcfg.get("block_size", 128))
    tied = bool(mcfg.get("tie_weights", True))
    layers = []
    tok = v * d
    layers.append(("tok_emb", tok))
    if mcfg.get("pos_encoding") == "learned":
        layers.append(("pos_emb", bs * d))
    per = 4 * d * d + 2 * d * f + 2 * _norm_params(d, mcfg.get("norm_type", "rmsnorm"))
    layers.append((f"blocks x{nl}", nl * per))
    if not tied:
        layers.append(("lm_head", v * d))
    total = sum(n for _, n in layers)
    return layers, total


# ---------------------------------------------------------------------------
# Training-shape heuristic
# ---------------------------------------------------------------------------

def balance(n_layer, embed_dim, n_head):
    """Lower is better. Penalises degenerate head dims and depth/width mismatch."""
    hd = embed_dim // max(1, n_head)
    if hd % 2:
        hd_pen = 1.5               # odd head_dim: rope covers one channel less
    elif hd in (32, 64):
        hd_pen = 0.0
    elif hd in (16, 128):
        hd_pen = 0.4
    else:
        hd_pen = 0.8 + min(0.7, abs(hd - 64) / 128.0)
    if n_head == 1:
        hd_pen += 1.0          # single-head attention barely learns
    if embed_dim < 16:
        hd_pen += 1.2          # below this you are in MLP territory
    ideal = max(1, round(embed_dim / 64.0))
    return abs(n_layer - ideal) / max(1, ideal) + hd_pen


def _pick_heads(embed_dim, prefer=64):
    """Choose a head count that divides embed_dim evenly.

    Prefers even head_dim: rotary embeddings rotate in pairs of channels, and
    an odd head_dim leaves one channel unrotated. Scoring every valid divisor
    and taking the best is cheap here (embed_dim is small) and avoids
    generating architectures that only half-work.
    """
    best, best_cost = 1, float("inf")
    for h in (1, 2, 4, 8, 16, 32, 64, 128):
        if h > embed_dim // 8:
            break
        if embed_dim % h:
            continue
        hd = embed_dim // h
        cost = abs(hd - prefer) / 32.0      # want head_dim near 64
        if hd % 2:
            cost += 3.0                    # strongly avoid odd head_dim
        if hd < 8:
            cost += 2.0
        if h == 1:
            cost += 1.0                    # single-head attention learns poorly
        if cost < best_cost:               # lower is better, as in balance()
            best, best_cost = h, cost
    return best


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

# (dim_lo, dim_hi, dim_step, layer_lo, layer_hi) tiers by target size. The tiny
# end goes down to width 4 / 1 layer because "very very tiny nets" is a first
# class target here, not an edge case.
_GPT_TIERS = [
    (2,    64,    2,  1, 4),
    (4,    128,   4,  1, 6),
    (16,   256,   8,  1, 12),
    (64,   512,   16, 2, 24),
    (256,  1024,  32, 2, 32),
    (512,  2048,  64, 4, 48),
    (1024, 4096,  128, 4, 64),
    (2048, 8192,  256, 8, 80),
    (4096, 16384, 256, 12, 96),
]


def gpt_floor_params(vocab_size, block_size, pos_encoding="learned", tie_weights=True):
    """Smallest param count any GPT with this vocab/context can have.

    The token embedding alone is vocab_size * embed_dim, and the minimum
    sensible embed_dim is 4. So a 32k-vocab GPT cannot go below ~128k params
    no matter how thin the stack is. `suggest_gpt` reports this rather than
    silently returning something 300000% over target.
    """
    v, d = int(vocab_size), 4
    tok = v * d * (1 if tie_weights else 2)
    pos = int(block_size) * d if pos_encoding == "learned" else 0
    return int(tok + pos + d)          # d for the final norm


def suggest_gpt(target, block_size=512, vocab_size=51200, tie_weights=True,
                pos_encoding="rope", ff_mult=4, tol=0.03):
    """Find a GPT config whose param count lands within `tol` of `target`.

    Returns the solved keys plus `estimated_params` and, when the target is
    unreachable, a `warning` explaining the floor.
    """
    target = max(1, int(target))
    floor = gpt_floor_params(vocab_size, block_size, pos_encoding, tie_weights)
    warning = ""
    if target < floor:
        target = floor
        warning = (f"a {vocab_size:,}-vocab GPT cannot be smaller than "
                   f"{floor:,} params (the embedding is vocab x width); "
                   f"clamped to the floor - use the MLP family for tiny nets")

    tier = _GPT_TIERS[0]
    for t in _GPT_TIERS:
        if t[1] * t[1] * t[3] * 12 >= target:
            tier = t
            break
    else:
        tier = _GPT_TIERS[-1]

    dim_lo, dim_hi, dim_step, lay_lo, lay_hi = tier
    # Below a thousand params the fixed 8-wide FFN dominates the count, so let
    # it scale down too rather than giving up on the search.
    ff_min = 4 if target < 2000 else 8

    def ff_for(dim):
        return max(ff_min, int(round(dim * ff_mult / 8.0)) * 8 if ff_min == 8
                   else int(round(dim * ff_mult)))

    best = None
    for dim in range(max(4, dim_lo), dim_hi + 1, dim_step):
        heads = _pick_heads(dim)
        for nl in range(lay_lo, lay_hi + 1):
            ff = ff_for(dim)
            est = estimate_gpt_params(vocab_size, block_size, nl, heads, dim, ff,
                                      pos_encoding=pos_encoding,
                                      tie_weights=tie_weights)
            if est > target * (1 + tol) * 3:
                break                      # deeper only grows it further
            err = abs(est - target) / max(1, target)
            if err > tol:
                continue
            key = (balance(nl, dim, heads), err, -nl)
            if best is None or key < best[0]:
                best = (key, dict(n_layer=nl, n_head=heads, embed_dim=dim,
                                  ff_dim=ff, estimated_params=est))
            break

    if best is None:
        # Nothing landed inside the band - take the thinnest config that is
        # still no bigger than the target, rather than something wildly over.
        dim = max(4, dim_lo)
        heads = _pick_heads(dim)
        ff = ff_for(dim)
        nl = lay_lo
        while nl > 1 and estimate_gpt_params(vocab_size, block_size, nl, heads,
                                             dim, ff, pos_encoding=pos_encoding,
                                             tie_weights=tie_weights) > target * 1.5:
            nl -= 1
        best = (None, dict(n_layer=nl, n_head=heads, embed_dim=dim, ff_dim=ff,
                           estimated_params=estimate_gpt_params(
                               vocab_size, block_size, nl, heads, dim, ff,
                               pos_encoding=pos_encoding,
                               tie_weights=tie_weights)))
    out = best[1]
    if warning:
        out["warning"] = warning
    return out


def suggest_mlp(target, n_features=8, n_outputs=1, max_layers=4, tol=0.05):
    """Find an MLP shape close to `target` params.

    Searched over (depth, width) with the width shared by all hidden layers.
    """
    target = max(1, int(target))
    best = None
    for depth in range(0, max_layers + 1):
        for width in range(1, 4097):
            hidden = [width] * depth
            if depth == 0 and width > 8:
                break                      # no hidden layers: width is a no-op
            est = estimate_mlp_params(n_features, hidden, n_outputs)
            if est > target * (1 + tol) * 3:
                break
            err = abs(est - target) / max(1, target)
            if err <= tol:
                key = (depth * 0.15 + abs(est - target) / max(1, target), -depth)
                if best is None or key < best[0]:
                    best = (key, dict(hidden=hidden, estimated_params=est))
                break
    if best is None:
        width = max(1, target // max(1, n_features * max(1, n_outputs)))
        hidden = [max(1, width)] * 2
        best = (None, dict(hidden=hidden,
                           estimated_params=estimate_mlp_params(n_features, hidden,
                                                               n_outputs)))
    return best[1]


def suggest_arch(target, family="gpt", block_size=512, vocab_size=51200,
                 tie_weights=True, pos_encoding="rope", ff_mult=4,
                 n_features=8, n_outputs=1, shrink_vocab=True):
    """Public entry point used by the GUI's "Auto-fit" button.

    Returns a dict of the model-cfg keys that were solved for, plus
    `estimated_params` and optionally `warning`.

    `shrink_vocab`: when the target is below what any stack can reach, shrink
    the vocabulary too rather than clamping to the floor. A user asking for a
    5k-param GPT wants 5k params, and the honest answer is "vocab 256, one
    layer, width 16" - not "here is 128k params and a warning".
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family {family!r} (choose from {MODEL_FAMILIES})")
    if family == "mlp":
        out = suggest_mlp(target, n_features=n_features, n_outputs=n_outputs)
        out["family"] = "mlp"
        return out

    target = max(1, int(target))
    orig_vocab, orig_ctx = int(vocab_size), int(block_size)
    floor = gpt_floor_params(vocab_size, block_size, pos_encoding, tie_weights)
    shrunk = False

    if shrink_vocab and target < floor:
        # Below ~1M params the vocabulary, not the stack, is the constraint.
        # Cap it at 512 - which is exactly a char tokenizer's natural size -
        # and hand the rest of the budget to the layers, where the search can
        # actually land inside the tolerance band.
        vocab_size = min(vocab_size, 512)
        if vocab_size * 4 >= target:
            # even 512 is too big: scale down to whatever the budget allows
            vocab_size = max(8, int(target * 0.5) // 4)
        block_size = min(block_size, max(16, (target // 64 // 64) * 64 or 16))
        shrunk = (vocab_size != orig_vocab) or (block_size != orig_ctx)

    out = suggest_gpt(target, block_size=block_size, vocab_size=vocab_size,
                      tie_weights=tie_weights, pos_encoding=pos_encoding,
                      ff_mult=ff_mult)
    out["family"] = "gpt"
    out["block_size"] = int(block_size)
    out["vocab_size"] = int(vocab_size)
    # carry the constraints the solve was made under, so the returned dict can
    # be handed straight back to estimate_params(). Without them it silently
    # re-estimates with the default pos_encoding and disagrees with the very
    # number it ships as 'estimated_params'.
    out["pos_encoding"] = pos_encoding
    out["tie_weights"] = bool(tie_weights)
    if shrunk:
        out["warning"] = (
            out.get("warning") or "") + (
            f" (vocab/context shrunk to {vocab_size:,}/{block_size} to reach "
            f"{target:,} params; a real corpus wants a char or small-BPE "
            f"tokenizer here)")
    return out


# ---------------------------------------------------------------------------
# Presets. Labels are what you would actually say out loud.
# ---------------------------------------------------------------------------

PRESETS = [
    # (label, family, param string, extra model-cfg overrides)
    ("Toy",        "mlp", "40",     dict(n_features=2, n_outputs=1)),
    ("Micro",      "mlp", "500",    dict(n_features=8)),
    ("Nano",       "gpt", "50k",    dict(block_size=128, vocab_size=512)),
    ("Small",      "gpt", "5m",     dict(block_size=256, vocab_size=2048)),
    ("GPT-2",      "gpt", "124m",   dict(block_size=1024, vocab_size=50257)),
    ("GPT-3",      "gpt", "175b",   dict(block_size=2048, vocab_size=50257)),
    ("Llama-7B",   "gpt", "7b",     dict(block_size=4096, vocab_size=32000)),
    ("Llama-70B",  "gpt", "70b",    dict(block_size=8192, vocab_size=128256)),
]

TIER_LABELS = ["<1K", "1K-100K", "100K-1M", "1M-10M", "10M-100M",
               "100M-1B", "1B-10B", ">10B"]