"""
The models.

Two families behind one interface, so the training loop never branches on
"is this a transformer or not":

  TinyGPT - decoder-only transformer. forward returns (logits, loss) and has a
            KV-cached generate().
  MLPNet  - plain feedforward. forward returns (y, loss). Handles regression
            and single/multi-class classification.

Both expose: cfg, n_params, loss_kind, forward(x, targets=None), param_summary().
`build_model` is the only factory; `unwrap` peels off torch.compile's wrapper.

Loss is computed *inside* the model, matching the tinyllm convention. Callers
always write `out, loss = model(x, targets=y)`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_POS = ("learned", "sinusoidal", "rope", "none")
VALID_NORM = ("layernorm", "rmsnorm")
VALID_ACT = ("gelu", "gelu_new", "relu", "silu", "tanh", "sigmoid")


def _norm_layer(kind, dim):
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm type {kind!r} (choose from {VALID_NORM})")


def _activation(name):
    if name == "gelu":
        return nn.GELU()
    if name == "gelu_new":
        return nn.GELU(approximate="tanh")
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "none":
        return nn.Identity()
    raise ValueError(f"unknown activation {name!r} (choose from {VALID_ACT})")


def _build_sinusoidal(n_pos, dim):
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_pos, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _apply_rope(x, offset=0):
    """Rotary position embedding over the last dim.

    x: (B, H, T, head_dim).

    An odd head_dim is rotated over its largest even prefix and the trailing
    channel is left alone. That matters because users can type any n_head,
    and embed_dim % n_head == 0 does not imply an even head_dim - 44/4 = 11.
    Silently producing a wrong-shaped tensor here is much worse than the tiny
    loss of rotational coverage on one channel, so `validate_model_cfg` warns
    about it too.
    """
    b, h, t, d = x.shape
    d_even = d - (d % 2)
    if d_even < 2:
        return x
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, d_even, 2, device=x.device, dtype=torch.float32) / d_even
    )
    pos = torch.arange(offset, offset + t, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freqs[None, :]                       # (T, d_even/2)
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    # our cat() puts all evens then all odds; stack then flatten to interleave
    head, tail = x[..., :d_even], x[..., d_even:]
    x1, x2 = head[..., 0::2], head[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    rotated = torch.stack((o1, o2), dim=-1).flatten(-2)
    return rotated if d_even == d else torch.cat((rotated, tail), dim=-1)


class RMSNorm(nn.Module):
    """Weight-only RMSNorm. No bias, so it saves a parameter per dim."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head, dropout, pos_encoding="learned"):
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, past_k=None, past_v=None, past_len=0):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        if self.pos_encoding == "rope":
            off = past_len if past_k is not None else 0
            q, k = _apply_rope(q, off), _apply_rope(k, off)

        # Fast path: no cache means SDPA can pick its flash kernel, which is
        # what keeps a 4k-context forward from materialising a 4k x 4k matrix.
        no_cache = past_k is None and past_v is None and past_len == 0
        if self.use_flash and no_cache:
            dp = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p=dp, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(b, t, c)
            return self.resid_drop(self.out(y)), k, v

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        s = k.size(2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        ik = torch.arange(s, device=x.device).unsqueeze(0)
        iq = torch.arange(past_len, past_len + t, device=x.device).unsqueeze(1)
        # float("-inf") not -1e10: fp16 overflows on large negative fills
        att = att.masked_fill(~(ik <= iq).unsqueeze(0).unsqueeze(0), float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.out(y)), k, v


class MLP(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout, activation):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.act = _activation(activation)
        self.fc2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, dropout, norm_type, activation,
                 pos_encoding):
        super().__init__()
        self.ln1 = _norm_layer(norm_type, embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head, dropout, pos_encoding)
        self.ln2 = _norm_layer(norm_type, embed_dim)
        self.mlp = MLP(embed_dim, ff_dim, dropout, activation)

    def forward(self, x, past_k=None, past_v=None, past_len=0):
        a, k, v = self.attn(self.ln1(x), past_k=past_k, past_v=past_v, past_len=past_len)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class TinyGPT(nn.Module):
    """A configurable decoder-only transformer.

    Configurable without editing source: vocab_size, embed_dim, n_layer, n_head,
    ff_dim / ff_mult, block_size, dropout, pos_encoding, norm_type, activation,
    tie_weights, dtype.
    """

    loss_kind = "next_token"

    def __init__(self, vocab_size, block_size, n_layer, n_head, embed_dim,
                 ff_mult=4, ff_dim=None, dropout=0.1, pos_encoding="learned",
                 norm_type="rmsnorm", activation="silu", tie_weights=True,
                 loss_chunk=0):
        super().__init__()
        assert pos_encoding in VALID_POS, f"choose from {VALID_POS}"
        assert norm_type in VALID_NORM, f"choose from {VALID_NORM}"
        assert activation in VALID_ACT, f"choose from {VALID_ACT}"
        cfg = dict(vocab_size=int(vocab_size), block_size=int(block_size),
                   n_layer=int(n_layer), n_head=int(n_head),
                   embed_dim=int(embed_dim),
                   ff_dim=int(ff_dim) if ff_dim else
                   max(8, int(round(int(embed_dim) * float(ff_mult)))),
                   dropout=float(dropout), pos_encoding=pos_encoding,
                   norm_type=norm_type, activation=activation,
                   tie_weights=bool(tie_weights))
        self.cfg = cfg
        self.block_size = cfg["block_size"]
        self.n_layer = cfg["n_layer"]
        self.ff_dim = cfg["ff_dim"]
        self.loss_chunk = int(loss_chunk or 0)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_encoding = pos_encoding
        if pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg["block_size"], cfg["embed_dim"])
        elif pos_encoding == "sinusoidal":
            self.register_buffer("pos_emb",
                                 _build_sinusoidal(cfg["block_size"], cfg["embed_dim"]),
                                 persistent=False)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([
            Block(cfg["embed_dim"], cfg["n_head"], cfg["ff_dim"], cfg["dropout"],
                  cfg["norm_type"], cfg["activation"], cfg["pos_encoding"])
            for _ in range(cfg["n_layer"])
        ])
        self.ln_f = _norm_layer(cfg["norm_type"], cfg["embed_dim"])
        self.head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)
        if cfg["tie_weights"]:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2 trick) - without it a deep
        # stack has variance that grows with depth and diverges.
        for name, p in self.named_parameters():
            if name.endswith("out.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg["n_layer"]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @property
    def n_params(self):
        """Count unique parameters - tied weights must not be double-counted."""
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, idx, targets=None, past_kv=None, past_len=0):
        b, t = idx.shape
        assert t + past_len <= self.block_size, \
            f"seq len {t + past_len} > block_size {self.block_size}"
        x = self.tok_emb(idx)
        if self.pos_encoding == "learned":
            pos = torch.arange(past_len, past_len + t, device=idx.device) \
                .unsqueeze(0) % self.block_size
            x = x + self.pos_emb(pos)
        elif self.pos_encoding == "sinusoidal":
            x = x + self.pos_emb[past_len:past_len + t].unsqueeze(0).to(x.dtype)
        x = self.drop(x)

        if past_kv is None:
            past_kv = [(None, None)] * self.n_layer
        new_kv = []
        for blk, (pk, pv) in zip(self.blocks, past_kv):
            x, k, v = blk(x, past_k=pk, past_v=pv, past_len=past_len)
            new_kv.append((k, v))

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = self._loss(logits, targets)
        return logits, loss, new_kv

    def _loss(self, logits, targets):
        flat = logits.reshape(-1, logits.size(-1))
        tgt = targets.reshape(-1)
        if tgt.dtype != torch.long:
            tgt = tgt.long()
        if self.loss_chunk and flat.size(0) > self.loss_chunk:
            # Chunked cross-entropy: recompute the logits per micro-batch so we
            # never hold B*T*V in fp32 at once. Costs one extra head matmul.
            total = flat.new_zeros((), dtype=torch.float32)
            n = 0
            for i in range(0, flat.size(0), self.loss_chunk):
                chunk = flat[i:i + self.loss_chunk]
                n_ch = chunk.size(0)
                total = total + F.cross_entropy(
                    chunk.float(), tgt[i:i + self.loss_chunk], reduction="sum")
                n += n_ch
            return total / max(1, n)
        return F.cross_entropy(flat.float(), tgt)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, stop_ids=None):
        """KV-cached sampling. Returns the full sequence including the prompt.

        The cache is fed one full prompt on the first step, then exactly one
        token per step after that, so a 200-token generation does 1 forward
        over T tokens and 199 forwards over 1 token instead of 200 forwards
        over T.

        Once the cache reaches block_size it slides: the oldest `drop`
        positions are evicted from K and V. Without that, generating past the
        context window is an assertion error rather than a working model with
        a slightly stale memory - which is what sliding-window attention
        actually is.
        """
        stop_ids = set(stop_ids or ())
        past = [(None, None)] * self.n_layer
        past_len = 0

        for _ in range(int(max_new_tokens)):
            step = idx[:, -self.block_size:]

            # slide the window before feeding, so past_len + t never exceeds ctx
            overflow = past_len + step.size(1) - self.block_size
            if overflow > 0 and past[0][0] is not None:
                past = [(k[:, :, overflow:, :], v[:, :, overflow:, :])
                        for k, v in past]
                past_len -= overflow

            logits, _, past = self.forward(step, past_kv=past, past_len=past_len)
            past_len += step.size(1)
            logits = logits[:, -1, :] / max(1e-8, float(temperature))
            if top_k:
                k = min(int(top_k), logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = torch.where(logits < v[:, -1:],
                                     torch.full_like(logits, float("-inf")), logits)
            if top_p:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                # keep the first token above the cutoff, drop the rest
                drop = cum - probs > float(top_p)
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() in stop_ids:
                break
        return idx

    def param_summary(self):
        groups = {}
        for n, p in self.named_parameters():
            key = n.split(".")[0] if "." in n else n
            groups[key] = groups.get(key, 0) + p.numel()
        return sorted(groups.items(), key=lambda kv: -kv[1])


class MLPNet(nn.Module):
    """Plain feedforward net. Reaches ~40 parameters or ~500M, no attention.

    The loss follows `out_activation`:
      out_activation == 'none'  -> regression, MSE loss
      out_activation == 'logit' or n_outputs > 1 -> classification
    A 1-output net with out_activation='none' is regression with MSE; with
    'logit' it is binary classification with BCE-with-logits (numerically
    stabler than routing 1 logit through softmax).
    """

    def __init__(self, n_features, hidden, n_outputs=1, activation="relu",
                 out_activation="none", dropout=0.0, residual=False):
        super().__init__()
        self.cfg = dict(n_features=int(n_features),
                        hidden=[int(h) for h in hidden],
                        n_outputs=int(n_outputs), activation=activation,
                        out_activation=out_activation, dropout=float(dropout),
                        residual=bool(residual))
        self.loss_kind = "regression" if out_activation in ("none", "linear") \
            else "classification"

        layers, widths = [], [self.cfg["n_features"]] + self.cfg["hidden"]
        if self.cfg["dropout"] > 0:
            layers.append(nn.Dropout(self.cfg["dropout"]))
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:        # no activation after the last hidden
                layers.append(_activation(activation))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], self.cfg["n_outputs"])
        self.out_act = _activation(out_activation)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, targets=None, past_kv=None, past_len=0):
        y = self.out_act(self.head(self.net(x)))
        loss = None
        if targets is not None:
            if self.loss_kind == "regression":
                loss = F.mse_loss(y, targets.reshape(y.shape).float())
            elif self.cfg["n_outputs"] == 1:
                loss = F.binary_cross_entropy_with_logits(
                    y.float(), targets.reshape(y.shape).float())
            else:
                loss = F.cross_entropy(y.float(), targets.reshape(-1).long())
        return y, loss, []

    def param_summary(self):
        return sorted(((n, p.numel()) for n, p in self.named_parameters()),
                      key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------

def build_model(mcfg):
    """Factory. The only place model classes get instantiated by config."""
    fam = mcfg.get("family", "gpt")
    if fam == "mlp":
        from .arch import _as_hidden
        return MLPNet(
            n_features=int(mcfg.get("n_features", 8)),
            hidden=_as_hidden(mcfg.get("hidden")),
            n_outputs=int(mcfg.get("n_outputs", 1)),
            activation=mcfg.get("activation", "relu"),
            out_activation=mcfg.get("out_activation", "none"),
            dropout=float(mcfg.get("dropout", 0.0)),
            residual=bool(mcfg.get("residual", False)),
        )
    if fam != "gpt":
        raise ValueError(f"unknown model family {fam!r} (gpt | mlp)")
    return TinyGPT(
        vocab_size=int(mcfg.get("vocab_size", 512)),
        block_size=int(mcfg.get("block_size", 128)),
        n_layer=int(mcfg.get("n_layer", 4)),
        n_head=int(mcfg.get("n_head", 4)),
        embed_dim=int(mcfg.get("embed_dim", 128)),
        ff_mult=float(mcfg.get("ff_mult", 4)),
        ff_dim=int(mcfg.get("ff_dim") or 0),
        dropout=float(mcfg.get("dropout", 0.1)),
        pos_encoding=mcfg.get("pos_encoding", "rope"),
        norm_type=mcfg.get("norm_type", "rmsnorm"),
        activation=mcfg.get("activation", "silu"),
        tie_weights=bool(mcfg.get("tie_weights", True)),
        loss_chunk=int(mcfg.get("loss_chunk", 0) or 0),
    )


def unwrap(model):
    """Peel torch.compile's OptimizedModule off, if present."""
    return getattr(model, "_orig_mod", model)


def model_dtype(name):
    return {"fp32": torch.float32, "float32": torch.float32,
            "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(
        str(name).lower(), torch.float32)
