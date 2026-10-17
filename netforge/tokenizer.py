"""
Tokenizers: zero dependencies, deterministic, pure Python.

Two classes, one duck-typed interface:
    .vocab_size  .encode(str) -> list[int]  .decode(list[int]) -> str
    .to_config() -> dict   .save(path)  .load(path)  .train(text)

CharTokenizer
    98 symbols (1 special + printable ASCII + \\t\\n). No training needed.
    Order of magnitude worse compression than BPE, but it round-trips
    losslessly for any byte string and is the right choice for a model with
    fewer than a few million parameters - a 128-token context is 128 real
    characters instead of 20 whole words.

BPETokenizer
    Byte-level BPE trained on the corpus. ID layout is
    [specials][256 raw bytes][merge results], so every possible byte is always
    representable and you can never produce <unk> from valid text.

Training uses evenly-spaced *contiguous* blocks rather than strided sampling:
BPE learns from adjacent-pair statistics, and striding the corpus destroys
exactly the adjacency it is trying to count.
"""

import json
import os
import re

SPECIALS = ("<unk>", "<bos>", "<eos>", "<pad>")


# ---------------------------------------------------------------------------
# Char
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, specials=None):
        self.specials = list(specials) if specials else list(SPECIALS)
        chars = [chr(c) for c in range(32, 127)] + ["\n", "\t"]
        self.itos = self.specials + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        unk = self.stoi.get("<unk>", 0)
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(self.itos) and self.itos[i] not in self.specials:
                out.append(self.itos[i])
        return "".join(out)

    def train(self, text, **kw):
        """No-op: the character set is fixed. Present for interface parity."""
        return self

    def to_config(self):
        return {"type": "char", "vocab_size": self.vocab_size,
                "specials": list(self.specials), "merges": []}

    def save(self, path):
        _write_json({"type": "char", "specials": self.specials}, path)

    @classmethod
    def load(cls, path):
        d = _read_json(path)
        return cls(specials=d.get("specials"))


# ---------------------------------------------------------------------------
# Byte-level BPE
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\s*\S+|\s+")


class BPETokenizer:
    """Byte-level BPE with a word-piece pre-tokenizer.

    Pre-tokenising keeps encoding linear: without it, BPE on a single 10 MB
    string degenerates. The regex keeps leading whitespace attached to the
    following word (GPT-style), so "the cat" and " the cat" tokenise the same
    way inside a longer sequence.
    """

    def __init__(self, merges=None, specials=None):
        self.specials = list(specials) if specials else list(SPECIALS)
        self.merges = [tuple(m) for m in (merges or [])]
        self._rebuild()

    def _rebuild(self):
        self.itos = list(self.specials) + \
            [chr(b) for b in range(256)] + \
            ["".join(p) for p in self.merges]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.ranks = {p: i for i, p in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return len(self.specials) + 256 + len(self.merges)

    # -- encoding -----------------------------------------------------------

    def _bpe(self, token):
        """Apply merges to a single word, repeatedly, lowest rank first.

        Operates in latin-1 space, where one character is exactly one byte.
        That is what makes the 256 single-byte entries in the vocabulary
        meaningful and lets a merge learned over bytes be looked up over a
        decoded string.
        """
        if len(token) <= 1:
            return [token]
        word = list(token)
        while len(word) > 1:
            pairs = [(self.ranks.get((word[i], word[i + 1]), -1), i)
                     for i in range(len(word) - 1)]
            pairs = [p for p in pairs if p[0] >= 0]
            if not pairs:
                break
            _, idx = min(pairs)
            word = word[:idx] + [word[idx] + word[idx + 1]] + word[idx + 2:]
        return word

    def encode(self, text):
        ids = []
        fallback = self.stoi.get(chr(0), 0)
        for chunk in _WORD_RE.findall(text):
            # latin-1 view: one char == one byte, matching how merges were
            # trained. A multi-byte UTF-8 character becomes several tokens,
            # which round-trips exactly through decode().
            lat = chunk.encode("utf-8").decode("latin-1")
            for piece in self._bpe(lat):
                i = self.stoi.get(piece)
                if i is not None:
                    ids.append(i)
                else:
                    ids.extend(self.stoi.get(chr(b), fallback)
                               for b in piece.encode("latin-1", errors="replace"))
        return ids

    def decode(self, ids):
        buf = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(self.itos):
                s = self.itos[i]
                if s not in self.specials:
                    buf.append(s)
        raw = "".join(buf)
        try:
            return raw.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
        except Exception:
            return raw

    # -- training -----------------------------------------------------------

    def train(self, text, vocab_size=8000, min_count=2, max_chars=256000,
              verbose=False):
        """Learn merges from `text`. Deterministic: ties break on the pair text.

        `max_chars` caps the corpus by taking evenly-spaced *contiguous*
        blocks. Striding single characters would look cheaper and is wrong -
        it destroys the adjacent-pair statistics BPE is trying to learn.
        """
        vocab_size = int(vocab_size)
        if vocab_size <= len(self.specials) + 256:
            return self          # 256 raw bytes already fill it; no merges fit
        text = _sample_contiguous(text, max_chars)
        target_merges = max(0, vocab_size - len(self.specials) - 256)

        chunks = _WORD_RE.findall(text)
        words = {}
        for c in chunks:
            # latin-1 view so a "symbol" here is exactly one byte, and the
            # merge pairs below are strings that _bpe() can look up directly
            b = list(c.encode("utf-8").decode("latin-1"))
            if len(b) < 2:
                continue
            key = tuple(b)
            words[key] = words.get(key, 0) + 1

        target_merges = vocab_size - len(self.specials) - 256
        merges = []
        for _ in range(target_merges):
            freq = {}
            for word, c in words.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    freq[p] = freq.get(p, 0) + c
            if not freq:
                break
            best_freq = max(freq.values())
            if best_freq < min_count:
                break
            # deterministic tie-break: smallest pair in tuple order
            best = min(p for p, c in freq.items() if c == best_freq)
            merges.append(best)

            merged, n = {}, 0
            for word, c in words.items():
                new_word, i = [], 0
                while i < len(word):
                    if (i < len(word) - 1 and
                            (word[i], word[i + 1]) == best):
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                key = tuple(new_word)
                merged[key] = merged.get(key, 0) + c
                n += c
            words = merged
            if verbose:
                print(f"  merge {len(merges)}/{target_merges}: "
                      f"{best[0]!r}+{best[1]!r} ({best_freq})")
            if n == 0:
                break

        self.merges = merges
        self._rebuild()
        return self

    # -- persistence --------------------------------------------------------

    def to_config(self):
        return {"type": "bpe", "vocab_size": self.vocab_size,
                "specials": list(self.specials), "merges": [list(m) for m in self.merges]}

    def save(self, path):
        _write_json({"type": "bpe", "specials": self.specials,
                     "merges": [list(m) for m in self.merges]}, path)

    @classmethod
    def load(cls, path):
        d = _read_json(path)
        return cls(merges=d.get("merges"), specials=d.get("specials"))


# ---------------------------------------------------------------------------

def build_tokenizer(tcfg, corpus_text=""):
    """Factory used by training and by the GUI's tokenizer tab.

    A saved tokenizer file wins. Otherwise a BPE tokenizer is trained on the
    corpus if the config asks for it; a char tokenizer needs no training.
    """
    path = (tcfg or {}).get("tokenizer_file") or ""
    kind = (tcfg or {}).get("type", "char")
    if path and os.path.exists(path):
        try:
            return (BPETokenizer.load(path) if kind == "bpe" else CharTokenizer.load(path))
        except Exception:
            pass
    if kind == "bpe":
        tok = BPETokenizer()
        if corpus_text and (tcfg or {}).get("train_from_corpus", True):
            tok.train(corpus_text,
                      vocab_size=int(tcfg.get("vocab_size", 8000)),
                      max_chars=int(tcfg.get("max_corpus_chars", 256000)))
        return tok
    return CharTokenizer()


def _sample_contiguous(text, max_chars):
    """Evenly-spaced contiguous blocks, capped at `max_chars` characters."""
    if not max_chars or len(text) <= max_chars:
        return text
    blocks = 64
    size = max_chars // blocks
    if size <= 0:
        return text[:max_chars]
    step = max(1, (len(text) - size) // (blocks - 1)) if blocks > 1 else 0
    return "".join(text[i * step:i * step + size] for i in range(blocks))


def _write_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)