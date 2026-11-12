"""A small deterministic char corpus - no downloads, reproducible."""

import numpy as np


def char_dataset(text, seq_len=32):
    """Encode text to ids; y is x shifted by one at batch time."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    return {"ids": ids, "vocab": chars, "stoi": stoi, "seq_len": int(seq_len)}


def batch(ds, batch_size, rng):
    """Random (x, y) batches with y the next character after x."""
    ids, L = ds["ids"], ds["seq_len"]
    n = ids.size - L - 1
    starts = rng.integers(0, n, size=batch_size)
    x = np.stack([ids[s:s + L] for s in starts])
    y = np.stack([ids[s + 1:s + L + 1] for s in starts])
    return x, y


def synthetic_corpus(seed=0, repeats=40):
    """Deterministic pseudo-text: word cycles plus number pairs.

    Same seed, same corpus - the LM tests never download anything."""
    rng = np.random.default_rng(seed)
    words = ["fluid", "band", "edge", "gate", "sheaf", "route", "jam", "spectral"]
    lines = []
    for _ in range(repeats):
        w = " ".join(rng.choice(words, size=6))
        a, b = rng.integers(10, 99, size=2)
        lines.append(w + " " + str(int(a)) + " " + str(int(b)) + ".")
    return "\n".join(lines)

