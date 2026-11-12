import numpy as np
import pytest

from superfluid.data import char_dataset, synthetic_corpus
from superfluid.losses import cross_entropy
from superfluid.model import SuperfluidLM, TinyTransformer, train_steps
from superfluid.selftest import CHECKS, run_all


def test_all_21_checks_pass():
    results = run_all()
    failed = [r["name"] for r in results if not r["ok"]]
    assert len(results) == 21
    assert not failed, failed


def test_check_names_are_unique_and_numbered():
    names = [fn.__name__ for fn in CHECKS]
    assert len(names) == len(set(names))
    for i, name in enumerate(names, start=1):
        assert name.startswith("check_" + str(i).zfill(2) + "_")


def test_lm_ab_short_run_both_improve():
    """The honest A/B: both models must learn; no winner claimed here."""
    corpus = synthetic_corpus(seed=2)
    ds = char_dataset(corpus, seq_len=16)
    V = len(ds["vocab"])
    sf = SuperfluidLM(V, d_model=16, n_layers=1, seed=30)
    tt = TinyTransformer(V, d_model=16, n_layers=1, seed=31)
    c_sf = train_steps(sf, corpus, steps=20, lr=3e-3, seed=30)
    c_tt = train_steps(tt, corpus, steps=20, lr=3e-3, seed=31)
    assert c_sf[-1] < c_sf[0]
    assert c_tt[-1] < c_tt[0]

