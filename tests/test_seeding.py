"""Seeding determinism - the reproduction promise of the release.

These guard the invariants that make same-seed runs repeatable; a regression
here (e.g. a dropped np.random.seed in seed_worker) would silently change
augmentation RNG and every reported number while other tests stay green.
"""

import random

import numpy as np
import torch


def test_set_seed_makes_draws_reproducible():
    from lvt_eg.common.seeding import set_seed

    set_seed(42)
    a = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    set_seed(42)
    b = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    assert a == b


def test_make_generator_deterministic_and_seed_sensitive():
    from lvt_eg.common.seeding import make_generator

    d1 = torch.rand(5, generator=make_generator(42)).tolist()
    d2 = torch.rand(5, generator=make_generator(42)).tolist()
    d3 = torch.rand(5, generator=make_generator(43)).tolist()
    assert d1 == d2      # same seed -> same shuffle stream
    assert d1 != d3      # different seed -> different stream


def test_seed_worker_reseeds_from_torch_initial_seed():
    from lvt_eg.common.seeding import seed_worker

    torch.manual_seed(123)          # sets torch.initial_seed() == 123
    seed_worker(0)
    a = (float(np.random.rand()), random.random())
    torch.manual_seed(123)
    seed_worker(0)
    b = (float(np.random.rand()), random.random())
    assert a == b                   # worker RNG derives deterministically from torch seed

    torch.manual_seed(999)
    seed_worker(0)
    c = (float(np.random.rand()), random.random())
    assert a != c                   # a different torch seed changes the worker RNG
