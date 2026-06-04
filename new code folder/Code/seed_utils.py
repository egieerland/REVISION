"""Utilities for seeding random number generators."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, *, set_hash_seed: bool = False) -> int:
    """Seed Python and NumPy RNGs and return the normalized seed."""
    normalized = int(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(normalized)
    return normalized


def make_random(seed: Optional[int]) -> random.Random:
    """Create a local RNG instance without mutating global state."""
    if seed is None:
        return random.Random()
    return random.Random(int(seed))
