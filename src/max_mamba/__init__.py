import random

import numpy as np

from max_mamba.config import Mamba2Config
from max_mamba.model import Mamba2Model, Mamba2ForCausalLM


def seed_all(seed: int):
    """Set the seed for numpy, random, and torch (if available)."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


seed_all(1234)
