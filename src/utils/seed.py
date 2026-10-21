"""Random seed utilities for reproducible experiments."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: The integer seed value to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
