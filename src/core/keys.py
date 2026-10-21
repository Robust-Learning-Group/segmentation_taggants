"""Domain-agnostic key generation logic and metadata schemas."""

import secrets
from typing import Any


def generate_random_seed() -> int:
    """Generate a cryptographically secure random integer seed.

    Returns:
        32-bit positive integer seed.
    """
    return secrets.randbits(32)
