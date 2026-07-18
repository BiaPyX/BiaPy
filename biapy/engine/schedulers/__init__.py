"""
biapy.engine.schedulers
=======================

Schedulers for training, including custom learning rate schedules.
"""

from . import warmup_cosine_decay
from . import warmup_reduce_on_plateau

__all__ = [
    "warmup_cosine_decay",
    "warmup_reduce_on_plateau",
]
