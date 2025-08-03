"""
Legacy wrapper for ResUNet++ model to maintain backward compatibility.

This module provides access to the original ResUNet++ implementation from resunet++.py
while providing a Python-importable module name (resunet_plus_plus.py).
"""

import importlib
import torch
from typing import Dict, Union

# Import the original module
_original_module = importlib.import_module('biapy.models.resunet++')

# Get the original class
_original_class = _original_module.ResUNetPlusPlus

# Create wrapper class that preserves all docstrings
class ResUNetPlusPlus(_original_class):
    __doc__ = _original_class.__doc__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x) -> Union[Dict, torch.Tensor]:
        return super().forward(x)

# Clean up namespace
del _original_module