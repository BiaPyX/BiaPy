# This file serves as a wrapper to allow Sphinx to import resunet++.py
#
# Since the original file name 'resunet++.py' is not a valid Python
# module name, we load it programmatically.
import sys
import importlib
import torch
from typing import Dict, List

# Dynamically import from the original implementation
_original_module = importlib.import_module('biapy.models.resunet++')

# Get the original class
_original_class = _original_module.ResUNetPlusPlus

# Create wrapper class that preserves all docstrings
class ResUNetPlusPlus(_original_class):
    """ResUNet++ model definition for 2D and 3D biomedical image segmentation.
    
    This implementation preserves all documentation from the original resunet++.py file.
    """
    __doc__ = _original_class.__doc__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    __init__.__doc__ = _original_class.__init__.__doc__
    
    def forward(self, x) -> Dict | torch.Tensor:
        return super().forward(x)
    forward.__doc__ = _original_class.forward.__doc__

# Re-export any other public members from original module
__all__ = ['ResUNetPlusPlus']

# Clean up
del _original_module