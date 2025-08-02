# This file serves as a wrapper to allow Sphinx to import resunet++.py
#
# Since the original file name 'resunet++.py' is not a valid Python
# module name, we load it programmatically.
import sys
import importlib

# Dynamically import from the original implementation
_original_module = importlib.import_module('biapy.models.resunet++')

# Re-export all symbols from the original module
globals().update({
    name: getattr(_original_module, name)
    for name in dir(_original_module)
    if not name.startswith('_')
})

# Clean up
del _original_module