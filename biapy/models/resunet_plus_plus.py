# This file serves as a wrapper to allow Sphinx to import resunet++.py
#
# Since the original file name 'resunet++.py' is not a valid Python
# module name, we use importlib to load it programmatically.

import importlib.util
import sys
import os

# Define the full path to the original file
mdl = importlib.util.import_module("biapy.models.resunet++")
model_file = os.path.abspath(mdl.__file__)
current_dir = os.path.dirname(model_file)
original_file_path = os.path.join(current_dir, "resunet++.py")

# Create a module spec for the original file
spec = importlib.util.spec_from_file_location("resunet++", original_file_path)

# Create a new module based on the spec
resunet_module = importlib.util.module_from_spec(spec)

# Add the module to the system path so other imports can find it
sys.modules["resunet++"] = resunet_module

# Execute the module to load its contents (e.g., classes, functions)
spec.loader.exec_module(resunet_module)

# Now, we can access the class from the loaded module and make it
# available in our wrapper.
from resunet_module import ResUNetPlusPlus