# This file serves as a wrapper to allow Sphinx to import resunet++.py
#
# Since the original file name 'resunet++.py' is not a valid Python
# module name, we load it programmatically.

from biapy.models.resunet_plus_plus import ResUNetPlusPlus

# Re-export the class with the same name
ResUNetPlusPlus = ResUNetPlusPlus