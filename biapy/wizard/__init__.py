"""
Logic behind the configuration wizards of BiaPy's front-ends (BiaPy-GUI, the Fiji plugin...), so all of them
check the data and create the configuration files in the same way:

* :mod:`biapy.wizard.data_check`: checks the data folders selected and gathers what is needed from them
  (channels, classes and the data fingerprint).
* :mod:`biapy.wizard.preview`: a few samples of the data checked, for the user to verify them.
* :mod:`biapy.wizard.config_builder`: configuration from the wizard's answers, adapted to the data
  (:mod:`biapy.wizard.planning`, in the spirit of nnU-Net).

Importing them is quick: BiaPy's readers (which load PyTorch, a few seconds) are imported on first use, so a
front-end can start a Python process and warm it up in the background while the user answers.
"""

# What the front-ends use of this package (module.name). tests/test_wizard_api.py compares their signatures and the
# keys of their results with tests/wizard_api.json, so any change to them is noticed in the CI.
INTERFACE = (
    "data_check.check_folder",
    "preview.preview",
    "config_builder.build_config",
    "config_builder.write_config",
    "config_builder.WizardError",
    "config_builder.to_jsonable",
    "planning.query_gpus",
    "instance_seg.REPRESENTATIONS",
)

# Version of INTERFACE, so the front-ends can tell whether an installed BiaPy works with them without importing it
# (they read this line) and whatever BiaPy's version number says (development installs keep the number of the last
# release). It must increase when INTERFACE changes in a way that breaks a front-end written for the previous
# version: something removed or renamed, an argument that becomes required, a result key removed... (the test above
# tells). The front-ends accept only the version they were written for, so they must be updated then. Additions
# (optional arguments, result keys, functions) don't need it. Neither does a change in the planning strategy.
API_VERSION = 1
