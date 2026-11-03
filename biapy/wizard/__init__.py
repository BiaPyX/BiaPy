"""
Logic of the configuration wizards of BiaPy's front-ends (BiaPy-GUI, the Fiji plugin), so all of them check the data
and build the configurations in the same way. The data is listed, read and checked with BiaPy's own functions
(:mod:`biapy.data`); this package adds what is specific to the wizard:

* :mod:`biapy.wizard.data_check`: checks a folder and gathers what is needed from it (channels, classes, fingerprint).
* :mod:`biapy.wizard.preview`: a few samples of the data checked, for the user to verify them.
* :mod:`biapy.wizard.config_builder`: configuration from the answers, adapted to the data (:mod:`biapy.wizard.planning`).

BiaPy's readers (which load PyTorch) are imported on first use, so a front-end can warm up a Python process while the
user answers.
"""

# What the front-ends use (module.name); tests/test_wizard_api.py checks it against tests/wizard_api.json
INTERFACE = (
    "data_check.check_folder",
    "preview.preview",
    "config_builder.build_config",
    "config_builder.write_config",
    "config_builder.WizardError",
    "config_builder.TRAINING_LENGTHS",
    "config_builder.to_jsonable",
    "planning.query_gpus",
    "instance_seg.REPRESENTATIONS",
)

# Version of INTERFACE, read from this line by the front-ends, which accept only the one they were written for.
# Increase it when a change breaks them (something removed or renamed, an argument made required...).
# 2 (BiaPy 3.7.3): tracking.
API_VERSION = 2
