"""
Reproduce the test outputs of a BioImage Model Zoo package with bioimageio.core and report how many output elements
disagree with the expected ones.

The result is printed as JSON on a line starting with ``BMZ_REPRODUCIBILITY`` so that ``run_checks.py`` does not
depend on how bioimageio.core words its validation messages. Exits with a non-zero code if the package can not be run.

Usage: python bmz_reproducibility.py <package.zip> [--weight_format pytorch_state_dict] [--device cuda]
"""
import argparse
import json

import numpy as np
from bioimageio.core import create_prediction_pipeline, load_model_description
from bioimageio.core.digest_spec import get_test_input_sample, get_test_output_sample

# Same tolerances BiaPy passes to bioimageio.core's test_model() when exporting a model
RELATIVE_TOLERANCE = 1e-3
ABSOLUTE_TOLERANCE = 1e-3
RESULT_PREFIX = "BMZ_REPRODUCIBILITY"

parser = argparse.ArgumentParser(
    description="Reproduce the test outputs of a BMZ package",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("package", help="Path to the BMZ package (.zip or rdf.yaml)")
parser.add_argument("--weight_format", default="pytorch_state_dict", help="Weight format to run")
parser.add_argument("--device", default="cpu", help="Device to run the model on, e.g. 'cpu' or 'cuda'")
args = parser.parse_args()

model = load_model_description(args.package, perform_io_checks=False)
test_input = get_test_input_sample(model)
expected = get_test_output_sample(model)
with create_prediction_pipeline(
    bioimageio_model=model, devices=[args.device], weight_format=args.weight_format
) as pipeline:
    actual = pipeline.predict_sample_without_blocking(test_input, skip_input_padding=True, skip_output_cropping=True)

result = {"weight_format": args.weight_format, "outputs": {}}
for member_id, expected_tensor in expected.members.items():
    expected_np = expected_tensor.data.to_numpy().astype(np.float32)
    actual_tensor = actual.members.get(member_id)
    if actual_tensor is None or actual_tensor.tagged_shape != expected_tensor.tagged_shape:
        result["outputs"][str(member_id)] = {
            "error": "output '{}' has shape {}, expected {}".format(
                member_id,
                None if actual_tensor is None else dict(actual_tensor.tagged_shape),
                dict(expected_tensor.tagged_shape),
            )
        }
        continue
    actual_np = actual_tensor.data.to_numpy().astype(np.float32)
    abs_diff = np.abs(actual_np - expected_np)
    # Same criterion as bioimageio.core's test_model()
    mismatched = abs_diff > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(expected_np)
    result["outputs"][str(member_id)] = {
        "mismatched": int(mismatched.sum()),
        "total": int(expected_np.size),
        "fraction": float(mismatched.mean()),
        "max_abs_diff": float(abs_diff.max()),
    }

print(RESULT_PREFIX, json.dumps(result), flush=True)
