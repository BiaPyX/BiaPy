"""N2V pixel manipulators must pickle: DataLoader workers on Windows/macOS ('spawn') pickle the generator."""

import pickle

import numpy as np
import pytest

from biapy.engine.denoising import get_value_manipulation

MANIPULATORS = ["normal_withoutCP", "mean", "median", "uniform_withCP", "uniform_withoutCP",
                "normal_additive", "normal_fitted", "identity"]


@pytest.mark.parametrize("name", MANIPULATORS)
def test_manipulator_pickles(name):
    fn = get_value_manipulation(name, 2)
    restored = pickle.loads(pickle.dumps(fn))
    patch = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    coords = (np.array([3, 8]), np.array([5, 12]))
    np.random.seed(0)
    expected = fn(patch, coords, 2)
    np.random.seed(0)
    assert np.allclose(restored(patch, coords, 2), expected)
