import sys
from pathlib import Path

# Put the repo root on sys.path so top-level packages (e.g. `refset`, used like in the
# notebooks) import under pytest, which otherwise only sees the tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxtyping import install_import_hook

install_import_hook("creativity_measure", "beartype.beartype")

import pytest

from creativity_measure.device import set_default_device

# Opt-in (not autouse: test_device.py needs default_device() with no override active, to test its
# own auto-detection) CPU pin for analytic-2D-toy test files, via `pytestmark = pytest.mark.usefixtures(...)`.
# Without it, Density.sample() lands on default_device()'s auto-selected CUDA on a GPU host, mismatching
# these files' hand-written CPU literal tensors -- invisible on a machine with no CUDA.


@pytest.fixture
def cpu_device():
    set_default_device("cpu")
    yield
    set_default_device(None)


@pytest.fixture(scope="module")
def cpu_device_module():
    set_default_device("cpu")
    yield
    set_default_device(None)
