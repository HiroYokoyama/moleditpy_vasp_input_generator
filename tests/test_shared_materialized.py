"""The shared periodic modules must be present before anything else runs.

They are not committed here — ``scripts/materialize_shared.py`` copies them
out of the ``_periodic_shared`` submodule. A clone that skipped
``git submodule update --init`` or the materialize step gets a clear failure
here instead of a confusing import error deeper in the suite.
"""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(ROOT, "vasp_input_generator")

_REQUIRED = ("cell_model.py", "elements.py", "cell_preview.py")


@pytest.mark.parametrize("filename", _REQUIRED)
def test_the_shared_module_is_materialized(filename):
    path = os.path.join(PACKAGE_DIR, filename)
    assert os.path.isfile(path), (
        f"{filename} is missing. Run `git submodule update --init` then "
        "`python scripts/materialize_shared.py`."
    )
