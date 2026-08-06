"""The vendored shared modules must match the release they claim to be.

Plugins cannot import from one another, so each carries a byte-identical copy of
the modules owned by ``moleditpy-periodic-shared``.  ``.shared-versions.json``
records which release each copy came from; ``scripts/sync_shared.py`` in that
repository writes the file and the manifest together, so a copy edited by hand
here fails this check instead of drifting quietly.
"""

import hashlib
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, ".shared-versions.json")

_NAME_RE = re.compile(r'^SHARED_MODULE_NAME\s*=\s*["\'](.+?)["\']', re.M)
_VERSION_RE = re.compile(r'^SHARED_MODULE_VERSION\s*=\s*["\'](.+?)["\']', re.M)


def _package_dir():
    for entry in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, entry)
        if entry in ("tests", "scripts"):
            continue
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            return path
    raise AssertionError("no plugin package found")


def _digest(path):
    """Hash with line endings normalised to LF.

    Git rewrites line endings on checkout, so the same commit has different
    bytes on a Windows runner than on a Linux one; hashing raw bytes made this
    check fail on Windows for files nobody had touched.
    """
    with open(path, "rb") as handle:
        content = handle.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_the_manifest_exists():
    assert os.path.isfile(MANIFEST_PATH), (
        "run scripts/sync_shared.py from moleditpy-periodic-shared"
    )


@pytest.mark.parametrize("filename", sorted(_manifest()))
def test_the_vendored_copy_matches_its_release(filename):
    expected = _manifest()[filename]
    path = os.path.join(_package_dir(), filename)
    assert os.path.isfile(path), f"{filename} is missing from the package"
    digest = _digest(path)
    assert digest == expected["sha256"], (
        f"{filename} does not match {expected['module']} {expected['version']}. "
        "Edit it in moleditpy-periodic-shared and re-run scripts/sync_shared.py."
    )


@pytest.mark.parametrize("filename", sorted(_manifest()))
def test_the_copy_declares_the_version_the_manifest_records(filename):
    expected = _manifest()[filename]
    text = open(os.path.join(_package_dir(), filename), encoding="utf-8").read()
    assert _NAME_RE.search(text).group(1) == expected["module"]
    assert _VERSION_RE.search(text).group(1) == expected["version"]
