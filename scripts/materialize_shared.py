#!/usr/bin/env python3
"""Copy the shared periodic modules from the ``_periodic_shared`` submodule
into the plugin package, so the package is self-contained for tests and for
the release zip.

The submodule pins one commit of moleditpy-periodic-shared; this script does
not edit or version anything itself, it only flattens what that commit holds.
Run it after ``git submodule update --init`` and again whenever the submodule
pointer moves:

    python scripts/materialize_shared.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(HERE, "_periodic_shared", "periodic_shared")
PACKAGE_DIR = os.path.join(HERE, "vasp_input_generator")

#: Not every plugin takes every module — the Slab Builder has its own dialog
#: and so has no use for the structure panel.
ALWAYS = ("cell_model.py", "elements.py", "cell_preview.py")
OPTIONAL = ("structure_panel.py",)


def files_to_copy() -> list[str]:
    names = list(ALWAYS)
    names += [n for n in OPTIONAL if os.path.isfile(os.path.join(PACKAGE_DIR, n))]
    return names


def main() -> int:
    if not os.path.isdir(SOURCE_DIR):
        print(
            "_periodic_shared is empty — run `git submodule update --init` first",
            file=sys.stderr,
        )
        return 1

    for filename in files_to_copy():
        source = os.path.join(SOURCE_DIR, filename)
        destination = os.path.join(PACKAGE_DIR, filename)
        with open(source, "rb") as handle:
            data = handle.read()
        with open(destination, "wb") as handle:
            handle.write(data)
        print(f"  {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
