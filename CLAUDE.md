# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`moleditpy_vasp_input_generator` — a MoleditPy plugin that writes POSCAR,
INCAR, KPOINTS and a POTCAR assembly note from the molecule open in the editor,
a CIF file, or the CIF Viewer panel. Entry point
`vasp_input_generator/__init__.py`, dialog in `main_dialog.py`, all text
generation in `writer.py` (POTCAR naming in `potentials.py`).

## Shared modules — do NOT edit them here

Some files in `vasp_input_generator/` are **vendored copies** owned by
[`moleditpy-periodic-shared`](https://github.com/HiroYokoyama/moleditpy-periodic-shared)
(`../moleditpy-periodic-shared/` locally). A plugin is installed as a
self-contained folder and cannot import from another plugin, so each carries a
byte-identical copy:

| File | `SHARED_MODULE_NAME` |
|---|---|
| `cell_model.py` | `periodic-cell-model` |
| `elements.py` | `periodic-elements` |
| `cell_preview.py` | `periodic-cell-preview` |
| `structure_panel.py` | `periodic-structure-panel` |

`.shared-versions.json` records which release each copy came from, and
`tests/test_shared_sync.py` fails if a copy no longer matches its hash. Editing
one of these files here is the mistake that check exists to catch.

**To change shared code:**

```bash
cd ../moleditpy-periodic-shared
# edit the module and its tests, bump SHARED_MODULE_VERSION in the file
python -m pytest tests/ -v
git tag cell-model-v0.8.0 && git push origin cell-model-v0.8.0
python scripts/sync_shared.py ../moleditpy_vasp_input_generator      # writes the copy + the manifest
```

Then run this repo's suite and release as usual. Their tests live in that
repository too, so they are written once rather than four times.

```bash
cd ../moleditpy-periodic-shared
python scripts/sync_shared.py ../moleditpy_vasp_input_generator --check   # drifted? non-zero exit
```

## Testing

```bash
python -m pytest tests/ -v
```

Headless; PyQt6 and RDKit are stubbed where needed. `tests/test_api.py` runs
`plugin_api_checker.py` against the main app when
`../python_molecular_editor/` exists, and skips otherwise.

`.coveragerc` omits the vendored shared modules, so the coverage figure here is
this plugin's own code. The shared modules are covered in their own repository.

## Conventions

- No module-level `run()`. The host adds its own Plugins-menu entry for any
  module exposing `run`, which would duplicate the entry registered in
  `initialize()`; the dialog opens through `_open_dialog()` instead.
- Only hard dependency beyond the host is numpy. pymatgen is optional and
  imported inside a `try` (space-group expansion for CIFs with no symop loop).
- Files are written with `newline="\n"`.
