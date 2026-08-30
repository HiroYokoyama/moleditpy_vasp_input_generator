# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`moleditpy_vasp_input_generator` — a MoleditPy plugin that writes POSCAR,
INCAR, KPOINTS and a POTCAR assembly note from the molecule open in the editor,
a CIF file, or the CIF Viewer panel. Entry point
`vasp_input_generator/__init__.py`, dialog in `main_dialog.py`, all text
generation in `writer.py` (POTCAR naming in `potentials.py`).

## Shared modules — do NOT edit them here

`cell_model.py`, `elements.py`, `cell_preview.py`, `structure_panel.py` in
`vasp_input_generator/` are **not committed to this repo**. They are
materialized at test/build time from the `_periodic_shared` git submodule,
which pins one commit of
[`moleditpy-periodic-shared`](https://github.com/HiroYokoyama/moleditpy-periodic-shared).
A plugin is installed as a self-contained folder and cannot import from
another plugin, so CI (and local dev) flattens the submodule's files into the
package directory before anything runs:

```bash
git submodule update --init
python scripts/materialize_shared.py
```

`tests/test_shared_materialized.py` fails loudly if this step was skipped.
`.gitignore` excludes the materialized copies — editing them in place is
pointless, `materialize_shared.py` overwrites them from the submodule on the
next run.

**To pick up new shared code:**

```bash
cd _periodic_shared && git pull origin main && cd ..
git add _periodic_shared
git commit -m "chore: pull in the latest moleditpy-periodic-shared"
python scripts/materialize_shared.py
python -m pytest tests/ -v
```

Editing the shared modules themselves happens in that repository, not here —
see its own `CLAUDE.md`. Their tests live there too, written once rather than
four times.

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
