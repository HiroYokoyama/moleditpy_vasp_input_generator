# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`moleditpy_vasp_input_generator` — a MoleditPy plugin that writes POSCAR,
INCAR, KPOINTS and a POTCAR assembly note from the molecule open in the editor,
a CIF file, or the CIF Viewer panel. Entry point
`vasp_input_generator/__init__.py`, dialog in `main_dialog.py`, all text
generation in `writer.py` (POTCAR naming in `potentials.py`).

## Shared modules — read this before editing

Some files in `vasp_input_generator/` are **not owned by this repository**. A
byte-identical copy of each lives in every periodic plugin, because these
plugins ship independently and cannot import from one another:

| File | `SHARED_MODULE_NAME` | Version | Also in |
|---|---|---|---|
| `cell_model.py` | `periodic-cell-model` | 0.7.0 | Quantum ESPRESSO, CP2K, Slab Builder |
| `elements.py` | `periodic-elements` | 0.1.0 | Quantum ESPRESSO, CP2K, Slab Builder |
| `cell_preview.py` | `periodic-cell-preview` | 0.1.0 | Quantum ESPRESSO, CP2K, Slab Builder |
| `structure_panel.py` | `periodic-structure-panel` | 0.5.0 | Quantum ESPRESSO, CP2K |

Sibling repositories under `DEV_MAIN/`:
`moleditpy_quantum_espresso_input_generator/qe_input_generator/`,
`moleditpy_cp2k_input_generator/cp2k_input_generator/`,
`moleditpy_slab_builder/slab_builder/`.

**The rule when you change one of these files:**

1. Bump its `SHARED_MODULE_VERSION` (the module's own version, independent of
   `PLUGIN_VERSION`).
2. Copy the file verbatim over every other copy listed above — the copies must
   stay byte-identical, so make the edit once and copy, never edit each in turn.
3. Update the pinned version in each repo's test suite
   (`tests/test_structure_panel.py`, or `tests/test_cell_model.py` in the Slab
   Builder). The pin is what makes a stale copy fail loudly instead of drifting.
4. Run all four test suites, not just this one.

```bash
cd G:/DEV_MAIN
md5sum moleditpy_*/*/cell_model.py    # every hash must match
```

`cell_model.py` imports `elements.py` for its element-symbol table, so the two
always travel together.  `cell_preview.py` imports RDKit and reaches the host's
PyVista plotter, but only inside its functions, so the plugin's declared
dependencies stay `numpy` alone. `structure_panel.py` is the only shared file that
imports PyQt6.

## Testing

```bash
python -m pytest tests/ -v
```

Headless; PyQt6 and RDKit are stubbed where needed. `tests/test_api.py` runs
`plugin_api_checker.py` against the main app when
`../python_molecular_editor/` exists, and skips otherwise.

## Conventions

- No module-level `run()`. The host adds its own Plugins-menu entry for any
  module exposing `run`, which would duplicate the entry registered in
  `initialize()`; the dialog opens through `_open_dialog()` instead.
- Only hard dependency beyond the host is numpy. pymatgen is optional and
  imported inside a `try` (space-group expansion for CIFs with no symop loop).
- Files are written with `newline="\n"`.
