# MoleditPy VASP Input Generator

A [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor) plugin that
writes **POSCAR / INCAR / KPOINTS** for VASP from a molecule or a crystal
structure, with a live preview of every file before you save.

## What it does

- **Three structure sources**
  - the molecule currently open in MoleditPy, wrapped in an orthorhombic vacuum box
  - a `.cif` file loaded from disk (asymmetric unit expanded with the CIF's own
    symmetry operations)
  - the structure already open in the **CIF Viewer** plugin panel, copied across
    without re-reading the file
- **Supercells** — independent a/b/c repeats applied to any source
- **Surface slab builder** — cut any (hkl) surface from a bulk structure: layers,
  vacuum thickness, a termination shift that slides the cut window through the
  cell, and an option to put **c** along the surface normal. Hexagonal
  Miller-Bravais indices `(hkil)` are accepted and folded to `(hkl)`
- **Vacuum per axis** — pad one axis only (the usual slab setup) instead of a
  uniform box
- **Task presets** — single point, ionic relaxation, ionic + cell relaxation,
  DOS, non-SCF band structure, NVT molecular dynamics
- **INCAR control** — functional (PBE / PBEsol / RPBE / LDA), ENCUT, EDIFF/EDIFFG,
  smearing, spin polarisation with an initial MAGMOM guess, dispersion
  (D2 / D3 / D3-BJ / TS), PREC / ALGO / LREAL / NELM / NCORE, plus a free-text
  block for anything else
- **K-points** — Gamma-only, Gamma-centred or Monkhorst-Pack meshes with shifts,
  or an automatic mesh from a target reciprocal-space spacing
- **Selective dynamics** — freeze the atoms currently selected in MoleditPy
- **POTCAR hint** — a ready-to-run `cat` command using the VASP *recommended*
  PAW set, in the same element order as the POSCAR (POTCAR files themselves are
  licensed VASP data and are never generated)

- **Checks** — a warning strip flags the classic mistakes: a molecule sampled
  with a dense k-mesh, k-points across a slab's vacuum, too little vacuum,
  cut-offs that are too low, a relaxation with zero steps, and charged cells

- **Charged systems** are detected from the molecule and flagged with the NELECT hint

## Install

Plugin Manager → install from the MoleditPy plugin registry, or drop the
`vasp_input_generator` folder into your MoleditPy plugins directory.

Requires `numpy` (already a MoleditPy dependency). `pymatgen` is optional and is
used only to expand a CIF Viewer structure that holds nothing but the asymmetric
unit; reading a `.cif` file directly never needs it.

## Use

**File → Export → VASP Input (POSCAR/INCAR)...**

Pick a structure source on the *Structure* tab, set up the run on *Calculation*
and *K-points*, check the *Preview* tab, then **Save Files...** to write
`POSCAR`, `INCAR`, `KPOINTS` and `POTCAR.readme` into a directory of your choice.

## Shared modules

`cell_model.py`, `elements.py` and `structure_panel.py` are shared byte-for-byte
with the Quantum ESPRESSO and CP2K input generator plugins. Each carries its own
`SHARED_MODULE_NAME` / `SHARED_MODULE_VERSION`, independent of `PLUGIN_VERSION`:
change one, bump its version, copy it to the sibling plugins, and update the pin
in each `tests/test_structure_panel.py`.

The CIF reader, lattice construction and symmetry de-duplication are derived from
the [MoleditPy CIF Viewer](https://github.com/HiroYokoyama/moleditpy_cif_viewer)
plugin's parser.

## Tests

```bash
python -m pytest tests/ -v
```

## Licence

GNU General Public License v3.0 — see [LICENSE](LICENSE).
