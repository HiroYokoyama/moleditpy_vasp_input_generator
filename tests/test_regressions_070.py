"""Regressions fixed in 0.7.0; each test fails on 0.6.3."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402
from vasp_input_generator import writer  # noqa: E402

from shared_fixtures import CUBIC_CIF, _FakeMol  # noqa: E402


def _incar_tags(text):
    tags = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("!"):
            key, _, value = line.partition("=")
            tags[key.strip()] = value.strip()
    return tags


def test_md_switches_symmetry_off():
    """Symmetrised forces would pin every atom to its symmetry element."""
    text = writer.build_incar({"task": "Molecular dynamics (NVT)", "isym": 2})
    assert _incar_tags(text)["ISYM"] == "0"
    assert text.count("ISYM") == 1


def test_static_run_keeps_the_chosen_symmetry():
    assert _incar_tags(writer.build_incar({"isym": 2}))["ISYM"] == "2"


def test_lda_uses_the_lda_potcars():
    cell = cm.parse_cif(CUBIC_CIF)
    notes = writer.build_potcar_notes(cell, {"functional": "LDA (CA)"})
    assert "potpaw_LDA" in notes and "potpaw_PBE" not in notes


def test_pbe_keeps_the_pbe_potcars():
    cell = cm.parse_cif(CUBIC_CIF)
    assert "potpaw_PBE" in writer.build_potcar_notes(cell, {"functional": "PBEsol"})


def _molecule_with_a_ghost():
    # O, a Bq probe, H, H: the probe is dropped, so the hydrogens move up a row.
    return _FakeMol(
        ["O", "*", "H", "H"],
        [[0, 0, 0], [0, 0, 1.0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        customs=[None, "Bq", None, None],
    )


def test_selection_follows_the_molecule_after_a_ghost_is_dropped():
    """Freezing MoleditPy atom 3 (the last H) must freeze that H, not row 3."""
    cell = cm.molecule_to_cell(_molecule_with_a_ghost(), padding=5.0)
    frozen = writer.frozen_cell_indices(
        cell, {"selective_dynamics": True, "frozen_indices": [3]}
    )
    assert [cell.atoms[i].source_index for i in frozen] == [3]


def test_selection_covers_every_supercell_image():
    cell = cm.make_supercell(cm.molecule_to_cell(_molecule_with_a_ghost(), padding=5.0), [2, 1, 1])
    frozen = writer.frozen_cell_indices(
        cell, {"selective_dynamics": True, "frozen_indices": [0], "supercell": [2, 1, 1]}
    )
    assert sorted(frozen) == [0, 3]
    assert all(cell.atoms[i].element == "O" for i in frozen)


def test_ghost_probe_is_not_written_as_boron():
    cell = cm.molecule_to_cell(_molecule_with_a_ghost(), padding=5.0)
    poscar = writer.build_poscar(cell)
    assert " B " not in poscar.splitlines()[5] + " "
    assert any("ghost" in message for message in writer.validate(cell))
