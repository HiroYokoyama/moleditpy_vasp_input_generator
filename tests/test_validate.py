import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402
from vasp_input_generator import writer  # noqa: E402


@pytest.fixture
def molecule_cell():
    return cm.cell_from_molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        padding=6.0,
    )


@pytest.fixture
def bulk_cell():
    lengths, angles = (4.0, 4.0, 4.0), (90.0, 90.0, 90.0)
    lattice = cm.cell_vectors(lengths, angles)
    atom = cm.CellAtom("Cu1", "Cu", np.zeros(3), np.zeros(3))
    return cm.Cell("sc", lengths, angles, lattice, (atom,), source="cif")


@pytest.fixture
def slab_cell():
    """A slab as it arrives from the Slab Builder: a CIF with a vacuum layer."""
    lengths, angles = (4.0, 4.0, 20.0), (90.0, 90.0, 90.0)
    lattice = cm.cell_vectors(lengths, angles)
    atoms = tuple(
        cm.CellAtom(
            f"Cu{index + 1}",
            "Cu",
            np.array([0.0, 0.0, height / 20.0]),
            np.array([0.0, 0.0, height]),
        )
        for index, height in enumerate((0.0, 2.0, 4.0))
    )
    return cm.Cell("slab", lengths, angles, lattice, atoms, source="cif")


def joined(messages):
    return " | ".join(messages)


# -- VASP ------------------------------------------------------------------


def test_no_warnings_for_a_sane_bulk_run(bulk_cell):
    assert writer.validate(bulk_cell, {"kmesh": [8, 8, 8]}) == []


def test_molecule_with_a_dense_mesh_is_flagged(molecule_cell):
    assert "Gamma-only" in joined(writer.validate(molecule_cell, {"kmesh": [4, 4, 4]}))


def test_molecule_with_gamma_only_is_quiet(molecule_cell):
    messages = writer.validate(molecule_cell, {"kpoint_mode": "Gamma-only", "padding": 8.0})
    assert "Gamma-only is enough" not in joined(messages)


def test_tetrahedron_with_too_few_kpoints(bulk_cell):
    messages = writer.validate(
        bulk_cell, {"kpoint_mode": "Gamma-only", "smearing": writer.SMEARING[3]}
    )
    assert "tetrahedron" in joined(messages)


def test_large_sigma_with_methfessel_paxton(bulk_cell):
    messages = writer.validate(
        bulk_cell, {"kmesh": [8, 8, 8], "smearing": writer.SMEARING[1], "sigma": 0.5}
    )
    assert "entropy" in joined(messages)


def test_metal_smearing_on_a_molecule(molecule_cell):
    messages = writer.validate(
        molecule_cell, {"kpoint_mode": "Gamma-only", "smearing": writer.SMEARING[1]}
    )
    assert "discrete levels" in joined(messages)


def test_relaxation_with_zero_steps(bulk_cell):
    messages = writer.validate(bulk_cell, {"task": "Relax ions", "nsw": 0, "kmesh": [8, 8, 8]})
    assert "NSW=0" in joined(messages)


def test_band_structure_reminds_about_chgcar(bulk_cell):
    messages = writer.validate(
        bulk_cell, {"task": "Band structure (non-SCF)", "kmesh": [8, 8, 8]}
    )
    assert "CHGCAR" in joined(messages)


def test_low_encut(bulk_cell):
    assert "ENCUT" in joined(writer.validate(bulk_cell, {"encut": 250.0, "kmesh": [8, 8, 8]}))


def test_variable_cell_pulay_warning(bulk_cell):
    messages = writer.validate(
        bulk_cell, {"task": "Relax ions + cell", "encut": 520.0, "kmesh": [8, 8, 8]}
    )
    assert "Pulay" in joined(messages)


def test_selective_dynamics_without_a_selection(bulk_cell):
    messages = writer.validate(
        bulk_cell, {"selective_dynamics": True, "frozen_indices": [], "kmesh": [8, 8, 8]}
    )
    assert "nothing is selected" in joined(messages)


def test_charged_cell_warns_about_nelect(molecule_cell):
    messages = writer.validate(
        molecule_cell, {"kpoint_mode": "Gamma-only", "padding": 8.0}, net_charge=-1
    )
    assert "NELECT" in joined(messages)


def test_thin_vacuum_is_flagged(molecule_cell):
    messages = writer.validate(molecule_cell, {"kpoint_mode": "Gamma-only", "padding": 3.0})
    assert "vacuum" in joined(messages)


def test_thick_vacuum_is_quiet(molecule_cell):
    messages = writer.validate(molecule_cell, {"kpoint_mode": "Gamma-only", "padding": 7.0})
    assert "vacuum separates" not in joined(messages)


def test_per_axis_vacuum_uses_the_smallest_axis(molecule_cell):
    messages = writer.validate(
        molecule_cell,
        {"kpoint_mode": "Gamma-only", "per_axis_padding": True, "padding_axes": [2.0, 9.0, 9.0]},
    )
    assert "4.0 A of vacuum" in joined(messages)


# -- structure faults reach the VASP warnings ---------------------------------


def test_a_partially_occupied_cif_is_flagged(bulk_cell):
    disordered = cm.Cell(
        bulk_cell.name,
        bulk_cell.lengths,
        bulk_cell.angles,
        bulk_cell.lattice,
        (cm.CellAtom("Fe1", "Fe", np.zeros(3), np.zeros(3), 0.5),),
        source="cif",
    )
    assert "partially occupied" in joined(writer.validate(disordered, {"kmesh": [8, 8, 8]}))


def test_a_left_handed_lattice_is_flagged():
    lattice = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -4.0]])
    cell = cm.Cell("lh", (4.0, 4.0, 4.0), (90.0, 90.0, 90.0), lattice, ())
    assert "left-handed" in joined(writer.validate(cell, {"kmesh": [8, 8, 8]}))


def test_overlapping_atoms_are_flagged(bulk_cell):
    doubled = cm.Cell(
        bulk_cell.name,
        bulk_cell.lengths,
        bulk_cell.angles,
        bulk_cell.lattice,
        (
            cm.CellAtom("Cu1", "Cu", np.zeros(3), np.zeros(3)),
            cm.CellAtom("Cu2", "Cu", np.array([0.01, 0.0, 0.0]), np.array([0.04, 0.0, 0.0])),
        ),
        source="cif",
    )
    assert "closer than" in joined(writer.validate(doubled, {"kmesh": [8, 8, 8]}))
