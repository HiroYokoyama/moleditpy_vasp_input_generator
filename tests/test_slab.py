import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402

from test_cell_model import CUBIC_CIF, _FakeMol  # noqa: E402


@pytest.fixture
def simple_cubic():
    lengths, angles = (4.0, 4.0, 4.0), (90.0, 90.0, 90.0)
    lattice = cm.cell_vectors(lengths, angles)
    atom = cm.CellAtom("Cu1", "Cu", np.zeros(3), np.zeros(3))
    return cm.Cell("sc", lengths, angles, lattice, (atom,), source="cif")


def surface_normal(cell):
    normal = np.cross(cell.lattice[0], cell.lattice[1])
    return normal / np.linalg.norm(normal)


def reciprocal_direction(cell, miller):
    vector = np.asarray(miller, dtype=float) @ np.linalg.inv(cell.lattice).T
    return vector / np.linalg.norm(vector)


# -- Miller indices --------------------------------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        ([0, 0, 1], (0, 0, 1)),
        ([1, -1, 0], (1, -1, 0)),
        ([1, 0, -1, 0], (1, 0, 0)),      # hexagonal (10-10) -> (100)
        ([1, 0, -1, 1], (1, 0, 1)),
        ([0, 1, -1, 0], (0, 1, 0)),
        ([0, 0, 0, 1], (0, 0, 1)),       # basal plane
    ],
)
def test_normalize_miller(given, expected):
    assert cm.normalize_miller(given) == expected


def test_normalize_miller_rejects_bad_bravais_index():
    with pytest.raises(ValueError, match=r"i must equal"):
        cm.normalize_miller([1, 0, 0, 1])


def test_normalize_miller_rejects_zero_and_wrong_length():
    with pytest.raises(ValueError):
        cm.normalize_miller([0, 0, 0])
    with pytest.raises(ValueError):
        cm.normalize_miller([1, 1])


# -- surface basis ---------------------------------------------------------


@pytest.mark.parametrize("miller", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)])
def test_surface_transformation_is_unimodular(simple_cubic, miller):
    """A surface basis must re-tile the same lattice, i.e. |det T| == 1."""
    matrix = cm.surface_transformation(simple_cubic.lattice, miller)
    assert abs(round(float(np.linalg.det(matrix)))) == 1


def test_surface_transformation_rejects_zero_indices(simple_cubic):
    with pytest.raises(ValueError):
        cm.surface_transformation(simple_cubic.lattice, (0, 0, 0))


# -- slab geometry ---------------------------------------------------------


@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 0), (1, 1, 1), (2, 1, 0)])
def test_slab_c_axis_follows_the_surface_normal(simple_cubic, miller):
    slab = cm.build_slab(simple_cubic, miller, layers=2, vacuum=10.0)
    normal = surface_normal(slab)
    expected = reciprocal_direction(simple_cubic, miller)
    assert abs(abs(float(np.dot(normal, expected))) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "miller,area",
    [
        ((0, 0, 1), 16.0),                    # a^2
        ((1, 1, 0), 16.0 * np.sqrt(2.0)),     # a^2 * sqrt(2)
        ((1, 1, 1), 16.0 * np.sqrt(3.0)),     # a^2 * sqrt(3)
    ],
)
def test_slab_surface_area(simple_cubic, miller, area):
    slab = cm.build_slab(simple_cubic, miller, layers=1, vacuum=10.0)
    assert cm.surface_area(slab) == pytest.approx(area, rel=1e-9)


@pytest.mark.parametrize(
    "miller,spacing",
    [
        ((0, 0, 1), 4.0),
        ((1, 1, 0), 4.0 / np.sqrt(2.0)),
        ((1, 1, 1), 4.0 / np.sqrt(3.0)),
    ],
)
def test_slab_layer_spacing_matches_d_hkl(simple_cubic, miller, spacing):
    """Two layers must be one interplanar spacing d_hkl apart."""
    slab = cm.build_slab(simple_cubic, miller, layers=2, vacuum=10.0)
    normal = surface_normal(slab)
    heights = sorted(float(np.dot(atom.cart, normal)) for atom in slab.atoms)
    assert heights[1] - heights[0] == pytest.approx(spacing, rel=1e-9)


def test_slab_atom_count_scales_with_layers(simple_cubic):
    for layers in (1, 3, 5):
        slab = cm.build_slab(simple_cubic, (0, 0, 1), layers=layers, vacuum=10.0)
        assert len(slab.atoms) == layers


def test_slab_c_length_is_thickness_plus_vacuum(simple_cubic):
    slab = cm.build_slab(simple_cubic, (0, 0, 1), layers=3, vacuum=10.0)
    # three layers span 2 spacings of atoms but 3 of cell repeat = 12 A
    assert slab.lengths[2] == pytest.approx(22.0)


def test_slab_vacuum_is_split_across_both_faces(simple_cubic):
    slab = cm.build_slab(simple_cubic, (0, 0, 1), layers=2, vacuum=12.0)
    normal = surface_normal(slab)
    heights = [float(np.dot(atom.cart, normal)) for atom in slab.atoms]
    below = min(heights)
    above = slab.lengths[2] - max(heights)
    assert below == pytest.approx(above, abs=1e-9)


def test_slab_atoms_stay_inside_the_cell(simple_cubic):
    slab = cm.build_slab(simple_cubic, (1, 1, 1), layers=4, vacuum=15.0)
    for atom in slab.atoms:
        assert -1e-9 <= atom.fract[2] <= 1.0 + 1e-9


def test_slab_marks_its_source(simple_cubic):
    slab = cm.build_slab(simple_cubic, (0, 0, 1))
    assert slab.source == "slab"
    assert slab.name.endswith("001")


def test_slab_zero_vacuum_is_the_bulk_thickness(simple_cubic):
    slab = cm.build_slab(simple_cubic, (0, 0, 1), layers=2, vacuum=0.0)
    assert slab.lengths[2] == pytest.approx(8.0)


def test_slab_non_orthogonal_keeps_the_stacking_vector(simple_cubic):
    slab = cm.build_slab(simple_cubic, (1, 1, 1), layers=2, vacuum=10.0, orthogonal_c=False)
    normal = surface_normal(slab)
    direction = slab.lattice[2] / np.linalg.norm(slab.lattice[2])
    assert abs(float(np.dot(direction, normal))) < 1.0 - 1e-6


def _stacking_pattern(slab):
    """Atom heights measured from the bottom of the slab, in order."""
    normal = surface_normal(slab)
    heights = sorted(float(np.dot(atom.cart, normal)) for atom in slab.atoms)
    return [round(height - heights[0], 6) for height in heights]


def _bottom_atom_xy(slab):
    normal = surface_normal(slab)
    lowest = min(slab.atoms, key=lambda atom: float(np.dot(atom.cart, normal)))
    return (round(float(lowest.cart[0]), 6), round(float(lowest.cart[1]), 6))


def test_slab_termination_shift_changes_the_exposed_plane():
    """A two-plane basis has two terminations; the shift picks between them.

    In body-centred (001) the planes are evenly spaced, so the heights are
    unchanged — what swaps is which sublattice (corner or body centre) is
    exposed at the surface.
    """
    cell = cm.parse_cif(CUBIC_CIF)  # atoms at z = 0 and z = 1/2
    plain = cm.build_slab(cell, (0, 0, 1), layers=2, vacuum=10.0)
    shifted = cm.build_slab(cell, (0, 0, 1), layers=2, vacuum=10.0, shift=0.25)
    assert len(plain.atoms) == len(shifted.atoms) == 4
    assert _bottom_atom_xy(plain) == (0.0, 0.0)
    assert _bottom_atom_xy(shifted) == (2.0, 2.0)


def test_slab_shift_keeps_the_atom_count():
    cell = cm.parse_cif(CUBIC_CIF)
    for shift in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        slab = cm.build_slab(cell, (0, 0, 1), layers=3, vacuum=10.0, shift=shift)
        assert len(slab.atoms) == 6


def test_slab_shift_is_a_no_op_for_a_single_plane_basis(simple_cubic):
    """Simple cubic exposes the same plane however the window slides."""
    plain = cm.build_slab(simple_cubic, (0, 0, 1), layers=2, vacuum=10.0)
    shifted = cm.build_slab(simple_cubic, (0, 0, 1), layers=2, vacuum=10.0, shift=0.5)
    assert _stacking_pattern(plain) == _stacking_pattern(shifted)


def test_slab_shift_wraps_past_one(simple_cubic):
    a = cm.build_slab(simple_cubic, (0, 0, 1), layers=2, vacuum=10.0, shift=0.25)
    b = cm.build_slab(simple_cubic, (0, 0, 1), layers=2, vacuum=10.0, shift=1.25)
    assert _stacking_pattern(a) == _stacking_pattern(b)


def test_slab_from_two_atom_cell_keeps_stoichiometry():
    cell = cm.parse_cif(CUBIC_CIF)  # body-centred, 2 atoms
    slab = cm.build_slab(cell, (0, 0, 1), layers=2, vacuum=10.0)
    assert len(slab.atoms) == 4
    assert all(atom.element == "Fe" for atom in slab.atoms)


def test_slab_layers_are_clamped(simple_cubic):
    assert len(cm.build_slab(simple_cubic, (0, 0, 1), layers=0).atoms) == 1


# -- CIF export ------------------------------------------------------------


def test_write_cif_roundtrips_through_the_parser(simple_cubic):
    slab = cm.build_slab(simple_cubic, (1, 1, 0), layers=2, vacuum=12.0)
    reparsed = cm.parse_cif(cm.write_cif(slab), expand=False)
    assert len(reparsed.atoms) == len(slab.atoms)
    assert reparsed.lengths == pytest.approx(slab.lengths, rel=1e-6)
    assert reparsed.angles == pytest.approx(slab.angles, rel=1e-6)
    for original, copy in zip(slab.atoms, reparsed.atoms):
        assert copy.element == original.element
        assert copy.fract == pytest.approx(cm.wrap_fractional(original.fract), abs=1e-7)


def test_write_cif_is_p1_with_unique_labels():
    cell = cm.cell_from_molecule(["O", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], padding=4.0)
    text = cm.write_cif(cell, name="water box")
    assert "data_water_box" in text
    assert "'P 1'" in text
    labels = [line.split()[0] for line in text.splitlines() if line.startswith("  ")]
    assert labels == ["'x,", "O1", "H1", "H2"] or labels[-3:] == ["O1", "H1", "H2"]


# -- charge / multiplicity -------------------------------------------------


class _ChargedAtom:
    def __init__(self, symbol, charge=0, radicals=0):
        self._symbol, self._charge, self._radicals = symbol, charge, radicals

    def GetSymbol(self):
        return self._symbol

    def GetFormalCharge(self):
        return self._charge

    def GetNumRadicalElectrons(self):
        return self._radicals

    def HasProp(self, name):
        return False

    def GetProp(self, name):
        return ""


class _ChargedMol(_FakeMol):
    def __init__(self, atoms):
        super().__init__([a.GetSymbol() for a in atoms], [[0, 0, 0]] * len(atoms))
        self._atoms = atoms


def test_molecule_charge_and_multiplicity_neutral():
    mol = _ChargedMol([_ChargedAtom("C"), _ChargedAtom("O")])
    assert cm.molecule_charge_and_multiplicity(mol) == (0, 1)


def test_molecule_charge_sums_formal_charges():
    mol = _ChargedMol([_ChargedAtom("N", charge=1), _ChargedAtom("O", charge=-1), _ChargedAtom("O", charge=-1)])
    assert cm.molecule_charge_and_multiplicity(mol)[0] == -1


def test_molecule_multiplicity_counts_radicals():
    mol = _ChargedMol([_ChargedAtom("C", radicals=1), _ChargedAtom("H")])
    assert cm.molecule_charge_and_multiplicity(mol)[1] == 2


def test_molecule_charge_requires_a_molecule():
    with pytest.raises(ValueError):
        cm.molecule_charge_and_multiplicity(None)


def test_molecule_charge_tolerates_atoms_without_the_api():
    mol = _FakeMol(["C"], [[0, 0, 0]])  # plain fake: no GetFormalCharge
    assert cm.molecule_charge_and_multiplicity(mol) == (0, 1)


# -- per-axis vacuum -------------------------------------------------------


def test_per_axis_padding_builds_a_slab_like_box():
    cell = cm.cell_from_molecule(
        ["C", "C"], [[0, 0, 0], [1.4, 0, 0]], padding=[0.0, 0.0, 12.0]
    )
    assert cell.lengths[0] == pytest.approx(1.4)
    assert cell.lengths[2] == pytest.approx(24.0)


def test_scalar_padding_still_works():
    cell = cm.cell_from_molecule(["H", "H"], [[0, 0, 0], [0, 0, 1.0]], padding=5.0)
    assert cell.lengths[0] == pytest.approx(10.0)


def test_padding_rejects_a_bad_shape():
    with pytest.raises(ValueError, match="three numbers"):
        cm.cell_from_molecule(["H"], [[0, 0, 0]], padding=[1.0, 2.0])
