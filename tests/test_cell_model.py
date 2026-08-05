import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402


CUBIC_CIF = """
data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'I m -3 m'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'x+1/2, y+1/2, z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0.0 0.0 0.0
"""


class _FakePosition:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeConformer:
    def __init__(self, coords):
        self._coords = coords

    def GetAtomPosition(self, index):
        return _FakePosition(*self._coords[index])


class _FakeAtom:
    def __init__(self, symbol, custom=None):
        self._symbol = symbol
        self._custom = custom

    def GetSymbol(self):
        return self._symbol

    def HasProp(self, name):
        return name == "custom_symbol" and self._custom is not None

    def GetProp(self, name):
        return self._custom


class _FakeMol:
    def __init__(self, symbols, coords, customs=None):
        customs = customs or [None] * len(symbols)
        self._atoms = [_FakeAtom(s, c) for s, c in zip(symbols, customs)]
        self._conf = _FakeConformer(coords)

    def GetNumAtoms(self):
        return len(self._atoms)

    def GetAtomWithIdx(self, index):
        return self._atoms[index]

    def GetConformer(self):
        return self._conf


# -- lattice ---------------------------------------------------------------


def test_cell_vectors_cubic():
    lattice = cm.cell_vectors((3.0, 3.0, 3.0), (90.0, 90.0, 90.0))
    assert np.allclose(lattice, np.diag([3.0, 3.0, 3.0]))


def test_cell_vectors_hexagonal_volume():
    a, c = 3.0, 5.0
    lattice = cm.cell_vectors((a, a, c), (90.0, 90.0, 120.0))
    expected = math.sqrt(3.0) / 2.0 * a * a * c
    assert abs(abs(np.linalg.det(lattice)) - expected) < 1e-9


def test_lattice_parameters_roundtrip():
    lengths = (4.1, 5.2, 6.3)
    angles = (78.0, 95.0, 112.0)
    lattice = cm.cell_vectors(lengths, angles)
    out_lengths, out_angles = cm.lattice_parameters(lattice)
    assert np.allclose(out_lengths, lengths)
    assert np.allclose(out_angles, angles)


def test_cell_vectors_rejects_singular_gamma():
    with pytest.raises(ValueError):
        cm.cell_vectors((1.0, 1.0, 1.0), (90.0, 90.0, 180.0))


def test_cell_vectors_rejects_inconsistent_angles():
    with pytest.raises(ValueError):
        cm.cell_vectors((1.0, 1.0, 1.0), (20.0, 20.0, 150.0))


def test_fractional_cartesian_roundtrip():
    lattice = cm.cell_vectors((4.0, 5.0, 6.0), (80.0, 100.0, 110.0))
    fract = np.array([0.13, 0.71, 0.42])
    cart = cm.fractional_to_cartesian(fract, lattice)
    assert np.allclose(cm.cartesian_to_fractional(cart, lattice), fract)


def test_wrap_fractional_folds_and_snaps():
    wrapped = cm.wrap_fractional([1.25, -0.25, 1.0 - 1e-12])
    assert np.allclose(wrapped, [0.25, 0.75, 0.0])


# -- symmetry --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,point,expected",
    [
        ("x, y, z", (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)),
        ("-x, -y, -z", (0.1, 0.2, 0.3), (-0.1, -0.2, -0.3)),
        ("x+1/2, y, z-1/4", (0.1, 0.2, 0.3), (0.6, 0.2, 0.05)),
        ("1/2-x, 1/2+y, z", (0.1, 0.2, 0.3), (0.4, 0.7, 0.3)),
        ("y, x, -z+3/4", (0.1, 0.2, 0.3), (0.2, 0.1, 0.45)),
        ("-y, x-y, z", (0.1, 0.2, 0.3), (-0.2, -0.1, 0.3)),
    ],
)
def test_parse_symmetry_operation(text, point, expected):
    rotation, translation = cm.parse_symmetry_operation(text)
    assert np.allclose(rotation @ np.array(point) + translation, expected)


def test_parse_symmetry_operation_rejects_short_string():
    with pytest.raises(ValueError):
        cm.parse_symmetry_operation("x, y")


def test_apply_symmetry_deduplicates_special_position():
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    atom = cm.CellAtom("Fe1", "Fe", np.zeros(3), np.zeros(3))
    operations = [cm.parse_symmetry_operation(op) for op in ("x, y, z", "-x, -y, -z")]
    assert len(cm.apply_symmetry([atom], lattice, operations)) == 1


def test_apply_symmetry_keeps_distinct_images():
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    fract = np.array([0.25, 0.25, 0.25])
    atom = cm.CellAtom("C1", "C", fract, cm.fractional_to_cartesian(fract, lattice))
    operations = [cm.parse_symmetry_operation(op) for op in ("x, y, z", "-x, -y, -z")]
    expanded = cm.apply_symmetry([atom], lattice, operations)
    assert len(expanded) == 2
    assert np.allclose(sorted(a.fract[0] for a in expanded), [0.25, 0.75])


def test_apply_symmetry_keeps_mixed_occupancy_site():
    """Two elements on one site must both survive (per-atom dedup scope)."""
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    atoms = [
        cm.CellAtom("Fe1", "Fe", np.zeros(3), np.zeros(3), 0.5),
        cm.CellAtom("Co1", "Co", np.zeros(3), np.zeros(3), 0.5),
    ]
    expanded = cm.apply_symmetry(atoms, lattice, [(np.eye(3), np.zeros(3))])
    assert {atom.element for atom in expanded} == {"Fe", "Co"}


def test_apply_symmetry_without_operations_is_identity():
    lattice = np.eye(3) * 3.0
    atom = cm.CellAtom("H1", "H", np.array([0.2, 0.2, 0.2]), np.zeros(3))
    assert len(cm.apply_symmetry([atom], lattice, [])) == 1


# -- CIF -------------------------------------------------------------------


def test_parse_cif_expands_body_centring():
    cell = cm.parse_cif(CUBIC_CIF)
    assert len(cell.atoms) == 2
    assert cell.space_group == "I m -3 m"
    assert np.allclose(sorted(atom.fract[0] for atom in cell.atoms), [0.0, 0.5])
    assert cell.volume == pytest.approx(64.0)


def test_parse_cif_without_expansion_keeps_asymmetric_unit():
    cell = cm.parse_cif(CUBIC_CIF, expand=False)
    assert len(cell.atoms) == 1


def test_parse_cif_reads_uncertainties_and_comments():
    text = CUBIC_CIF.replace("_cell_length_a 4.0", "_cell_length_a 4.0(3)  # esd")
    assert cm.parse_cif(text).lengths[0] == pytest.approx(4.0)


def test_parse_cif_requires_cell_tags():
    with pytest.raises(ValueError, match="missing required tag"):
        cm.parse_cif("data_x\n_cell_length_a 1.0\n")


def test_parse_cif_requires_atoms():
    text = "\n".join(CUBIC_CIF.splitlines()[:9])
    with pytest.raises(ValueError, match="atom positions"):
        cm.parse_cif(text)


def test_parse_cif_accepts_cartesian_sites():
    text = CUBIC_CIF.replace("fract_x", "Cartn_x").replace("fract_y", "Cartn_y")
    text = text.replace("fract_z", "Cartn_z")
    cell = cm.parse_cif(text, expand=False)
    assert np.allclose(cell.atoms[0].cart, [0.0, 0.0, 0.0])


def test_parse_cif_file(tmp_path):
    path = tmp_path / "test.cif"
    path.write_text(CUBIC_CIF, encoding="utf-8")
    assert len(cm.parse_cif_file(str(path)).atoms) == 2


def test_normalize_element_and_number():
    assert cm.normalize_element("FE2+") == "Fe"
    assert cm.normalize_element("c") == "C"
    assert cm.normalize_element("1") == "X"
    assert cm.parse_cif_number("1.5(2)") == 1.5
    with pytest.raises(ValueError):
        cm.parse_cif_number("?")


# -- supercell / molecule --------------------------------------------------


def test_make_supercell_scales_lattice_and_atoms():
    cell = cm.parse_cif(CUBIC_CIF)
    super_cell = cm.make_supercell(cell, [2, 1, 3])
    assert len(super_cell.atoms) == len(cell.atoms) * 6
    assert super_cell.lengths == pytest.approx((8.0, 4.0, 12.0))
    assert all(0.0 <= value < 1.0 + 1e-9 for atom in super_cell.atoms for value in atom.fract)


def test_make_supercell_identity_returns_same_object():
    cell = cm.parse_cif(CUBIC_CIF)
    assert cm.make_supercell(cell, [1, 1, 1]) is cell


def test_make_supercell_preserves_cartesian_positions():
    cell = cm.parse_cif(CUBIC_CIF)
    super_cell = cm.make_supercell(cell, [2, 2, 2])
    originals = {tuple(np.round(atom.cart, 6)) for atom in cell.atoms}
    images = {tuple(np.round(atom.cart, 6)) for atom in super_cell.atoms}
    assert originals <= images


def test_cell_from_molecule_pads_and_centres():
    cell = cm.cell_from_molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], padding=5.0)
    assert cell.lengths[2] == pytest.approx(11.0)
    assert cell.lengths[0] == pytest.approx(10.0)
    assert cell.angles == (90.0, 90.0, 90.0)
    centre = np.mean([atom.cart for atom in cell.atoms], axis=0)
    assert np.allclose(centre, np.array(cell.lengths) / 2.0)


def test_cell_from_molecule_cubic_box():
    cell = cm.cell_from_molecule(["H", "H"], [[0, 0, 0], [0, 0, 4.0]], padding=3.0, cubic=True)
    assert cell.lengths[0] == cell.lengths[1] == cell.lengths[2] == pytest.approx(10.0)


def test_cell_from_molecule_single_atom_gets_finite_box():
    cell = cm.cell_from_molecule(["Ne"], [[0.0, 0.0, 0.0]], padding=0.0)
    assert min(cell.lengths) >= 1.0


def test_cell_from_molecule_validates_input():
    with pytest.raises(ValueError):
        cm.cell_from_molecule([], [], padding=1.0)
    with pytest.raises(ValueError):
        cm.cell_from_molecule(["H"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_molecule_to_cell_uses_custom_symbol():
    mol = _FakeMol(["C", "C"], [[0, 0, 0], [1.5, 0, 0]], customs=[None, "Si"])
    cell = cm.molecule_to_cell(mol, padding=4.0)
    assert [atom.element for atom in cell.atoms] == ["C", "Si"]
    assert cell.source == "molecule"


def test_molecule_to_cell_requires_molecule():
    with pytest.raises(ValueError):
        cm.molecule_to_cell(None)


# -- helpers ---------------------------------------------------------------


def test_group_and_sort_by_species():
    cell = cm.cell_from_molecule(
        ["O", "H", "O", "H"], [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 3, 0]], padding=2.0
    )
    groups = cm.group_by_species(cell)
    assert [element for element, _ in groups] == ["O", "H"]
    atoms, counts = cm.sorted_by_species(cell)
    assert counts == [("O", 2), ("H", 2)]
    assert [atom.element for atom in atoms] == ["O", "O", "H", "H"]
    assert cm.formula(cell) == "O2 H2"


def test_cell_with_lattice_keeps_fractional_positions():
    cell = cm.parse_cif(CUBIC_CIF)
    recast = cm.cell_with_lattice(cell, (8.0, 8.0, 8.0), (90.0, 90.0, 90.0))
    assert np.allclose(recast.atoms[1].fract, cell.atoms[1].fract)
    assert recast.atoms[1].cart[0] == pytest.approx(4.0)


def test_kpoint_mesh_from_density():
    cell = cm.parse_cif(CUBIC_CIF)  # 4 A cube
    assert cm.kpoint_mesh_from_density(cell, 0.05) == (5, 5, 5)
    assert cm.kpoint_mesh_from_density(cell, 10.0) == (1, 1, 1)
