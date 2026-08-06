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


# -- slab detection --------------------------------------------------------


def _layered_cell(c_length, heights, source="cif"):
    lengths, angles = (4.0, 4.0, c_length), (90.0, 90.0, 90.0)
    lattice = cm.cell_vectors(lengths, angles)
    atoms = tuple(
        cm.CellAtom(
            f"Cu{index + 1}",
            "Cu",
            np.array([0.0, 0.0, height / c_length]),
            np.array([0.0, 0.0, height]),
        )
        for index, height in enumerate(heights)
    )
    return cm.Cell("x", lengths, angles, lattice, atoms, source=source)


def test_vacuum_gap_measures_the_empty_space():
    cell = _layered_cell(20.0, [0.0, 2.0, 4.0])
    assert cm.vacuum_gap(cell) == pytest.approx(16.0)


def test_vacuum_gap_is_zero_for_a_filled_cell():
    cell = _layered_cell(4.0, [0.0, 2.0, 4.0])
    assert cm.vacuum_gap(cell) == pytest.approx(0.0)


def test_vacuum_gap_handles_an_empty_cell():
    lengths, angles = (4.0, 4.0, 4.0), (90.0, 90.0, 90.0)
    empty = cm.Cell("x", lengths, angles, cm.cell_vectors(lengths, angles), ())
    assert cm.vacuum_gap(empty) == 0.0


def test_looks_like_slab_detects_a_cif_borne_slab():
    """A slab loaded from a CIF has no marker, so geometry must give it away."""
    assert cm.looks_like_slab(_layered_cell(20.0, [0.0, 2.0, 4.0]))


def test_looks_like_slab_rejects_a_dense_bulk_cell():
    assert not cm.looks_like_slab(_layered_cell(4.0, [0.0, 2.0]))


def test_looks_like_slab_honours_the_threshold():
    cell = _layered_cell(10.0, [0.0, 2.0, 4.0])  # 6 A of vacuum
    assert cm.looks_like_slab(cell, minimum_vacuum=5.0)
    assert not cm.looks_like_slab(cell, minimum_vacuum=8.0)


def test_looks_like_slab_never_fires_for_a_molecule_box():
    cell = cm.cell_from_molecule(["H", "H"], [[0, 0, 0], [0, 0, 1.0]], padding=8.0)
    assert not cm.looks_like_slab(cell)


def test_looks_like_slab_trusts_an_explicit_slab_source():
    assert cm.looks_like_slab(_layered_cell(4.0, [0.0, 2.0], source="slab"))


# -- reciprocal lattice / k-mesh -------------------------------------------


def test_reciprocal_lengths_invert_an_orthogonal_cell():
    lattice = cm.cell_vectors((4.0, 5.0, 6.0), (90.0, 90.0, 90.0))
    assert cm.reciprocal_lengths(lattice) == pytest.approx((0.25, 0.2, 1.0 / 6.0))


def test_reciprocal_lengths_are_not_one_over_a_when_the_cell_is_hexagonal():
    """|b1| = 1/(a sin(gamma)), which is what the k-mesh must follow."""
    lattice = cm.cell_vectors((3.0, 3.0, 5.0), (90.0, 90.0, 120.0))
    b1, b2, b3 = cm.reciprocal_lengths(lattice)
    expected = 1.0 / (3.0 * math.sin(math.radians(120.0)))
    assert b1 == pytest.approx(expected)
    assert b2 == pytest.approx(expected)
    assert b3 == pytest.approx(0.2)
    assert b1 > 1.0 / 3.0  # the direct length would under-sample by 15%


def test_reciprocal_lengths_reject_a_flat_cell():
    with pytest.raises(ValueError):
        cm.reciprocal_lengths(np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]]))


def test_kpoint_mesh_uses_the_reciprocal_lattice_for_a_hexagonal_cell():
    lengths, angles = (3.0, 3.0, 5.0), (90.0, 90.0, 120.0)
    lattice = cm.cell_vectors(lengths, angles)
    cell = cm.Cell("hex", lengths, angles, lattice, (cm.CellAtom("C1", "C", np.zeros(3), np.zeros(3)),))
    # 1/(3 * 0.05) = 6.67 -> 7 from the direct length; the reciprocal one gives 8.
    assert cm.kpoint_mesh_from_density(cell, 0.05)[:2] == (8, 8)


def test_kpoint_mesh_matches_the_exact_ratio_without_rounding_up():
    cell = cm.parse_cif(CUBIC_CIF)  # 4 A cube, |b| = 0.25
    assert cm.kpoint_mesh_from_density(cell, 0.25) == (1, 1, 1)
    assert cm.kpoint_mesh_from_density(cell, 0.125) == (2, 2, 2)


# -- structure audit -------------------------------------------------------


def _cell_with(atoms, lengths=(6.0, 6.0, 6.0), angles=(90.0, 90.0, 90.0), **kwargs):
    lattice = cm.cell_vectors(lengths, angles)
    built = []
    for label, element, fract, occupancy in atoms:
        fract = np.asarray(fract, dtype=float)
        built.append(
            cm.CellAtom(label, element, fract, cm.fractional_to_cartesian(fract, lattice), occupancy)
        )
    return cm.Cell("test", lengths, angles, lattice, tuple(built), **kwargs)


def test_minimum_image_distance_crosses_the_cell_boundary():
    lattice = cm.cell_vectors((5.0, 5.0, 5.0), (90.0, 90.0, 90.0))
    # 0.05 and 0.95 are 0.5 A apart through the boundary, not 4.5 A across the cell.
    assert cm.minimum_image_distance([0.05, 0, 0], [0.95, 0, 0], lattice) == pytest.approx(0.5)


def test_close_contacts_finds_overlapping_symmetry_images():
    cell = _cell_with(
        [("Fe1", "Fe", [0.0, 0.0, 0.0], 1.0), ("Fe1", "Fe", [0.01, 0.0, 0.0], 1.0)]
    )
    contacts = cm.close_contacts(cell)
    assert len(contacts) == 1
    assert contacts[0][2] == pytest.approx(0.06)


def test_close_contacts_leaves_a_normal_bond_alone():
    cell = cm.cell_from_molecule(["O", "H"], [[0, 0, 0], [0.96, 0, 0]], padding=6.0)
    assert cm.close_contacts(cell) == []


def test_close_contacts_skips_a_cell_that_is_too_large():
    atoms = [(f"C{i}", "C", [i / 600.0, 0.0, 0.0], 1.0) for i in range(600)]
    assert cm.close_contacts(_cell_with(atoms)) == []


def test_partial_occupancy_sites_reports_only_the_disordered_ones():
    cell = _cell_with(
        [("Fe1", "Fe", [0.0, 0.0, 0.0], 0.5), ("O1", "O", [0.5, 0.5, 0.5], 1.0)]
    )
    assert cm.partial_occupancy_sites(cell) == [("Fe1", "Fe", 0.5)]


def test_structure_warnings_are_silent_for_a_clean_cell():
    assert cm.structure_warnings(cm.parse_cif(CUBIC_CIF)) == []


def test_structure_warnings_flag_a_left_handed_lattice():
    lattice = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, -4.0]])
    cell = cm.Cell("lh", (4.0, 4.0, 4.0), (90.0, 90.0, 90.0), lattice, ())
    assert any("left-handed" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_flag_a_flat_lattice():
    lattice = np.array([[4.0, 0, 0], [8.0, 0, 0], [0, 0, 4.0]])
    cell = cm.Cell("flat", (4.0, 8.0, 4.0), (90.0, 90.0, 90.0), lattice, ())
    assert any("no volume" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_flag_partial_occupancy():
    cell = _cell_with([("Fe1", "Fe", [0.0, 0.0, 0.0], 0.5)], source="cif")
    assert any("partially occupied" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_flag_overlapping_atoms():
    cell = _cell_with(
        [("Fe1", "Fe", [0.0, 0.0, 0.0], 1.0), ("Fe2", "Fe", [0.01, 0.0, 0.0], 1.0)],
        source="cif",
    )
    assert any("closer than" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_flag_a_non_element():
    cell = _cell_with([("Q1", "Q", [0.0, 0.0, 0.0], 1.0)], source="cif")
    assert any("Not chemical elements" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_flag_an_unexpandable_space_group():
    """A CIF that names a space group but lists no operations is half a cell.

    The symbol is deliberately unresolvable so the test does not depend on
    whether the optional pymatgen fallback is installed.
    """
    text = CUBIC_CIF.replace("loop_\n_symmetry_equiv_pos_as_xyz\n'x, y, z'\n'x+1/2, y+1/2, z+1/2'\n", "")
    text = text.replace("'I m -3 m'", "'Not a group'")
    cell = cm.parse_cif(text)
    assert cell.symmetry_operations == 1
    assert any("no symmetry operations" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_flag_expansion_that_was_switched_off():
    cell = cm.parse_cif(CUBIC_CIF, expand=False)
    assert cell.symmetry_operations == 0
    assert any("switched off" in message for message in cm.structure_warnings(cell))


def test_structure_warnings_stay_quiet_for_a_p1_cif():
    text = CUBIC_CIF.replace("'I m -3 m'", "'P 1'")
    assert cm.structure_warnings(cm.parse_cif(text, expand=False)) == []


# -- CIF robustness --------------------------------------------------------


def test_parse_cif_records_the_operation_count():
    assert cm.parse_cif(CUBIC_CIF).symmetry_operations == 2


def test_parse_cif_skips_a_row_with_unreadable_coordinates():
    """One '?' site must not take the whole file down with it."""
    text = CUBIC_CIF.replace("Fe1 Fe 0.0 0.0 0.0", "Fe1 Fe 0.0 0.0 0.0\nO1 O ? ? ?")
    cell = cm.parse_cif(text)
    assert [atom.element for atom in cell.atoms] == ["Fe", "Fe"]


def test_parse_cif_wraps_positions_even_without_expansion():
    text = CUBIC_CIF.replace("Fe1 Fe 0.0 0.0 0.0", "Fe1 Fe -0.25 1.25 0.0")
    cell = cm.parse_cif(text, expand=False)
    assert np.allclose(cell.atoms[0].fract, [0.75, 0.25, 0.0])
    assert np.allclose(cell.atoms[0].cart, [3.0, 1.0, 0.0])


def test_normalize_element_falls_back_when_two_letters_are_not_an_element():
    assert cm.normalize_element("OW1") == "O"     # a water oxygen label
    assert cm.normalize_element("Cl2") == "Cl"    # a real two-letter element stays
    assert cm.normalize_element("Zz") == "Zz"     # nothing sensible to fall back to


def test_is_element():
    assert cm.is_element("Fe")
    assert not cm.is_element("Ow")


def test_make_supercell_keeps_the_operation_count():
    cell = cm.make_supercell(cm.parse_cif(CUBIC_CIF), [2, 1, 1])
    assert cell.symmetry_operations == 2


# -- primitive-cell reduction ----------------------------------------------


_F_OPS = ["x, y, z", "x, y+1/2, z+1/2", "x+1/2, y, z+1/2", "x+1/2, y+1/2, z"]
_I_OPS = ["x, y, z", "x+1/2, y+1/2, z+1/2"]


def _centred_cif(a, ops, sites):
    lines = "\n".join(f"'{op}'" for op in ops)
    rows = "\n".join(sites)
    return f"""data_x
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
{lines}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{rows}
"""


def _shortest_contact(cell):
    """Nearest neighbour including an atom's own periodic images."""
    best = min(
        float(np.linalg.norm(np.array(v, dtype=float) @ cell.lattice))
        for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                  [1, -1, 0], [1, 0, -1], [0, 1, -1], [1, 1, 1], [1, 1, -1])
    )
    for index, atom in enumerate(cell.atoms):
        for other in cell.atoms[index + 1:]:
            best = min(best, cm.minimum_image_distance(atom.fract, other.fract, cell.lattice))
    return best


def test_translation_symmetries_find_the_body_centring():
    cell = cm.parse_cif(_centred_cif(2.87, _I_OPS, ["Fe1 Fe 0 0 0"]))
    found = cm.translation_symmetries(cell)
    assert len(found) == 1
    assert np.allclose(sorted(found[0]), [0.5, 0.5, 0.5])


def test_translation_symmetries_find_the_face_centring():
    cell = cm.parse_cif(_centred_cif(3.615, _F_OPS, ["Cu1 Cu 0 0 0"]))
    assert len(cm.translation_symmetries(cell)) == 3


def test_translation_symmetries_are_empty_for_a_primitive_cell():
    cell = cm.parse_cif(_centred_cif(4.0, ["x, y, z"], ["Cs1 Cs 0 0 0", "Cl1 Cl 0.5 0.5 0.5"]))
    assert cm.translation_symmetries(cell) == []


def test_a_translation_must_map_every_element_onto_its_own_kind():
    """CsCl looks body-centred until the elements are taken into account."""
    cell = cm.parse_cif(_centred_cif(4.11, ["x, y, z"], ["Cs1 Cs 0 0 0", "Cl1 Cl 0.5 0.5 0.5"]))
    assert cm.primitive_cell(cell) is cell


def test_bcc_reduces_to_one_atom():
    cell = cm.parse_cif(_centred_cif(2.8665, _I_OPS, ["Fe1 Fe 0 0 0"]))
    primitive = cm.primitive_cell(cell)
    assert len(primitive.atoms) == 1
    assert cell.volume / primitive.volume == pytest.approx(2.0)
    # a_prim = a * sqrt(3) / 2, angles 109.47 / 70.53
    assert primitive.lengths[0] == pytest.approx(2.8665 * math.sqrt(3) / 2)


def test_fcc_reduces_to_the_textbook_60_degree_cell():
    cell = cm.parse_cif(_centred_cif(3.615, _F_OPS, ["Cu1 Cu 0 0 0"]))
    primitive = cm.primitive_cell(cell)
    assert len(primitive.atoms) == 1
    assert cell.volume / primitive.volume == pytest.approx(4.0)
    assert primitive.lengths[0] == pytest.approx(3.615 / math.sqrt(2))
    for angle in primitive.angles:
        assert angle == pytest.approx(60.0)


def test_reduction_preserves_the_number_density_and_contacts():
    cell = cm.parse_cif(_centred_cif(5.64, _F_OPS, ["Na1 Na 0 0 0", "Cl1 Cl 0.5 0.5 0.5"]))
    primitive = cm.primitive_cell(cell)
    assert len(primitive.atoms) == 2
    assert len(cell.atoms) / cell.volume == pytest.approx(len(primitive.atoms) / primitive.volume)
    assert _shortest_contact(primitive) == pytest.approx(_shortest_contact(cell))


def test_reduction_undoes_a_supercell():
    """An exact supercell is translational symmetry too, so it collapses."""
    cell = cm.parse_cif(_centred_cif(5.64, _F_OPS, ["Na1 Na 0 0 0", "Cl1 Cl 0.5 0.5 0.5"]))
    doubled = cm.make_supercell(cell, [2, 2, 1])
    assert len(doubled.atoms) == 32
    assert len(cm.primitive_cell(doubled).atoms) == 2


def test_a_primitive_cell_is_returned_unchanged():
    cell = cm.parse_cif(_centred_cif(4.0, ["x, y, z"], ["Po1 Po 0 0 0"]))
    assert cm.primitive_cell(cell) is cell


def test_reduction_keeps_the_cell_right_handed():
    for ops in (_I_OPS, _F_OPS):
        primitive = cm.primitive_cell(cm.parse_cif(_centred_cif(3.5, ops, ["Cu1 Cu 0 0 0"])))
        assert np.linalg.det(primitive.lattice) > 0
        assert cm.structure_warnings(primitive) == []


def test_a_partially_occupied_site_does_not_match_a_full_one():
    """A vacancy-ordered cell is not centred, however alike the sites look."""
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    atoms = (
        cm.CellAtom("Fe1", "Fe", np.zeros(3), np.zeros(3), 1.0),
        cm.CellAtom("Fe2", "Fe", np.full(3, 0.5), cm.fractional_to_cartesian([0.5] * 3, lattice), 0.5),
    )
    cell = cm.Cell("x", (4.0,) * 3, (90.0,) * 3, lattice, atoms, source="cif")
    assert cm.translation_symmetries(cell) == []


# -- LLL reduction ----------------------------------------------------------


def test_lll_shortens_a_skewed_basis_without_changing_the_lattice():
    lattice = np.array([[1.0, 0.0, 0.0], [10.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    reduced, transformation = cm.lll_reduce(lattice)
    assert abs(np.linalg.det(reduced)) == pytest.approx(abs(np.linalg.det(lattice)))
    assert np.allclose(transformation @ lattice, reduced)
    assert np.linalg.norm(reduced[1]) < np.linalg.norm(lattice[1])


def test_lll_keeps_the_basis_right_handed():
    lattice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    reduced, _ = cm.lll_reduce(lattice)
    assert np.linalg.det(reduced) > 0


def test_lll_leaves_an_already_reduced_basis_alone():
    lattice = cm.cell_vectors((3.0, 4.0, 5.0), (90.0, 90.0, 90.0))
    reduced, _ = cm.lll_reduce(lattice)
    assert sorted(round(float(np.linalg.norm(row)), 6) for row in reduced) == [3.0, 4.0, 5.0]


# -- guards and edge paths --------------------------------------------------


def test_cell_elements_property():
    cell = cm.parse_cif(CUBIC_CIF)
    assert cell.elements == ("Fe", "Fe")


def test_cell_vectors_rejects_a_zero_length():
    with pytest.raises(ValueError, match="lengths must be positive"):
        cm.cell_vectors((0.0, 1.0, 1.0), (90.0, 90.0, 90.0))


def test_lattice_parameters_reject_a_zero_vector():
    with pytest.raises(ValueError, match="zero length"):
        cm.lattice_parameters(np.array([[0.0, 0.0, 0.0], [0, 1.0, 0], [0, 0, 1.0]]))


def test_symmetry_operation_rejects_an_empty_component():
    with pytest.raises(ValueError, match="Empty component"):
        cm.parse_symmetry_operation("x, , z")


def test_symmetry_operation_reads_a_numeric_coefficient():
    rotation, _ = cm.parse_symmetry_operation("2x, y, z")
    assert rotation[0, 0] == pytest.approx(2.0)


def test_a_broken_symmetry_row_is_skipped_not_fatal():
    text = CUBIC_CIF.replace("'x, y, z'", "'x, y, z'\n'nonsense'")
    assert len(cm.parse_cif(text).atoms) == 2


def test_spacegroup_operations_without_a_symbol():
    assert cm._spacegroup_operations(None) == []


def test_parse_cif_number_falls_through_to_float():
    assert cm.parse_cif_number("1e3") == pytest.approx(1000.0)


def test_a_tag_whose_value_is_on_the_next_line():
    text = CUBIC_CIF.replace("_cell_length_a 4.0", "_cell_length_a\n4.0")
    assert cm.parse_cif(text).lengths[0] == pytest.approx(4.0)


def test_a_semicolon_text_field_is_read_as_one_value():
    text = CUBIC_CIF.replace(
        "data_test", "data_test\n_chemical_name\n;\nsome long\nname\n;"
    )
    assert len(cm.parse_cif(text).atoms) == 2


def test_an_unreadable_occupancy_is_dropped_not_fatal():
    text = CUBIC_CIF.replace(
        "_atom_site_fract_z\nFe1 Fe 0.0 0.0 0.0",
        "_atom_site_fract_z\n_atom_site_occupancy\nFe1 Fe 0.0 0.0 0.0 ?",
    )
    cell = cm.parse_cif(text)
    assert cell.atoms[0].occupancy is None


def test_a_space_group_tag_of_question_mark_is_ignored():
    text = CUBIC_CIF.replace("'I m -3 m'", "?")
    assert cm.parse_cif(text).space_group is None


def test_a_cif_without_a_space_group_tag():
    text = "\n".join(
        line for line in CUBIC_CIF.splitlines() if "space_group" not in line
    )
    assert cm.parse_cif(text).space_group is None


def test_cell_from_molecule_rejects_a_bad_padding_shape():
    with pytest.raises(ValueError, match="three numbers"):
        cm.cell_from_molecule(["H"], [[0.0, 0.0, 0.0]], padding=[1.0, 2.0])


def test_molecule_arrays_rejects_an_empty_molecule():
    class _Empty:
        def GetNumAtoms(self):
            return 0

        def GetConformer(self):
            return None

    with pytest.raises(ValueError, match="no atoms"):
        cm.molecule_arrays(_Empty())


def test_molecule_arrays_rejects_none():
    with pytest.raises(ValueError, match="No molecule"):
        cm.molecule_arrays(None)


# -- charge and multiplicity ------------------------------------------------


class _ChargedAtom:
    def __init__(self, charge=0, radicals=0):
        self._charge, self._radicals = charge, radicals

    def GetFormalCharge(self):
        return self._charge

    def GetNumRadicalElectrons(self):
        return self._radicals


class _ChargedMol:
    def __init__(self, atoms):
        self._atoms = atoms

    def GetNumAtoms(self):
        return len(self._atoms)

    def GetAtomWithIdx(self, index):
        return self._atoms[index]


def test_charge_and_multiplicity_of_a_closed_shell():
    assert cm.molecule_charge_and_multiplicity(_ChargedMol([_ChargedAtom()])) == (0, 1)


def test_charge_and_multiplicity_of_a_radical_anion():
    mol = _ChargedMol([_ChargedAtom(charge=-1, radicals=1), _ChargedAtom()])
    assert cm.molecule_charge_and_multiplicity(mol) == (-1, 2)


def test_charge_and_multiplicity_ignores_an_atom_that_cannot_answer():
    class _Mute:
        def GetFormalCharge(self):
            raise AttributeError

        def GetNumRadicalElectrons(self):
            raise TypeError

    assert cm.molecule_charge_and_multiplicity(_ChargedMol([_Mute()])) == (0, 1)


def test_charge_and_multiplicity_rejects_none():
    with pytest.raises(ValueError, match="No molecule"):
        cm.molecule_charge_and_multiplicity(None)


# -- write_cif --------------------------------------------------------------


def test_write_cif_round_trips_through_the_parser():
    cell = cm.parse_cif(CUBIC_CIF)
    reparsed = cm.parse_cif(cm.write_cif(cell))
    assert len(reparsed.atoms) == len(cell.atoms)
    assert reparsed.lengths == pytest.approx(cell.lengths)


def test_write_cif_names_the_data_block():
    text = cm.write_cif(cm.parse_cif(CUBIC_CIF), name="my cell")
    assert text.startswith("data_my_cell")


def test_write_cif_defaults_the_occupancy():
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    atoms = (cm.CellAtom("Fe1", "Fe", np.zeros(3), np.zeros(3), None),)
    text = cm.write_cif(cm.Cell("x", (4.0,) * 3, (90.0,) * 3, lattice, atoms))
    assert "1.0000" in text


# -- translation symmetry guards --------------------------------------------


def test_translation_symmetries_skip_a_huge_cell():
    lattice = cm.cell_vectors((500.0, 500.0, 500.0), (90.0, 90.0, 90.0))
    atoms = tuple(
        cm.CellAtom(f"C{i}", "C", np.array([i / 500.0, 0.0, 0.0]), np.zeros(3), 1.0)
        for i in range(500)
    )
    cell = cm.Cell("big", (500.0,) * 3, (90.0,) * 3, lattice, atoms)
    assert cm.translation_symmetries(cell) == []


def test_translation_symmetries_need_two_atoms():
    cell = cm.parse_cif(CUBIC_CIF.replace("'x+1/2, y+1/2, z+1/2'", "'x, y, z'"))
    assert cm.translation_symmetries(cell) == []


def test_primitive_transformation_is_none_without_translations():
    cell = cm.parse_cif(CUBIC_CIF.replace("'x+1/2, y+1/2, z+1/2'", "'x, y, z'"))
    assert cm.primitive_transformation(cell) is None


def test_parse_cif_number_accepts_what_the_esd_pattern_rejects():
    assert cm.parse_cif_number("1_000") == pytest.approx(1000.0)


def test_a_site_row_with_neither_fractional_nor_cartesian_columns_is_skipped():
    text = CUBIC_CIF.replace("_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n",
                             "_atom_site_u_iso\n_atom_site_b_iso\n_atom_site_wyckoff\n")
    with pytest.raises(ValueError, match="readable atom positions"):
        cm.parse_cif(text)


def test_a_candidate_translation_that_does_not_map_the_cell_is_rejected():
    """Two unlike sublattices look centred until every atom is checked."""
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    fract = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.0, 0.0]]
    atoms = tuple(
        cm.CellAtom(f"Fe{i}", "Fe", np.array(f), cm.fractional_to_cartesian(f, lattice), 1.0)
        for i, f in enumerate(fract)
    )
    cell = cm.Cell("x", (4.0,) * 3, (90.0,) * 3, lattice, atoms)
    # (1/2,1/2,1/2) would map atoms 0 and 1 but leaves atom 2 with no partner.
    assert all(not np.allclose(shift, [0.5, 0.5, 0.5]) for shift in cm.translation_symmetries(cell))


def test_duplicate_candidate_translations_are_only_tried_once():
    """A 2x1x1 supercell offers the same shift from several atoms."""
    cell = cm.parse_cif(CUBIC_CIF)
    doubled = cm.make_supercell(cell, [2, 1, 1])
    found = cm.translation_symmetries(doubled)
    for index, shift in enumerate(found):
        for other in found[index + 1:]:
            assert not np.allclose(shift, other, atol=1e-4)


def test_primitive_transformation_rejects_a_non_integer_multiplicity():
    """A basis has to divide the cell a whole number of times."""
    lattice = cm.cell_vectors((6.0, 6.0, 6.0), (90.0, 90.0, 90.0))
    fract = [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0]]
    atoms = tuple(
        cm.CellAtom(f"C{i}", "C", np.array(f), cm.fractional_to_cartesian(f, lattice), 1.0)
        for i, f in enumerate(fract)
    )
    cell = cm.Cell("x", (6.0,) * 3, (90.0,) * 3, lattice, atoms)
    matrix = cm.primitive_transformation(cell)
    assert matrix is None or abs(round(1 / abs(np.linalg.det(matrix))) - 1 / abs(np.linalg.det(matrix))) < 1e-6


def test_primitive_cell_keeps_a_cell_it_cannot_divide_evenly():
    """A near-miss translation must not be allowed to delete atoms."""
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    fract = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]
    atoms = tuple(
        cm.CellAtom(f"Fe{i}", "Fe", np.array(f), cm.fractional_to_cartesian(f, lattice), 1.0)
        for i, f in enumerate(fract)
    )
    cell = cm.Cell("x", (4.0,) * 3, (90.0,) * 3, lattice, atoms)
    assert len(cm.primitive_cell(cell).atoms) == 3


def test_atomic_number_and_is_known():
    from vasp_input_generator import elements

    assert elements.atomic_number("Fe") == 26
    assert elements.atomic_number("Xx") is None
    assert elements.is_known("O") and not elements.is_known("Xx")


def _cell_of(fract_occ, length=4.0):
    lattice = cm.cell_vectors((length,) * 3, (90.0,) * 3)
    atoms = tuple(
        cm.CellAtom(f"Fe{i}", "Fe", np.array(f, dtype=float),
                    cm.fractional_to_cartesian(f, lattice), occ)
        for i, (f, occ) in enumerate(fract_occ)
    )
    return cm.Cell("x", (length,) * 3, (90.0,) * 3, lattice, atoms)


def test_a_translation_must_match_occupancies_atom_by_atom():
    """Half-occupied sites map onto half-occupied ones, not onto full ones."""
    cell = _cell_of([
        ([0.0, 0.0, 0.0], 1.0),
        ([0.5, 0.5, 0.5], 1.0),
        ([0.25, 0.25, 0.25], 0.5),
        ([0.75, 0.75, 0.75], 0.5),
    ])
    found = cm.translation_symmetries(cell)
    assert any(np.allclose(shift, [0.5, 0.5, 0.5]) for shift in found)


def test_a_duplicate_site_offers_no_translation():
    """An atom sitting on atom 0 gives a zero shift, which is not a symmetry."""
    cell = _cell_of([([0.0, 0.0, 0.0], 1.0), ([0.0, 0.0, 0.0], 1.0)])
    assert cm.translation_symmetries(cell) == []


def test_two_coincident_sites_are_only_tried_once():
    cell = _cell_of([
        ([0.0, 0.0, 0.0], 1.0),
        ([0.5, 0.5, 0.5], 1.0),
        ([0.5, 0.5, 0.5], 1.0),
    ])
    found = cm.translation_symmetries(cell)
    assert len(found) == len({tuple(np.round(s, 4)) for s in found})
