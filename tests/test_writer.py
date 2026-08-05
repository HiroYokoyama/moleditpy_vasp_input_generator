import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402
from vasp_input_generator import potentials, writer  # noqa: E402


@pytest.fixture
def water_cell():
    return cm.cell_from_molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        padding=5.0,
        name="water",
    )


@pytest.fixture
def rutile_cell():
    return cm.cell_from_molecule(
        ["Ti", "O", "O", "Ti"],
        [[0, 0, 0], [1.0, 1.0, 0], [2.0, 0, 0], [3.0, 0, 0]],
        padding=2.0,
    )


def incar_tags(text):
    tags = {}
    for line in text.splitlines():
        if "=" in line and not line.strip().startswith("!"):
            key, _, value = line.partition("=")
            tags[key.strip()] = value.strip()
    return tags


# -- POSCAR ----------------------------------------------------------------


def test_poscar_header_and_species(water_cell):
    text = writer.build_poscar(water_cell, {"title": "water box"})
    lines = text.splitlines()
    assert lines[0] == "water box"
    assert float(lines[1]) == pytest.approx(1.0)
    assert lines[5].split() == ["O", "H"]
    assert lines[6].split() == ["1", "2"]
    assert lines[7] == "Direct"
    assert len(lines) == 11


def test_poscar_lattice_rows_match_cell(water_cell):
    lines = writer.build_poscar(water_cell).splitlines()
    for row in range(3):
        values = [float(v) for v in lines[2 + row].split()]
        assert values == pytest.approx(list(water_cell.lattice[row]), abs=1e-9)


def test_poscar_cartesian_mode(water_cell):
    text = writer.build_poscar(water_cell, {"coord_mode": writer.COORD_MODES[1]})
    assert "Cartesian" in text.splitlines()[7]
    first = [float(v) for v in text.splitlines()[8].split()[:3]]
    assert first == pytest.approx(list(water_cell.atoms[0].cart), abs=1e-9)


def test_poscar_direct_coordinates_are_fractional(water_cell):
    text = writer.build_poscar(water_cell)
    for line in text.splitlines()[8:]:
        for value in [float(v) for v in line.split()[:3]]:
            assert 0.0 <= value <= 1.0


def test_poscar_selective_dynamics_freezes_selection(water_cell):
    text = writer.build_poscar(
        water_cell, {"selective_dynamics": True, "frozen_indices": [0]}
    )
    lines = text.splitlines()
    assert lines[7] == "Selective dynamics"
    assert lines[8] == "Direct"
    assert "F   F   F" in lines[9]  # the O atom sorts first
    assert lines[10].count("T   T   T") == 1


def test_poscar_freeze_maps_through_species_sort():
    """frozen_indices index the original cell, not the species-sorted rows."""
    cell = cm.cell_from_molecule(
        ["H", "O", "H"], [[0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], padding=3.0
    )
    text = writer.build_poscar(cell, {"selective_dynamics": True, "frozen_indices": [1]})
    rows = text.splitlines()[9:]
    # Species order is first-appearance (H, then O), so the frozen O is last.
    assert [row.split()[-1] for row in rows] == ["H1", "H3", "O2"]
    assert "F   F   F" in rows[2]
    assert all("T   T   T" in row for row in rows[:2])


def test_poscar_labels_are_commented(water_cell):
    assert "! O1" in writer.build_poscar(water_cell)


# -- INCAR -----------------------------------------------------------------


def test_incar_defaults():
    tags = incar_tags(writer.build_incar())
    assert tags["PREC"] == "Accurate"
    assert tags["ENCUT"] == "520"
    assert tags["EDIFF"] == "1.0E-06"
    assert tags["IBRION"] == "-1"
    assert tags["NSW"] == "0"
    assert tags["ISMEAR"] == "0"
    assert tags["SIGMA"] == "0.05"
    assert "GGA" not in tags  # PBE POTCARs already default to GGA = PE


@pytest.mark.parametrize(
    "task,expected",
    [
        ("Relax ions", {"IBRION": "2", "ISIF": "2", "NSW": "100"}),
        ("Relax ions + cell", {"IBRION": "2", "ISIF": "3"}),
        ("Density of states", {"LORBIT": "11", "NEDOS": "3001", "ISMEAR": "-5"}),
        ("Band structure (non-SCF)", {"ICHARG": "11", "LORBIT": "11"}),
        ("Molecular dynamics (NVT)", {"IBRION": "0", "MDALGO": "2", "TEBEG": "300"}),
    ],
)
def test_incar_task_presets(task, expected):
    tags = incar_tags(writer.build_incar({"task": task}))
    for key, value in expected.items():
        assert tags[key] == value


def test_incar_dos_forces_tetrahedron_and_drops_sigma():
    tags = incar_tags(writer.build_incar({"task": "Density of states"}))
    assert tags["ISMEAR"] == "-5"
    assert "SIGMA" not in tags


def test_incar_relax_carries_ediffg():
    tags = incar_tags(writer.build_incar({"task": "Relax ions", "ediffg": -0.01}))
    assert tags["EDIFFG"] == "-0.01"


@pytest.mark.parametrize(
    "functional,gga",
    [("PBEsol", "PS"), ("RPBE", "RP"), ("LDA (CA)", "CA"), ("PS (PBEsol)", "PS")],
)
def test_incar_functional_tags(functional, gga):
    assert incar_tags(writer.build_incar({"functional": functional}))["GGA"] == gga


@pytest.mark.parametrize(
    "label,value",
    [
        ("DFT-D2 (IVDW=1)", "1"),
        ("TS (IVDW=2)", "2"),
        ("DFT-D3 (IVDW=11)", "11"),
        ("DFT-D3(BJ) (IVDW=12)", "12"),
    ],
)
def test_incar_dispersion(label, value):
    assert incar_tags(writer.build_incar({"vdw": label}))["IVDW"] == value


def test_incar_no_ivdw_when_disabled():
    assert "IVDW" not in incar_tags(writer.build_incar())


def test_incar_spin_and_magmom():
    counts = [("Fe", 2), ("O", 3)]
    tags = incar_tags(writer.build_incar({"ispin": True}, counts))
    assert tags["ISPIN"] == "2"
    assert tags["MAGMOM"] == "2*5.0 3*0.6"


def test_incar_magmom_omitted_without_counts():
    assert "MAGMOM" not in incar_tags(writer.build_incar({"ispin": True}))


def test_incar_ncore_only_when_above_one():
    assert "NCORE" not in incar_tags(writer.build_incar({"ncore": 1}))
    assert incar_tags(writer.build_incar({"ncore": 8}))["NCORE"] == "8"


def test_incar_write_flags():
    tags = incar_tags(writer.build_incar({"lwave": True, "lcharg": False}))
    assert tags["LWAVE"] == ".TRUE."
    assert tags["LCHARG"] == ".FALSE."


def test_incar_extra_tags_appended():
    text = writer.build_incar({"extra_incar": "LDAU = .TRUE.\nLDAUTYPE = 2"})
    assert text.rstrip().endswith("LDAUTYPE = 2")
    assert incar_tags(text)["LDAU"] == ".TRUE."


def test_incar_title_newlines_are_flattened():
    tags = incar_tags(writer.build_incar({"title": "line1\nline2"}))
    assert tags["SYSTEM"] == "line1 line2"


# -- KPOINTS ---------------------------------------------------------------


def test_kpoints_gamma_only(water_cell):
    lines = writer.build_kpoints(water_cell, {"kpoint_mode": "Gamma-only"}).splitlines()
    assert lines[1] == "0"
    assert lines[2] == "Gamma"
    assert lines[3].split() == ["1", "1", "1"]


def test_kpoints_monkhorst_mesh(water_cell):
    text = writer.build_kpoints(
        water_cell,
        {"kpoint_mode": "Monkhorst-Pack mesh", "kmesh": [3, 4, 5], "kshift": [0.5, 0, 0]},
    )
    lines = text.splitlines()
    assert lines[2] == "Monkhorst-Pack"
    assert lines[3].split() == ["3", "4", "5"]
    assert lines[4].split()[0] == "0.5"


def test_kpoints_automatic_spacing_scales_with_cell(water_cell):
    dense = writer.build_kpoints(
        water_cell, {"kpoint_mode": "Automatic (spacing)", "kspacing": 0.01}
    ).splitlines()[3]
    coarse = writer.build_kpoints(
        water_cell, {"kpoint_mode": "Automatic (spacing)", "kspacing": 0.5}
    ).splitlines()[3]
    assert [int(v) for v in dense.split()] > [int(v) for v in coarse.split()]
    assert coarse.split() == ["1", "1", "1"]


def test_kpoints_mesh_is_clamped_to_one(water_cell):
    text = writer.build_kpoints(water_cell, {"kmesh": [0, -3, 2]})
    assert text.splitlines()[3].split() == ["1", "1", "2"]


# -- POTCAR ----------------------------------------------------------------


def test_potcar_notes_use_recommended_names(rutile_cell):
    text = writer.build_potcar_notes(rutile_cell)
    assert "Ti_sv" in text
    assert "/O/POTCAR" in text
    assert text.index("Ti_sv") < text.index("/O/POTCAR")  # POSCAR species order


def test_potcar_notes_plain_symbols_when_disabled(rutile_cell):
    text = writer.build_potcar_notes(rutile_cell, {"recommended_potcar": False})
    assert "Ti_sv" not in text
    assert "/Ti/POTCAR" in text


def test_potcar_notes_flag_unmapped_elements():
    cell = cm.cell_from_molecule(["Bk"], [[0.0, 0.0, 0.0]], padding=2.0)
    assert "No recommended-set entry for: Bk" in writer.build_potcar_notes(cell)


def test_potential_helpers():
    assert potentials.potcar_name("li") == "Li_sv"
    assert potentials.potcar_name("Li", recommended=False) == "Li"
    assert potentials.potcar_name("Zz") == "Zz"
    assert potentials.unmapped_elements(["O", "Zz"]) == ["Zz"]
    assert potentials.potcar_names(["O", "Ga"]) == ["O", "Ga_d"]


def test_recommended_table_matches_vasp_wiki_spot_checks():
    for element, expected in [
        ("Li", "Li_sv"), ("Na", "Na_pv"), ("K", "K_sv"), ("Ti", "Ti_sv"),
        ("Cr", "Cr_pv"), ("Fe", "Fe"), ("Ga", "Ga_d"), ("Mo", "Mo_sv"),
        ("Cs", "Cs_sv"), ("W", "W_sv"), ("Pb", "Pb_d"), ("Bi", "Bi_d"),
    ]:
        assert potentials.RECOMMENDED_PAW[element] == expected


# -- bundle ----------------------------------------------------------------


def test_build_all_returns_four_files(water_cell):
    files = writer.build_all(water_cell)
    assert set(files) == {"POSCAR", "INCAR", "KPOINTS", "POTCAR.readme"}
    assert all(text.endswith("\n") for text in files.values())


def test_build_preview_contains_every_section(water_cell):
    preview = writer.build_preview(water_cell)
    for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR.readme"):
        assert name in preview


def test_default_settings_are_independent_copies():
    first = writer.default_settings()
    first["encut"] = 1.0
    assert writer.default_settings()["encut"] == 520.0
