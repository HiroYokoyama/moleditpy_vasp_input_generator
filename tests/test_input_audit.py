"""Regressions for the documented VASP input format and smearing options."""

import ast
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from vasp_input_generator import cell_model, writer
from vasp_input_generator.main_dialog import VaspInputDialog
from shared_fixtures import _FakeMol


@pytest.fixture
def cell():
    return cell_model.cell_from_molecule(["H"], [[0, 0, 0]], padding=8)


@pytest.mark.parametrize("smearing,expected", zip(writer.SMEARING, (0, 1, -1, -5)))
def test_dos_respects_selected_smearing(smearing, expected):
    incar = writer.build_incar({"task": "Density of states", "smearing": smearing})
    assert f"ISMEAR  = {expected}\n" in incar


def test_gamma_only_dos_with_default_smearing_is_valid(cell):
    files = writer.build_all(cell, {"task": "Density of states", "kpoint_mode": "Gamma-only"})
    assert "ISMEAR  = 0\n" in files["INCAR"]


@pytest.mark.parametrize("ismear", (-4, -5, -14, -15))
def test_extra_tetrahedron_with_gamma_only_is_rejected(cell, ismear):
    with pytest.raises(ValueError, match="at least 4 k-points"):
        writer.build_all(cell, {"kpoint_mode": "Gamma-only", "extra_incar": f"ISMEAR={ismear}"})


def test_gaussian_override_of_tetrahedron_is_valid_and_unique(cell):
    files = writer.build_all(cell, {
        "kpoint_mode": "Gamma-only", "smearing": writer.SMEARING[3],
        "extra_incar": "ismear=0; SIGMA=0.05 ! Gaussian override",
    })
    assert "ISMEAR  = -5" not in files["INCAR"]
    assert "ismear=0" in files["INCAR"]


def test_extra_tags_accept_comments_continuations_and_quoted_blocks():
    text = 'ISMEAR=-1; SIGMA=0.1 # ISMEAR=-5\nMAGMOM=1 \\\n  2\nWANNIER90_WIN="\nISMEAR=-5; # content\n"\nPLUGINS {\nSTRUCTURE=T\n}\n'
    tags = writer._extra_tags(text)
    assert tags["ISMEAR"] == "-1"
    assert tags["MAGMOM"].split() == ["1", "2"]
    assert "STRUCTURE" not in tags
    assert "# content" in tags["WANNIER90_WIN"]


@pytest.mark.parametrize("extra", ('ISMEAR=-1; ismear=0', 'SYSTEM="unclosed', 'PLUGINS { STRUCTURE=T'))
def test_ambiguous_or_unclosed_extra_tags_are_rejected(extra):
    with pytest.raises(ValueError):
        writer.build_incar({"extra_incar": extra})


def test_nested_tags_do_not_replace_global_tags():
    incar = writer.build_incar({"extra_incar": "IMAGES {\nISMEAR=-1\n}\n"})
    assert "ISMEAR  = 0" in incar


@pytest.mark.parametrize("value", (1.23456789012345e-8, 520.123456789012, -0.0123456789012345))
def test_numeric_values_retain_precision(value):
    assert float(writer._format_value(value)) == pytest.approx(value, rel=1e-14)


def test_title_cannot_add_an_incar_assignment(cell):
    settings = {"title": 'title\r\nline; ISMEAR=-5 # "comment"'}
    incar = writer.build_incar(settings)
    title = incar.splitlines()[0].partition("=")[2]
    assert all(char not in title for char in ';#!="')
    poscar = writer.build_poscar(cell, settings)
    assert poscar.splitlines()[1].strip() == "1.00000000000000"


def test_tetrahedron_shift_warning(cell):
    warnings = writer.validate(cell, {"smearing": writer.SMEARING[3], "kshift": [0.5, 0, 0]})
    assert any("unshifted Gamma" in warning for warning in warnings)


def test_invalid_preview_recovers_and_save_stops_before_directory_selection(qapp, monkeypatch):
    mol = _FakeMol(["H"], [[0, 0, 0]])
    dialog = VaspInputDialog(persistent_settings={}, get_molecule=lambda: mol)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    def unexpected_directory(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before choosing a directory")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", unexpected_directory)
    dialog.kmode_combo.setCurrentText("Gamma-only")
    dialog.smearing_combo.setCurrentText(writer.SMEARING[3])
    assert "Invalid input" in dialog.preview.toPlainText()
    assert not dialog.save_button.isEnabled()
    dialog.save_files()
    assert warnings
    dialog.smearing_combo.setCurrentText(writer.SMEARING[0])
    assert dialog.save_button.isEnabled()
    assert "INCAR" in dialog.preview.toPlainText()
    dialog.deleteLater()


def test_main_plugin_imports_are_at_module_scope():
    root = Path(writer.__file__).parent
    for filename in ("__init__.py", "main_dialog.py", "writer.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node)), filename
