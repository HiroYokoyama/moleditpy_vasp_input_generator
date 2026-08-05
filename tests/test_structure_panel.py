import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402
from vasp_input_generator import structure_panel as sp  # noqa: E402

from test_cell_model import CUBIC_CIF, _FakeMol  # noqa: E402


# -- shared module identity ------------------------------------------------


@pytest.mark.parametrize(
    "module,name,version",
    [
        (cm, "periodic-cell-model", "0.1.0"),
        (sp, "periodic-structure-panel", "0.1.0"),
        (__import__("vasp_input_generator.elements", fromlist=["x"]), "periodic-elements", "0.1.0"),
    ],
)
def test_shared_module_identity(module, name, version):
    """A shared file copied from another plugin must match the pinned version."""
    assert module.SHARED_MODULE_NAME == name
    assert module.SHARED_MODULE_VERSION == version


# -- CIF Viewer interop ----------------------------------------------------


def _viewer_structure(is_asymmetric=False, atoms=None):
    lattice = cm.cell_vectors((4.0, 4.0, 4.0), (90.0, 90.0, 90.0))
    if atoms is None:
        atoms = [
            types.SimpleNamespace(
                label="Na1", element="Na", fract=np.array([0.0, 0.0, 0.0]), occupancy=1.0
            ),
            types.SimpleNamespace(
                label="Cl1", element="Cl", fract=np.array([0.5, 0.5, 0.5]), occupancy=1.0
            ),
        ]
    return types.SimpleNamespace(
        name="NaCl",
        cell_lengths=(4.0, 4.0, 4.0),
        cell_angles=(90.0, 90.0, 90.0),
        lattice=lattice,
        atoms=atoms,
        space_group="F m -3 m",
        is_asymmetric_unit_only=is_asymmetric,
    )


def test_cell_from_viewer_structure():
    cell = cm.cell_from_viewer_structure(_viewer_structure())
    assert cell.source == "cif_viewer"
    assert cell.name == "NaCl"
    assert [atom.element for atom in cell.atoms] == ["Na", "Cl"]
    assert cell.lengths == (4.0, 4.0, 4.0)
    assert np.allclose(cell.atoms[1].cart, [2.0, 2.0, 2.0])


def test_cell_from_viewer_structure_derives_missing_parameters():
    structure = _viewer_structure()
    structure.cell_lengths = None
    structure.cell_angles = None
    cell = cm.cell_from_viewer_structure(structure)
    assert cell.lengths == pytest.approx((4.0, 4.0, 4.0))
    assert cell.angles == pytest.approx((90.0, 90.0, 90.0))


def test_cell_from_viewer_structure_wraps_positions():
    atoms = [types.SimpleNamespace(label="H1", element="H", fract=np.array([1.25, -0.5, 0.0]))]
    cell = cm.cell_from_viewer_structure(_viewer_structure(atoms=atoms))
    assert np.allclose(cell.atoms[0].fract, [0.25, 0.5, 0.0])


def test_cell_from_viewer_structure_rejects_empty():
    with pytest.raises(ValueError):
        cm.cell_from_viewer_structure(None)
    with pytest.raises(ValueError, match="no atoms"):
        cm.cell_from_viewer_structure(_viewer_structure(atoms=[]))


def test_cell_from_viewer_structure_rejects_bad_lattice():
    structure = _viewer_structure()
    structure.lattice = np.zeros((2, 2))
    with pytest.raises(ValueError, match="lattice"):
        cm.cell_from_viewer_structure(structure)


def test_cell_from_viewer_asymmetric_needs_spacegroup_expansion(monkeypatch):
    monkeypatch.setattr(cm, "_spacegroup_operations", lambda symbol: [])
    with pytest.raises(ValueError, match="asymmetric unit"):
        cm.cell_from_viewer_structure(_viewer_structure(is_asymmetric=True))


def test_cell_from_viewer_asymmetric_expands_when_ops_available(monkeypatch):
    ops = [cm.parse_symmetry_operation(op) for op in ("x, y, z", "x+1/2, y+1/2, z+1/2")]
    monkeypatch.setattr(cm, "_spacegroup_operations", lambda symbol: ops)
    cell = cm.cell_from_viewer_structure(_viewer_structure(is_asymmetric=True))
    assert len(cell.atoms) == 4


def test_cell_from_viewer_asymmetric_can_be_taken_as_is():
    cell = cm.cell_from_viewer_structure(
        _viewer_structure(is_asymmetric=True), expand_asymmetric=False
    )
    assert len(cell.atoms) == 2


# -- viewer lookup ---------------------------------------------------------


class _FakeDock:
    def __init__(self, inner):
        self._inner = inner

    def widget(self):
        return self._inner


def test_find_cif_viewer_widget_via_plugin_manager():
    panel = types.SimpleNamespace(structure=_viewer_structure())
    main_window = types.SimpleNamespace(
        plugin_manager=types.SimpleNamespace(
            plugin_windows={"CIF Viewer": {"cif_viewer_panel": _FakeDock(panel)}}
        )
    )
    assert sp.find_cif_viewer_widget(main_window) is panel


def test_find_cif_viewer_widget_accepts_direct_holder():
    panel = types.SimpleNamespace(structure=_viewer_structure())
    main_window = types.SimpleNamespace(
        plugin_manager=types.SimpleNamespace(
            plugin_windows={"CIF Viewer": {"cif_viewer_panel": panel}}
        )
    )
    assert sp.find_cif_viewer_widget(main_window) is panel


def test_find_cif_viewer_widget_ignores_other_plugins():
    other = types.SimpleNamespace(structure=_viewer_structure())
    main_window = types.SimpleNamespace(
        plugin_manager=types.SimpleNamespace(
            plugin_windows={"Cube Viewer": {"cube_panel": other}}
        ),
        findChildren=lambda _cls: [],
    )
    assert sp.find_cif_viewer_widget(main_window) is None


def test_find_cif_viewer_widget_handles_missing_host():
    assert sp.find_cif_viewer_widget(None) is None


# -- panel behaviour (real Qt) --------------------------------------------


@pytest.fixture
def panel(qapp, tmp_path):
    cif = tmp_path / "cubic.cif"
    cif.write_text(CUBIC_CIF, encoding="utf-8")
    mol = _FakeMol(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    viewer = types.SimpleNamespace(structure=_viewer_structure())
    widget = sp.StructurePanel(get_molecule=lambda: mol, get_cif_viewer=lambda: viewer)
    widget._cif_path = str(cif)
    yield widget
    widget.deleteLater()


def test_panel_defaults_to_molecule_source(panel):
    cell = panel.build_cell()
    assert cell.source == "molecule"
    assert len(cell.atoms) == 3


def test_panel_molecule_padding_changes_box(panel):
    panel.padding_spin.setValue(10.0)
    assert panel.build_cell().lengths[1] > 20.0


def test_panel_cif_source(panel):
    panel.source_combo.setCurrentText(sp.SOURCE_CIF)
    panel.cif_edit.setText(panel._cif_path)
    cell = panel.build_cell()
    assert cell.source == "cif"
    assert len(cell.atoms) == 2


def test_panel_cif_source_requires_path(panel):
    panel.source_combo.setCurrentText(sp.SOURCE_CIF)
    panel.cif_edit.setText("")
    with pytest.raises(ValueError, match="Choose a CIF file"):
        panel.build_cell()


def test_panel_cif_source_reports_missing_file(panel, tmp_path):
    panel.source_combo.setCurrentText(sp.SOURCE_CIF)
    panel.cif_edit.setText(str(tmp_path / "nope.cif"))
    with pytest.raises(ValueError, match="not found"):
        panel.build_cell()


def test_panel_viewer_source(panel):
    panel.source_combo.setCurrentText(sp.SOURCE_VIEWER)
    cell = panel.build_cell()
    assert cell.source == "cif_viewer"
    assert len(cell.atoms) == 2


def test_panel_viewer_source_without_panel(qapp):
    widget = sp.StructurePanel(get_molecule=lambda: None, get_cif_viewer=lambda: None)
    widget.source_combo.setCurrentText(sp.SOURCE_VIEWER)
    with pytest.raises(ValueError, match="not open"):
        widget.build_cell()
    widget.deleteLater()


def test_panel_requires_a_molecule(qapp):
    widget = sp.StructurePanel(get_molecule=lambda: None)
    with pytest.raises(ValueError, match="No molecule"):
        widget.build_cell()
    widget.deleteLater()


def test_panel_applies_supercell(panel):
    panel.repeat_spins[0].setValue(2)
    panel.repeat_spins[2].setValue(3)
    assert len(panel.build_cell().atoms) == 3 * 6


def test_panel_visibility_follows_source(panel):
    panel.source_combo.setCurrentText(sp.SOURCE_CIF)
    assert panel.cif_widget.isVisibleTo(panel)
    assert not panel.mol_widget.isVisibleTo(panel)
    assert panel.expand_check.isVisibleTo(panel)

    panel.source_combo.setCurrentText(sp.SOURCE_VIEWER)
    assert panel.viewer_widget.isVisibleTo(panel)
    assert not panel.cif_widget.isVisibleTo(panel)

    panel.source_combo.setCurrentText(sp.SOURCE_MOLECULE)
    assert panel.mol_widget.isVisibleTo(panel)
    assert not panel.expand_check.isVisibleTo(panel)


def test_panel_settings_roundtrip(panel):
    settings = {
        "structure_source": sp.SOURCE_CIF,
        "padding": 8.5,
        "cubic_box": True,
        "cif_path": panel._cif_path,
        "expand_symmetry": False,
        "supercell": [2, 3, 4],
    }
    panel.apply_settings(settings)
    assert panel.read_settings() == settings


def test_panel_settings_ignore_unknown_source(panel):
    panel.apply_settings({"structure_source": "nonsense"})
    assert panel.source_combo.currentText() in sp.SOURCES


def test_panel_emits_changed_on_edit(panel):
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.padding_spin.setValue(7.5)
    assert seen


def test_panel_summary_reports_cell_and_errors(panel):
    panel.refresh_summary(panel.build_cell())
    assert "atoms" in panel.summary_label.text()
    panel.refresh_summary(error="boom")
    assert "boom" in panel.summary_label.text()
    panel.refresh_summary(None)
    assert "No structure" in panel.summary_label.text()


def test_panel_summary_labels_viewer_source(panel):
    panel.source_combo.setCurrentText(sp.SOURCE_VIEWER)
    panel.refresh_summary(panel.build_cell())
    assert "CIF Viewer" in panel.viewer_label.text()
