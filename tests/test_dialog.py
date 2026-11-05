import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import writer  # noqa: E402
from vasp_input_generator.main_dialog import VaspInputDialog  # noqa: E402

from shared_fixtures import _FakeMol  # noqa: E402


@pytest.fixture
def dialog(qapp):
    mol = _FakeMol(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    settings = writer.default_settings()
    dlg = VaspInputDialog(
        persistent_settings=settings,
        get_molecule=lambda: mol,
        get_selected_indices=lambda: [0],
    )
    yield dlg
    dlg.deleteLater()


def test_dialog_builds_preview_from_the_molecule(dialog):
    text = dialog.preview.toPlainText()
    assert "INCAR" in text and "POSCAR" in text
    assert dialog._cell is not None
    assert dialog.save_button.isEnabled()


def test_dialog_preview_tracks_task_changes(dialog):
    dialog.task_combo.setCurrentText("Relax ions")
    assert "IBRION  = 2" in dialog.preview.toPlainText()


def test_dialog_preview_tracks_kpoint_changes(dialog):
    dialog.kmode_combo.setCurrentText("Gamma-only")
    assert "Gamma-only" in dialog.preview.toPlainText()


def test_dialog_updates_persistent_settings(dialog):
    dialog.encut_spin.setValue(650.0)
    assert dialog.persistent_settings["encut"] == 650.0


def test_dialog_marks_the_project_modified(qapp):
    seen = []
    mol = _FakeMol(["H"], [[0.0, 0.0, 0.0]])
    dlg = VaspInputDialog(
        persistent_settings={}, get_molecule=lambda: mol, mark_modified=lambda: seen.append(1)
    )
    dlg.encut_spin.setValue(400.0)
    assert seen
    dlg.deleteLater()


def test_dialog_settings_roundtrip(dialog):
    settings = dialog.read_settings()
    settings.update({"task": "Relax ions + cell", "encut": 480.0, "kmesh": [2, 3, 4]})
    dialog.apply_settings(settings)
    out = dialog.read_settings()
    assert out["task"] == "Relax ions + cell"
    assert out["encut"] == 480.0
    assert out["kmesh"] == [2, 3, 4]


def test_dialog_invalid_ediff_falls_back(dialog):
    dialog.ediff_edit.setText("not a number")
    assert dialog.read_settings()["ediff"] == 1e-6


def test_dialog_selection_drives_selective_dynamics(dialog):
    dialog.selective_check.setChecked(True)
    assert dialog.read_settings()["frozen_indices"] == [0]
    assert "Selective dynamics" in dialog.preview.toPlainText()


def test_dialog_no_selection_when_disabled(dialog):
    dialog.selective_check.setChecked(False)
    assert dialog.read_settings()["frozen_indices"] == []


def test_dialog_reports_a_missing_molecule(qapp):
    dlg = VaspInputDialog(persistent_settings={}, get_molecule=lambda: None)
    assert "No molecule" in dlg.preview.toPlainText()
    assert not dlg.save_button.isEnabled()
    assert dlg._cell is None
    dlg.deleteLater()


def test_dialog_save_writes_every_file(dialog, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dialog.save_files()

    for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR.readme"):
        assert (tmp_path / name).exists()
    assert "\r" not in (tmp_path / "POSCAR").read_bytes().decode()


def test_dialog_save_is_cancellable(dialog, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    dialog.save_files()
    assert not list(tmp_path.iterdir())


def test_dialog_save_asks_before_overwriting(dialog, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    (tmp_path / "INCAR").write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog.save_files()
    assert (tmp_path / "INCAR").read_text(encoding="utf-8") == "keep me"


def test_dialog_save_without_structure_warns(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(a))
    dlg = VaspInputDialog(persistent_settings={}, get_molecule=lambda: None)
    dlg.save_files()
    assert seen
    dlg.deleteLater()


def test_dialog_copy_preview(dialog, qapp):
    dialog.copy_preview()
    assert "INCAR" in qapp.clipboard().text()


def DIALOG_FACTORY(context):
    mol = _FakeMol(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    return VaspInputDialog(persistent_settings=writer.default_settings(), get_molecule=lambda: mol, context=context)


# -- drag and drop ----------------------------------------------------------


def test_the_dialog_accepts_a_dropped_cif(dialog, tmp_path):
    from shared_fixtures import _FakeDropEvent, _FakeMime

    path = tmp_path / "dropped.cif"
    path.write_text("data_x", encoding="utf-8")
    assert dialog.acceptDrops()
    event = _FakeDropEvent(_FakeMime([str(path)]))
    dialog.dropEvent(event)
    assert event.accepted
    assert dialog.structure_panel.cif_edit.text() == str(path)


def test_a_drag_without_a_cif_is_refused_by_the_dialog(dialog):
    from shared_fixtures import _FakeDropEvent, _FakeMime

    event = _FakeDropEvent(_FakeMime(["/tmp/notes.txt"]))
    dialog.dragEnterEvent(event)
    assert event.ignored and not event.accepted
    dialog.dropEvent(event)
    assert not event.accepted


def test_a_drag_move_follows_the_same_rule(dialog):
    from shared_fixtures import _FakeDropEvent, _FakeMime

    event = _FakeDropEvent(_FakeMime(["/tmp/x.cif"]))
    dialog.dragMoveEvent(event)
    assert event.accepted


def test_the_box_is_drawn_as_soon_as_the_dialog_opens(qapp):
    """Opening the generator should show the cell, not an empty viewer."""
    pytest.importorskip("rdkit")
    from shared_fixtures import _RecordingContext

    context = _RecordingContext()
    dlg = DIALOG_FACTORY(context)
    assert context.current_molecule is not None
    assert len(context.plotter.lines) == 12
    dlg.deleteLater()


from PySide6.QtCore import Qt  # noqa: E402


COPY_OR_MOVE = Qt.DropAction.CopyAction | Qt.DropAction.MoveAction


def _shift_event(kind, path, actions=COPY_OR_MOVE):
    """A real Qt event with Shift held, the way Explorer asks for a move."""
    from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    left, shift = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier
    if kind == "drop":
        event = QDropEvent(QPointF(1, 1), actions, mime, left, shift)
    elif kind == "move":
        event = QDragMoveEvent(QPoint(1, 1), actions, mime, left, shift)
    else:
        event = QDragEnterEvent(QPoint(1, 1), actions, mime, left, shift)
    # The event does not own its mime data, which has to outlive it.
    event.keepalive = mime
    return event


def test_a_shift_drop_on_the_dialog_is_a_copy(dialog, tmp_path):
    path = tmp_path / "x.cif"
    enter = _shift_event("enter", path)
    assert enter.proposedAction() == Qt.DropAction.MoveAction
    dialog.dragEnterEvent(enter)
    assert enter.dropAction() == Qt.DropAction.CopyAction
    event = _shift_event("drop", path)
    dialog.dropEvent(event)
    assert event.isAccepted()
    assert event.dropAction() == Qt.DropAction.CopyAction
    assert dialog.structure_panel.cif_edit.text() == path.as_posix()


def test_a_source_that_can_only_move_is_refused_by_the_dialog(dialog, tmp_path):
    before = dialog.structure_panel.cif_edit.text()
    event = _shift_event("drop", tmp_path / "x.cif", actions=Qt.DropAction.MoveAction)
    dialog.dropEvent(event)
    assert not event.isAccepted()
    assert dialog.structure_panel.cif_edit.text() == before
