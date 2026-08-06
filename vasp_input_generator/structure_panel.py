"""Structure source panel shared by the periodic input generators.

Builds a :class:`cell_model.Cell` from the molecule currently open in MoleditPy
(wrapped in a vacuum box), from a CIF file, or from an open CIF Viewer panel —
optionally expanded to a supercell.

SHARED FILE.  A byte-identical copy lives in every periodic input generator
plugin (VASP / Quantum ESPRESSO / CP2K).  Bump ``SHARED_MODULE_VERSION`` on any
change and copy the file to the other plugins; each plugin's test suite pins the
version it expects, so a stale copy fails loudly.
"""

from __future__ import annotations

import logging
import os

SHARED_MODULE_NAME = "periodic-structure-panel"
SHARED_MODULE_VERSION = "0.11.0"

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import cell_preview
from .cell_model import (
    Cell,
    cell_from_viewer_structure,
    make_supercell,
    molecule_to_cell,
    parse_cif_file,
    primitive_cell,
)

SOURCE_MOLECULE = "Current molecule (vacuum box)"
SOURCE_CIF = "CIF file"
SOURCE_VIEWER = "CIF Viewer panel (currently loaded)"
SOURCES = (SOURCE_MOLECULE, SOURCE_CIF, SOURCE_VIEWER)

#: Suffixes accepted by drag and drop.  mmCIF uses .mmcif or .cif alike.
CIF_SUFFIXES = (".cif", ".mmcif")


def find_cif_viewer_widget(main_window):
    """Locate the CIF Viewer plugin's panel widget, or None.

    Plugin windows are namespaced per plugin, so ``context.get_window()`` cannot
    reach another plugin's dock — go through the plugin manager's registry and
    fall back to a widget scan.
    """
    if main_window is None:
        return None

    def _holder(window):
        if window is None:
            return None
        if getattr(window, "structure", None) is not None:
            return window
        inner = window.widget() if hasattr(window, "widget") else None
        if inner is not None and hasattr(inner, "structure"):
            return inner
        return None

    try:
        registry = getattr(getattr(main_window, "plugin_manager", None), "plugin_windows", None)
        if isinstance(registry, dict):
            for plugin_name, windows in registry.items():
                if not isinstance(windows, dict):
                    continue
                for window_id, window in windows.items():
                    key = f"{plugin_name} {window_id}".lower()
                    if "cif" not in key or "viewer" not in key:
                        continue
                    holder = _holder(window)
                    if holder is not None:
                        return holder
    except Exception as exc:  # pragma: no cover - host internals guard
        logging.debug("CIF Viewer lookup via the plugin manager failed: %s", exc)

    try:
        for child in main_window.findChildren(QWidget):
            if type(child).__name__ == "CifViewerWidget" and hasattr(child, "structure"):
                return child
    except Exception as exc:  # pragma: no cover - host internals guard
        logging.debug("CIF Viewer widget scan failed: %s", exc)
    return None


def dropped_cif_path(mime) -> str:
    """Local path of the first .cif in a drag, or "" when there is none."""
    if mime is None or not mime.hasUrls():
        return ""
    for url in mime.urls():
        path = url.toLocalFile()
        if path and os.path.splitext(path)[1].lower() in CIF_SUFFIXES:
            return path
    return ""


class StructurePanel(QWidget):
    """Source selector + cell builder.  Emits ``changed`` on every edit."""

    changed = pyqtSignal()
    previewed = pyqtSignal()

    def __init__(self, get_molecule=None, get_cif_viewer=None, parent=None, context=None):
        super().__init__(parent)
        self.get_molecule = get_molecule
        self.get_cif_viewer = get_cif_viewer
        self.context = context
        self._last_error = ""
        self._preview_actors = []
        self._auto_previewed_key = None
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        source_box = QGroupBox("Structure source")
        source_form = QFormLayout(source_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SOURCES))
        source_form.addRow("Source:", self.source_combo)

        self.mol_widget = QWidget()
        mol_form = QFormLayout(self.mol_widget)
        mol_form.setContentsMargins(0, 0, 0, 0)
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 50.0)
        self.padding_spin.setSingleStep(0.5)
        self.padding_spin.setValue(6.0)
        self.padding_spin.setSuffix(" A")
        mol_form.addRow("Vacuum padding:", self.padding_spin)
        self.cubic_check = QCheckBox("Force a cubic box")
        mol_form.addRow("", self.cubic_check)
        self.per_axis_check = QCheckBox("Vacuum per axis (slab: pad c only)")
        mol_form.addRow("", self.per_axis_check)
        self.axis_widget = QWidget()
        axis_layout = QHBoxLayout(self.axis_widget)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        self.axis_padding_spins = []
        for axis in "abc":
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 50.0)
            spin.setSingleStep(0.5)
            spin.setValue(6.0)
            spin.setSuffix(" A")
            axis_layout.addWidget(QLabel(f"{axis}:"))
            axis_layout.addWidget(spin)
            self.axis_padding_spins.append(spin)
        mol_form.addRow("", self.axis_widget)
        source_form.addRow(self.mol_widget)

        self.cif_widget = QWidget()
        cif_layout = QVBoxLayout(self.cif_widget)
        cif_layout.setContentsMargins(0, 0, 0, 0)
        path_row = QHBoxLayout()
        self.cif_edit = QLineEdit()
        self.cif_edit.setPlaceholderText("Path to a .cif file")
        self.load_button = QPushButton("Load CIF...")
        self.load_button.clicked.connect(self._browse_cif)
        path_row.addWidget(self.cif_edit, 1)
        path_row.addWidget(self.load_button)
        cif_layout.addLayout(path_row)
        source_form.addRow(self.cif_widget)

        self.expand_check = QCheckBox("Expand the asymmetric unit with the symmetry operations")
        self.expand_check.setChecked(True)
        source_form.addRow("", self.expand_check)

        self.primitive_check = QCheckBox("Reduce to the primitive cell")
        self.primitive_check.setToolTip(
            "Detects the translational symmetry of the structure and drops the repeats: "
            "a face-centred cell of 4 atoms becomes 1.\n"
            "The physics is unchanged and the code still finds the point group, but the "
            "run is far cheaper.\n"
            "Only translations are used, so this is not a full symmetry analysis."
        )
        source_form.addRow("", self.primitive_check)

        self.viewer_widget = QWidget()
        viewer_layout = QHBoxLayout(self.viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_label = QLabel("Uses the structure currently open in the CIF Viewer panel.")
        self.viewer_label.setWordWrap(True)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self._emit_changed)
        viewer_layout.addWidget(self.viewer_label, 1)
        viewer_layout.addWidget(self.reload_button)
        source_form.addRow(self.viewer_widget)

        layout.addWidget(source_box)

        super_box = QGroupBox("Supercell")
        super_layout = QGridLayout(super_box)
        self.repeat_spins = []
        for column, axis in enumerate("abc"):
            spin = QSpinBox()
            spin.setRange(1, 20)
            spin.setValue(1)
            super_layout.addWidget(QLabel(f"{axis}:"), 0, column * 2)
            super_layout.addWidget(spin, 0, column * 2 + 1)
            self.repeat_spins.append(spin)
        layout.addWidget(super_box)

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_button = QPushButton("Show in 3D view")
        self.preview_button.setToolTip(
            "Draw this cell in MoleditPy's own 3D view, with the unit-cell box."
        )
        self.preview_button.clicked.connect(self.preview_in_3d)
        self.clear_preview_button = QPushButton("Clear box")
        self.clear_preview_button.setToolTip("Remove the cell box from the 3D view.")
        self.clear_preview_button.clicked.connect(self.clear_3d_preview)
        self.bonds_check = QCheckBox("Bonds")
        self.bonds_check.setTristate(True)
        self.bonds_check.setCheckState(Qt.CheckState.PartiallyChecked)
        self.bonds_check.setToolTip(
            "Draw bonds inferred from covalent radii.\n"
            "Off by default: a periodic cell is cut at its faces, so every bond "
            "crossing a face is missing its partner and simply will not be drawn."
        )
        self.auto_preview_check = QCheckBox("Show automatically when a CIF is loaded")
        self.auto_preview_check.setChecked(True)
        self.auto_preview_check.setToolTip(
            "Draw the structure as soon as a CIF file (or the CIF Viewer panel) is read.\n"
            "Only a new structure triggers it, not every settings change."
        )
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.clear_preview_button)
        preview_layout.addWidget(self.bonds_check)
        preview_layout.addWidget(self.auto_preview_check)
        preview_layout.addStretch(1)
        layout.addWidget(preview_row)

        self.summary_label = QLabel("No structure yet.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        self.padding_spin.valueChanged.connect(self._emit_changed)
        self.cubic_check.toggled.connect(self._emit_changed)
        self.per_axis_check.toggled.connect(self._on_padding_mode_changed)
        for spin in self.axis_padding_spins:
            spin.valueChanged.connect(self._emit_changed)
        self.cif_edit.textChanged.connect(self._emit_changed)
        self.expand_check.toggled.connect(self._emit_changed)
        self.primitive_check.toggled.connect(self._emit_changed)
        self.bonds_check.toggled.connect(self._emit_changed)
        for spin in self.repeat_spins:
            spin.valueChanged.connect(self._emit_changed)

        self._on_source_changed(self.source_combo.currentText())
        self._on_padding_mode_changed(False)

    # -- state ------------------------------------------------------------

    def read_settings(self) -> dict:
        return {
            "structure_source": self.source_combo.currentText(),
            "padding": self.padding_spin.value(),
            "per_axis_padding": self.per_axis_check.isChecked(),
            "padding_axes": [spin.value() for spin in self.axis_padding_spins],
            "cubic_box": self.cubic_check.isChecked(),
            "cif_path": self.cif_edit.text(),
            "expand_symmetry": self.expand_check.isChecked(),
            "primitive_cell": self.primitive_check.isChecked(),
            "auto_preview_3d": self.auto_preview_check.isChecked(),
            "preview_bonds": self.bond_choice(),
            "supercell": [spin.value() for spin in self.repeat_spins],
        }

    def apply_settings(self, settings: dict) -> None:
        source = settings.get("structure_source")
        if source in SOURCES:
            self.source_combo.setCurrentText(source)
        self.padding_spin.setValue(float(settings.get("padding", 6.0)))
        self.per_axis_check.setChecked(bool(settings.get("per_axis_padding", False)))
        for spin, value in zip(self.axis_padding_spins, settings.get("padding_axes") or [6.0, 6.0, 6.0]):
            spin.setValue(float(value))
        self.cubic_check.setChecked(bool(settings.get("cubic_box", False)))
        self.cif_edit.setText(str(settings.get("cif_path", "") or ""))
        self.expand_check.setChecked(bool(settings.get("expand_symmetry", True)))
        self.primitive_check.setChecked(bool(settings.get("primitive_cell", False)))
        self.auto_preview_check.setChecked(bool(settings.get("auto_preview_3d", True)))
        choice = settings.get("preview_bonds", None)
        self.bonds_check.setCheckState(
            Qt.CheckState.PartiallyChecked if choice is None
            else (Qt.CheckState.Checked if choice else Qt.CheckState.Unchecked)
        )
        repeats = settings.get("supercell") or [1, 1, 1]
        for spin, value in zip(self.repeat_spins, repeats):
            spin.setValue(max(1, int(value)))

    # -- cell -------------------------------------------------------------

    def build_cell(self) -> Cell:
        """Build the cell for the current settings.  Raises ValueError on failure."""
        source = self.source_combo.currentText()
        if source == SOURCE_VIEWER:
            dock = self.get_cif_viewer() if self.get_cif_viewer is not None else None
            if dock is None:
                raise ValueError(
                    "The CIF Viewer panel is not open. Open it from View > CIF Viewer "
                    "Panel, or choose the CIF file source."
                )
            cell = cell_from_viewer_structure(
                getattr(dock, "structure", None),
                expand_asymmetric=self.expand_check.isChecked(),
            )
        elif source == SOURCE_CIF:
            path = self.cif_edit.text().strip()
            if not path:
                raise ValueError("Choose a CIF file first.")
            if not os.path.isfile(path):
                raise ValueError(f"CIF file not found:\n{path}")
            cell = parse_cif_file(path, expand=self.expand_check.isChecked())
        else:
            mol = self.get_molecule() if self.get_molecule is not None else None
            if mol is None:
                raise ValueError("No molecule is loaded in MoleditPy.")
            cell = molecule_to_cell(
                mol,
                padding=self.padding(),
                cubic=self.cubic_check.isChecked(),
            )
        if self.primitive_check.isChecked():
            # Reduce first: a supercell of the primitive cell is what the user
            # asked for, and reducing afterwards would just undo it.
            cell = primitive_cell(cell)
        return make_supercell(cell, [spin.value() for spin in self.repeat_spins])

    def refresh_summary(self, cell=None, error: str = "") -> None:
        if error:
            self.summary_label.setText(f"<b>{error}</b>")
            return
        if cell is None:
            self.summary_label.setText("No structure yet.")
            return
        from .cell_model import formula

        # Every dialog calls this after a successful build, which makes it the
        # one place that sees a new structure arrive.
        self.auto_preview(cell)

        if cell.source == "cif_viewer":
            self.viewer_label.setText(f"Copied from the CIF Viewer panel: <b>{cell.name}</b>")
        a, b, c = cell.lengths
        alpha, beta, gamma = cell.angles
        self.summary_label.setText(
            f"{len(cell.atoms)} atoms — {formula(cell)}<br>"
            f"a={a:.4f} b={b:.4f} c={c:.4f} A, "
            f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f} deg, "
            f"V={cell.volume:.2f} A<sup>3</sup>"
        )

    # -- drag and drop ----------------------------------------------------

    def bond_choice(self):
        """True/False when the user has decided, None to leave it automatic."""
        state = self.bonds_check.checkState()
        if state == Qt.CheckState.PartiallyChecked:
            return None
        return state == Qt.CheckState.Checked

    def load_cif_path(self, path: str) -> None:
        """Point the panel at a CIF, switching the source over to match.

        Dropping a file is a request to use it, so leaving the source on
        "current molecule" would silently ignore the drop.
        """
        self.source_combo.setCurrentText(SOURCE_CIF)
        self.cif_edit.setText(str(path))

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Drops land on this panel only, not on the whole dialog."""
        if dropped_cif_path(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        path = dropped_cif_path(event.mimeData())
        if not path:
            event.ignore()
            return
        self.load_cif_path(path)
        event.acceptProposedAction()

    # -- 3D preview -------------------------------------------------------

    def preview_in_3d(self) -> None:
        """Draw the current cell, box included, in MoleditPy's 3D view.

        The view is only brought to the front for a crystal.  A molecule in a
        vacuum box is the one being edited, so the 2D editor is left in place
        even when the button is pressed deliberately.
        """
        try:
            cell = self.build_cell()
            self._preview_actors = cell_preview.show_cell(
                self.context, cell, self._preview_actors,
                show_bonds=self.bond_choice(),
            )
        except (ValueError, OSError, AttributeError, ImportError, RuntimeError) as exc:
            QMessageBox.warning(self, "3D preview", str(exc))
            return
        self._auto_previewed_key = None
        self.previewed.emit()

    def auto_preview(self, cell) -> None:
        """Show a newly loaded crystal without being asked.

        Fires only when the structure itself changes, so tweaking a k-mesh or a
        supercell does not redraw the 3D view on every keystroke.  Failures stay
        silent: this was not something the user clicked.
        """
        if cell is None or not self.auto_preview_check.isChecked():
            return
        # A molecule wrapped in a vacuum box is shown too: its boundary is
        # exactly what the padding settings are there to control.
        key = (cell.name, cell.source, len(cell.atoms),
               tuple(round(v, 6) for v in cell.lengths),
               self.bond_choice())
        if key == self._auto_previewed_key:
            return
        self._auto_previewed_key = key
        try:
            self._preview_actors = cell_preview.show_cell(
                self.context, cell, self._preview_actors,
                show_bonds=self.bond_choice(),
            )
        except (ValueError, OSError, AttributeError, ImportError, RuntimeError) as exc:
            logging.debug("Automatic 3D preview skipped: %s", exc)

    def clear_3d_preview(self) -> None:
        """Remove the cell box.  Safe to call when nothing was drawn."""
        self._preview_actors = cell_preview.clear_cell_box(
            self.context, self._preview_actors
        )

    # -- internals --------------------------------------------------------

    def padding(self):
        """Scalar padding, or one value per axis when per-axis mode is on."""
        if self.per_axis_check.isChecked():
            return [spin.value() for spin in self.axis_padding_spins]
        return self.padding_spin.value()

    def _on_padding_mode_changed(self, checked) -> None:
        per_axis = self.per_axis_check.isChecked()
        self.axis_widget.setVisible(per_axis)
        self.padding_spin.setEnabled(not per_axis)
        # A cubic box and per-axis vacuum are contradictory requests.
        self.cubic_check.setEnabled(not per_axis)
        self._emit_changed()

    def _on_source_changed(self, text: str) -> None:
        self.mol_widget.setVisible(text == SOURCE_MOLECULE)
        self.cif_widget.setVisible(text == SOURCE_CIF)
        self.viewer_widget.setVisible(text == SOURCE_VIEWER)
        # The asymmetric-unit toggle drives both crystal sources.
        self.expand_check.setVisible(text in (SOURCE_CIF, SOURCE_VIEWER))
        self._emit_changed()

    def _browse_cif(self) -> None:  # pragma: no cover - file dialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CIF", self.cif_edit.text(), "CIF files (*.cif);;All files (*)"
        )
        if path:
            self.cif_edit.setText(path)

    def _emit_changed(self, *_args) -> None:
        self.changed.emit()
