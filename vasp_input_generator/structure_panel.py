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
SHARED_MODULE_VERSION = "0.3.0"

from PyQt6.QtCore import pyqtSignal
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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .cell_model import (
    Cell,
    build_slab,
    cell_from_viewer_structure,
    make_supercell,
    molecule_to_cell,
    normalize_miller,
    parse_cif_file,
)

SOURCE_MOLECULE = "Current molecule (vacuum box)"
SOURCE_CIF = "CIF file"
SOURCE_VIEWER = "CIF Viewer panel (currently loaded)"
SOURCES = (SOURCE_MOLECULE, SOURCE_CIF, SOURCE_VIEWER)


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


class StructurePanel(QWidget):
    """Source selector + cell builder.  Emits ``changed`` on every edit."""

    changed = pyqtSignal()

    def __init__(self, get_molecule=None, get_cif_viewer=None, parent=None):
        super().__init__(parent)
        self.get_molecule = get_molecule
        self.get_cif_viewer = get_cif_viewer
        self._last_error = ""

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

        self.slab_box = QGroupBox("Surface slab")
        self.slab_box.setCheckable(True)
        self.slab_box.setChecked(False)
        slab_form = QFormLayout(self.slab_box)

        miller_widget = QWidget()
        miller_layout = QHBoxLayout(miller_widget)
        miller_layout.setContentsMargins(0, 0, 0, 0)
        self.miller_spins = []
        for label in ("h", "k", "l"):
            spin = QSpinBox()
            spin.setRange(-9, 9)
            spin.setValue(1 if label == "l" else 0)
            miller_layout.addWidget(QLabel(f"{label}:"))
            miller_layout.addWidget(spin)
            self.miller_spins.append(spin)
        self.four_index_check = QCheckBox("(hkil)")
        self.four_index_check.setToolTip(
            "Hexagonal Miller-Bravais indices. i is fixed at -(h+k) and dropped, "
            "so (1 0 -1 0) is the same surface as (1 0 0)."
        )
        miller_layout.addWidget(self.four_index_check)
        self.i_label = QLabel("i: 0")
        miller_layout.addWidget(self.i_label)
        miller_layout.addStretch(1)
        slab_form.addRow("Miller indices:", miller_widget)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 100)
        self.layers_spin.setValue(6)
        slab_form.addRow("Layers:", self.layers_spin)

        self.vacuum_spin = QDoubleSpinBox()
        self.vacuum_spin.setRange(0.0, 100.0)
        self.vacuum_spin.setSingleStep(1.0)
        self.vacuum_spin.setValue(15.0)
        self.vacuum_spin.setSuffix(" A")
        slab_form.addRow("Vacuum:", self.vacuum_spin)

        self.termination_spin = QDoubleSpinBox()
        self.termination_spin.setRange(0.0, 1.0)
        self.termination_spin.setSingleStep(0.05)
        self.termination_spin.setDecimals(3)
        self.termination_spin.setToolTip(
            "Slides the cut plane through the bulk cell to expose a different termination."
        )
        slab_form.addRow("Termination shift:", self.termination_spin)

        self.orthogonal_check = QCheckBox("Put c along the surface normal")
        self.orthogonal_check.setChecked(True)
        slab_form.addRow("", self.orthogonal_check)

        layout.addWidget(self.slab_box)

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
        for spin in self.repeat_spins:
            spin.valueChanged.connect(self._emit_changed)
        self.slab_box.toggled.connect(self._emit_changed)
        for spin in self.miller_spins:
            spin.valueChanged.connect(self._on_miller_changed)
        self.four_index_check.toggled.connect(self._on_miller_changed)
        self.layers_spin.valueChanged.connect(self._emit_changed)
        self.vacuum_spin.valueChanged.connect(self._emit_changed)
        self.termination_spin.valueChanged.connect(self._emit_changed)
        self.orthogonal_check.toggled.connect(self._emit_changed)

        self._on_source_changed(self.source_combo.currentText())
        self._on_padding_mode_changed(False)
        self._on_miller_changed()

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
            "supercell": [spin.value() for spin in self.repeat_spins],
            "slab_enabled": self.slab_box.isChecked(),
            "miller": [spin.value() for spin in self.miller_spins],
            "miller_four_index": self.four_index_check.isChecked(),
            "slab_layers": self.layers_spin.value(),
            "slab_vacuum": self.vacuum_spin.value(),
            "slab_shift": self.termination_spin.value(),
            "slab_orthogonal_c": self.orthogonal_check.isChecked(),
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
        repeats = settings.get("supercell") or [1, 1, 1]
        for spin, value in zip(self.repeat_spins, repeats):
            spin.setValue(max(1, int(value)))
        self.slab_box.setChecked(bool(settings.get("slab_enabled", False)))
        for spin, value in zip(self.miller_spins, settings.get("miller") or [0, 0, 1]):
            spin.setValue(int(value))
        self.four_index_check.setChecked(bool(settings.get("miller_four_index", False)))
        self.layers_spin.setValue(max(1, int(settings.get("slab_layers", 6))))
        self.vacuum_spin.setValue(float(settings.get("slab_vacuum", 15.0)))
        self.termination_spin.setValue(float(settings.get("slab_shift", 0.0)))
        self.orthogonal_check.setChecked(bool(settings.get("slab_orthogonal_c", True)))

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
        if self.slab_box.isChecked():
            if cell.source == "molecule":
                raise ValueError(
                    "A slab is cut from a periodic bulk structure - load a CIF, or turn "
                    "the slab off to keep the molecule in a box."
                )
            cell = build_slab(
                cell,
                miller=self.miller(),
                layers=self.layers_spin.value(),
                vacuum=self.vacuum_spin.value(),
                shift=self.termination_spin.value(),
                orthogonal_c=self.orthogonal_check.isChecked(),
            )

        return make_supercell(cell, [spin.value() for spin in self.repeat_spins])

    def miller(self):
        """The (hkl) triple, folding a hexagonal (hkil) entry down to three indices."""
        h, k, l = (spin.value() for spin in self.miller_spins)
        if self.four_index_check.isChecked():
            return normalize_miller([h, k, -(h + k), l])
        return normalize_miller([h, k, l])

    def refresh_summary(self, cell=None, error: str = "") -> None:
        if error:
            self.summary_label.setText(f"<b>{error}</b>")
            return
        if cell is None:
            self.summary_label.setText("No structure yet.")
            return
        from .cell_model import formula

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

    # -- internals --------------------------------------------------------

    def padding(self):
        """Scalar padding, or one value per axis when per-axis mode is on."""
        if self.per_axis_check.isChecked():
            return [spin.value() for spin in self.axis_padding_spins]
        return self.padding_spin.value()

    def _on_miller_changed(self, *_args) -> None:
        h, k = self.miller_spins[0].value(), self.miller_spins[1].value()
        self.i_label.setText(f"i: {-(h + k)}")
        self.i_label.setVisible(self.four_index_check.isChecked())
        self._emit_changed()

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
        self.slab_box.setVisible(text in (SOURCE_CIF, SOURCE_VIEWER))
        self._emit_changed()

    def _browse_cif(self) -> None:  # pragma: no cover - file dialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CIF", self.cif_edit.text(), "CIF files (*.cif);;All files (*)"
        )
        if path:
            self.cif_edit.setText(path)

    def _emit_changed(self, *_args) -> None:
        self.changed.emit()
