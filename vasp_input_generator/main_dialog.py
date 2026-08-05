"""VASP input generator dialog."""

from __future__ import annotations

import logging
import os

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import writer
from .structure_panel import StructurePanel

_FILE_ORDER = ("INCAR", "KPOINTS", "POSCAR", "POTCAR.readme")


class VaspInputDialog(QDialog):
    def __init__(
        self,
        parent=None,
        persistent_settings=None,
        get_molecule=None,
        get_selected_indices=None,
        mark_modified=None,
        get_cif_viewer=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("VASP Input Generator")
        self.resize(940, 720)

        self.persistent_settings = persistent_settings if persistent_settings is not None else {}
        self.get_selected_indices = get_selected_indices
        self.mark_modified = mark_modified
        self._updating = False
        self._cell = None

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.structure_panel = StructurePanel(
            get_molecule=get_molecule, get_cif_viewer=get_cif_viewer
        )
        structure_tab = QWidget()
        structure_layout = QVBoxLayout(structure_tab)
        structure_layout.addWidget(self.structure_panel)
        structure_layout.addWidget(self._build_poscar_box())
        structure_layout.addStretch(1)
        self.tabs.addTab(structure_tab, "Structure")

        self.tabs.addTab(self._build_calculation_tab(), "Calculation")
        self.tabs.addTab(self._build_kpoints_tab(), "K-points")

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Courier New", 9))
        self.tabs.addTab(self.preview, "Preview")

        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton("Save Files...", QDialogButtonBox.ButtonRole.AcceptRole)
        self.copy_button = buttons.addButton("Copy Preview", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.save_button.clicked.connect(self.save_files)
        self.copy_button.clicked.connect(self.copy_preview)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.apply_settings(self.persistent_settings)
        self.structure_panel.changed.connect(self.update_preview)
        self.update_preview()

    # -- tab construction -------------------------------------------------

    def _build_poscar_box(self) -> QGroupBox:
        box = QGroupBox("POSCAR")
        form = QFormLayout(box)
        self.coord_combo = QComboBox()
        self.coord_combo.addItems(writer.COORD_MODES)
        form.addRow("Coordinates:", self.coord_combo)
        self.selective_check = QCheckBox("Selective dynamics (freeze the atoms selected in MoleditPy)")
        form.addRow("", self.selective_check)
        self.title_edit = QLineEdit()
        form.addRow("Title:", self.title_edit)

        self.coord_combo.currentTextChanged.connect(self.update_preview)
        self.selective_check.toggled.connect(self.update_preview)
        self.title_edit.textChanged.connect(self.update_preview)
        return box

    def _build_calculation_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        main_box = QGroupBox("Calculation")
        form = QFormLayout(main_box)
        self.task_combo = QComboBox()
        self.task_combo.addItems(writer.TASKS)
        form.addRow("Task:", self.task_combo)
        self.functional_combo = QComboBox()
        self.functional_combo.addItems(writer.FUNCTIONALS)
        form.addRow("Functional:", self.functional_combo)
        self.encut_spin = QDoubleSpinBox()
        self.encut_spin.setRange(100.0, 2000.0)
        self.encut_spin.setSingleStep(10.0)
        self.encut_spin.setSuffix(" eV")
        form.addRow("ENCUT:", self.encut_spin)
        self.ediff_edit = QLineEdit()
        form.addRow("EDIFF:", self.ediff_edit)
        self.ediffg_spin = QDoubleSpinBox()
        self.ediffg_spin.setRange(-10.0, 10.0)
        self.ediffg_spin.setDecimals(4)
        self.ediffg_spin.setSingleStep(0.01)
        form.addRow("EDIFFG:", self.ediffg_spin)
        self.nsw_spin = QSpinBox()
        self.nsw_spin.setRange(0, 100000)
        form.addRow("NSW:", self.nsw_spin)
        self.vdw_combo = QComboBox()
        self.vdw_combo.addItems(writer.VDW_OPTIONS)
        form.addRow("Dispersion:", self.vdw_combo)
        outer.addWidget(main_box)

        elec_box = QGroupBox("Electronic structure")
        elec_form = QFormLayout(elec_box)
        self.smearing_combo = QComboBox()
        self.smearing_combo.addItems(writer.SMEARING)
        elec_form.addRow("Smearing:", self.smearing_combo)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.001, 2.0)
        self.sigma_spin.setDecimals(3)
        self.sigma_spin.setSingleStep(0.01)
        elec_form.addRow("SIGMA:", self.sigma_spin)
        self.spin_check = QCheckBox("Spin polarised (ISPIN=2)")
        elec_form.addRow("", self.spin_check)
        self.magmom_check = QCheckBox("Write an initial MAGMOM guess")
        elec_form.addRow("", self.magmom_check)
        self.prec_combo = QComboBox()
        self.prec_combo.addItems(["Normal", "Accurate", "Single", "Low"])
        elec_form.addRow("PREC:", self.prec_combo)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Normal", "Fast", "VeryFast", "All", "Damped"])
        elec_form.addRow("ALGO:", self.algo_combo)
        self.lreal_combo = QComboBox()
        self.lreal_combo.addItems(["Auto", ".FALSE.", "On"])
        elec_form.addRow("LREAL:", self.lreal_combo)
        self.nelm_spin = QSpinBox()
        self.nelm_spin.setRange(1, 10000)
        elec_form.addRow("NELM:", self.nelm_spin)
        self.ncore_spin = QSpinBox()
        self.ncore_spin.setRange(0, 1024)
        self.ncore_spin.setSpecialValueText("off")
        elec_form.addRow("NCORE:", self.ncore_spin)
        self.lwave_check = QCheckBox("LWAVE")
        self.lcharg_check = QCheckBox("LCHARG")
        flags = QHBoxLayout()
        flags.addWidget(self.lwave_check)
        flags.addWidget(self.lcharg_check)
        flags.addStretch(1)
        elec_form.addRow("Write:", flags)
        outer.addWidget(elec_box)

        md_box = QGroupBox("Molecular dynamics")
        md_form = QFormLayout(md_box)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 10000.0)
        self.temp_spin.setSuffix(" K")
        md_form.addRow("Temperature:", self.temp_spin)
        self.potim_spin = QDoubleSpinBox()
        self.potim_spin.setRange(0.01, 100.0)
        self.potim_spin.setSingleStep(0.1)
        self.potim_spin.setSuffix(" fs")
        md_form.addRow("POTIM:", self.potim_spin)
        outer.addWidget(md_box)

        extra_box = QGroupBox("Additional INCAR tags")
        extra_layout = QVBoxLayout(extra_box)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("LDAU = .TRUE.\nLDAUTYPE = 2")
        self.extra_edit.setMaximumHeight(110)
        extra_layout.addWidget(self.extra_edit)
        outer.addWidget(extra_box)

        self.potcar_check = QCheckBox("Use the VASP recommended PAW set in the POTCAR hint")
        outer.addWidget(self.potcar_check)
        outer.addStretch(1)

        for widget in (
            self.task_combo,
            self.functional_combo,
            self.vdw_combo,
            self.smearing_combo,
            self.prec_combo,
            self.algo_combo,
            self.lreal_combo,
        ):
            widget.currentTextChanged.connect(self.update_preview)
        for widget in (self.encut_spin, self.ediffg_spin, self.sigma_spin, self.temp_spin, self.potim_spin):
            widget.valueChanged.connect(self.update_preview)
        for widget in (self.nsw_spin, self.nelm_spin, self.ncore_spin):
            widget.valueChanged.connect(self.update_preview)
        for widget in (
            self.spin_check,
            self.magmom_check,
            self.lwave_check,
            self.lcharg_check,
            self.potcar_check,
        ):
            widget.toggled.connect(self.update_preview)
        self.ediff_edit.textChanged.connect(self.update_preview)
        self.extra_edit.textChanged.connect(self.update_preview)
        return tab

    def _build_kpoints_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        box = QGroupBox("K-point sampling")
        form = QFormLayout(box)
        self.kmode_combo = QComboBox()
        self.kmode_combo.addItems(writer.KPOINT_MODES)
        form.addRow("Mode:", self.kmode_combo)

        mesh_widget = QWidget()
        mesh_grid = QGridLayout(mesh_widget)
        mesh_grid.setContentsMargins(0, 0, 0, 0)
        self.kmesh_spins = []
        self.kshift_spins = []
        for column, axis in enumerate("123"):
            spin = QSpinBox()
            spin.setRange(1, 200)
            mesh_grid.addWidget(QLabel(f"n{axis}:"), 0, column * 2)
            mesh_grid.addWidget(spin, 0, column * 2 + 1)
            self.kmesh_spins.append(spin)
            shift = QDoubleSpinBox()
            shift.setRange(-1.0, 1.0)
            shift.setSingleStep(0.5)
            shift.setDecimals(2)
            mesh_grid.addWidget(QLabel(f"s{axis}:"), 1, column * 2)
            mesh_grid.addWidget(shift, 1, column * 2 + 1)
            self.kshift_spins.append(shift)
        form.addRow("Mesh / shift:", mesh_widget)

        self.kspacing_spin = QDoubleSpinBox()
        self.kspacing_spin.setRange(0.001, 1.0)
        self.kspacing_spin.setDecimals(4)
        self.kspacing_spin.setSingleStep(0.005)
        self.kspacing_spin.setSuffix(" 1/A")
        form.addRow("Automatic spacing:", self.kspacing_spin)
        outer.addWidget(box)

        hint = QLabel(
            "Gamma-only suits a molecule in a large box. For metals use a dense "
            "Gamma-centred mesh with Methfessel-Paxton smearing; for a DOS run the "
            "tetrahedron method is forced."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)
        outer.addStretch(1)

        self.kmode_combo.currentTextChanged.connect(self.update_preview)
        for spin in self.kmesh_spins + self.kshift_spins:
            spin.valueChanged.connect(self.update_preview)
        self.kspacing_spin.valueChanged.connect(self.update_preview)
        return tab

    # -- settings ---------------------------------------------------------

    def apply_settings(self, settings) -> None:
        settings = {**writer.default_settings(), **(settings or {})}
        self._updating = True
        try:
            self.title_edit.setText(str(settings.get("title", "")))
            self.coord_combo.setCurrentText(settings.get("coord_mode", writer.COORD_MODES[0]))
            self.selective_check.setChecked(bool(settings.get("selective_dynamics")))
            self.task_combo.setCurrentText(settings.get("task", writer.TASKS[0]))
            self.functional_combo.setCurrentText(settings.get("functional", writer.FUNCTIONALS[0]))
            self.encut_spin.setValue(float(settings.get("encut", 520.0)))
            self.ediff_edit.setText(str(settings.get("ediff", 1e-6)))
            self.ediffg_spin.setValue(float(settings.get("ediffg", -0.02)))
            self.nsw_spin.setValue(int(settings.get("nsw", 100)))
            self.vdw_combo.setCurrentText(settings.get("vdw", writer.VDW_OPTIONS[0]))
            self.smearing_combo.setCurrentText(settings.get("smearing", writer.SMEARING[0]))
            self.sigma_spin.setValue(float(settings.get("sigma", 0.05)))
            self.spin_check.setChecked(bool(settings.get("ispin")))
            self.magmom_check.setChecked(bool(settings.get("magmom_auto", True)))
            self.prec_combo.setCurrentText(settings.get("prec", "Accurate"))
            self.algo_combo.setCurrentText(settings.get("algo", "Normal"))
            self.lreal_combo.setCurrentText(settings.get("lreal", "Auto"))
            self.nelm_spin.setValue(int(settings.get("nelm", 120)))
            self.ncore_spin.setValue(int(settings.get("ncore", 4)))
            self.lwave_check.setChecked(bool(settings.get("lwave")))
            self.lcharg_check.setChecked(bool(settings.get("lcharg", True)))
            self.temp_spin.setValue(float(settings.get("temperature", 300.0)))
            self.potim_spin.setValue(float(settings.get("potim", 1.0)))
            self.extra_edit.setPlainText(str(settings.get("extra_incar", "") or ""))
            self.potcar_check.setChecked(bool(settings.get("recommended_potcar", True)))
            self.kmode_combo.setCurrentText(settings.get("kpoint_mode", writer.KPOINT_MODES[1]))
            for spin, value in zip(self.kmesh_spins, settings.get("kmesh", [4, 4, 4])):
                spin.setValue(max(1, int(value)))
            for spin, value in zip(self.kshift_spins, settings.get("kshift", [0.0, 0.0, 0.0])):
                spin.setValue(float(value))
            self.kspacing_spin.setValue(float(settings.get("kspacing", 0.03)))
            self.structure_panel.apply_settings(settings)
        finally:
            self._updating = False
        self.update_preview()

    def read_settings(self) -> dict:
        try:
            ediff = float(self.ediff_edit.text())
        except (TypeError, ValueError):
            ediff = 1e-6
        settings = {
            "title": self.title_edit.text(),
            "coord_mode": self.coord_combo.currentText(),
            "selective_dynamics": self.selective_check.isChecked(),
            "frozen_indices": self._frozen_indices(),
            "task": self.task_combo.currentText(),
            "functional": self.functional_combo.currentText(),
            "encut": self.encut_spin.value(),
            "ediff": ediff,
            "ediffg": self.ediffg_spin.value(),
            "nsw": self.nsw_spin.value(),
            "vdw": self.vdw_combo.currentText(),
            "smearing": self.smearing_combo.currentText(),
            "sigma": self.sigma_spin.value(),
            "ispin": self.spin_check.isChecked(),
            "magmom_auto": self.magmom_check.isChecked(),
            "prec": self.prec_combo.currentText(),
            "algo": self.algo_combo.currentText(),
            "lreal": self.lreal_combo.currentText(),
            "nelm": self.nelm_spin.value(),
            "ncore": self.ncore_spin.value(),
            "lwave": self.lwave_check.isChecked(),
            "lcharg": self.lcharg_check.isChecked(),
            "temperature": self.temp_spin.value(),
            "potim": self.potim_spin.value(),
            "extra_incar": self.extra_edit.toPlainText(),
            "recommended_potcar": self.potcar_check.isChecked(),
            "kpoint_mode": self.kmode_combo.currentText(),
            "kmesh": [spin.value() for spin in self.kmesh_spins],
            "kshift": [spin.value() for spin in self.kshift_spins],
            "kspacing": self.kspacing_spin.value(),
        }
        settings.update(self.structure_panel.read_settings())
        return settings

    def _frozen_indices(self):
        if not self.selective_check.isChecked() or self.get_selected_indices is None:
            return []
        try:
            return list(self.get_selected_indices())
        except Exception as exc:  # pragma: no cover - host API guard
            logging.warning("VASP plugin: could not read the selection: %s", exc)
            return []

    # -- preview / output -------------------------------------------------

    def update_preview(self, *_args) -> None:
        if self._updating:
            return
        settings = self.read_settings()
        self.persistent_settings.update(settings)
        if self.mark_modified is not None:
            try:
                self.mark_modified()
            except Exception:  # pragma: no cover - host API guard
                pass

        try:
            self._cell = self.structure_panel.build_cell()
        except (ValueError, OSError) as exc:
            self._cell = None
            self.structure_panel.refresh_summary(error=str(exc))
            self.preview.setPlainText(f"! {exc}")
            self.save_button.setEnabled(False)
            return

        self.structure_panel.refresh_summary(self._cell)
        self.preview.setPlainText(writer.build_preview(self._cell, settings))
        self.save_button.setEnabled(True)

    def copy_preview(self) -> None:
        from PyQt6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.preview.toPlainText())

    def save_files(self) -> None:
        if self._cell is None:
            QMessageBox.warning(self, "VASP Input Generator", "There is no valid structure to write.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose an output directory")
        if not directory:
            return

        files = writer.build_all(self._cell, self.read_settings())
        existing = [name for name in _FILE_ORDER if os.path.exists(os.path.join(directory, name))]
        if existing:
            answer = QMessageBox.question(
                self,
                "Overwrite?",
                "These files already exist and will be overwritten:\n  " + "\n  ".join(existing),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            for name in _FILE_ORDER:
                with open(os.path.join(directory, name), "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(files[name])
        except OSError as exc:
            QMessageBox.critical(self, "VASP Input Generator", f"Could not write the files:\n{exc}")
            return

        QMessageBox.information(
            self, "VASP Input Generator", f"Wrote {', '.join(_FILE_ORDER)} to\n{directory}"
        )
