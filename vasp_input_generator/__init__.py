"""VASP Input Generator plugin for MoleditPy."""

import logging
import os

from PyQt6.QtWidgets import QMessageBox

PLUGIN_NAME = "VASP Input Generator"
PLUGIN_VERSION = "0.3.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Generate VASP POSCAR/INCAR/KPOINTS inputs from the current molecule "
    "(in a vacuum box) or from a CIF crystal structure, with supercells and "
    "recommended PAW potential hints."
)
PLUGIN_CATEGORY = "Export"
PLUGIN_TAGS = ["DFT", "Generator"]
PLUGIN_DEPENDENCIES = ["numpy"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
WINDOW_ID = "vasp_input_generator_dialog"

_context = None
_dialog_opened = False
current_settings = None


def get_default_settings():
    from .writer import default_settings

    return default_settings()


current_settings = get_default_settings()


def run(mw):
    global _dialog_opened

    if _context is not None:
        mw = _context.get_main_window()

    from .main_dialog import VaspInputDialog

    if _context is not None:
        existing = _context.get_window(WINDOW_ID)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

    def _get_molecule():
        try:
            if _context is not None:
                return _context.current_molecule
        except Exception as exc:  # pragma: no cover - host API guard
            logging.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
        return getattr(mw, "current_mol", None)

    def _mark_modified():
        if _context is not None:
            try:
                _context.mark_project_modified()
            except Exception:  # pragma: no cover - host API guard
                pass

    def _selected_indices():
        try:
            if _context is not None:
                return list(_context.get_selected_atom_indices())
        except Exception:  # pragma: no cover - host API guard
            pass
        return []

    def _get_cif_viewer():
        from .structure_panel import find_cif_viewer_widget

        return find_cif_viewer_widget(mw)

    _dialog_opened = True
    dlg = VaspInputDialog(
        parent=mw,
        persistent_settings=current_settings,
        get_molecule=_get_molecule,
        get_selected_indices=_selected_indices,
        mark_modified=_mark_modified,
        get_cif_viewer=_get_cif_viewer,
    )
    if _context is not None:
        _context.register_window(WINDOW_ID, dlg)
    dlg.show()


def initialize(context):
    global _context
    _context = context

    def show_dialog():
        run(context.get_main_window())

    context.add_export_action("VASP Input (POSCAR/INCAR)...", show_dialog)

    def save_state():
        if not _dialog_opened:
            return {}
        return {"settings": dict(current_settings)}

    def load_state(data):
        if not isinstance(data, dict):
            return
        saved = data.get("settings")
        if isinstance(saved, dict):
            current_settings.update(saved)
            dlg = context.get_window(WINDOW_ID)
            if dlg is not None:
                try:
                    dlg.apply_settings(current_settings)
                except Exception as exc:  # pragma: no cover - host API guard
                    logging.warning("%s: could not apply loaded state: %s", PLUGIN_NAME, exc)

    def handle_reset():
        global _dialog_opened
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None and dlg.isVisible():
            # Leave an open dialog alone: the user may still be editing.
            return
        current_settings.clear()
        current_settings.update(get_default_settings())
        _dialog_opened = False

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)


def _warn(parent, message):  # pragma: no cover - trivial UI helper
    QMessageBox.warning(parent, PLUGIN_NAME, message)
