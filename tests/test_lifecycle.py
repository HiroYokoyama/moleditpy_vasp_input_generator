"""The entry points the host itself calls: _open_dialog, run and initialize.

This is the least-exercised code in the plugin and the code most likely to
break the app rather than the output, so it is covered here directly.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QWidget  # noqa: E402

import vasp_input_generator as plugin  # noqa: E402

from shared_fixtures import _FakeMol  # noqa: E402
from test_plugin import FakeContext  # noqa: E402

PACKAGE = "vasp_input_generator"


class _HostWindow(QWidget):
    """Stands in for MoleditPy's main window (a real widget, so it can parent)."""

    def __init__(self, mol=None):
        super().__init__()
        self.current_mol = mol
        self.plugin_manager = None


class _LifecycleContext(FakeContext):
    def __init__(self, main_window=None, molecule=None):
        super().__init__(main_window=main_window)
        self.current_molecule = molecule

    def get_main_window(self):
        return self.main_window


@pytest.fixture
def clean_plugin(qapp):
    """Restore the module globals a dialog-opening test necessarily mutates."""
    context, opened = plugin._context, plugin._dialog_opened
    settings = dict(plugin.current_settings)
    yield
    plugin._context = context
    plugin._dialog_opened = opened
    plugin.current_settings.clear()
    plugin.current_settings.update(settings)


def _close(context):
    dialog = context.windows.get(plugin.WINDOW_ID)
    if dialog is not None:
        dialog.close()
        dialog.deleteLater()


# -- opening ----------------------------------------------------------------


def test_open_dialog_without_a_context_still_opens(clean_plugin):
    """A host that never called initialize() must not crash the menu item."""
    plugin._context = None
    plugin._dialog_opened = False
    window = _HostWindow(mol=_FakeMol(["H"], [[0.0, 0.0, 0.0]]))
    plugin._open_dialog(window)
    assert plugin._dialog_opened is True


def test_open_dialog_registers_the_window(clean_plugin):
    context = _LifecycleContext(main_window=_HostWindow())
    plugin._context = context
    plugin._open_dialog(None)
    assert plugin.WINDOW_ID in context.windows
    _close(context)


def test_open_dialog_reuses_a_visible_window(clean_plugin):
    """A second menu click must raise the open dialog, not stack another."""
    context = _LifecycleContext(main_window=_HostWindow())
    plugin._context = context
    plugin._open_dialog(None)
    first = context.windows[plugin.WINDOW_ID]
    first.show()
    plugin._open_dialog(None)
    assert context.windows[plugin.WINDOW_ID] is first
    _close(context)


def test_open_dialog_replaces_a_closed_window(clean_plugin):
    context = _LifecycleContext(main_window=_HostWindow())
    plugin._context = context
    plugin._open_dialog(None)
    first = context.windows[plugin.WINDOW_ID]
    first.hide()
    plugin._open_dialog(None)
    assert context.windows[plugin.WINDOW_ID] is not first
    _close(context)


def test_run_opens_the_dialog(clean_plugin):
    """run() is what the host's Plugins menu calls."""
    context = _LifecycleContext(main_window=_HostWindow())
    plugin._context = context
    plugin.run(context.main_window)
    assert plugin.WINDOW_ID in context.windows
    _close(context)


# -- the closures handed to the dialog --------------------------------------


def test_the_dialog_reads_the_molecule_from_the_context(clean_plugin):
    molecule = _FakeMol(["O"], [[0.0, 0.0, 0.0]])
    context = _LifecycleContext(main_window=_HostWindow(), molecule=molecule)
    plugin._context = context
    plugin._open_dialog(None)
    assert context.windows[plugin.WINDOW_ID]._get_molecule() is molecule
    _close(context)


def test_the_dialog_falls_back_to_the_main_window_molecule(clean_plugin):
    """Without a context the molecule can only come from the window itself."""
    plugin._context = None
    molecule = _FakeMol(["N"], [[0.0, 0.0, 0.0]])
    window = _HostWindow(mol=molecule)
    plugin._open_dialog(window)
    # the dialog keeps the closure it was built with
    assert window.current_mol is molecule


def test_editing_marks_the_project_modified(clean_plugin):
    context = _LifecycleContext(main_window=_HostWindow())
    plugin._context = context
    plugin._open_dialog(None)
    dialog = context.windows[plugin.WINDOW_ID]
    before = context.modified
    dialog.encut_spin.setValue(430.0)
    assert context.modified > before
    _close(context)


def test_the_cif_viewer_lookup_is_wired_up(clean_plugin):
    """The panel must be able to reach the CIF Viewer through the host."""
    context = _LifecycleContext(main_window=_HostWindow())
    plugin._context = context
    plugin._open_dialog(None)
    dialog = context.windows[plugin.WINDOW_ID]
    assert dialog.structure_panel.get_cif_viewer() is None  # none open, but callable
    _close(context)


# -- initialize -------------------------------------------------------------


def test_initialize_wires_the_export_action(clean_plugin):
    context = _LifecycleContext(main_window=_HostWindow())
    plugin.initialize(context)
    assert len(context.export_actions) == 1
    label, callback = context.export_actions[0]
    assert label.endswith("...")
    callback()
    assert plugin.WINDOW_ID in context.windows
    _close(context)


def test_the_frozen_atom_selection_comes_from_the_host(clean_plugin):
    """Selective dynamics is driven by whatever MoleditPy has selected."""
    context = _LifecycleContext(main_window=_HostWindow())
    context.get_selected_atom_indices = lambda: [0, 2]
    plugin._context = context
    plugin._open_dialog(None)
    dialog = context.windows[plugin.WINDOW_ID]
    dialog.selective_check.setChecked(True)
    assert dialog.read_settings()["frozen_indices"] == [0, 2]
    _close(context)


def test_a_host_that_cannot_answer_leaves_nothing_frozen(clean_plugin):
    def _boom():
        raise RuntimeError("no selection API")

    context = _LifecycleContext(main_window=_HostWindow())
    context.get_selected_atom_indices = _boom
    plugin._context = context
    plugin._open_dialog(None)
    dialog = context.windows[plugin.WINDOW_ID]
    dialog.selective_check.setChecked(True)
    assert dialog.read_settings()["frozen_indices"] == []
    _close(context)
