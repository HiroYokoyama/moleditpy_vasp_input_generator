import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vasp_input_generator as plugin  # noqa: E402


class FakeContext:
    def __init__(self, main_window=None):
        self.main_window = main_window
        self.export_actions = []
        self.save_handlers = []
        self.load_handlers = []
        self.reset_handlers = []
        self.windows = {}
        self.current_molecule = None
        self.modified = 0

    def add_export_action(self, label, callback):
        self.export_actions.append((label, callback))

    def register_save_handler(self, callback):
        self.save_handlers.append(callback)

    def register_load_handler(self, callback):
        self.load_handlers.append(callback)

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def register_window(self, window_id, window):
        self.windows[window_id] = window

    def get_window(self, window_id):
        return self.windows.get(window_id)

    def get_main_window(self):
        return self.main_window

    def get_selected_atom_indices(self):
        return []

    def mark_project_modified(self):
        self.modified += 1


@pytest.fixture
def context():
    original = dict(plugin.current_settings)
    ctx = FakeContext()
    plugin.initialize(ctx)
    yield ctx
    plugin._context = None
    plugin._dialog_opened = False
    plugin.current_settings.clear()
    plugin.current_settings.update(original)


# -- metadata --------------------------------------------------------------


def test_plugin_metadata():
    assert plugin.PLUGIN_NAME == "VASP Input Generator"
    assert plugin.PLUGIN_VERSION == "0.4.2"
    assert plugin.PLUGIN_AUTHOR == "HiroYokoyama"
    assert plugin.PLUGIN_CATEGORY == "Export"
    assert plugin.PLUGIN_DEPENDENCIES == ["numpy", "pyvista", "rdkit"]
    assert plugin.PLUGIN_TAGS == ["DFT", "Generator"]
    assert plugin.PLUGIN_DESCRIPTION.strip()


def test_plugin_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)


def test_supported_version_range():
    assert plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION == ">=4.0.0, <5.0.0"


def test_default_settings_shape():
    settings = plugin.get_default_settings()
    for key in ("task", "encut", "kpoint_mode", "coord_mode", "supercell"):
        assert key in settings


def test_module_exposes_run_for_the_plugins_menu():
    """run() is what puts the plugin in the host's Plugins menu (manual 7.1).

    It does not duplicate the entry initialize() registers, because that one
    lands in the Export menu instead.
    """
    assert callable(plugin.run)


def test_module_has_no_autorun_attribute():
    """autorun() executes at startup; this plugin has nothing to do there."""
    assert not hasattr(plugin, "autorun")


# -- registration ----------------------------------------------------------


def test_initialize_registers_export_action(context):
    labels = [label for label, _ in context.export_actions]
    assert labels == ["VASP Input (POSCAR/INCAR)..."]


def test_initialize_registers_persistence(context):
    assert len(context.save_handlers) == 1
    assert len(context.load_handlers) == 1
    assert len(context.reset_handlers) == 1


def test_save_handler_is_silent_until_the_dialog_opens(context):
    assert context.save_handlers[0]() == {}


def test_save_handler_emits_settings_after_use(context):
    plugin._dialog_opened = True
    plugin.current_settings["encut"] = 700.0
    state = context.save_handlers[0]()
    assert state["settings"]["encut"] == 700.0


def test_load_handler_updates_settings(context):
    context.load_handlers[0]({"settings": {"encut": 333.0}})
    assert plugin.current_settings["encut"] == 333.0


def test_load_handler_ignores_junk(context):
    before = dict(plugin.current_settings)
    context.load_handlers[0](None)
    context.load_handlers[0]({})
    context.load_handlers[0]({"settings": "nope"})
    assert plugin.current_settings == before


def test_reset_handler_restores_defaults(context):
    plugin._dialog_opened = True
    plugin.current_settings["encut"] = 999.0
    context.reset_handlers[0]()
    assert plugin.current_settings["encut"] == plugin.get_default_settings()["encut"]
    assert plugin._dialog_opened is False


def test_reset_handler_leaves_an_open_dialog_alone(context):
    class _Dialog:
        def isVisible(self):
            return True

    context.windows[plugin.WINDOW_ID] = _Dialog()
    plugin.current_settings["encut"] = 999.0
    context.reset_handlers[0]()
    assert plugin.current_settings["encut"] == 999.0


def test_load_handler_pushes_into_an_open_dialog(context):
    applied = {}

    class _Dialog:
        def apply_settings(self, settings):
            applied.update(settings)

    context.windows[plugin.WINDOW_ID] = _Dialog()
    context.load_handlers[0]({"settings": {"encut": 456.0}})
    assert applied["encut"] == 456.0


def test_load_handler_survives_a_broken_dialog(context):
    class _Dialog:
        def apply_settings(self, settings):
            raise RuntimeError("wrapped C/C++ object deleted")

    context.windows[plugin.WINDOW_ID] = _Dialog()
    context.load_handlers[0]({"settings": {"encut": 456.0}})
    assert plugin.current_settings["encut"] == 456.0
