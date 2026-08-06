"""The 3D preview talks to the host, so the host is faked and its calls recorded."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vasp_input_generator import cell_model as cm  # noqa: E402
from vasp_input_generator import cell_preview as cp  # noqa: E402


def test_shared_module_identity():
    assert cp.SHARED_MODULE_NAME == "periodic-cell-preview"
    assert cp.SHARED_MODULE_VERSION == "0.2.0"


class FakePlotter:
    def __init__(self, fail_on=()):
        self.lines = []
        self.labels = []
        self.removed = []
        self.renders = 0
        self._fail_on = set(fail_on)

    def add_lines(self, points, color=None, width=None, name=None):
        if "add_lines" in self._fail_on:
            raise RuntimeError("no render window")
        self.lines.append((np.asarray(points), color, width, name))

    def add_point_labels(self, points, labels, **kwargs):
        self.labels.append((list(points), list(labels), kwargs.get("name")))

    def remove_actor(self, name):
        self.removed.append(name)

    def render(self):
        self.renders += 1


class FakeContext:
    def __init__(self, plotter=None):
        self.plotter = plotter
        self.current_molecule = None

    def get_main_window(self):
        return None


@pytest.fixture
def cell():
    return cm.cell_from_molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        padding=4.0,
    )


# -- geometry ---------------------------------------------------------------


def test_cell_corners_span_the_parallelepiped():
    lattice = cm.cell_vectors((2.0, 3.0, 4.0), (90.0, 90.0, 90.0))
    corners = cp.cell_corners(lattice)
    assert len(corners) == 8
    assert np.allclose(corners[0], [0, 0, 0])
    assert np.allclose(corners[7], [2.0, 3.0, 4.0])


def test_cell_segments_describe_all_twelve_edges():
    lattice = cm.cell_vectors((2.0, 3.0, 4.0), (90.0, 90.0, 90.0))
    segments = cp.cell_segments(lattice)
    assert len(segments) == 12
    assert [label for *_, label in segments][:3] == ["a", "b", "c"]
    # every edge must be a lattice vector long
    lengths = sorted(round(float(np.linalg.norm(end - start)), 6) for start, end, _, _ in segments)
    assert lengths == sorted([2.0] * 4 + [3.0] * 4 + [4.0] * 4)


def test_cell_segments_follow_a_triclinic_lattice():
    lattice = cm.cell_vectors((4.0, 5.0, 6.0), (80.0, 100.0, 110.0))
    start, end, _, label = cp.cell_segments(lattice)[1]
    assert label == "b"
    assert np.allclose(end - start, lattice[1])


# -- drawing ----------------------------------------------------------------


def test_draw_cell_box_adds_edges_and_axis_labels(cell):
    plotter = FakePlotter()
    names = cp.draw_cell_box(FakeContext(plotter), cell)
    assert len(plotter.lines) == 12
    assert len(plotter.labels) == 3
    assert len(names) == 15
    assert all(name.startswith(cp.ACTOR_PREFIX) for name in names)
    assert plotter.renders == 1


def test_drawing_again_removes_the_previous_box(cell):
    plotter = FakePlotter()
    context = FakeContext(plotter)
    names = cp.draw_cell_box(context, cell)
    cp.draw_cell_box(context, cell, names)
    assert plotter.removed == names


def test_clear_cell_box_removes_every_actor(cell):
    plotter = FakePlotter()
    context = FakeContext(plotter)
    names = cp.draw_cell_box(context, cell)
    assert cp.clear_cell_box(context, names) == []
    assert plotter.removed == names


def test_drawing_survives_a_plotter_that_refuses(cell):
    """A closed render window must warn, not crash the dialog."""
    plotter = FakePlotter(fail_on=["add_lines"])
    assert cp.draw_cell_box(FakeContext(plotter), cell) == []


def test_nothing_is_drawn_without_a_plotter(cell):
    assert cp.draw_cell_box(FakeContext(None), cell) == []


def test_clearing_without_a_plotter_is_safe():
    assert cp.clear_cell_box(FakeContext(None), ["a", "b"]) == []


# -- molecule ---------------------------------------------------------------


def test_build_molecule_places_every_atom(cell):
    pytest.importorskip("rdkit")
    molecule = cp.build_molecule(cell)
    assert molecule.GetNumAtoms() == len(cell.atoms)
    conformer = molecule.GetConformer()
    for index, atom in enumerate(cell.atoms):
        position = conformer.GetAtomPosition(index)
        assert (position.x, position.y, position.z) == pytest.approx(tuple(atom.cart))


def test_build_molecule_keeps_the_elements(cell):
    pytest.importorskip("rdkit")
    molecule = cp.build_molecule(cell)
    assert [a.GetSymbol() for a in molecule.GetAtoms()] == [a.element for a in cell.atoms]


def test_build_molecule_survives_a_placeholder_element():
    pytest.importorskip("rdkit")
    lattice = cm.cell_vectors((5.0, 5.0, 5.0), (90.0, 90.0, 90.0))
    atoms = (cm.CellAtom("Q1", "Q", np.zeros(3), np.zeros(3), 1.0),)
    cell = cm.Cell("x", (5.0,) * 3, (90.0,) * 3, lattice, atoms)
    assert cp.build_molecule(cell).GetNumAtoms() == 1


def test_show_cell_hands_the_molecule_to_the_host(cell):
    pytest.importorskip("rdkit")
    plotter = FakePlotter()
    context = FakeContext(plotter)
    cp.show_cell(context, cell)
    assert context.current_molecule is not None
    assert context.current_molecule.GetNumAtoms() == len(cell.atoms)
    assert len(plotter.lines) == 12


def test_show_cell_rejects_an_empty_cell():
    lattice = cm.cell_vectors((5.0, 5.0, 5.0), (90.0, 90.0, 90.0))
    empty = cm.Cell("x", (5.0,) * 3, (90.0,) * 3, lattice, ())
    with pytest.raises(ValueError, match="no structure"):
        cp.show_cell(FakeContext(FakePlotter()), empty)


def test_show_cell_reports_a_missing_viewer(cell):
    """Silently drawing nothing is indistinguishable from a broken feature."""
    pytest.importorskip("rdkit")

    class Bare:
        pass

    with pytest.raises(ValueError, match="3D view is not available"):
        cp.show_cell(None, cell, main_window=Bare())
    with pytest.raises(ValueError, match="3D view is not available"):
        cp.show_cell(FakeContext(None), cell)


# -- the molecule must survive the host's renderer ---------------------------


def test_the_molecule_answers_the_calls_the_host_renderer_makes(cell):
    """An unsanitised molecule blanked the 3D view.

    view_3d_logic clears the plotter before it draws, then calls GetRingInfo()
    (for aromatic circles) and GetTotalNumHs() (for labels).  Both raise a
    Pre-condition Violation on a molecule that was never sanitised, so the draw
    died half-way and left an empty viewer.
    """
    pytest.importorskip("rdkit")
    molecule = cp.build_molecule(cell)
    assert molecule.GetRingInfo().NumRings() == 0
    for atom in molecule.GetAtoms():
        assert atom.GetTotalNumHs() == 0


def test_a_crystal_site_gains_no_implicit_hydrogens():
    """A bare oxygen must not be drawn or labelled as if it were water."""
    pytest.importorskip("rdkit")
    lattice = cm.cell_vectors((5.0, 5.0, 5.0), (90.0, 90.0, 90.0))
    atoms = (cm.CellAtom("O1", "O", np.zeros(3), np.zeros(3), 1.0),)
    molecule = cp.build_molecule(cm.Cell("x", (5.0,) * 3, (90.0,) * 3, lattice, atoms))
    assert molecule.GetAtomWithIdx(0).GetTotalNumHs() == 0
    assert molecule.GetAtomWithIdx(0).GetNoImplicit()


def test_show_cell_refits_the_camera(cell):
    """The host restores the old camera, so a big cell would land off-screen."""
    pytest.importorskip("rdkit")

    class CountingContext(FakeContext):
        def __init__(self, plotter):
            super().__init__(plotter)
            self.resets = 0

        def reset_3d_camera(self):
            self.resets += 1

    context = CountingContext(FakePlotter())
    cp.show_cell(context, cell)
    assert context.resets == 1


def test_show_cell_works_without_the_camera_helper(cell):
    """Older hosts have no reset_3d_camera; the preview must still work."""
    pytest.importorskip("rdkit")
    context = FakeContext(FakePlotter())
    assert len(cp.show_cell(context, cell)) == 15


# -- switching the app to the 3D view ---------------------------------------


class _SwitchingContext(FakeContext):
    def __init__(self, plotter):
        super().__init__(plotter)
        self.entered = 0

    def enter_3d_viewer_mode(self):
        self.entered += 1


def test_show_cell_brings_the_3d_view_to_the_front(cell):
    """Drawing into a hidden 3D view looks like nothing happened."""
    pytest.importorskip("rdkit")
    context = _SwitchingContext(FakePlotter())
    cp.show_cell(context, cell)
    assert context.entered == 1


def test_switching_to_3d_can_be_declined(cell):
    pytest.importorskip("rdkit")
    context = _SwitchingContext(FakePlotter())
    cp.show_cell(context, cell, enter_3d=False)
    assert context.entered == 0


def test_show_cell_works_on_a_host_without_the_switch(cell):
    """Older hosts have no enter_3d_viewer_mode; the preview must still work."""
    pytest.importorskip("rdkit")
    assert len(cp.show_cell(FakeContext(FakePlotter()), cell)) == 15
