"""Show a :class:`cell_model.Cell` in MoleditPy's own 3D viewer, with a cell box.

Follows the CIF Viewer plugin's approach (``cif_viewer/viewer.py``): the atoms
go in by assigning ``context.current_molecule`` so the host keeps ownership and
style switches still redraw, and the box is drawn straight onto the host's
PyVista plotter as named line actors that can be removed again.

RDKit and PyVista are imported only inside the functions.  Both are supplied by
the host, so importing this module needs nothing beyond numpy and the plugin
keeps ``PLUGIN_DEPENDENCIES = ["numpy"]``.

SHARED FILE.  A byte-identical copy lives in every periodic plugin (VASP /
Quantum ESPRESSO / CP2K input generators and the Slab Builder).  Bump
``SHARED_MODULE_VERSION`` on any change and copy the file to the other plugins;
each plugin's test suite pins the version it expects, so a stale copy fails
loudly.
"""

from __future__ import annotations

SHARED_MODULE_NAME = "periodic-cell-preview"
SHARED_MODULE_VERSION = "0.6.0"

import logging
from typing import List, Optional, Sequence

import numpy as np

from .elements import bond_cutoff

#: Actor names are namespaced so clearing never touches another plugin's overlay.
ACTOR_PREFIX = "periodic_cell_preview"

AXIS_COLORS = ("#ff4444", "#44dd44", "#4488ff")
EDGE_COLOR = "#cccccc"

# The 12 edges of a parallelepiped: three axes from the origin, then the rest.
_EDGE_INDICES = (
    (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
)


def cell_corners(lattice) -> np.ndarray:
    """The 8 corners of the cell, origin first."""
    rows = np.asarray(lattice, dtype=float)
    origin = np.zeros(3)
    return np.array(
        [
            origin,
            origin + rows[0],
            origin + rows[1],
            origin + rows[2],
            origin + rows[0] + rows[1],
            origin + rows[0] + rows[2],
            origin + rows[1] + rows[2],
            origin + rows[0] + rows[1] + rows[2],
        ]
    )


def cell_segments(lattice):
    """(start, end, colour, label) for every edge; the three axes are labelled."""
    corners = cell_corners(lattice)
    segments = [
        (corners[0], corners[1], AXIS_COLORS[0], "a"),
        (corners[0], corners[2], AXIS_COLORS[1], "b"),
        (corners[0], corners[3], AXIS_COLORS[2], "c"),
    ]
    segments.extend(
        (corners[start], corners[end], EDGE_COLOR, "") for start, end in _EDGE_INDICES
    )
    return segments


def infer_bonds(cell, tolerance: float = 0.45, limit: int = 2000):
    """(i, j) pairs closer than the sum of their covalent radii + ``tolerance``.

    Distance-based on purpose.  RDKit's own perception works from valences and
    routinely fails on a periodic cell, whose contents are cut at the faces and
    so full of atoms with a fraction of their neighbours.  Bonds are not
    followed across the boundary either, for the same reason: the partner is in
    the next image, not in this list.  Matches the rule the CIF Viewer uses.
    """
    atoms = cell.atoms
    count = len(atoms)
    if count < 2 or count > int(limit):
        return []

    positions = np.array([atom.cart for atom in atoms], dtype=float)
    elements = [atom.element for atom in atoms]
    radii = np.array([bond_cutoff(element, element, 0.0) / 2.0 for element in elements])

    rows, columns = np.triu_indices(count, k=1)
    delta = positions[rows] - positions[columns]
    squared = np.einsum("ij,ij->i", delta, delta)
    cutoff = radii[rows] + radii[columns] + float(tolerance)
    # The 0.25 A floor drops overlapping images that would otherwise "bond".
    hits = np.nonzero((squared <= cutoff * cutoff) & (squared >= 0.0625))[0]
    return [(int(rows[index]), int(columns[index])) for index in hits]


def enters_3d_by_default(cell) -> bool:
    """False for a molecule in a vacuum box, which the user is probably editing."""
    return getattr(cell, "source", "") != "molecule"


def bonds_by_default(cell) -> bool:
    """True for a molecule boxed in vacuum, False for a periodic structure.

    An isolated molecule is complete, so drawing its bonds is simply right; a
    crystal or slab is cut at its faces, where bonds lose their partner.
    """
    return getattr(cell, "source", "") == "molecule"


def build_molecule(cell, bonds: Optional[Sequence] = None, template=None):
    """An RDKit molecule holding the cell's atoms at their Cartesian positions.

    Bonds are optional: a periodic cell is cut at its faces, so every bond
    crossing a face is missing its partner, and the host draws bond-less atoms
    as spheres — a fair picture of a crystal.

    ``template`` is the molecule the cell was built from, when there is one.
    Boxing a molecule only moves it, so the original is reused with its
    coordinates replaced; rebuilding it from scratch would hand the viewer
    single bonds and lose the double bonds, aromaticity and charges the user
    drew.  It is ignored unless the atom count still matches.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    atoms = list(cell.atoms)

    if template is not None:
        try:
            if template.GetNumAtoms() == len(atoms):
                moved = Chem.Mol(template)
                conformer = moved.GetConformer()
                for index, atom in enumerate(atoms):
                    x, y, z = (float(value) for value in atom.cart)
                    conformer.SetAtomPosition(index, Point3D(x, y, z))
                return moved
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # TypeError covers Boost.Python.ArgumentError: a host may hand over
            # something molecule-shaped that RDKit will not copy.
            logging.debug("Could not reuse the source molecule: %s", exc)
    editable = Chem.RWMol()
    for atom in atoms:
        try:
            editable.AddAtom(Chem.Atom(str(atom.element)))
        except RuntimeError:
            # A placeholder element the CIF invented; show it as a dummy atom.
            editable.AddAtom(Chem.Atom(0))

    for left, right in bonds or ():
        editable.AddBond(int(left), int(right), Chem.BondType.SINGLE)

    molecule = editable.GetMol()

    # The host's renderer calls GetRingInfo() and GetTotalNumHs(), and both
    # raise a Pre-condition Violation on a molecule that was never sanitised —
    # after the plotter has already been cleared, so the 3D view ends up empty.
    # Full sanitisation is wrong here (a crystal is full of valences RDKit would
    # reject), so initialise just the two caches the renderer needs.
    for atom in molecule.GetAtoms():
        # A crystal site carries no implicit hydrogens; without this a bare
        # oxygen would be drawn and labelled as if it were water.
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    molecule.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(molecule)

    conformer = Chem.Conformer(len(atoms))
    for index, atom in enumerate(atoms):
        x, y, z = (float(value) for value in atom.cart)
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def _plotter(context, main_window=None):
    """The host's PyVista plotter, or None when the 3D view is unavailable."""
    if context is not None:
        plotter = getattr(context, "plotter", None)
        if plotter is not None:
            return plotter
        if main_window is None and hasattr(context, "get_main_window"):
            try:
                main_window = context.get_main_window()
            except Exception as exc:  # pragma: no cover - host API guard
                logging.debug("Could not reach the main window: %s", exc)
    if main_window is None:
        return None
    if hasattr(main_window, "plotter"):
        return main_window.plotter
    manager = getattr(main_window, "view_3d_manager", None)
    return getattr(manager, "plotter", None)


def clear_cell_box(context, actor_names: Sequence[str], main_window=None) -> List[str]:
    """Remove previously drawn box actors; returns the new (empty) name list."""
    if not actor_names:
        return []
    plotter = _plotter(context, main_window)
    if plotter is not None:
        for name in actor_names:
            try:
                plotter.remove_actor(name)
            except Exception as exc:  # pragma: no cover - host API guard
                logging.debug("Could not remove the actor %s: %s", name, exc)
        try:
            plotter.render()
        except Exception as exc:  # pragma: no cover - host API guard
            logging.debug("Could not render after clearing: %s", exc)
    return []


def draw_cell_box(
    context,
    cell,
    actor_names: Sequence[str] = (),
    main_window=None,
    width: int = 3,
    show_labels: bool = True,
) -> List[str]:
    """Draw the cell box; returns the actor names to hand back when clearing."""
    names = clear_cell_box(context, actor_names, main_window)
    plotter = _plotter(context, main_window)
    if plotter is None or cell is None:
        return names

    for index, (start, end, color, label) in enumerate(cell_segments(cell.lattice)):
        name = f"{ACTOR_PREFIX}_line_{index}"
        try:
            plotter.add_lines(
                np.array([start, end]),
                color=color,
                width=width if label else max(1, width - 2),
                name=name,
            )
        except Exception as exc:  # pragma: no cover - host API guard
            logging.debug("Could not draw the cell edge %s: %s", name, exc)
            continue
        names.append(name)

        if label and show_labels:
            label_name = f"{ACTOR_PREFIX}_label_{label}"
            try:
                plotter.add_point_labels(
                    [end],
                    [label],
                    point_size=0,
                    font_size=14,
                    text_color=color,
                    bold=True,
                    always_visible=True,
                    shape=None,
                    shape_opacity=0.0,
                    name=label_name,
                )
            except Exception as exc:  # pragma: no cover - host API guard
                logging.debug("Could not draw the axis label %s: %s", label_name, exc)
                continue
            names.append(label_name)

    try:
        plotter.render()
    except Exception as exc:  # pragma: no cover - host API guard
        logging.debug("Could not render the cell box: %s", exc)
    return names


def show_cell(
    context,
    cell,
    actor_names: Sequence[str] = (),
    main_window=None,
    enter_3d: Optional[bool] = None,
    show_bonds: Optional[bool] = None,
    template=None,
) -> List[str]:
    """Put the cell in the host's 3D view: atoms first, then the box.

    ``enter_3d`` brings the 3D view to the front, which is what makes the
    preview visible when the app is sitting in the 2D editor.  Left automatic it
    does that for a crystal but not for a molecule boxed in vacuum: that
    molecule is the one being edited, and yanking the user out of the 2D editor
    mid-edit is not a fair trade for a preview they did not ask for.

    ``show_bonds`` defaults to :func:`bonds_by_default`: a molecule in a vacuum
    box is whole, so its bonds are all real, while a crystal is cut at its faces
    and every bond crossing one would be missing its partner.

    Raises ValueError if the cell cannot be shown, so the caller can report it.
    """
    if cell is None or not cell.atoms:
        raise ValueError("There is no structure to preview.")
    if _plotter(context, main_window) is None:
        # Without this the atoms are handed over, no box is drawn, and nothing
        # is said — which looks exactly like the feature being broken.
        raise ValueError(
            "MoleditPy's 3D view is not available, so there is nowhere to draw the "
            "cell. Switch to the 3D viewer and try again."
        )

    wanted = bonds_by_default(cell) if show_bonds is None else bool(show_bonds)
    molecule = build_molecule(
        cell, infer_bonds(cell) if wanted else None, template=template
    )
    if context is not None and hasattr(context, "current_molecule"):
        # Assigning current_molecule (rather than calling draw_molecule_3d)
        # keeps the host's own record in step, so later style changes redraw
        # this structure instead of the previous one.
        context.current_molecule = molecule
    else:
        window = main_window
        if window is None and context is not None and hasattr(context, "get_main_window"):
            window = context.get_main_window()
        if window is None or not hasattr(window, "draw_molecule_3d"):
            raise ValueError("MoleditPy's 3D view is not available.")
        window.draw_molecule_3d(molecule)

    names = draw_cell_box(context, cell, actor_names, main_window)

    # Drawing into the 3D view is pointless while the app is showing the 2D
    # editor, so bring that view to the front.
    enter = enters_3d_by_default(cell) if enter_3d is None else bool(enter_3d)
    if enter and context is not None and hasattr(context, "enter_3d_viewer_mode"):
        try:
            context.enter_3d_viewer_mode()
        except Exception as exc:  # pragma: no cover - host API guard
            logging.debug("Could not switch to the 3D viewer: %s", exc)

    # The host restores the previous camera after a redraw, so a cell much
    # larger than whatever was on screen before would sit outside the view.
    if context is not None and hasattr(context, "reset_3d_camera"):
        try:
            context.reset_3d_camera()
        except Exception as exc:  # pragma: no cover - host API guard
            logging.debug("Could not reset the camera: %s", exc)
    return names
