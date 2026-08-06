"""Fakes shared by this plugin's own tests.

The shared modules themselves are tested once, in
``moleditpy-periodic-shared``; what stays here is only what the plugin's own
tests need to stand a molecule or a drag event up.
"""

from __future__ import annotations

CUBIC_CIF = """
data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'I m -3 m'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'x+1/2, y+1/2, z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0.0 0.0 0.0
"""


# -- a molecule the host might hand over ------------------------------------


class _FakePosition:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeConformer:
    def __init__(self, coords):
        self._coords = coords

    def GetAtomPosition(self, index):
        return _FakePosition(*self._coords[index])


class _FakeAtom:
    def __init__(self, symbol, custom=None):
        self._symbol = symbol
        self._custom = custom

    def GetSymbol(self):
        return self._symbol

    def HasProp(self, name):
        return name == "custom_symbol" and self._custom is not None

    def GetProp(self, name):
        return self._custom


class _FakeMol:
    def __init__(self, symbols, coords, customs=None):
        customs = customs or [None] * len(symbols)
        self._atoms = [_FakeAtom(s, c) for s, c in zip(symbols, customs)]
        self._conf = _FakeConformer(coords)

    def GetNumAtoms(self):
        return len(self._atoms)

    def GetAtomWithIdx(self, index):
        return self._atoms[index]

    def GetConformer(self):
        return self._conf


# -- a drag carrying files --------------------------------------------------


class _FakeUrl:
    def __init__(self, path):
        self._path = path

    def toLocalFile(self):
        return self._path


class _FakeMime:
    def __init__(self, paths):
        self._urls = [_FakeUrl(p) for p in paths]

    def hasUrls(self):
        return bool(self._urls)

    def urls(self):
        return self._urls


class _FakeDropEvent:
    def __init__(self, mime):
        self._mime = mime
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


# -- a host that records what the preview asked of it -----------------------


class _RecordingPlotter:
    def __init__(self):
        self.lines = []

    def add_lines(self, points, color=None, width=None, name=None):
        self.lines.append(name)

    def add_point_labels(self, *args, **kwargs):
        pass

    def remove_actor(self, name):
        pass

    def render(self):
        pass


class _RecordingContext:
    def __init__(self):
        self.plotter = _RecordingPlotter()
        self.current_molecule = None
        self.shown = 0
        self.entered = 0

    def get_main_window(self):
        return None

    def enter_3d_viewer_mode(self):
        self.entered += 1

    def reset_3d_camera(self):
        self.shown += 1
