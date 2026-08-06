"""Element data shared by the periodic input generators.

Masses are IUPAC 2021 standard atomic weights (conventional values for elements
with no stable isotope, i.e. the longest-lived isotope's mass number).

SHARED FILE.  A byte-identical copy lives in every periodic plugin (VASP /
Quantum ESPRESSO / CP2K input generators and the Slab Builder).  Bump
``SHARED_MODULE_VERSION`` on any change and copy the file to the other plugins;
each plugin's test suite pins the version it expects, so a stale copy fails
loudly.
"""

from __future__ import annotations

SHARED_MODULE_NAME = "periodic-elements"
SHARED_MODULE_VERSION = "0.1.0"

from typing import Dict, List, Optional

ATOMIC_NUMBER: Dict[str, int] = {}

_MASSES: List[float] = [
    1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,
    22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.95, 39.098, 40.078,
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
    69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224,
    92.906, 95.95, 97.0, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,
    121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,
    145.0, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05,
    174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,
    204.38, 207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04,
    231.04, 238.03, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,
    258.0, 259.0, 262.0,
]

SYMBOLS: List[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
]

ATOMIC_MASS: Dict[str, float] = dict(zip(SYMBOLS, _MASSES))
for _index, _symbol in enumerate(SYMBOLS, start=1):
    ATOMIC_NUMBER[_symbol] = _index


def atomic_number(element: str) -> Optional[int]:
    return ATOMIC_NUMBER.get(str(element).strip().capitalize())


def atomic_mass(element: str, default: float = 1.0) -> float:
    return ATOMIC_MASS.get(str(element).strip().capitalize(), default)


def is_known(element: str) -> bool:
    return str(element).strip().capitalize() in ATOMIC_NUMBER
