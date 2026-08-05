"""Recommended VASP PAW POTCAR names.

Source: VASP wiki, "Available pseudopotentials" — the recommended set for the
potpaw.64 PBE distribution (verified 2026-08-05).  Elements absent from the
table fall back to the plain element symbol.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

RECOMMENDED_PAW: Dict[str, str] = {
    "H": "H", "He": "He", "Li": "Li_sv", "Be": "Be", "B": "B", "C": "C",
    "N": "N", "O": "O", "F": "F", "Ne": "Ne", "Na": "Na_pv", "Mg": "Mg",
    "Al": "Al", "Si": "Si", "P": "P", "S": "S", "Cl": "Cl", "Ar": "Ar",
    "K": "K_sv", "Ca": "Ca_sv", "Sc": "Sc_sv", "Ti": "Ti_sv", "V": "V_sv",
    "Cr": "Cr_pv", "Mn": "Mn_pv", "Fe": "Fe", "Co": "Co", "Ni": "Ni",
    "Cu": "Cu", "Zn": "Zn", "Ga": "Ga_d", "Ge": "Ge_d", "As": "As",
    "Se": "Se", "Br": "Br", "Kr": "Kr", "Rb": "Rb_sv", "Sr": "Sr_sv",
    "Y": "Y_sv", "Zr": "Zr_sv", "Nb": "Nb_sv", "Mo": "Mo_sv", "Tc": "Tc_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd", "Ag": "Ag", "Cd": "Cd",
    "In": "In_d", "Sn": "Sn_d", "Sb": "Sb", "Te": "Te", "I": "I", "Xe": "Xe",
    "Cs": "Cs_sv", "Ba": "Ba_sv", "La": "La", "Ce": "Ce", "Hf": "Hf_pv",
    "Ta": "Ta_pv", "W": "W_sv", "Re": "Re", "Os": "Os", "Ir": "Ir",
    "Pt": "Pt", "Au": "Au", "Hg": "Hg", "Tl": "Tl_d", "Pb": "Pb_d",
    "Bi": "Bi_d", "Po": "Po_d", "At": "At", "Rn": "Rn", "Fr": "Fr_sv",
    "Ra": "Ra_sv", "Ac": "Ac", "Th": "Th", "Pa": "Pa", "U": "U",
    "Np": "Np", "Pu": "Pu", "Am": "Am", "Cm": "Cm",
}


def potcar_name(element: str, recommended: bool = True) -> str:
    element = str(element).strip().capitalize()
    if not recommended:
        return element
    return RECOMMENDED_PAW.get(element, element)


def potcar_names(elements: Sequence[str], recommended: bool = True) -> List[str]:
    return [potcar_name(element, recommended) for element in elements]


def unmapped_elements(elements: Sequence[str]) -> List[str]:
    """Elements with no entry in the recommended table (caller should warn)."""
    return [
        element
        for element in elements
        if str(element).strip().capitalize() not in RECOMMENDED_PAW
    ]


def concat_command(names: Sequence[str], potcar_dir: str = "$VASP_PP_PATH/potpaw_PBE") -> str:
    parts = " ".join(f'"{potcar_dir}/{name}/POTCAR"' for name in names)
    return f"cat {parts} > POTCAR"


def magmom_defaults(counts: Sequence[Tuple[str, int]]) -> str:
    """A conservative MAGMOM guess: 5 µB for 3d/4d/5d metals, 0.6 otherwise."""
    magnetic = set(
        "Sc Ti V Cr Mn Fe Co Ni Cu Y Zr Nb Mo Tc Ru Rh Pd Ag "
        "Hf Ta W Re Os Ir Pt Au".split()
    )
    chunks = []
    for element, count in counts:
        moment = 5.0 if element in magnetic else 0.6
        chunks.append(f"{count}*{moment}")
    return " ".join(chunks)
