"""Periodic cell model: CIF parsing, lattice math, supercells and molecule boxing.

The CIF reader, lattice construction and symmetry de-duplication are derived from
the MoleditPy CIF Viewer plugin (``cif_viewer/parser.py``).  Two deliberate
differences: symmetry operations are parsed from the CIF's own
``_symmetry_equiv_pos_as_xyz`` loop so pymatgen stays optional, and the
crystallographic metadata the viewer carries for display is dropped.

SHARED FILE.  A byte-identical copy lives in every periodic input generator
plugin (VASP / Quantum ESPRESSO / CP2K).  Bump ``SHARED_MODULE_VERSION`` on any
change and copy the file to the other plugins; each plugin's test suite pins the
version it expects, so a stale copy fails loudly.
"""

from __future__ import annotations

SHARED_MODULE_NAME = "periodic-cell-model"
SHARED_MODULE_VERSION = "0.5.0"

from dataclasses import dataclass
import math
import re
import shlex
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Two symmetry images closer than this (Angstrom) are the same site.
_DUPLICATE_TOL_SQ = 0.0025

_UNCERTAINTY_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?$"
)

_FRACT_KEYS = (
    "_atom_site.fract_x",
    "_atom_site.fract_y",
    "_atom_site.fract_z",
)
_CART_KEYS = (
    "_atom_site.cartn_x",
    "_atom_site.cartn_y",
    "_atom_site.cartn_z",
)
_SYMOP_KEYS = (
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop_operation_xyz",
    "_space_group_symop.operation_xyz",
    "_symmetry_equiv.pos_as_xyz",
)


@dataclass(frozen=True)
class CellAtom:
    label: str
    element: str
    fract: np.ndarray
    cart: np.ndarray
    occupancy: Optional[float] = None


@dataclass(frozen=True)
class Cell:
    name: str
    lengths: Tuple[float, float, float]
    angles: Tuple[float, float, float]
    lattice: np.ndarray
    atoms: Tuple[CellAtom, ...]
    space_group: Optional[str] = None
    source: str = "cif"

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def elements(self) -> Tuple[str, ...]:
        return tuple(atom.element for atom in self.atoms)


# --------------------------------------------------------------------------
# lattice math
# --------------------------------------------------------------------------


def cell_vectors(lengths: Sequence[float], angles_deg: Sequence[float]) -> np.ndarray:
    """Standard crystallographic setting: a along x, b in the xy plane."""
    a_len, b_len, c_len = (float(value) for value in lengths)
    alpha, beta, gamma = [math.radians(float(angle)) for angle in angles_deg]

    if min(a_len, b_len, c_len) <= 0.0:
        raise ValueError("Invalid cell: lengths must be positive.")

    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-8:
        raise ValueError("Invalid cell: gamma angle makes the cell singular.")

    a_vec = np.array([a_len, 0.0, 0.0], dtype=float)
    b_vec = np.array([b_len * math.cos(gamma), b_len * sin_gamma, 0.0])

    c_x = c_len * math.cos(beta)
    c_y = c_len * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    c_z_sq = c_len * c_len - c_x * c_x - c_y * c_y
    if c_z_sq < -1e-6:
        raise ValueError("Invalid cell: angles and lengths are inconsistent.")
    c_vec = np.array([c_x, c_y, math.sqrt(max(c_z_sq, 0.0))])
    return np.vstack([a_vec, b_vec, c_vec])


def lattice_parameters(
    lattice: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Inverse of :func:`cell_vectors` — (a, b, c), (alpha, beta, gamma in deg)."""
    rows = np.asarray(lattice, dtype=float)
    lengths = tuple(float(np.linalg.norm(row)) for row in rows)
    if min(lengths) <= 0.0:
        raise ValueError("Invalid lattice: a cell vector has zero length.")

    def _angle(i: int, j: int) -> float:
        cosine = float(np.dot(rows[i], rows[j]) / (lengths[i] * lengths[j]))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    return lengths, (_angle(1, 2), _angle(0, 2), _angle(0, 1))


def fractional_to_cartesian(fract: Sequence[float], lattice: np.ndarray) -> np.ndarray:
    return np.asarray(fract, dtype=float) @ np.asarray(lattice, dtype=float)


def cartesian_to_fractional(cart: Sequence[float], lattice: np.ndarray) -> np.ndarray:
    return np.asarray(cart, dtype=float) @ np.linalg.inv(np.asarray(lattice, dtype=float))


def wrap_fractional(fract: Sequence[float]) -> np.ndarray:
    """Fold into [0, 1); values within 1e-9 of 1.0 wrap to 0.0, not 0.999…"""
    values = np.asarray(fract, dtype=float) % 1.0
    values[np.abs(values - 1.0) < 1e-9] = 0.0
    return values


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------


def _split_terms(text: str) -> List[Tuple[float, str]]:
    terms: List[Tuple[float, str]] = []
    sign = 1.0
    current = ""
    for char in text.replace(" ", ""):
        if char in "+-":
            if current:
                terms.append((sign, current))
                current = ""
            sign = 1.0 if char == "+" else -1.0
            continue
        current += char
    if current:
        terms.append((sign, current))
    return terms


def parse_symmetry_operation(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an ``x, -y, z+1/2`` style symmetry string into (rotation, translation)."""
    parts = [part for part in str(text).strip().strip("'\"").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Symmetry operation needs three components: {text!r}")

    rotation = np.zeros((3, 3), dtype=float)
    translation = np.zeros(3, dtype=float)

    for row, component in enumerate(parts):
        component = component.strip().lower()
        if not component:
            raise ValueError(f"Empty component in symmetry operation {text!r}")
        for sign, body in _split_terms(component):
            variable = None
            for index, name in enumerate("xyz"):
                if name in body:
                    variable = index
                    body = body.replace(name, "")
                    break
            body = body.replace("*", "").strip()
            if body in ("", "+", "-"):
                magnitude = 1.0
            elif "/" in body:
                numerator, _, denominator = body.partition("/")
                magnitude = float(numerator) / float(denominator)
            else:
                magnitude = float(body)
            value = sign * magnitude
            if variable is None:
                translation[row] += value
            else:
                rotation[row, variable] += value

    return rotation, translation


def _symmetry_operations_from_loops(loops) -> List[Tuple[np.ndarray, np.ndarray]]:
    for headers, rows in loops:
        for header in headers:
            if header in _SYMOP_KEYS:
                operations = []
                for row in rows:
                    try:
                        operations.append(parse_symmetry_operation(row[header]))
                    except (ValueError, KeyError):
                        continue
                return operations
    return []


def _spacegroup_operations(symbol: Optional[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Optional pymatgen fallback for CIFs that omit the symop loop."""
    if not symbol:
        return []
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        return [
            (np.asarray(op.rotation_matrix, dtype=float),
             np.asarray(op.translation_vector, dtype=float))
            for op in SpaceGroup(symbol).symmetry_ops
        ]
    except Exception:  # pragma: no cover - depends on optional pymatgen
        return []


def apply_symmetry(
    atoms: Sequence[CellAtom],
    lattice: np.ndarray,
    operations: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[CellAtom]:
    """Expand an asymmetric unit to the full unit cell.

    De-duplication is scoped per source atom so that a shared site holding two
    different elements keeps both (the CIF Viewer lost mixed Fe/Co sites to a
    global seen-list).
    """
    if not operations:
        operations = [(np.eye(3), np.zeros(3))]

    expanded: List[CellAtom] = []
    for atom in atoms:
        seen: List[np.ndarray] = []
        for rotation, translation in operations:
            fract = wrap_fractional(rotation @ np.asarray(atom.fract, dtype=float) + translation)

            duplicate = False
            for previous in seen:
                delta = fract - previous
                delta -= np.round(delta)
                offset = delta @ lattice
                if float(offset @ offset) < _DUPLICATE_TOL_SQ:
                    duplicate = True
                    break
            if duplicate:
                continue

            seen.append(fract.copy())
            expanded.append(
                CellAtom(
                    label=atom.label,
                    element=atom.element,
                    fract=fract,
                    cart=fractional_to_cartesian(fract, lattice),
                    occupancy=atom.occupancy,
                )
            )
    return expanded


# --------------------------------------------------------------------------
# CIF reading
# --------------------------------------------------------------------------


def normalize_element(value: str) -> str:
    match = re.match(r"([A-Za-z]{1,2})", str(value).strip())
    if not match:
        return "X"
    raw = match.group(1)
    return raw[0].upper() + raw[1:].lower()


def parse_cif_number(value: str) -> float:
    cleaned = str(value).strip().strip("'\"")
    if cleaned in {"?", "."}:
        raise ValueError("Missing numeric CIF value.")
    match = _UNCERTAINTY_RE.match(cleaned)
    if match:
        return float(match.group(1))
    return float(cleaned)


def _strip_comment(line: str) -> str:
    quote = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _split_cif_line(line: str) -> List[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _normalize_tag(tag: str) -> str:
    lowered = tag.lower()
    if lowered.startswith("_atom_site_"):
        return "_atom_site." + lowered[len("_atom_site_") :]
    return lowered


def _logical_lines(text: str) -> Iterable[str]:
    in_text_field = False
    value_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if in_text_field:
            if line.startswith(";"):
                yield "\n".join(value_lines).strip()
                in_text_field = False
                value_lines = []
            else:
                value_lines.append(line)
            continue

        stripped = _strip_comment(line).strip()
        if not stripped:
            continue
        if line.startswith(";"):
            in_text_field = True
            value_lines = [line[1:]] if line[1:].strip() else []
            continue
        yield stripped


def read_cif_tokens(text: str):
    lines = list(_logical_lines(text))
    tags: Dict[str, str] = {}
    loops = []
    data_name = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        lower = line.lower()
        if lower.startswith("data_"):
            data_name = line[5:].strip() or data_name
            index += 1
            continue
        if lower == "loop_":
            index += 1
            headers = []
            while index < len(lines) and lines[index].startswith("_"):
                headers.append(_normalize_tag(_split_cif_line(lines[index])[0]))
                index += 1

            values = []
            while index < len(lines):
                current = lines[index]
                current_lower = current.lower()
                if (
                    current_lower == "loop_"
                    or current.startswith("_")
                    or current_lower.startswith("data_")
                ):
                    break
                values.extend(_split_cif_line(current))
                index += 1

            rows = []
            if headers:
                width = len(headers)
                for start in range(0, len(values), width):
                    row_values = values[start : start + width]
                    if len(row_values) == width:
                        rows.append(dict(zip(headers, row_values)))
            loops.append((headers, rows))
            continue
        if line.startswith("_"):
            tokens = _split_cif_line(line)
            if len(tokens) >= 2:
                tags[_normalize_tag(tokens[0])] = tokens[1]
            elif len(tokens) == 1 and index + 1 < len(lines):
                tags[_normalize_tag(tokens[0])] = lines[index + 1]
                index += 1
        index += 1

    return tags, loops, data_name


def _required_float(tags: Dict[str, str], key: str) -> float:
    try:
        return parse_cif_number(tags[key])
    except KeyError as exc:
        raise ValueError(f"CIF is missing required tag {key}.") from exc


def _find_atom_loop(loops):
    for headers, rows in loops:
        if any(header.startswith("_atom_site.") for header in headers):
            return rows
    return []


def _atoms_from_loop(rows, lattice: np.ndarray) -> List[CellAtom]:
    atoms: List[CellAtom] = []
    for row in rows:
        label = row.get("_atom_site.label") or row.get("_atom_site.id") or "Atom"
        element = normalize_element(
            row.get("_atom_site.type_symbol") or row.get("_atom_site.label") or label
        )

        if all(key in row for key in _FRACT_KEYS):
            fract = np.array(
                [parse_cif_number(row[key]) for key in _FRACT_KEYS], dtype=float
            )
            cart = fractional_to_cartesian(fract, lattice)
        elif all(key in row for key in _CART_KEYS):
            cart = np.array(
                [parse_cif_number(row[key]) for key in _CART_KEYS], dtype=float
            )
            fract = cartesian_to_fractional(cart, lattice)
        else:
            continue

        occupancy = None
        if "_atom_site.occupancy" in row:
            try:
                occupancy = parse_cif_number(row["_atom_site.occupancy"])
            except ValueError:
                occupancy = None

        atoms.append(CellAtom(str(label), element, fract, cart, occupancy))
    return atoms


def _first_tag(tags: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in tags:
            cleaned = str(tags[key]).strip().strip("'\"")
            if cleaned not in {".", "?", ""}:
                return cleaned
    return None


def parse_cif(text: str, name: str = "CIF", expand: bool = True) -> Cell:
    """Read a CIF into a :class:`Cell`, expanding the asymmetric unit by default."""
    tags, loops, data_name = read_cif_tokens(text)

    lengths = (
        _required_float(tags, "_cell_length_a"),
        _required_float(tags, "_cell_length_b"),
        _required_float(tags, "_cell_length_c"),
    )
    angles = (
        _required_float(tags, "_cell_angle_alpha"),
        _required_float(tags, "_cell_angle_beta"),
        _required_float(tags, "_cell_angle_gamma"),
    )
    lattice = cell_vectors(lengths, angles)

    atoms = _atoms_from_loop(_find_atom_loop(loops), lattice)
    if not atoms:
        raise ValueError("CIF does not contain readable atom positions.")

    space_group = _first_tag(
        tags,
        [
            "_space_group_name_h-m_alt",
            "_symmetry_space_group_name_h-m",
            "_space_group.symmetry_space_group_name_h-m",
        ],
    )

    if expand:
        operations = _symmetry_operations_from_loops(loops)
        if not operations:
            operations = _spacegroup_operations(space_group)
        atoms = apply_symmetry(atoms, lattice, operations)

    return Cell(
        name=data_name or name,
        lengths=lengths,
        angles=angles,
        lattice=lattice,
        atoms=tuple(atoms),
        space_group=space_group,
        source="cif",
    )


def parse_cif_file(path: str, expand: bool = True) -> Cell:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_cif(handle.read(), name=path, expand=expand)


# --------------------------------------------------------------------------
# construction / transformation
# --------------------------------------------------------------------------


def make_supercell(cell: Cell, repeats: Sequence[int]) -> Cell:
    counts = [max(1, int(value)) for value in repeats]
    if counts == [1, 1, 1]:
        return cell

    lattice = np.asarray(cell.lattice, dtype=float) * np.array(counts, dtype=float)[:, None]
    scale = np.array(counts, dtype=float)

    atoms: List[CellAtom] = []
    for ia in range(counts[0]):
        for ib in range(counts[1]):
            for ic in range(counts[2]):
                offset = np.array([ia, ib, ic], dtype=float)
                for atom in cell.atoms:
                    fract = (np.asarray(atom.fract, dtype=float) + offset) / scale
                    atoms.append(
                        CellAtom(
                            label=atom.label,
                            element=atom.element,
                            fract=fract,
                            cart=fractional_to_cartesian(fract, lattice),
                            occupancy=atom.occupancy,
                        )
                    )

    lengths, angles = lattice_parameters(lattice)
    suffix = "x".join(str(count) for count in counts)
    return Cell(
        name=f"{cell.name}_{suffix}",
        lengths=lengths,
        angles=angles,
        lattice=lattice,
        atoms=tuple(atoms),
        space_group=None if any(count > 1 for count in counts) else cell.space_group,
        source=cell.source,
    )


def cell_from_molecule(
    elements: Sequence[str],
    coords: Sequence[Sequence[float]],
    padding: Union[float, Sequence[float]] = 6.0,
    cubic: bool = False,
    name: str = "molecule",
    labels: Optional[Sequence[str]] = None,
) -> Cell:
    """Wrap a non-periodic molecule in an orthorhombic box with vacuum padding.

    ``padding`` is a single value or one per axis — per-axis padding is how a
    slab is set up (vacuum along c only).
    """
    positions = np.asarray(coords, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("Molecule coordinates must be a non-empty (N, 3) array.")
    if len(elements) != len(positions):
        raise ValueError("Element and coordinate counts differ.")

    pads = np.full(3, float(padding), dtype=float) if np.isscalar(padding) else np.asarray(padding, dtype=float)
    if pads.shape != (3,):
        raise ValueError("padding must be a number or three numbers.")
    pads = np.maximum(pads, 0.0)
    extent = positions.max(axis=0) - positions.min(axis=0)
    lengths = extent + 2.0 * pads
    # A single atom (or a planar/linear molecule) would give a zero-width axis.
    lengths = np.maximum(lengths, 1.0)
    if cubic:
        lengths = np.full(3, float(lengths.max()))

    lattice = np.diag(lengths)
    origin = positions.min(axis=0) - (lengths - extent) / 2.0
    shifted = positions - origin

    atoms = tuple(
        CellAtom(
            label=str(labels[index]) if labels is not None else f"{elements[index]}{index + 1}",
            element=normalize_element(elements[index]),
            fract=shifted[index] / lengths,
            cart=shifted[index].copy(),
            occupancy=1.0,
        )
        for index in range(len(positions))
    )

    return Cell(
        name=name,
        lengths=tuple(float(value) for value in lengths),
        angles=(90.0, 90.0, 90.0),
        lattice=lattice,
        atoms=atoms,
        space_group="P 1",
        source="molecule",
    )


def cell_with_lattice(cell: Cell, lengths: Sequence[float], angles: Sequence[float]) -> Cell:
    """Re-cast a cell onto a user-supplied lattice, keeping fractional positions."""
    lattice = cell_vectors(lengths, angles)
    atoms = tuple(
        CellAtom(
            label=atom.label,
            element=atom.element,
            fract=np.asarray(atom.fract, dtype=float),
            cart=fractional_to_cartesian(atom.fract, lattice),
            occupancy=atom.occupancy,
        )
        for atom in cell.atoms
    )
    return Cell(
        name=cell.name,
        lengths=tuple(float(value) for value in lengths),
        angles=tuple(float(value) for value in angles),
        lattice=lattice,
        atoms=atoms,
        space_group=cell.space_group,
        source=cell.source,
    )


def group_by_species(cell: Cell) -> List[Tuple[str, List[int]]]:
    """Atom indices grouped by element, in first-appearance order (POSCAR needs this)."""
    order: List[str] = []
    groups: Dict[str, List[int]] = {}
    for index, atom in enumerate(cell.atoms):
        if atom.element not in groups:
            groups[atom.element] = []
            order.append(atom.element)
        groups[atom.element].append(index)
    return [(element, groups[element]) for element in order]


def sorted_by_species(cell: Cell) -> Tuple[List[CellAtom], List[Tuple[str, int]]]:
    """Atoms reordered so each element is contiguous, plus (element, count) pairs."""
    atoms: List[CellAtom] = []
    counts: List[Tuple[str, int]] = []
    for element, indices in group_by_species(cell):
        atoms.extend(cell.atoms[index] for index in indices)
        counts.append((element, len(indices)))
    return atoms, counts


def formula(cell: Cell) -> str:
    return " ".join(
        f"{element}{count}" if count > 1 else element
        for element, count in sorted_by_species(cell)[1]
    )


def kpoint_mesh_from_density(
    cell: Cell, density: float, minimum: int = 1
) -> Tuple[int, int, int]:
    """Mesh where n_i ≈ density / |a_i*|⁻¹, i.e. density measured in Å⁻¹ spacing.

    ``density`` is the target maximum spacing between k-points in reciprocal
    space (2π/Å is not applied — the convention here is n_i = ceil(1 / (|a_i| * d))).
    """
    density = max(1e-6, float(density))
    mesh = []
    for length in cell.lengths:
        mesh.append(max(int(minimum), int(math.ceil(1.0 / (float(length) * density)))))
    return tuple(mesh)  # type: ignore[return-value]


def molecule_arrays(mol):
    """Extract (labels, elements, coords) from an RDKit-like molecule.

    Duck-typed on purpose: the XYZ Editor's ``custom_symbol`` property wins over
    the element symbol, and no rdkit import is needed here.
    """
    if mol is None:
        raise ValueError("No molecule is loaded.")
    conformer = mol.GetConformer()
    labels: List[str] = []
    elements: List[str] = []
    coords: List[List[float]] = []
    for index in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(index)
        symbol = (
            atom.GetProp("custom_symbol")
            if atom.HasProp("custom_symbol")
            else atom.GetSymbol()
        )
        position = conformer.GetAtomPosition(index)
        labels.append(f"{symbol}{index + 1}")
        elements.append(normalize_element(symbol))
        coords.append([float(position.x), float(position.y), float(position.z)])
    if not coords:
        raise ValueError("The molecule has no atoms.")
    return labels, elements, coords


def molecule_to_cell(
    mol,
    padding: Union[float, Sequence[float]] = 6.0,
    cubic: bool = False,
    name: str = "molecule",
) -> Cell:
    labels, elements, coords = molecule_arrays(mol)
    return cell_from_molecule(
        elements, coords, padding=padding, cubic=cubic, name=name, labels=labels
    )


def molecule_charge_and_multiplicity(mol) -> Tuple[int, int]:
    """Net formal charge and spin multiplicity of an RDKit-like molecule.

    Multiplicity is 2S+1 from the radical electron count, so a closed shell gives
    1 and a doublet radical 2.  Duck-typed like :func:`molecule_arrays`.
    """
    if mol is None:
        raise ValueError("No molecule is loaded.")
    charge = 0
    radicals = 0
    for index in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(index)
        try:
            charge += int(atom.GetFormalCharge())
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            radicals += int(atom.GetNumRadicalElectrons())
        except (AttributeError, TypeError, ValueError):
            pass
    return charge, radicals + 1


def cell_from_viewer_structure(structure, expand_asymmetric: bool = True) -> Cell:
    """Convert a CIF Viewer ``CifStructure`` into a :class:`Cell`.

    Duck-typed so the CIF Viewer plugin never has to be importable.  A panel
    holding only the asymmetric unit needs the space group to be expanded, which
    is pymatgen's job here — the viewer does not keep the raw symop strings.
    """
    if structure is None:
        raise ValueError("The CIF Viewer panel has no structure loaded.")

    lattice = np.asarray(getattr(structure, "lattice", None), dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError("The CIF Viewer structure has no usable lattice.")

    lengths = tuple(float(v) for v in getattr(structure, "cell_lengths", ()) or ())
    angles = tuple(float(v) for v in getattr(structure, "cell_angles", ()) or ())
    if len(lengths) != 3 or len(angles) != 3:
        lengths, angles = lattice_parameters(lattice)

    source_atoms = getattr(structure, "atoms", ()) or ()
    atoms = [
        CellAtom(
            label=str(getattr(atom, "label", "") or "Atom"),
            element=normalize_element(getattr(atom, "element", "X")),
            fract=wrap_fractional(np.asarray(atom.fract, dtype=float)),
            cart=fractional_to_cartesian(
                wrap_fractional(np.asarray(atom.fract, dtype=float)), lattice
            ),
            occupancy=getattr(atom, "occupancy", None),
        )
        for atom in source_atoms
    ]
    if not atoms:
        raise ValueError("The CIF Viewer structure contains no atoms.")

    space_group = getattr(structure, "space_group", None)
    if expand_asymmetric and getattr(structure, "is_asymmetric_unit_only", False):
        operations = _spacegroup_operations(space_group)
        if not operations:
            raise ValueError(
                "The CIF Viewer panel holds only the asymmetric unit and the space "
                "group could not be expanded (pymatgen is required).\n"
                "Load the .cif file directly instead."
            )
        atoms = apply_symmetry(atoms, lattice, operations)

    return Cell(
        name=str(getattr(structure, "name", "cif_viewer") or "cif_viewer"),
        lengths=lengths,
        angles=angles,
        lattice=lattice,
        atoms=tuple(atoms),
        space_group=space_group,
        source="cif_viewer",
    )


def write_cif(cell: Cell, name: Optional[str] = None) -> str:
    """Serialise a cell as P1 CIF text (no external writer needed)."""
    name = (name or cell.name or "cell").strip().replace(" ", "_") or "cell"
    a, b, c = cell.lengths
    alpha, beta, gamma = cell.angles
    lines = [
        f"data_{name}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        f"_cell_length_a    {a:.8f}",
        f"_cell_length_b    {b:.8f}",
        f"_cell_length_c    {c:.8f}",
        f"_cell_angle_alpha {alpha:.6f}",
        f"_cell_angle_beta  {beta:.6f}",
        f"_cell_angle_gamma {gamma:.6f}",
        f"_cell_volume      {cell.volume:.6f}",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "  'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    counters: Dict[str, int] = {}
    for atom in cell.atoms:
        counters[atom.element] = counters.get(atom.element, 0) + 1
        label = f"{atom.element}{counters[atom.element]}"
        fract = wrap_fractional(atom.fract)
        occupancy = 1.0 if atom.occupancy is None else float(atom.occupancy)
        lines.append(
            f"  {label:<6} {atom.element:<3} "
            f"{fract[0]:.8f} {fract[1]:.8f} {fract[2]:.8f} {occupancy:.4f}"
        )
    return "\n".join(lines) + "\n"




def vacuum_gap(cell: Cell) -> float:
    """Empty space along c, in Angstrom, between an atom and its periodic image.

    Measured along the surface normal, so it is meaningful for a tilted c axis.
    """
    lattice = np.asarray(cell.lattice, dtype=float)
    normal = np.cross(lattice[0], lattice[1])
    norm = np.linalg.norm(normal)
    if norm < 1e-12 or not cell.atoms:
        return 0.0
    normal = normal / norm

    spacing = abs(float(np.dot(lattice[2], normal)))
    heights = [float(np.dot(np.asarray(atom.cart, dtype=float), normal)) for atom in cell.atoms]
    return max(0.0, spacing - (max(heights) - min(heights)))


def looks_like_slab(cell: Cell, minimum_vacuum: float = 5.0) -> bool:
    """True for a cell with a real vacuum layer along c.

    A slab arriving as a plain CIF carries no marker, so it is recognised by its
    geometry rather than by ``cell.source``.
    """
    if cell.source == "slab":
        return True
    if cell.source == "molecule":
        return False
    return vacuum_gap(cell) >= float(minimum_vacuum)
