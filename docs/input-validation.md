# Input keyword validation — 2026-10-12

Version 0.8.0 fixes DOS smearing selection, tetrahedron mesh validation,
numeric precision, title delimiters and conflicting additional INCAR tags.
Imports in the plugin entry point, dialog and writer are at module scope.

## Official references

- [ISMEAR](https://vasp.at/wiki/ISMEAR): Gaussian 0, MP 1, Fermi -1,
  tetrahedron -5; tetrahedron methods use a Gamma-centred mesh.
- [Smearing technique](https://vasp.at/wiki/Smearing_technique): tetrahedron
  interpolation requires at least four k-points. DOS keeps the user's selected
  smearing rather than forcing tetrahedron for an isolated molecule.
- [INCAR format](https://vasp.at/wiki/INCAR): semicolon-separated assignments,
  comments, backslash continuations, quoted multiline values and nested tags.
  Additional outer assignments replace generated tags without duplicates;
  repeated additional outer tags and unclosed quotes/braces are rejected.
- [GGA](https://vasp.at/wiki/GGA): PE, PS, RP and CA are valid existing mappings.

The tests exercise text generation, additional-tag parsing, invalid input
rejection, real Qt preview recovery and saving. The four-point check is a
necessary minimum, not a proof that a mesh is converged or suitable for a
particular structure. No VASP executable or licensed POTCAR data is available
in this environment, so these checks do not validate a calculation in VASP.
