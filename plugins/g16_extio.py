#!/usr/bin/env python3
"""Gaussian External input/output helpers.

Reference: Gaussian documentation in `$g16root/g16/doc/extern.txt` and `$g16root/g16/doc/extgau`.
"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np

BOHR_PER_ANG = 1.8897261254578281


class G16IOError(RuntimeError):
    pass


def _parse_int_fields_fixed_width(line):
    # Gaussian docs specify (4I10)
    return [
        int(line[0:10]),
        int(line[10:20]),
        int(line[20:30]),
        int(line[30:40]),
    ]


def _parse_header(line):
    parts = line.split()
    if len(parts) >= 4:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    return tuple(_parse_int_fields_fixed_width(line))


def _atomic_number_to_symbol(z):
    try:
        from ase.data import chemical_symbols
    except Exception as exc:
        raise G16IOError("ASE is required to map atomic numbers to symbols.") from exc

    z = int(z)
    if z <= 0 or z >= len(chemical_symbols):
        raise G16IOError("Unsupported atomic number in Gaussian external input: {}".format(z))
    return chemical_symbols[z]


def read_g16_external_input(input_path):
    """Read Gaussian external input file.

    Format from Gaussian external interface docs:
    NAtoms, IGrd, ICharg, Multip  (4I10)
    then NAtoms lines:
    IAn, X, Y, Z, MMCharge
    followed by connectivity block (ignored here).

    NOTE:
    In Gaussian16 C.02 External calls, X/Y/Z in the generated input are
    observed in Bohr units. Backends in this project consume Angstrom, so
    we convert Bohr -> Angstrom here.
    """
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise G16IOError("Gaussian external input file not found: {}".format(input_path))

    with open(input_path, "r") as handle:
        lines = [ln.rstrip("\n") for ln in handle if ln.strip()]

    if not lines:
        raise G16IOError("Gaussian external input is empty: {}".format(input_path))

    nat, igrd, charge, multiplicity = _parse_header(lines[0])
    if nat <= 0:
        raise G16IOError("NAtoms must be positive, got {}".format(nat))

    atom_lines = lines[1 : 1 + nat]
    if len(atom_lines) != nat:
        raise G16IOError("Gaussian external input ended before all atom lines were read.")

    symbols = []
    coords = []
    mm_charges = []
    for row in atom_lines:
        parts = row.split()
        if len(parts) < 4:
            raise G16IOError("Malformed atom line in Gaussian external input: '{}'".format(row))
        ian = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        qmm = float(parts[4]) if len(parts) >= 5 else 0.0

        symbols.append(_atomic_number_to_symbol(ian))
        coords.append([x, y, z])
        mm_charges.append(qmm)

    coords_bohr = np.asarray(coords, dtype=np.float64)
    coords_ang = coords_bohr / BOHR_PER_ANG

    return {
        "input_path": input_path,
        "natoms": nat,
        "igrd": int(igrd),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "symbols": symbols,
        "coords_bohr": coords_bohr,
        "coords_ang": coords_ang,
        "mm_charges": np.asarray(mm_charges, dtype=np.float64),
    }


def _fmt_d(value):
    # Gaussian examples use D exponent with width 20, precision 12.
    return "{:20.12E}".format(float(value)).replace("E", "D")


def _write_series(handle, series, per_line):
    vals = list(series)
    for idx in range(0, len(vals), int(per_line)):
        chunk = vals[idx : idx + int(per_line)]
        handle.write("".join(_fmt_d(v) for v in chunk) + "\n")


def _pack_lower_triangle(mat):
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise G16IOError("Hessian must be a square 2D array for Gaussian output.")
    n = m.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1):
            out.append(float(m[i, j]))
    return out


def write_g16_external_output(output_path, natoms, energy_ha, gradient_ha_bohr, hessian_ha_bohr2):
    """Write Gaussian external output file.

    Output sequence (unless iofchk route is used):
    - Energy + Dipole(3)         (4D20.12)
    - Gradient (if IGrd>=1)      (3D20.12), length 3N
    - Polar(6) (if IGrd==2)      (3D20.12)
    - DDip(9N) (if IGrd==2)      (3D20.12)
    - FFX lower-triangle Hessian (if IGrd==2) (3D20.12), length 3N*(3N+1)/2
    """
    out_path = os.path.abspath(output_path)

    with open(out_path, "w") as handle:
        # Energy and dipole; dipole not provided by these plugins -> zeros.
        handle.write(
            "".join(
                [_fmt_d(energy_ha), _fmt_d(0.0), _fmt_d(0.0), _fmt_d(0.0)]
            )
            + "\n"
        )

        if gradient_ha_bohr is not None:
            grad = np.asarray(gradient_ha_bohr, dtype=np.float64).reshape(int(natoms) * 3)
            _write_series(handle, grad, per_line=3)

        if hessian_ha_bohr2 is not None:
            # Polarizability and dipole derivatives are not available here -> zeros.
            _write_series(handle, [0.0] * 6, per_line=3)
            _write_series(handle, [0.0] * (int(natoms) * 9), per_line=3)

            h2 = np.asarray(hessian_ha_bohr2, dtype=np.float64).reshape(int(natoms) * 3, int(natoms) * 3)
            h_lower = _pack_lower_triangle(h2)
            _write_series(handle, h_lower, per_line=3)


def write_msg(msg_path, text):
    msg_path = os.path.abspath(msg_path)
    with open(msg_path, "w") as handle:
        handle.write(str(text))
        if not str(text).endswith("\n"):
            handle.write("\n")
