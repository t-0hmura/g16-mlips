#!/usr/bin/env python3
"""Shared Gaussian plugin runner."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import traceback

try:
    from importlib import metadata as _importlib_metadata
except Exception:
    try:
        import importlib_metadata as _importlib_metadata
    except Exception:
        _importlib_metadata = None

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from g16_extio import read_g16_external_input, write_g16_external_output, write_msg
    from mlip_backends import (
        ev_to_ha,
        forces_ev_ang_to_gradient_ha_bohr,
        hessian_ev_ang2_to_ha_bohr2,
    )
else:
    from .g16_extio import read_g16_external_input, write_g16_external_output, write_msg
    from .mlip_backends import (
        ev_to_ha,
        forces_ev_ang_to_gradient_ha_bohr,
        hessian_ev_ang2_to_ha_bohr2,
    )


class RunnerError(RuntimeError):
    pass


def _backend_alias(plugin_name):
    low = os.path.basename(str(plugin_name or "")).lower()
    if "orb" in low:
        return "orb"
    if "mace" in low:
        return "mace"
    return "uma"


def _find_orca_extinp_arg(argv):
    for arg in argv:
        text = str(arg).strip().lower()
        if text.endswith(".extinp.tmp"):
            return arg
    return None


def _version_text(plugin_name):
    version = "dev"
    if _importlib_metadata is not None:
        try:
            version = _importlib_metadata.version("mlips4g16")
        except Exception:
            pass
    return "{} (mlips4g16 {})".format(plugin_name, version)


def _split_gaussian_tail(argv):
    """Split argv into custom args and Gaussian-generated tail args.

    Gaussian passes 6 generated arguments to the script:
        LAYER InputFile OutputFile MsgFile FChkFile MatElFile
    where LAYER is one of R/M/S.
    """
    if len(argv) < 6:
        return None, None

    tail = argv[-6:]
    custom = argv[:-6]
    layer, input_file, output_file, msg_file, fchk_file, matel_file = tail
    return custom, {
        "layer": layer,
        "input_file": input_file,
        "output_file": output_file,
        "msg_file": msg_file,
        "fchk_file": fchk_file,
        "matel_file": matel_file,
    }


def run_g16_plugin(
    argv,
    plugin_name,
    make_evaluator,
    available_models,
    default_model,
    add_extra_args,
):
    parser = argparse.ArgumentParser(
        prog=plugin_name,
        description=(
            "Gaussian External plugin for {}.\n"
            "Gaussian call signature: {} [custom-options] LAYER InputFile OutputFile MsgFile FChkFile MatElFile"
        ).format(plugin_name.replace("_", " "), plugin_name)
    )
    parser.add_argument("--model", default=default_model, help="Model name/alias/path")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--hessian-mode", choices=["Analytical", "Numerical"], default="Analytical")
    parser.add_argument("--hessian-step", type=float, default=1.0e-3, help="Finite-difference step in Angstrom")
    parser.add_argument(
        "--strict-hessian",
        action="store_true",
        help="Fail instead of falling back to numerical Hessian when analytical Hessian is unavailable.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print model aliases and exit")
    parser.add_argument(
        "--version",
        action="version",
        version=_version_text(plugin_name),
    )

    if add_extra_args is not None:
        add_extra_args(parser)

    # Help mode should work without Gaussian-generated tail arguments.
    if ("-h" in argv) or ("--help" in argv):
        parser.parse_args(["--help"])
        return 0

    # Manual utility mode (no Gaussian tail needed)
    if len(argv) <= 3 and "--list-models" in argv:
        args = parser.parse_args(argv)
        if args.list_models:
            for item in available_models():
                print(item)
        return 0

    custom_args, gtail = _split_gaussian_tail(argv)
    if custom_args is None or gtail is None:
        extinp_like = _find_orca_extinp_arg(argv)
        if extinp_like is not None:
            backend = _backend_alias(plugin_name)
            parser.error(
                "Detected ORCA-style input '{}'. "
                "This command appears to be a Gaussian plugin, but it was called in ORCA style. "
                "If short aliases are conflicting, set ORCA ProgExt to '{}'.".format(
                    extinp_like, "mlips4orca-" + backend
                )
            )
        parser.error(
            "Gaussian-generated 6 tail args are missing. "
            "Use --list-models for standalone listing."
        )

    args = parser.parse_args(custom_args)

    if args.list_models:
        for item in available_models():
            print(item)
        return 0

    input_file = gtail["input_file"]
    output_file = gtail["output_file"]
    msg_file = gtail["msg_file"]

    ext = read_g16_external_input(input_file)

    evaluator = make_evaluator(args)

    need_grad = int(ext["igrd"]) >= 1
    need_hess = int(ext["igrd"]) >= 2

    energy_ev, forces_ev_ang, hess_ev_ang2 = evaluator.evaluate(
        symbols=ext["symbols"],
        coords_ang=ext["coords_ang"],
        charge=ext["charge"],
        multiplicity=ext["multiplicity"],
        need_forces=need_grad,
        need_hessian=need_hess,
        hessian_mode=args.hessian_mode,
        hessian_step=float(args.hessian_step),
        strict_hessian=bool(args.strict_hessian),
    )

    grad_ha_bohr = None
    if need_grad:
        if forces_ev_ang is None:
            raise RunnerError("Backend returned no forces although Gaussian requested gradient/Hessian.")
        grad_ha_bohr = forces_ev_ang_to_gradient_ha_bohr(forces_ev_ang)

    hess_ha_bohr2 = None
    if need_hess:
        if hess_ev_ang2 is None:
            raise RunnerError("Backend returned no Hessian although Gaussian requested IGrd=2.")
        hess_ha_bohr2 = hessian_ev_ang2_to_ha_bohr2(hess_ev_ang2)

    write_g16_external_output(
        output_path=output_file,
        natoms=ext["natoms"],
        energy_ha=ev_to_ha(energy_ev),
        gradient_ha_bohr=grad_ha_bohr,
        hessian_ha_bohr2=hess_ha_bohr2,
    )

    msg = []
    msg.append("[{}] Completed".format(plugin_name))
    msg.append("layer={}".format(gtail["layer"]))
    msg.append("input={}".format(os.path.abspath(input_file)))
    msg.append("output={}".format(os.path.abspath(output_file)))
    msg.append("model={}".format(args.model))
    msg.append("igrd={}".format(ext["igrd"]))
    msg.append("hessian_mode={}".format(args.hessian_mode))
    write_msg(msg_file, "\n".join(msg) + "\n")

    return 0


def main_entry(entry_fn):
    try:
        code = int(entry_fn(sys.argv[1:]))
        raise SystemExit(code)
    except Exception as exc:
        msg_file = None
        if len(sys.argv) >= 3:
            # Best effort: Gaussian msg file is 3rd from the end in full argv.
            try:
                msg_file = sys.argv[-3]
            except Exception:
                msg_file = None

        tb = traceback.format_exc()
        text = "[ERROR] {}\n{}".format(exc, tb)
        if msg_file:
            try:
                write_msg(msg_file, text)
            except Exception:
                pass

        print(text, file=sys.stderr)
        raise SystemExit(1)
