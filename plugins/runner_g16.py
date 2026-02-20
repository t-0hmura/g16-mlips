#!/usr/bin/env python3
"""Shared Gaussian plugin runner."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import traceback

import numpy as np

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
    from mlip_server import (
        MLIPServer,
        ServerError,
        auto_server_socket,
        client_evaluate,
        ensure_server,
        send_shutdown,
        server_is_alive,
    )
    from xtb_alpb_correction import (
        XTBError,
        delta_alpb_minus_vac,
        resolve_xtb_ncores,
        solvent_correction_enabled,
    )
else:
    from .g16_extio import read_g16_external_input, write_g16_external_output, write_msg
    from .mlip_backends import (
        ev_to_ha,
        forces_ev_ang_to_gradient_ha_bohr,
        hessian_ev_ang2_to_ha_bohr2,
    )
    from .mlip_server import (
        MLIPServer,
        ServerError,
        auto_server_socket,
        client_evaluate,
        ensure_server,
        send_shutdown,
        server_is_alive,
    )
    from .xtb_alpb_correction import (
        XTBError,
        delta_alpb_minus_vac,
        resolve_xtb_ncores,
        solvent_correction_enabled,
    )


class RunnerError(RuntimeError):
    pass


def _backend_alias(plugin_name):
    low = os.path.basename(str(plugin_name or "")).lower()
    if "orb" in low:
        return "orb"
    if "mace" in low:
        return "mace"
    if "aimnet2" in low:
        return "aimnet2"
    return "uma"


def _find_orca_extinp_arg(argv):
    for arg in argv:
        text = str(arg).strip().lower()
        if text.endswith(".extinp.tmp"):
            return arg
    return None


def _version_text(plugin_name):
    version = "dev"
    dist_name = "g16-mlips"
    if _importlib_metadata is not None:
        try:
            version = _importlib_metadata.version(dist_name)
        except Exception:
            pass
    return "{} ({} {})".format(plugin_name, dist_name, version)


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


def _add_server_args(parser):
    """Add server-related arguments to the parser."""
    parser.add_argument(
        "--server-socket", default=None,
        help="Path to Unix domain socket for persistent model server.",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start as a persistent model server (internal use).",
    )
    parser.add_argument(
        "--stop-server", action="store_true",
        help="Send shutdown signal to a running server.",
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="Disable auto server mode; load model directly each time.",
    )
    parser.add_argument(
        "--server-idle-timeout", type=int, default=600,
        help="Server idle timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--server-parent-pid", type=int, default=None, help=argparse.SUPPRESS
    )


def _handle_serve(args, make_evaluator):
    if not args.server_socket:
        raise SystemExit("--serve requires --server-socket PATH")
    evaluator = make_evaluator(args)
    server = MLIPServer(
        evaluator=evaluator,
        socket_path=args.server_socket,
        idle_timeout=args.server_idle_timeout,
        parent_pid=args.server_parent_pid,
    )
    server.serve_forever()
    return 0


def _handle_stop_server(args):
    if not args.server_socket:
        raise SystemExit("--stop-server requires --server-socket PATH")
    if not server_is_alive(args.server_socket):
        print("No server running at {}".format(args.server_socket), file=sys.stderr)
        return 1
    resp = send_shutdown(args.server_socket)
    print("Server response: {}".format(resp), file=sys.stderr)
    return 0


def _evaluate_via_server(socket_path, ext):
    """Evaluate using the persistent server."""
    need_grad = int(ext["igrd"]) >= 1
    need_hess = int(ext["igrd"]) >= 2
    return client_evaluate(
        socket_path=socket_path,
        symbols=ext["symbols"],
        coords_ang=ext["coords_ang"],
        charge=ext["charge"],
        multiplicity=ext["multiplicity"],
        need_forces=need_grad,
        need_hessian=need_hess,
        hessian_mode="Analytical",
        hessian_step=1.0e-3,
    )


def _evaluate_direct(make_evaluator, args, ext):
    """Evaluate by loading the model directly in this process."""
    need_grad = int(ext["igrd"]) >= 1
    need_hess = int(ext["igrd"]) >= 2
    evaluator = make_evaluator(args)
    return evaluator.evaluate(
        symbols=ext["symbols"],
        coords_ang=ext["coords_ang"],
        charge=ext["charge"],
        multiplicity=ext["multiplicity"],
        need_forces=need_grad,
        need_hessian=need_hess,
        hessian_mode="Analytical",
        hessian_step=1.0e-3,
    )


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
    parser.add_argument(
        "--solvent",
        default="none",
        help="xTB implicit-solvent name (set 'none' to disable solvent correction).",
    )
    parser.add_argument(
        "--solvent-model",
        default="alpb",
        choices=("alpb", "cpcmx"),
        help=(
            "Implicit solvent model for xTB correction: "
            "alpb -> --alpb, cpcmx -> --cpcmx."
        ),
    )
    parser.add_argument(
        "--xtb-cmd",
        default="xtb",
        help="xTB executable path/command for solvent correction (default: xtb).",
    )
    parser.add_argument(
        "--xtb-acc",
        type=float,
        default=0.2,
        help="xTB --acc value used for ALPB correction (default: 0.2).",
    )
    parser.add_argument(
        "--xtb-workdir",
        default="tmp",
        help="xTB scratch base dir: 'tmp' or a directory path (default: tmp).",
    )
    parser.add_argument(
        "--xtb-keep-files",
        action="store_true",
        help="Keep xTB temporary files for debugging.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print model aliases and exit")
    parser.add_argument(
        "--version",
        action="version",
        version=_version_text(plugin_name),
    )

    _add_server_args(parser)

    if add_extra_args is not None:
        add_extra_args(parser)

    # --serve / --stop-server: no Gaussian tail needed
    if "--serve" in argv or "--stop-server" in argv:
        args = parser.parse_args(argv)
        if args.stop_server:
            return _handle_stop_server(args)
        if args.serve:
            return _handle_serve(args, make_evaluator)
        return 0

    # Help mode should work without Gaussian-generated tail arguments.
    if ("-h" in argv) or ("--help" in argv):
        parser.parse_args(["--help"])
        return 0

    # Manual utility mode (no Gaussian tail needed)
    if len(argv) <= 3 and (("--list-models" in argv) or ("--version" in argv)):
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
                    extinp_like, "orca-mlips-" + backend
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

    need_grad = int(ext["igrd"]) >= 1
    need_hess = int(ext["igrd"]) >= 2

    # --- Evaluation: auto server mode (default) or direct mode ---
    if not args.no_server:
        parent_pid = os.getppid()
        socket_path = args.server_socket or auto_server_socket(args, parent_pid=parent_pid)
        server_ready = ensure_server(
            executable=sys.argv[0],
            custom_args=custom_args,
            socket_path=socket_path,
            idle_timeout=args.server_idle_timeout,
            parent_pid=parent_pid,
        )
        if server_ready:
            try:
                energy_ev, forces_ev_ang, hess_ev_ang2 = _evaluate_via_server(
                    socket_path, ext
                )
            except ServerError as exc:
                print(
                    "[mlip-client] WARNING: Server error: {}. "
                    "Falling back to direct mode.".format(exc),
                    file=sys.stderr, flush=True,
                )
                energy_ev, forces_ev_ang, hess_ev_ang2 = _evaluate_direct(
                    make_evaluator, args, ext
                )
        else:
            print(
                "[mlip-client] WARNING: Server not available, "
                "loading model directly.",
                file=sys.stderr, flush=True,
            )
            energy_ev, forces_ev_ang, hess_ev_ang2 = _evaluate_direct(
                make_evaluator, args, ext
            )
    else:
        energy_ev, forces_ev_ang, hess_ev_ang2 = _evaluate_direct(
            make_evaluator, args, ext
        )

    # --- Optional xTB implicit-solvent-vacuum correction ---
    if solvent_correction_enabled(args.solvent):
        try:
            de_ev, df_ev_ang, dh_ev_ang2 = delta_alpb_minus_vac(
                symbols=ext["symbols"],
                coords_ang=ext["coords_ang"],
                charge=ext["charge"],
                multiplicity=ext["multiplicity"],
                solvent=args.solvent,
                solvent_model=args.solvent_model,
                need_forces=need_grad,
                need_hessian=need_hess,
                xtb_cmd=args.xtb_cmd,
                xtb_acc=args.xtb_acc,
                xtb_workdir=args.xtb_workdir,
                xtb_keep_files=args.xtb_keep_files,
                ncores=resolve_xtb_ncores(),
            )
        except XTBError as exc:
            raise RunnerError("xTB solvent correction failed: {}".format(exc))

        energy_ev = float(energy_ev) + float(de_ev)

        if need_grad:
            if df_ev_ang is None:
                raise RunnerError("xTB solvent correction returned no force delta.")
            if forces_ev_ang is not None:
                forces_ev_ang = np.asarray(forces_ev_ang, dtype=np.float64) + np.asarray(
                    df_ev_ang, dtype=np.float64
                )

        if need_hess:
            if dh_ev_ang2 is None:
                raise RunnerError("xTB solvent correction returned no Hessian delta.")
            if hess_ev_ang2 is not None:
                hess_ev_ang2 = np.asarray(hess_ev_ang2, dtype=np.float64) + np.asarray(
                    dh_ev_ang2, dtype=np.float64
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

    # Keep Gaussian external message compact to reduce per-step log noise.
    msg = "[{}] ok igrd={} server={}".format(
        plugin_name,
        ext["igrd"],
        "off" if args.no_server else "on",
    )
    write_msg(msg_file, msg + "\n")

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
