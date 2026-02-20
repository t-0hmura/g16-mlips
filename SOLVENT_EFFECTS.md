# Solvent Effects (xTB/ALPB Delta Correction)

`g16-mlips` supports an implicit-solvent correction that adds only solvent
contributions to MLIP vacuum predictions.

## What Is Added

For each geometry `R`, the plugin evaluates xTB twice:

- vacuum
- ALPB solvent (`--solvent <name>`)

Then it builds:

- `dE(R) = E_xTB(ALPB) - E_xTB(vac)`
- `dF(R) = F_xTB(ALPB) - F_xTB(vac)`
- `dH(R) = H_xTB(ALPB) - H_xTB(vac)`

and returns:

- `E_total = E_MLIP(vac) + dE`
- `F_total = F_MLIP(vac) + dF`
- `H_total = H_MLIP(vac) + dH`

This keeps the MLIP model in vacuum mode and adds only solvent-origin terms.

## CLI Options

- `--solvent <name|none>`: enable ALPB correction (`none` disables it)
- `--xtb-cmd <path_or_cmd>`: xTB executable (default: `xtb`)
- `--xtb-acc <float>`: xTB accuracy value (default: `0.2`)
- `--xtb-workdir <tmp|path>`: per-call xTB scratch base (default: `tmp`)
- `--xtb-keep-files`: keep xTB temporary files

## Gaussian Usage

### Geometry optimization

```text
%nprocshared=8
%mem=32GB
%chk=water_solv.chk
#p external="uma --solvent water" opt(nomicro)
```

### Frequency (solvent-corrected Hessian)

```text
%nprocshared=8
%mem=32GB
%chk=water_solv.chk
#p external="uma --solvent water" freq
```

Gaussian requests Hessian only for `freq` (`igrd=2`), where `dH` is added.

## Performance Notes

- Solvent correction runs two xTB calculations per geometry point (vacuum + ALPB).
- Hessian correction is expensive: each state needs xTB Hessian.
- Use solvent-corrected Hessian only where needed (`freq`/`readfc` workflow).

## Troubleshooting

- `xTB command not found`:
  - install xTB in the active environment
  - or set `--xtb-cmd /full/path/to/xtb`
- `xTB solvent correction failed`:
  - verify solvent spelling (`water`, `thf`, `toluene`, ...)
  - rerun with `--xtb-keep-files --xtb-workdir <path>` and inspect files
