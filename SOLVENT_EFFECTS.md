# Solvent Effects (xTB Implicit-Solvent Delta Correction)

`g16-mlips` supports an implicit-solvent correction that adds only solvent
contributions to MLIP vacuum predictions.

## What Is Added

For each geometry `R`, the plugin evaluates xTB twice:

- vacuum
- implicit solvent (`--solvent <name>`, model selected by `--solvent-model`)

Then it builds:

- `dE(R) = E_xTB(solv) - E_xTB(vac)`
- `dF(R) = F_xTB(solv) - F_xTB(vac)`
- `dH(R) = H_xTB(solv) - H_xTB(vac)`

and returns:

- `E_total = E_MLIP(vac) + dE`
- `F_total = F_MLIP(vac) + dF`
- `H_total = H_MLIP(vac) + dH`

This keeps the MLIP model in vacuum mode and adds only solvent-origin terms.

## CLI Options

- `--solvent <name|none>`: enable implicit-solvent correction (`none` disables it)
- `--solvent-model <alpb|cpcmx>`: solvent model selector (default: `alpb`)
  - `alpb` uses xTB `--alpb`
  - `cpcmx` uses xTB `--cpcmx`
- `--xtb-cmd <path_or_cmd>`: xTB executable (default: `xtb`)
- `--xtb-acc <float>`: xTB accuracy value (default: `0.2`)
- `--xtb-workdir <tmp|path>`: per-call xTB scratch base (default: `tmp`)
- `--xtb-keep-files`: keep xTB temporary files

## CPCM-X Setup

CPCM-X installation is done by cloning the repository from
`grimme-lab/CPCM-X` and setting `CPXHOME` to that directory.
Place parameter and COSMO file databases under that setup, and use the
command-line workflow as recommended by CPCM-X.

```bash
git clone git@github.com:grimme-lab/CPCM-X.git
cd CPCM-X
export CPXHOME="$PWD"
```

For full installation/build details, see:
- https://github.com/grimme-lab/CPCM-X

## Gaussian Usage

### Geometry optimization

```text
%nprocshared=8
%mem=32GB
%chk=water_solv.chk
#p external="uma --solvent water --solvent-model alpb" opt(nomicro)
```

### Frequency (solvent-corrected Hessian)

```text
%nprocshared=8
%mem=32GB
%chk=water_solv.chk
#p external="uma --solvent water --solvent-model cpcmx" freq
```

Gaussian requests Hessian only for `freq` (`igrd=2`), where `dH` is added.

## Performance Notes

- Solvent correction runs two xTB calculations per geometry point (vacuum + solvated state).
- Hessian correction is expensive: each state needs xTB Hessian.
- Use solvent-corrected Hessian only where needed (`freq`/`readfc` workflow).

## Troubleshooting

- `xTB command not found`:
  - install xTB in the active environment
  - or set `--xtb-cmd /full/path/to/xtb`
- `xTB solvent correction failed`:
  - verify solvent spelling (`water`, `thf`, `toluene`, ...)
  - for `--solvent-model cpcmx`, use an xTB build with CPCM-X support
    (see https://github.com/grimme-lab/CPCM-X)
  - rerun with `--xtb-keep-files --xtb-workdir <path>` and inspect files
