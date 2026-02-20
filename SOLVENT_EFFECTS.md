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

CPCM-X requires xTB to be built from source with CPCM-X linked in.
The conda-forge `xtb` package does not include CPCM-X support.

**Step 1: Build xTB with `-DWITH_CPCMX=ON`**

CPCM-X is bundled in the xTB source tree (`subprojects/cpx.wrap`) and is fetched automatically during the CMake configure step. Requires GCC >= 10 (gfortran 8 causes internal compiler errors).

```bash
git clone --depth 1 https://github.com/grimme-lab/xtb.git
cd xtb
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPCMX=ON \
  -DBLAS_LIBRARIES=/path/to/libblas.so \
  -DLAPACK_LIBRARIES=/path/to/liblapack.so
make -C build tblite-lib -j8   # build tblite first to avoid a parallel build race
make -C build xtb-exe -j8
```

**Step 2: Use the custom xTB via `--xtb-cmd`**

```text
#p external="uma --solvent water --solvent-model cpcmx --xtb-cmd /path/to/xtb" freq
```

`CPXHOME` must be set at runtime to point to the CPCM-X source directory (containing `DB/`). When xTB fetches CPCM-X during build, the source is placed under `build/_deps/cpcmx-src/`.

For full details, see:
- https://github.com/grimme-lab/xtb
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

Gaussian requests the Hessian only for `freq` (`igrd=2`), where `dH` is added.

## Performance Notes

- Solvent correction runs two xTB calculations per geometry point (vacuum + solvated state).
- Hessian correction is expensive: each state needs an xTB Hessian.
- Use solvent-corrected Hessian only where needed (`freq`/`readfc` workflow).

## Troubleshooting

- `xTB command not found`:
  - Install xTB in the active environment.
  - Or set `--xtb-cmd /full/path/to/xtb`.
- `xTB solvent correction failed`:
  - Verify the solvent spelling (`water`, `thf`, `toluene`, ...).
  - For `--solvent-model cpcmx`, use an xTB build with CPCM-X support
    (see https://github.com/grimme-lab/CPCM-X).
  - Rerun with `--xtb-keep-files --xtb-workdir <path>` and inspect the files.
