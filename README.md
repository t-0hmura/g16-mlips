# mlips4g16

MLIP (Machine Learning Interatomic Potential) plugins for Gaussian16 `External` interface.

Three model families are supported:
- **UMA** (FAIR-Chem) — default model: `uma-s-1p1`
- **OrbMol** (orb-models) — default model: `orb_v3_conservative_omol`
- **MACE** — default model: `MACE-OMOL-0`

All backends provide energy, gradient, and analytical Hessian. The model server starts automatically and stays resident, so repeated calls during optimization are fast.

## Quick Start (Default = UMA)

1. Install PyTorch (CUDA 12.9 build).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with UMA profile.
```bash
pip install "mlips4g16[uma]"
```

3. Log in to Hugging Face for UMA model access.
```bash
huggingface-cli login
```

4. Use in a Gaussian input file.
```text
%nprocshared=8
%mem=32GB
%chk=water_ext.chk
#p external="uma" opt(nomicro)

Water external UMA example

0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
```

Other backends:
```text
#p external="orb" opt(nomicro)
#p external="mace" opt(nomicro)
```

> **Important:** For Gaussian `External` geometry optimization, always include `nomicro` in `opt(...)`.

### TS Search (Recommended: freq + readfc)

> **Why two steps?** Gaussian's `opt(calcfc)` computes the initial Hessian by numerical finite difference (igrd=1), calling the plugin many times. Only `freq` sends igrd=2 and receives the exact analytical Hessian from the plugin in a single call. By running `freq` first and then `opt(readfc)`, the optimizer starts with the accurate analytical Hessian, leading to faster and more reliable convergence.

Two-step workflow using the MLIP analytical Hessian as the initial Hessian:

**Step 1: Compute analytical Hessian via `freq`**
```text
%nprocshared=8
%mem=32GB
%chk=cla_ext.chk
#p external="uma" freq

CLA freq UMA

0 1
...
```
Gaussian sends igrd=2, and the plugin returns the analytical Hessian. The result is stored in the `.chk` file.

**Step 2: TS optimization reading Hessian from `.chk`**
```text
%nprocshared=8
%mem=32GB
%chk=cla_ext.chk
#p external="uma" opt(readfc,noeigentest,ts,nomicro)

CLA TS opt UMA

0 1
...
```
`readfc` reads the initial Hessian from the checkpoint file. `noeigentest` skips the eigenvalue check (MLIP Hessians may have extra negative eigenvalues near the TS).

> **Important: Gaussian External 2-step limit.** Gaussian's `External` interface limits optimization to 2 steps per run. If the geometry has not converged, Gaussian exits with a non-zero exit code and the optimization must be continued with `opt(restart)`:
>
> ```text
> %chk=cla_ext.chk
> #p external="uma" opt(restart,nomicro)
>
> ```
> `opt(restart)` reads the geometry, force constants, optimization history, and TS/noeigentest settings from the checkpoint file — do not re-specify `ts` or `noeigentest` in the restart input. No title or molecule specification is needed. Run this in a loop until Gaussian exits with code 0 (converged).

### Geometry Optimization (with analytical Hessian)

Same two-step workflow with `opt(readfc)` instead of `opt(readfc,noeigentest,ts)`:
```text
%chk=water_ext.chk
#p external="mace" freq
```
then:
```text
%chk=water_ext.chk
#p external="mace" opt(readfc,nomicro)
```
Restart if not converged:
```text
%chk=water_ext.chk
#p external="mace" opt(restart,nomicro)
```

### Frequency Calculation

```text
#p external="uma" freq
```

With `freq`, Gaussian requests the analytical Hessian directly (igrd=2) from the plugin.

> **Note:** Run `uma --list-models` to see available models. If the `uma` alias conflicts in your environment, use `mlips4g16-uma` instead.

Additional examples: `examples/cla_freq.gjf` + `examples/cla_external.gjf`, `examples/sn2_freq.gjf` + `examples/sn2_external.gjf`, `examples/water_freq.gjf` + `examples/water_external.gjf`

## Installing Model Families

```bash
pip install "mlips4g16[uma]"         # UMA (default)
pip install "mlips4g16[orb]"         # OrbMol
pip install "mlips4g16[mace]"        # MACE
pip install "mlips4g16[orb,mace]"    # OrbMol + MACE
pip install mlips4g16                # core only
```

> **Note:** UMA and MACE conflict at dependency level (`e3nn`). Use separate environments.

Local install:
```bash
git clone https://github.com/t-0hmura/mlips4g16.git
cd mlips4g16
pip install ".[uma]"
```

Model download notes:
- **UMA**: Hosted on Hugging Face Hub. Run `huggingface-cli login` once.
- **OrbMol / MACE**: Downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- OrbMol / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace

## Advanced Options

See `OPTIONS.md` for backend-specific tuning parameters.

Command aliases:
- Short: `uma`, `orb`, `mace`
- Prefixed: `mlips4g16-uma`, `mlips4g16-orb`, `mlips4g16-mace`

## Troubleshooting

- **`external="uma"` runs the wrong plugin** — Use `external="mlips4g16-uma"` to avoid alias conflicts.
- **`uma` command not found** — Activate the conda environment where the package is installed.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **Works interactively but fails in PBS jobs** — Use absolute path from `which uma` in the Gaussian input.

## References

- Gaussian External: `$g16root/g16/doc/extern.txt`, `$g16root/g16/doc/extgau`
