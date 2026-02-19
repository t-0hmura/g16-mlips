# g16-mlips

MLIP (Machine Learning Interatomic Potential) plugins for Gaussian 16 `External` interface.

Three model families are supported:
- **UMA** ([fairchem](https://github.com/facebookresearch/fairchem)) — default model: `uma-s-1p1`
- **ORB** ([orb-models](https://github.com/orbital-materials/orb-models)) — default model: `orb_v3_conservative_omol`
- **MACE** ([mace](https://github.com/ACEsuit/mace)) — default model: `MACE-OMOL-0`

All backends provide energy, gradient, and **analytical Hessian** to **Gaussian 16**.  
> The model server starts automatically and stays resident, so repeated calls during optimization are fast.  

## Quick Start (Default = UMA)

1. Install PyTorch (CUDA 12.9 build).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with UMA profile. If you need ORB/MACE, use `g16-mlips[orb]`/`g16-mlips[mace]`.
```bash
pip install "g16-mlips[uma]"
```

3. Log in to Hugging Face for UMA model access. (No need for ORB/MACE)
```bash
huggingface-cli login
```

4. Use in a Gaussian input file. `nomicro` is required. If you use ORB/MACE, use `external="orb"`/`external="mace"`.
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

### Analytical Hessian

Gaussian `freq` (with `external=...`) is the only path that requests the plugin's analytical Hessian directly.

**Frequency calculation**

```text
%nprocshared=8
%mem=32GB
%chk=cla_ext.chk
#p external="uma" freq

CLA freq UMA

0 1
...
```
Gaussian sends `igrd=2` and stores the result in the `.chk` file.

### Using analytical Hessian in optimization jobs

To use MLIP analytical Hessian in `opt`/`irc`, read the Hessian from an existing checkpoint using Gaussian `%oldchk` + `readfc`.

```text
%nprocshared=8
%mem=32GB
%chk=cla_ext.chk
%oldchk=cla_ext.chk

#p external="uma" opt(readfc,nomicro)

CLA opt UMA

0 1
...
```

`readfc` reads the force constants from `%oldchk`. This applies to `opt` and `irc` runs.
Note that `freq` is the only job type that requests analytical Hessian (`igrd=2`) from the plugin. `opt` and `irc` themselves never request it directly.

> **Important: Gaussian External 2-step limit.** Gaussian's `External` interface limits optimization to 2 steps per run. If the geometry has not converged, Gaussian exits with a non-zero exit code and the optimization must be continued with `opt(restart)`:
>
> ```text
> %nprocshared=8
> %mem=32GB
> %chk=cla_ext.chk
> %oldchk=cla_ext.chk
>
> #p external="uma" opt(restart,nomicro)
> ```
> `opt(restart)` reads the optimization history from the checkpoint; do not re-specify `ts` or `noeigentest` in the restart input.

> **Note:** Run `uma --list-models` to see available models. If the `uma` alias conflicts in your environment, use `g16-mlips-uma` instead.

Additional examples: `examples/cla_freq.gjf` + `examples/cla_external.gjf`, `examples/sn2_freq.gjf` + `examples/sn2_external.gjf`, `examples/water_freq.gjf` + `examples/water_external.gjf`

## Installing Model Families

```bash
pip install "g16-mlips[uma]"         # UMA (default)
pip install "g16-mlips[orb]"         # ORB
pip install "g16-mlips[mace]"        # MACE
pip install "g16-mlips[orb,mace]"    # ORB + MACE
pip install g16-mlips                # core only
```

> **Note:** UMA and MACE conflict at dependency level (`e3nn`). Use separate environments.

Local install:
```bash
git clone https://github.com/t-0hmura/g16-mlips.git
cd g16-mlips
pip install ".[uma]"
```

Model download notes:
- **UMA**: Hosted on Hugging Face Hub. Run `huggingface-cli login` once.
- **ORB / MACE**: Downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- ORB / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace

## Advanced Options

See `OPTIONS.md` for backend-specific tuning parameters.

Command aliases:
- Short: `uma`, `orb`, `mace`
- Prefixed: `g16-mlips-uma`, `g16-mlips-orb`, `g16-mlips-mace`

## Troubleshooting

- **`external="uma"` runs the wrong plugin** — Use `external="g16-mlips-uma"` to avoid alias conflicts.
- **`uma` command not found** — Activate the conda environment where the package is installed.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **Works interactively but fails in PBS jobs** — Use absolute path from `which uma` in the Gaussian input.

## References

- Gaussian External: `$g16root/g16/doc/extern.txt`, `$g16root/g16/doc/extgau`
