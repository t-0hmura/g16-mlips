# g16-mlips

[![DOI](https://zenodo.org/badge/1160316483.svg)](https://zenodo.org/badge/latestdoi/1160316483)

MLIP (Machine Learning Interatomic Potential) plugins for Gaussian 16 `External` interface.

Four model families are currently supported:
- **UMA** ([fairchem](https://github.com/facebookresearch/fairchem)) — default model: `uma-s-1p1`
- **ORB** ([orb-models](https://github.com/orbital-materials/orb-models)) — default model: `orb_v3_conservative_omol`
- **MACE** ([mace](https://github.com/ACEsuit/mace)) — default model: `MACE-OMOL-0`
- **AIMNet2** ([aimnetcentral](https://github.com/isayevlab/aimnetcentral)) — default model: `aimnet2`

All backends provide energy, gradient, and **analytical Hessian** for **Gaussian 16**.
An optional implicit-solvent correction (`xTB`) is also available via `--solvent`.

> The model server starts automatically and stays resident, so repeated calls during optimization are fast.  

Requires **Python 3.9** or later.

## Quick Start (Default = UMA)

1. Install PyTorch for your environment (CUDA/CPU).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with UMA profile. If you need ORB/MACE/AIMNet2, use `g16-mlips[orb]`/`g16-mlips[mace]`/`g16-mlips[aimnet2]`.
```bash
pip install "g16-mlips[uma]"
```

3. Log in to Hugging Face for UMA model access. (Not required for ORB/MACE/AIMNet2)
```bash
huggingface-cli login
```

4. Use in a Gaussian input file. `nomicro` is required. If you use ORB/MACE/AIMNet2, use `external="orb"`/`external="mace"`/`external="aimnet2"`.
For detailed Gaussian `External` usage, see https://gaussian.com/external/
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
#p external="aimnet2" opt(nomicro)
```

> **Important:** For Gaussian `External` geometry optimization, always include `nomicro` in `opt(...)`.
> Without it, Gaussian uses micro-iterations that assume an internal gradient routine, which is incompatible with the external interface.

### Analytical Hessian (optional)

Optimization and IRC can run without providing an initial Hessian — Gaussian builds one internally using estimated force constants. Providing an MLIP analytical Hessian via `freq` + `readfc` improves convergence, especially for TS searches.

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

## Implicit Solvent Correction (xTB)

Install xTB in your conda environment (easy path):

```bash
conda install xtb
```

Use `--solvent <name>` in `external="..."` (examples: `water`, `thf`):

```text
#p external="uma --solvent water" opt(nomicro)
#p external="uma --solvent thf" freq
```

For details, see `SOLVENT_EFFECTS.md`.

## Installing Model Families

```bash
pip install "g16-mlips[uma]"         # UMA (default)
pip install "g16-mlips[orb]"         # ORB
pip install "g16-mlips[mace]"        # MACE
pip install "g16-mlips[orb,mace]"    # ORB + MACE
pip install "g16-mlips[aimnet2]"     # AIMNet2
pip install "g16-mlips[orb,mace,aimnet2]"  # ORB + MACE + AIMNet2
pip install g16-mlips                # core only
```

> **Note:** UMA and MACE have a dependency conflict (`e3nn`). Use separate environments.

Local install:
```bash
git clone https://github.com/t-0hmura/g16-mlips.git
cd g16-mlips
pip install ".[uma]"
```

Model download notes:
- **UMA**: Hosted on Hugging Face Hub. Run `huggingface-cli login` once.
- **ORB / MACE / AIMNet2**: Downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- ORB / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace
- AIMNet2: https://github.com/isayevlab/aimnetcentral

## Advanced Options

See `OPTIONS.md` for backend-specific tuning parameters.
For solvent correction options, see `SOLVENT_EFFECTS.md`.

Command aliases:
- Short: `uma`, `orb`, `mace`, `aimnet2`
- Prefixed: `g16-mlips-uma`, `g16-mlips-orb`, `g16-mlips-mace`, `g16-mlips-aimnet2`

## Troubleshooting

- **`external="uma"` runs the wrong plugin** — Use `external="g16-mlips-uma"` to avoid alias conflicts.
- **`external="aimnet2"` runs the wrong plugin** — Use `external="g16-mlips-aimnet2"` to avoid alias conflicts.
- **`uma` command not found** — Activate the conda environment where the package is installed.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **Works interactively but fails in PBS jobs** — Use absolute path from `which uma` in the Gaussian input.

## Citation

If you use this package, please cite:

```bibtex
@software{ohmura2026g16mlips,
  author       = {Ohmura, Takuto},
  title        = {g16-mlips},
  year         = {2026},
  month        = {2},
  version      = {1.0.0},
  url          = {https://github.com/t-0hmura/g16-mlips},
  license      = {MIT},
  doi          = {10.5281/zenodo.18695243}
}
```

## References

- Gaussian External interface (official): https://gaussian.com/external/
- Gaussian External: `$g16root/g16/doc/extern.txt`, `$g16root/g16/doc/extgau`
