# mlips4g16

This plugin enables Machine Learning Interatomic Potentials (MLIPs) in Gaussian16 through the External interface.
Because analytical Hessians are available, this plugin can provide more accurate TS searches, IRC, and vibrational analysis than numerical Hessians.
If your environment has limited GPU VRAM, Numerical Hessian mode (`--hessian-mode Numerical`) is recommended.

MLIP plugins for Gaussian16 `External` with three model families:
- UMA (FAIR-Chem)
- OrbMol (orb-models)
- MACE

Default models:
- UMA: `uma-s-1p1`
- OrbMol: `orb_v3_conservative_omol`
- MACE: `MACE-OMOL-0`

## Quick Start (Default = UMA)

1. Install PyTorch (CUDA 12.9 build).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install package with UMA profile.
```bash
pip install "mlips4g16[uma]"
```
This install creates the commands `uma`, `orb`, `mace` (and prefixed aliases).

3. Log in once to Hugging Face for UMA model access.
```bash
huggingface-cli login
```

4. Confirm commands and model list.
```bash
uma --list-models
uma --list-tasks
```
If `uma` alias conflicts in your environment, use `mlips4g16-uma`.

5. Confirm plugin version.
```bash
uma --version
```

6. Use in Gaussian input (`External`).
```text
%chk=water_ext.chk
#p external="uma" opt

Water external UMA example

0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
```

Optional explicit options:
```bash
# model/task/hessian mode can still be passed explicitly
#p external="uma --model uma-s-1p1 --task omol --hessian-mode Analytical" opt
```

Other backends with defaults:
```bash
#p external="orb" opt
#p external="mace" opt
```

Additional example inputs:
- `examples/water_external.gjf`
- `examples/cla_external.gjf`
- `examples/sn2_external.gjf`

## Install Model Families

PyPI install:
```bash
# Default profile (UMA)
pip install "mlips4g16[uma]"

# Add OrbMol
pip install "mlips4g16[orb]"

# Add MACE
pip install "mlips4g16[mace]"

# Add both OrbMol + MACE
pip install "mlips4g16[orb,mace]"

# Core package only (no backend dependencies)
pip install mlips4g16
```

Important compatibility note:
- UMA and MACE are currently not compatible in a single environment due to `e3nn` dependency constraints.
- Use separate environments (for example: one env for `mlips4g16[uma]`, another env for `mlips4g16[mace]` or `mlips4g16[orb,mace]`).

Local source install:
```bash
git clone https://github.com/t-0hmura/mlips4g16.git
cd mlips4g16
pip install ".[uma]"
pip install ".[orb]"     # optional
pip install ".[mace]"    # optional
pip install .            # core only
```

Family-specific commands:
```bash
uma --list-models
orb --list-models
mace --list-models
```

Family notes:
- UMA: models are served from Hugging Face Hub. Run `huggingface-cli login` once.
- OrbMol: models are provided by `orb-models` and downloaded automatically on first use.
- MACE: models are provided by `mace-torch` and downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- OrbMol / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace

## Advanced Usage

### Backend Commands
- Short aliases: `uma`, `orb`, `mace`
- Prefixed aliases: `mlips4g16-uma`, `mlips4g16-orb`, `mlips4g16-mace`

Detailed and low-impact tuning options are documented in `OPTIONS.md`.

## Troubleshooting

- `external="uma"` runs the wrong plugin:
  Use prefixed aliases to avoid collisions, for example `external="mlips4g16-uma"`.
- `uma` command is not found after install:
  Activate the same environment where you installed the package, then reinstall with `python -m pip install "mlips4g16[uma]"`.
- UMA model download fails with 401/403:
  Run `huggingface-cli login`. Some UMA model repos are gated and require manual access approval on Hugging Face.
- Works interactively but fails in scheduler jobs:
  Job shells may have reduced `PATH`. Use an absolute command path in Gaussian from `which uma`.

## Notes

- Gaussian External references:
  - `$g16root/g16/doc/extern.txt`
  - `$g16root/g16/doc/extgau`
- UMA and MACE profiles currently conflict at dependency level (`e3nn`); use separate environments.
- `run.sh` contains a PBS smoke test template (`qsub run.sh`).
