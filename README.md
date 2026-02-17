# mlips4g16

Path-independent MLIP plugins for Gaussian16 `External`.

Included plugins:
- `plugins/uma_g16.py` (FAIR-Chem / UMA)
- `plugins/orbmol_g16.py` (orb-models / OrbMol)
- `plugins/mace_g16.py` (MACE)

Default models:
- UMA: `uma-s-1p1`
- OrbMol: `orb_v3_conservative_omol`
- MACE: `MACE-OMOL-0` (alias of `omol:extra_large`)

## Quick Start

1. Clone and enter this repository.
```bash
git clone https://github.com/t-0hmura/mlips4g16.git
cd mlips4g16
```

2. (Optional) Create a clean environment.
```bash
python3 -m venv .venv
. .venv/bin/activate
```

3. Install base requirements.
```bash
pip install -r requirements.txt
```

4. Install only the backend you need.
```bash
# UMA
pip install fairchem-core

# OrbMol
pip install orb-models

# MACE
pip install mace-torch
```

5. Verify model listing.
```bash
python3 plugins/uma_g16.py --list-models
python3 plugins/orbmol_g16.py --list-models
python3 plugins/mace_g16.py --list-models
```

## Gaussian Input Example

Replace `/path/to/mlips4g16` with your local clone path.

```text
%chk=water_ext.chk
#p external="/path/to/mlips4g16/plugins/uma_g16.py --model uma-s-1p1 --task omol --hessian-mode Analytical" opt

Water external UMA example

0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
```

Switch backend by replacing the script path:
- `.../plugins/orbmol_g16.py`
- `.../plugins/mace_g16.py`

## Model Selection

UMA:
```bash
python3 plugins/uma_g16.py --list-models
python3 plugins/uma_g16.py --list-tasks
```

OrbMol:
```bash
python3 plugins/orbmol_g16.py --list-models
```
Both dashed and underscored names are accepted, for example:
- `orb-v3-conservative-omol`
- `orb_v3_conservative_omol`

MACE:
```bash
python3 plugins/mace_g16.py --list-models
```
Accepted forms include:
- `MACE-OMOL-0`
- `mp:<alias>` or `<alias>` (for MP aliases)
- `off:<alias>` / `off-small|off-medium|off-large`
- `omol:extra_large`
- `anicc`
- local model path or model URL

## Hessian Mode

When Gaussian requests Hessian (`IGrd=2`), you can choose:
- `--hessian-mode Analytical`
- `--hessian-mode Numerical`

Use `--strict-hessian` to fail instead of falling back to numerical Hessian.

## Notes

- This implementation follows Gaussian External format from the docs distributed with Gaussian, typically:
  - `$g16root/g16/doc/extern.txt`
  - `$g16root/g16/doc/extgau`
- Upstream model sources:
  - https://github.com/facebookresearch/fairchem
  - https://github.com/orbital-materials/orb-models
  - https://github.com/ACEsuit/mace

## Cluster Smoke Test

Edit and submit:
```bash
qsub run.sh
```

`run.sh` is intentionally generic and only loads modules if your environment provides the module system.
