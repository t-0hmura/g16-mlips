#!/bin/sh
#PBS -N mlips4g16_smoke
#PBS -q default
#PBS -l nodes=1:ppn=32:gpus=1,mem=120GB,walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null

set -eu

test "${PBS_O_WORKDIR:-}" && cd "$PBS_O_WORKDIR"

. /home/apps/Modules/init/profile.sh
module load gaussian16.C02
module load orca/6.1.1

# Optional: activate env containing torch/ase and backend packages.
# source /home/tohmura/miniconda3/etc/profile.d/conda.sh
# conda activate pdb2reaction

python3 -m py_compile plugins/*.py

python3 plugins/uma_g16.py --list-models | head -n 20
python3 plugins/orbmol_g16.py --list-models | head -n 20
python3 plugins/mace_g16.py --list-models | head -n 20

# Full Gaussian run example (after dependency installation):
# g16 < examples/water_external.gjf > water_external.log
