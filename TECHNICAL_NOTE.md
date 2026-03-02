# Technical Note

## Solvent-Only Delta Injection

The plugin adds xTB implicit-solvent-vacuum deltas to backend MLIP outputs:

- `dE = E_xTB(solv) - E_xTB(vac)`
- `dF = F_xTB(solv) - F_xTB(vac)`
- `dH = H_xTB(solv) - H_xTB(vac)`

Injected quantities:

- `E_total = E_MLIP + dE`
- `F_total = F_MLIP + dF`
- `H_total = H_MLIP + dH`

This is implemented in `plugins/runner_g16.py` through
`plugins/xtb_alpb_correction.py`.

## Point-Charge Embedding Injection (`--embedcharge`)

For ONIOM `AllAtoms`, Gaussian can pass MM point-charge rows (`IAn=0`).
The plugin evaluates MLIP on real atoms (`IAn>0`) and optionally adds xTB
point-charge embedding corrections:

- `dE = E_xTB(embed) - E_xTB(no-embed)`
- `dF_Q = F_Q(embed) - F_Q(no-embed)`
- `dF_M = F_M(embed)`

Injected quantities:

- `E_total = E_MLIP + dE` (plus solvent delta if enabled)
- force/Hessian are assembled on the full Gaussian atom index space
  (real + MM point-charge rows)

Implementation:

- input split/reindex: `plugins/g16_extio.py`
- embedcharge correction: `plugins/xtb_embedcharge_correction.py`
- assembly and final output: `plugins/runner_g16.py`

## Units

Units parsed from xTB output:

- energy: `Eh`
- gradient: `Eh/Bohr`
- Hessian: `Eh/Bohr^2`

Converted to MLIP units before addition:

- energy: `eV`
- forces: `eV/Ang`
- Hessian: `eV/Ang^2`

Force conversion uses `F = -grad`.

## Backend-Specific Hessian Paths

The shared backend implementation is in `plugins/mlip_backends.py`.

### UMA / ORB

UMA/ORB analytical Hessians are computed via autograd on the energy with model-state
management to avoid graph/dropout issues:

1. Call `_prepare_model_for_autograd_hessian(...)`.
2. Compute the Hessian with `torch.autograd.functional.hessian(...)`.
3. Call `_restore_model_after_autograd_hessian(...)`.

Key points:

- The model is switched to train mode for reliable autograd graph construction.
- Dropout modules are effectively disabled (`p=0`, eval behavior).
- Original training/dropout/`requires_grad` states are restored after Hessian computation.

### MACE

Uses the calculator-native Hessian path (`get_hessian`) when available.

### AIMNet2

Requests the Hessian from AIMNet2 calculator outputs and reshapes to `(3N, 3N)`.

## Gaussian External Integration

`runner_g16.py` flow:

1. read Gaussian external input and split real atoms (`IAn>0`) / MM rows (`IAn=0`)
2. evaluate MLIP on real atoms only
3. reassemble MLIP forces/Hessian to full Gaussian atom indexing
4. if `--solvent != none`, add solvent `dE/dF/dH` on real-atom block
5. if `--embedcharge`, add embedding `dE/dF/dH` including MM point-charge terms
6. convert to Gaussian units and write external output

Available solvent models:
- `--solvent-model alpb` -> xTB `--alpb`
- `--solvent-model cpcmx` -> xTB `--cpcmx`

For `igrd=2` (`freq`), the solvent-corrected Hessian is required.
