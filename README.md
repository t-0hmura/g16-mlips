# mlips4g16

Gaussian16 `External` キーワード用の MLIP プラグイン集です。

提供プラグイン:
- `plugins/uma_g16.py` (FAIR-Chem / UMA)
- `plugins/orbmol_g16.py` (orb-models / OrbMol)
- `plugins/mace_g16.py` (MACE)

## 1. 仕様調査の根拠

この実装は以下に基づいています。
- Gaussian External仕様（ローカル同梱ドキュメント）:
  - `/home/apps/g16_C02_zen3/g16/doc/extern.txt`
  - `/home/apps/g16_C02_zen3/g16/doc/extgau`
- UMA モデル/API:
  - https://github.com/facebookresearch/fairchem
- OrbMol モデル/API:
  - https://github.com/orbital-materials/orb-models
- MACE モデル/API:
  - https://github.com/ACEsuit/mace

## 2. インストール

```bash
cd /data2/tohmura/pdb2reaction_workspace/mlips4g16
python3 -m pip install -r requirements.txt
```

追加依存（バックエンドごと）:
- UMA: `pip install fairchem-core`
- OrbMol: `pip install orb-models`
- MACE: `pip install mace-torch`

## 3. Gaussian入力例

```text
%chk=water_ext.chk
#p external="/data2/tohmura/pdb2reaction_workspace/mlips4g16/plugins/uma_g16.py --model uma-s-1p1 --task omol --hessian-mode Analytical" opt

Water external UMA example

0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
```

OrbMol/MACEを使う場合は `external=".../orbmol_g16.py ..."` または `external=".../mace_g16.py ..."` に置換します。

## 4. モデル指定

### UMA
```bash
python3 plugins/uma_g16.py --list-models
python3 plugins/uma_g16.py --list-tasks
```
- デフォルト: `uma-s-1p1`
- `--list-models` は、インストール済み `fairchem` の `available_models` を優先し、未導入時はリポジトリ由来のフォールバック一覧を返します。

### OrbMol
```bash
python3 plugins/orbmol_g16.py --list-models
```
- デフォルト: `orb_v3_conservative_omol`
- `-` と `_` の両形式を受理します（例: `orb-v3-conservative-omol`, `orb_v3_conservative_omol`）。

### MACE
```bash
python3 plugins/mace_g16.py --list-models
```
- デフォルト: `MACE-OMOL-0` (内部的に `omol:extra_large` と同義)

MACEは以下を受け付けます。
- `MACE-OMOL-0` (OMOL-0のデフォルトエイリアス)
- `mp:<alias>` / `<alias>` 例 `mp:medium-mpa-0`, `medium-mpa-0`
- `off:<alias>` 例 `off:medium`
- `off-small|off-medium|off-large`
- `omol:extra_large`
- `anicc`
- ローカルモデルパス / URL

## 5. Hessianモード

`IGrd=2`（Gaussian が Hessian を要求するケース）では、以下を選択できます。
- `--hessian-mode Analytical`
- `--hessian-mode Numerical`

`--strict-hessian` を付けると、Analytical失敗時にNumericalへフォールバックせずエラーにします。

## 6. ジョブ投入

`run.sh` を編集して投入:

```bash
cd /data2/tohmura/pdb2reaction_workspace/mlips4g16
qsub run.sh
```

`run.sh` には以下を含めています。
- `. /home/apps/Modules/init/profile.sh`
- `module load gaussian16.C02`
- `module load orca/6.1.1`
