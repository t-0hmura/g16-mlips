# OPTIONS.md (g16-mlips)

For most users, defaults are sufficient.

> **Note:** UMA and MACE currently conflict at dependency level (`e3nn`). Use separate environments.

## Common Options (all backends)

- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--list-models`
- `--version`

`readfc` や `%oldchk`, `nomicro`, `freq`, `calcfc`, `ts` などの挙動は Gaussian 入力の制御項目であり、CLIオプションではありません。`readfc` は `opt`/`irc` で既存の `.chk` から Hessian を再利用する場合に使用します（詳細は `README.md`）。

## Server Options

The model server starts automatically on first use and stops after idle timeout. These options are for advanced use only.

- `--no-server` — Disable auto server; load model directly each time.
- `--server-socket <path>` — Manual socket path.
- `--stop-server` — Send shutdown to a running server.
- `--server-idle-timeout <int>` — Idle timeout in seconds (default: 600).

Auto-started servers are scoped per parent Gaussian process and stop automatically when that parent exits.

## UMA Options (`uma` / `g16-mlips-uma`)

- `--task <omol|omat|odac|oc20|oc25|omc>`
- `--list-tasks`
- `--workers <int>` — Predictor worker count.
- `--workers-per-node <int>` — Worker cap per node.
- `--max-neigh <int>` — Override graph neighbor cap.
- `--radius <float>` — Override graph cutoff radius (Angstrom).
- `--r-edges` — Enable distance edge attributes.
- `--otf-graph` / `--no-otf-graph` — Toggle OTF graph collation (default: on).

## OrbMol Options (`orb` / `g16-mlips-orb`)

Only conservative Orb models are supported.

- `--precision <str>` (default: `float32-high`)
- `--compile-model`
- `--loader-opt KEY=VALUE` (repeatable) — Extra kwargs for Orb loader.
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for `ORBCalculator`.

## MACE Options (`mace` / `g16-mlips-mace`)

- `--dtype float32|float64` (default: `float64`)
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for MACE calculator.

## `KEY=VALUE` Parsing Rules

For `--loader-opt` / `--calc-opt`:

- `true` / `false` -> boolean
- `none` / `null` -> `None`
- integer/float strings -> numeric type
- otherwise -> string
