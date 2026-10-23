<p align="center"><b>English</b> · <a href="./README.zh-CN.md">简体中文</a></p>

# battery-soh-fl-benchmark

**A federated-learning research plan and executable harness for estimating battery SOH from crowdsourced charging data — without uploading the raw curves.**

[![Stage](https://img.shields.io/badge/stage-plan%20only%20·%20no%20results-yellow)](./docs/plan.md)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue)](./pyproject.toml)
[![flwr](https://img.shields.io/badge/flwr-1.41.0-blue)](./docs/plan.md)
[![torch](https://img.shields.io/badge/torch-2.10.0-blue)](./docs/plan.md)
[![Tests](https://img.shields.io/badge/tests-274%20passed-brightgreen)](./tests)
[![CI](https://github.com/ilo-love7/battery-soh-fl-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/ilo-love7/battery-soh-fl-benchmark/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

> **Status: plan stage. This repository contains no experimental results.** Every number in `docs/plan.md` §10.2 is a placeholder. Filling it in before the runs exist would be fabrication, and CI enforces that.

---

## Why this repo exists

Most federated-learning tutorials stop at "MNIST across 10 clients, works great".
Battery SOH estimation is a much harsher case, and the interesting question is not
whether federated training *runs* but **how much accuracy you give up when raw
charging curves never leave the device**:

- Data is split by **battery**, so the natural Non-IID structure is *structural*, not injected.
- The whole premise (a crowdsourced app that only sees charging segments) makes **global normalization a privacy leak** — every client must standardize locally.
- Battery-level splits make **sequence leakage** trivially easy and invisible: one cell in two clients inflates your metrics, and nothing in the accuracy curve tells you so.

So this repo treats the *checkers* as the deliverable, not the training loop.

## What is actually here

| | |
|---|---|
| **`docs/plan.md`** | The full execution plan (v2.1, 17 sections): falsifiable hypotheses H1–H4, module contracts, partitioning rules, risk register, milestone DODs |
| **`src/flsoh/partition.py`** | Cell-level partitioning + leakage detection. Pure standard library — no numpy, no torch |
| **`src/flsoh/protocol.py`** | Experiment-matrix validation. Refuses configs missing a centralized baseline, missing an IID control, or with fewer than 3 seeds |
| **`src/flsoh/model.py`** | MLP 1000→128→64→1. `parameter_bytes()` / `wire_bytes()` compute the communication table instead of hand-copying it |
| **`src/flsoh/matr.py`** | Remote random-access reader for the 7.7 GiB MATR `.mat` files: HTTP Range + h5py `fileobj`. Pulls ~33 MB instead of 3 GB per batch |
| **`src/flsoh/data.py`** | Feature-table I/O, per-client loaders, and the `LocalStandardizer` that makes "no global normalization" an assertion rather than a comment |
| **`src/flsoh/app/`** | Flower 1.41 client/server apps (`ClientApp` + `@app.train()` + `flwr.serverapp.strategy`) |
| **`scripts/check_continuations.py`** | Proves that 5 batch-2 cells are continuations of 5 batch-1 cells, using four independent lines of evidence. Wrong pairing would delete healthy cells invisibly |
| **`scripts/build_manifest.py`** | Generates `cells.txt` / `manifest.json` **from the data**, so `P-UNKNOWN` cannot pass against an invented ID list |
| **`scripts/build_features.py`** | Builds the per-cycle ΔQ(V) feature table, **resumably** — one parquet part per cell, so an interrupted run resumes instead of restarting. Records every dropped curve in `features.spec.json` |
| **`scripts/check_partition.py`** | Audits a partition file for cross-client leakage, test-set contamination, empty clients |
| **`scripts/check_protocol.py`** | Audits a TOML matrix before you burn GPU-hours on it |
| **`scripts/run_local_baseline.py`** | C0 (centralized ceiling) and C1 (local-only floor) baselines — no federated runtime needed |
| **`configs/mvp.toml`, `configs/full.toml`** | The machine-readable experiment matrix: 135 MIT-Stanford cells, 20 held out, k ∈ {2, 5}, α scan, DP scan |
| **`tests/`** | 274 tests. Every checker is proven to *fail* on a broken input — a checker that only ever prints PASS is worse than none |

## Quick start

```bash
# Python ≥ 3.11, no dependencies for the checkers
python3 scripts/check_protocol.py --config configs/mvp.toml
python3 scripts/check_partition.py --config configs/mvp.toml
```

Expected output:

```text
PASS configs/mvp.toml  (mvp)  errors=0 warnings=0
  客户端数：k=5→N=23
  协议齐备：划分原子、iid 对照、centralized 基线、seed 数、判据阈值均在场
mvp.toml  cells=135  test=20
  PASS  k=5 iid                    N=23   5颗×23
  PASS  k=5 dirichlet alpha=0.5    N=23   1颗×8 2颗×3 3颗×2 4颗×1 5颗×2 6颗×1 8颗×2 9颗×1 14颗×1 15颗×1 21颗×1

全 PASS：无跨客户端泄漏、无测试集污染、无空客户端、无未覆盖电池。
```

Emit a partition for a run, then audit the file that was actually used:

```bash
python3 scripts/check_partition.py --config configs/mvp.toml \
    --emit results/partition.json --emit-k 5 --emit-scheme iid
python3 scripts/check_partition.py --config configs/mvp.toml \
    --partition-json results/partition.json
```

Run the baselines (needs `pip install -e ".[train]"`):

```bash
python3 scripts/run_local_baseline.py --config configs/mvp.toml \
    --features data/mit-stanford/features.csv \
    --partition-json results/partition.json --out results/
```

## The seven checks that gate every experiment

| Code | What it catches | Cost of missing it |
|---|---|---|
| `P-LEAK` | One cell assigned to two clients | Inflated metrics; conclusions void |
| `P-TEST` | A global test cell inside any client | Same, and much harder to notice |
| `P-EMPTY` | A client with zero cells | Broken aggregation weights |
| `P-COVER` | A cell in neither clients nor test set | Wasted data |
| `P-COUNT` | Actual client count ≠ declared N | The N in your paper is wrong |
| `P-UNKNOWN` | An ID not in the dataset manifest | Wrong data source entirely |
| `P-CONT` | One physical cell under two IDs (cross-batch continuation) | `P-LEAK` compares IDs, so it structurally cannot catch this |

## Three numbers this repo refuses to hand-copy

`docs/plan.md` quotes these repeatedly, so `src/flsoh/model.py` computes them and
`tests/test_model.py` pins them:

| Quantity | Value | Source |
|---|---|---|
| Parameters | 136,449 | `expected_param_count(1000, (128, 64))` |
| Payload per update | 545,796 B | `parameter_bytes()` |
| **Wire size per update** | **546,630 B** | `wire_bytes()` — flwr's `ArrayRecord.count_bytes()`, 139 B/tensor of NumPy metadata |
| Round trip per client/round | 1,091,592 B ≈ 1.04 MiB | `round_trip_bytes()` |
| Total, N=23 × 100 rounds | 2,510,661,600 B ≈ 2.34 GiB | derived; **not** the 2.60 GiB v1 claimed |

## What is deliberately absent

- **No results.** See above.
- **No absolute accuracy target.** Conclusions are reported as *relative* degradation vs. the centralized baseline (C0). See `docs/plan.md` §11.
- **No claim of deployability.** This is a simulation. `docs/plan.md` §14 pitfall 6 is about exactly that.
- **The heavy tests are not in default CI.** `test_model.py` / `test_data.py` / `test_app.py` need torch + flwr, so they run in the manual `heavy-tests` job (Actions → CI → Run workflow). Default CI stays runnable in seconds.

## Documentation

- **Execution plan (v2.1)** — [`docs/plan.md`](./docs/plan.md), the primary document
- **§3.4 API calibration** — what was verified against the installed `flwr` 1.41.0, including the two discrepancies found
- **v1 archive** — [`docs/archive/plan-v1.md`](./docs/archive/plan-v1.md), kept for diffing
- **Flower version notes** — `docs/plan.md` §2: `flwr` 1.41.0 uses `ClientApp` + `@app.train()` + `flwr.serverapp.strategy`, not `NumPyClient`

## Contributing

Contributions are welcome, with two rules that are not negotiable:

1. **Every hard constraint gets an executable check.** If you add a rule to the plan, add the checker or a test that fails when the rule is broken. "We should verify X" without a checker is how `docs/archive/plan-v1.md` came to be.
2. **No numbers without runs.** §10.2 stays `—` until `results/*.json` exists.

```bash
python3 -m pytest tests -v        # stdlib-only, runs in CI
pip install -e ".[train,dev]"     # for the torch/flwr tests
python3 -m pytest tests -v        # all 200
```

## License

[MIT](./LICENSE). Datasets are governed by their own licenses — see [`data/README.md`](./data/README.md).
