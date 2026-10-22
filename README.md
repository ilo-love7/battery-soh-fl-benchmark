<p align="center"><b>English</b> · <a href="./README.zh-CN.md">简体中文</a></p>

# battery-soh-fl-benchmark

**A federated-learning research plan and executable harness for estimating battery SOH from crowdsourced charging data — without uploading the raw curves.**

[![Stage](https://img.shields.io/badge/stage-plan%20only%20·%20no%20results-yellow)](./docs/plan.md)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue)](./pyproject.toml)
[![flwr](https://img.shields.io/badge/flwr-1.41.0-blue)](./docs/plan.md)
[![torch](https://img.shields.io/badge/torch-2.10.0-blue)](./docs/plan.md)
[![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)](./tests)
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
| **`docs/plan.md`** | The full execution plan (v2.0, 17 sections): falsifiable hypotheses H1–H4, module contracts, partitioning rules, risk register, milestone DODs |
| **`src/flsoh/partition.py`** | Cell-level partitioning + leakage detection. Pure standard library — no numpy, no torch |
| **`src/flsoh/protocol.py`** | Experiment-matrix validation. Refuses configs missing a centralized baseline, missing an IID control, or with fewer than 3 seeds |
| **`scripts/check_partition.py`** | Audits a partition file for cross-client leakage, test-set contamination, empty clients |
| **`scripts/check_protocol.py`** | Audits a TOML matrix before you burn GPU-hours on it |
| **`configs/mvp.toml`, `configs/full.toml`** | The machine-readable experiment matrix: 124 MIT-Stanford cells, 24 held out, k ∈ {2, 5}, α scan, DP scan |
| **`tests/`** | 87 tests. Every checker is proven to *fail* on a broken input — a checker that only ever prints PASS is worse than none |

## Quick start

```bash
# Python ≥ 3.11, no dependencies for the checkers
python3 scripts/check_protocol.py --config configs/mvp.toml
python3 scripts/check_partition.py --config configs/mvp.toml
```

Expected output:

```text
PASS configs/mvp.toml  (mvp)  errors=0 warnings=0
  客户端数：k=5→N=20
  协议齐备：划分原子、iid 对照、centralized 基线、seed 数、判据阈值均在场
mvp.toml  cells=124  test=24
  PASS  k=5 iid                    N=20   5颗×20
  PASS  k=5 dirichlet alpha=0.5    N=20   1颗×7 2颗×2 3颗×3 5颗×2 6颗×1 7颗×1 9颗×2 14颗×1 25颗×1

全 PASS：无跨客户端泄漏、无测试集污染、无空客户端、无未覆盖电池。
```

Audit a partition that was actually used in a run, not just the generator:

```bash
python3 scripts/check_partition.py --config configs/mvp.toml \
    --partition-json results/partition.json
```

## The five checks that gate every experiment

| Code | What it catches | Cost of missing it |
|---|---|---|
| `P-LEAK` | One cell assigned to two clients | Inflated metrics; conclusions void |
| `P-TEST` | A global test cell inside any client | Same, and much harder to notice |
| `P-EMPTY` | A client with zero cells | Broken aggregation weights |
| `P-COVER` | A cell in neither clients nor test set | Wasted data |
| `P-COUNT` | Actual client count ≠ declared N | The N in your paper is wrong |
| `P-UNKNOWN` | An ID not in the dataset manifest | Wrong data source entirely |

## What is deliberately absent

- **No results.** See above.
- **No `data.py` / `model.py` / Flower apps yet.** They need torch + flwr, so they stay out of CI and out of this first commit. The module contracts in `docs/plan.md` §3.2 are the spec.
- **No absolute accuracy target.** Conclusions are reported as *relative* degradation vs. the centralized baseline (C0). See `docs/plan.md` §11.
- **No claim of deployability.** This is a simulation. `docs/plan.md` §14 pitfall 6 is about exactly that.

## Documentation

- **Execution plan (v2.0)** — [`docs/plan.md`](./docs/plan.md), the primary document
- **v1 archive** — [`docs/archive/plan-v1.md`](./docs/archive/plan-v1.md), kept for diffing
- **Flower version notes** — `docs/plan.md` §2: `flwr` 1.41.0 uses `ClientApp` + `@app.train()` + `flwr.serverapp.strategy`, not `NumPyClient`

## Contributing

Contributions are welcome, with two rules that are not negotiable:

1. **Every hard constraint gets an executable check.** If you add a rule to the plan, add the checker or a test that fails when the rule is broken. "We should verify X" without a checker is how `docs/archive/plan-v1.md` came to be.
2. **No numbers without runs.** §10.2 stays `—` until `results/*.json` exists.

```bash
python3 -m pytest tests -v
```

## License

[MIT](./LICENSE). Datasets are governed by their own licenses — see [`data/README.md`](./data/README.md).
