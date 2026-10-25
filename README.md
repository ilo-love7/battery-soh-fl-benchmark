<p align="center"><b>English</b> · <a href="./README.zh-CN.md">简体中文</a></p>

# battery-soh-fl-benchmark

**A federated-learning research plan and executable harness for estimating battery SOH from crowdsourced charging data — without uploading the raw curves.**

[![Stage](https://img.shields.io/badge/stage-third%20batch%20·%20results%20provisional-yellow)](./docs/plan.md)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue)](./pyproject.toml)
[![flwr](https://img.shields.io/badge/flwr-1.41.0-blue)](./docs/plan.md)
[![torch](https://img.shields.io/badge/torch-2.10.0-blue)](./docs/plan.md)
[![Tests](https://img.shields.io/badge/tests-702-blue)](./tests)
[![CI](https://github.com/ilo-love7/battery-soh-fl-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/ilo-love7/battery-soh-fl-benchmark/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

> **Status: third experimental batch complete (2026-10-25), results still provisional.**
> With 8 seeds per configuration all three hypotheses are now settled:
> **H1 rejected** (federated beats centralized by 23.5% at 3.3× the training
> budget; at equal budget it is 6.4× worse — the two framings disagree in sign),
> **H2 accepted** (SOH-sorted shards at 47.1% label skew cost +24.2% against a
> same-budget IID control, p_t=0.0013), **H3 rejected** (μ=0.01 and μ=0.1 are
> both worse than μ=0, with no usable interior point). Dirichlet α=0.5 stays
> *undecided (treatment not applied)*: its label skew (12.2%) is below the IID
> baseline (17.3%).
> M3 added the skew sweep (k=1/2/5/10, each tier against a same-k IID control)
> and **FedAdam**: server-side momentum recovers 86.1% of the accuracy lost to
> `local_epochs=3` (p_t=0.0056, 8/8 seeds) and reaches centralized-level MAE at
> 1/3.3 of the training budget — but adds nothing once local epochs suffice.
> FedBN is deferred (the model deliberately has no BatchNorm, and no second
> chemistry's ΔQ(V) features are available locally).
> **M4 added local DP-SGD** (`docs/memo-m4.md`): H4 splits in two. Monotonicity
> is **supported** (MAE rises strictly with the noise multiplier, 8/8 seeds,
> p_sign=0.0078), but usability at ε ≤ 8 is **rejected** — the only arm that
> reaches it (nm=8, ε=7.84) degrades +186.3% against the mechanism control and
> is **34.7% worse than not federating at all**. ε is not a free variable:
> q=0.5 and 600 steps put nm=1 at ε=152.31, so the DOD's literal "ε ≤ 8 at
> noise=1.0" is structurally unreachable under this protocol.
> Every number in the table is regenerated from `results/*.json` by CI,
> so nothing in it can be hand-edited.
>
> **M5 (current) found a data defect that invalidates those numbers.** The
> feature extractor was feeding the **whole cycle** (charge + discharge) into
> the voltage-sorted interpolator, so the discharge segment — whose `Qc` channel
> is constant — was scattered into the charge segment's voltage range. Measured
> on real data: total variation 269.60 vs 0.21 for the correct pipeline, and a
> correlation of **−0.019** with the true ΔQ(V). The table above was computed on
> that broken table. It is being rebuilt (`scripts/check_features.py` now gates
> every build at TV/amplitude ≤ 20, and the extraction version is part of the
> resume fingerprint so old shards cannot be silently reused). Methodology
> conclusions — training-budget accounting, seed-noise magnitude, the ε
> derivation, "stability ≠ correctness" — are unaffected; every MAE is pending
> re-verification. See `docs/report.md` §0.

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
| **`src/flsoh/dp.py`** | Opacus DP-SGD wrapper. One accountant **per client**, never multiplied by N (the joint mechanism is a product, so `ε_joint = max_i ε_i`) — and the accountant must outlive the rounds, or ε is undercounted 100× |
| **`src/flsoh/schema.py`** | Schema v1, the only contract between the phone app and the simulation. Pure standard library. `session_fingerprint` hashes the whole session, not its ID — a reinstalled app reuses IDs |
| **`src/flsoh/segments.py`** | The app-side path: raw records → discharge segments → (V, Q) pairs, then straight into `delta_q_curve` — the same function `build_features.py` calls, per plan §8.2 |
| **`src/flsoh/simulate.py`** | Synthetic app sessions with **ground truth**. Deliberately simple physics (OCV power law, IR sag, √-fade) — its job is to exercise the pipeline, not to prove the method |
| **`src/flsoh/ingest.py`** | Validate → deduplicate → assemble. Deduplication is by **content fingerprint**, not session ID: a reinstalled app reuses IDs |
| **`scripts/check_features.py`** | The gate that would have caught the 2026-10-25 feature bug: per-row total variation / amplitude. Correct data scores 5.9, mis-paired points score 122.4 |
| **`scripts/simulate_sessions.py`** | Emits Schema v1 JSONL + a truth file. Validates its own output against the schema — synthetic data does not get a pass |
| **`scripts/run_e2e.py`** | End-to-end: sessions → ingest → features → smoothness gate → split → federated training → results JSON. Five gates, each with a non-zero exit code. Reports a constant baseline alongside MAE, because MAE alone is unreadable |
| **`scripts/rerun_all.py`** | Reconstructs each run's command line from its result JSON and replays all 172 runs. Refuses to run when the assembled feature table is stale, when the interpreter's torch version differs from the recorded one, or when the thread count differs — all three change the numbers without changing the output's shape |
| **`scripts/check_schema.py`** | Validates a JSONL session dump against Schema v1. Exit 1 for a bad session, 2 for unparseable input — never a silent skip |
| **`scripts/check_dp_budget.py`** | Computes ε **before** you sweep `noise_multiplier`. ε is not a free variable: `q` and the step count are fixed by the protocol, and nm=1 lands at ε=152.31 here |
| **`scripts/check_continuations.py`** | Proves that 5 batch-2 cells are continuations of 5 batch-1 cells, using four independent lines of evidence. Wrong pairing would delete healthy cells invisibly |
| **`scripts/build_manifest.py`** | Generates `cells.txt` / `manifest.json` **from the data**, so `P-UNKNOWN` cannot pass against an invented ID list |
| **`scripts/build_features.py`** | Builds the per-cycle ΔQ(V) feature table, **one parquet part per cell** — an interrupted run resumes instead of restarting. Records every dropped curve in `features.spec.json` |
| **`scripts/check_partition.py`** | Audits a partition file for cross-client leakage, test-set contamination, empty clients |
| **`scripts/check_protocol.py`** | Audits a TOML matrix before you burn GPU-hours on it |
| **`scripts/run_local_baseline.py`** | C0 (centralized ceiling) and C1 (local-only floor) baselines — no federated runtime needed |
| **`configs/mvp.toml`, `configs/full.toml`, `configs/dp.toml`** | The machine-readable experiment matrix: 135 MIT-Stanford cells, 20 held out, k ∈ {2, 5}, α scan, DP scan |
| **`tests/`** | 653 test functions across 25 files. Every checker is proven to *fail* on a broken input — a checker that only ever prints PASS is worse than none |

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
    --cells-file data/mit-stanford/cells.txt \
    --emit results/partition.json --emit-k 5 --emit-scheme iid
python3 scripts/check_partition.py --config configs/mvp.toml \
    --cells-file data/mit-stanford/cells.txt \
    --partition-json results/partition.json
```

`--cells-file` is not optional when you emit: without it the checkers fall back
to placeholder IDs (`cell-000`…), and the audit will then report 135 unknown
cells — pointing at the partition instead of at the missing flag.

Run the baselines (needs `pip install -e ".[train]"`):

```bash
python3 scripts/run_local_baseline.py --config configs/mvp.toml \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition.json --out results/
```

Run one federated configuration, then summarize everything into Table 1:

```bash
python3 scripts/run_federated.py --config configs/mvp.toml \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition.json \
    --strategy fedavg --rounds 100 --seed 0 --out results/
python3 scripts/summarize_results.py --results results/ --paired
```

`--paired` subtracts per-seed differences between configurations. With the
seed-to-seed spread measured here (5%–63%), that is the only comparison with
enough resolution to settle a hypothesis.

Table 1 in `docs/plan.md` §10.2 is a **generated block**. Do not hand-edit it:

```bash
python3 scripts/summarize_results.py --write-plan docs/plan.md  # regenerate
python3 scripts/summarize_results.py --check-plan docs/plan.md  # verify (CI runs this)
```

`--check-plan` recomputes the table from `results/*.json` and fails if the
document differs by even one digit. Commit the result JSONs and the updated
`docs/plan.md` together — a table without its raw runs cannot be verified.

### Before judging H2, measure the label skew

A partition being *size*-skewed does not make it *label*-skewed. On this dataset
Dirichlet α=0.5 produces **less** label skew than IID (12.2% vs 17.3% of label
variance explained by client identity), because one client taking 21 cells
averages the between-cell SOH differences away. Reporting "Non-IID did not hurt"
on that partition would be reporting on a treatment that was never applied.

```bash
python3 scripts/measure_heterogeneity.py \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition-mvp-k5-iid.json \
    --partition-json results/partition-mvp-k5-dirichlet0.5.json \
    --out results/heterogeneity.json
```

Then pass it to the judge, which refuses to reject H2 on any partition whose
skew does not exceed the IID baseline:

```bash
python3 scripts/summarize_results.py --judge \
    --hetero results/heterogeneity.json
```

`results/heterogeneity.json` must be committed alongside the results. CI checks
that it covers every partition any run actually used.

The partition that *does* apply the treatment is
`scripts/check_partition.py --emit-scheme sohrank`, which shards cells by their
mean SOH. N, cells per client, total data and training budget are all unchanged
— only *which* cells go where changes — and the skew rises to 47.1%. It uses the
labels, so it is a stress test, not a deployable scenario.

### ε is not a free variable — compute the budget before sweeping `noise_multiplier`

`noise_multiplier` is the knob you turn; ε is what you report. They are not
freely exchangeable. ε is determined by three quantities, two of which the
training protocol pins down:

    q     = 1 / ceil(rows per client / batch_size)   # 1/2 = 0.5 here
    steps = rounds × local_epochs × batches          # 100 × 3 × 2 = 600
    ε     = RDP(q, noise_multiplier, steps) at δ

With q=0.5, subsampling barely amplifies anything, so ε runs into the hundreds
fast. Sweeping {0.5, 1, 2} without checking first yields three arms that are all
ε > 8 — and the acceptance criterion ("≤ 30% degradation at ε ≤ 8") can then
never be applied to any of them.

```bash
python3 scripts/check_dp_budget.py --config configs/dp.toml
```

Two further traps that this repo pays for explicitly:

- **The accountant is per-client and is never multiplied by N.** The joint
  mechanism is a *product* of mechanisms: changing one record in client *i*
  changes only client *i*'s update (the other N−1 are bit-identical), so
  `ε_joint = max_i ε_i`. Multiplying by N inflates the guarantee 23× and *looks*
  more rigorous — which is what makes it hard to catch.
- **`noise_multiplier=0` is a control arm, not a DP arm.** It still does
  per-sample clipping and Poisson sampling but adds no noise, so ε = ∞. Without
  it you cannot separate "what DP cost in total" from "what the noise alone
  cost" — here the mechanism costs +39.2% (not significant) and the noise adds
  another +82%–186% on top.

Full analysis: [`docs/memo-m4.md`](./docs/memo-m4.md).

### On `flwr run`

`scripts/run_federated.py` is a **local federation loop**, not the Flower
runtime. It exists because Flower 1.41's simulation backend is ray-only, and
ray's raylet aborts on this development machine: the hostname is GBK-encoded
Chinese, and protobuf rejects it as invalid UTF-8. `RAY_raylet_start_wait_time_s`,
`_node_ip_address` and a `COMPUTERNAME` override all fail to work around it.

The official runtime is exercised in CI instead
(`scripts/ci_smoke_flwr.py`, Linux runner, 2 rounds on synthetic data). Both
paths share `flsoh/train.py`, so the only thing that can differ is aggregation,
and aggregation is pinned by `tests/test_train.py`. See
[`docs/env-limitations.md`](docs/env-limitations.md) for the full trace.

On Windows, use **Python 3.11 or 3.12**: `flwr[simulation]==1.41.0` pins
`ray==2.55.1`, which ships no cp313 Windows wheel.

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

- **No run of the official simulation on this machine.** The ray backend aborts on a non-ASCII hostname; see `docs/env-limitations.md` and `docs/plan.md` §14 pitfall 14 for the workaround and the equivalence argument.
- **No absolute accuracy target.** Conclusions are reported as *relative* degradation vs. the centralized baseline (C0). See `docs/plan.md` §11.
- **No claim of deployability.** This is a simulation. `docs/plan.md` §14 pitfall 6 is about exactly that.
- **The heavy tests are not in default CI.** `test_model.py` / `test_data.py` / `test_app.py` need torch + flwr, so they run in the manual `heavy-tests` job (Actions → CI → Run workflow). Default CI stays runnable in seconds.

## Documentation

- **Execution plan (v2.1)** — [`docs/plan.md`](./docs/plan.md), the primary document
- **Report** — [`docs/report.md`](./docs/report.md): threat model, protocol, results, and the simulated-vs-deployed gap (quantified: 2 mV of voltage noise costs 46.6% of the ΔQ amplitude, vs 5.6% for 10 s sampling alone)
- **§3.4 API calibration** — what was verified against the installed `flwr` 1.41.0, including the two discrepancies found
- **v1 archive** — [`docs/archive/plan-v1.md`](./docs/archive/plan-v1.md), kept for diffing
- **Milestone memos** — `docs/memo-m2.md` (H1–H3), `docs/memo-m3.md` (skew sweep + FedAdam), `docs/memo-m4.md` (local DP)
- **Figures** — `docs/figures/fig1-rounds-mae.png` (H1), `fig2-skew-mae.png` (H2), `fig3-privacy-utility.png` (H4); each is generated by its `scripts/make_figure*.py`, never hand-drawn
- **Flower version notes** — `docs/plan.md` §2: `flwr` 1.41.0 uses `ClientApp` + `@app.train()` + `flwr.serverapp.strategy`, not `NumPyClient`

## Contributing

Contributions are welcome, with two rules that are not negotiable:

1. **Every hard constraint gets an executable check.** If you add a rule to the plan, add the checker or a test that fails when the rule is broken. "We should verify X" without a checker is how `docs/archive/plan-v1.md` came to be.
2. **No numbers without runs.** §10.2 stays `—` until `results/*.json` exists.

```bash
python3 -m pytest tests -v        # stdlib-only: the default CI job (no numpy/flwr installed)
pip install -e ".[train,dev]"     # for the torch/flwr/numpy tests
python3 -m pytest tests -v        # everything
```

The badge counts **test functions in the repo**, not a run — the number
`pytest` reports depends on which optional dependencies happen to be
installed (a bare Python collects ~300, a full install ~490). A test
(`tests/test_readme_badge.py`) recomputes the static count and fails if the
badge drifts.

## License

[MIT](./LICENSE). Datasets are governed by their own licenses — see [`data/README.md`](./data/README.md).
