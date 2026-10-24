<p align="center"><a href="./README.md">English</a> · <b>简体中文</b></p>

# battery-soh-fl-benchmark

**面向众包充电数据的电池 SOH 估计联邦学习研究平台——原始充电曲线不上传，只联合训练模型。**

[![Stage](https://img.shields.io/badge/stage-second%20batch%20·%20results%20provisional-yellow)](./docs/plan.md)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue)](./pyproject.toml)
[![flwr](https://img.shields.io/badge/flwr-1.41.0-blue)](./docs/plan.md)
[![torch](https://img.shields.io/badge/torch-2.10.0-blue)](./docs/plan.md)
[![Tests](https://img.shields.io/badge/tests-210%20passed-brightgreen)](./tests)
[![CI](https://github.com/ilo-love7/battery-soh-fl-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/ilo-love7/battery-soh-fl-benchmark/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

> **状态：第二批实验已完成（2026-10-24），结果仍暂定。** 每个配置补到 8 个 seed 后，
> 三个假设都有了判定：**H1 推翻**（联邦在 3.3 倍训练量下比中心化好 23.5%，等训练量下
> 差 6.4 倍——两种口径反号）、**H2 接受**（SOH 排序分片偏斜 47.1%，相对同预算 IID
> 劣化 +24.2%，p_t=0.0013）、**H3 推翻**（μ=0.01 与 μ=0.1 都差于 μ=0，不存在可用
> 内点）。Dirichlet α=0.5 记**未判定（处理未施加）**：它的标签偏斜 12.2% 低于 IID
> 基线 17.3%。M3 已补上偏斜扫描（k=1/2/5/10，每档配同 k IID 对照）与 **FedAdam**：
> 服务端动量补回了 `local_epochs=3` 损失的 86.1%（p_t=0.0056，8/8 同向），用三分之
> 一的训练量达到中心化水平的 MAE；但本地 epoch 已经够用时没有额外收益。FedBN 暂缓
> （模型刻意不含 BatchNorm，且本地没有第二个化学体系的 ΔQ(V) 特征）。
> 表里每个数都由 CI 从 `results/*.json` 重新生成，手改不了。

---

## 为什么要有这个仓库

大多数联邦学习教程止步于「MNIST 分给 10 个客户端，效果不错」。
电池 SOH 估计是难得多的情况，而且真正的问题不是联邦训练**能不能跑通**，
而是**原始充电曲线不出设备的前提下，要付出多少精度代价**：

- 数据按**电池**划分，Non-IID 是结构自带的，不是人为注入的。
- 整个设定（众包 App 只采充电片段）决定了**全局归一化本身就是隐私泄露**——
  每个客户端必须本地标准化。
- 按电池划分让**序列泄漏**变得极易发生且难以察觉：一颗电池被分进两个客户端，
  指标会虚高，而精度曲线上看不出任何异常。

所以这个仓库把**检查器**当作交付物，而不是训练循环。

## 这里已经有什么

| | |
|---|---|
| **`docs/plan.md`** | 完整执行方案（v2.1，17 节）：可证伪假设 H1–H4、模块契约、划分规则、风险登记册、里程碑 DOD |
| **`src/flsoh/partition.py`** | 电池级划分 + 泄漏检测。纯标准库——不依赖 numpy，也不依赖 torch |
| **`src/flsoh/protocol.py`** | 实验矩阵校验。缺 centralized 基线、缺 IID 对照、seed 少于 3 个，一律拒绝 |
| **`src/flsoh/model.py`** | MLP 1000→128→64→1。`parameter_bytes()` / `wire_bytes()` 把通信量表算出来，不手抄 |
| **`src/flsoh/matr.py`** | 7.7 GiB MATR `.mat` 的远端随机读：HTTP Range + h5py `fileobj`。每批只拉约 33 MB，不是 3 GB |
| **`src/flsoh/data.py`** | 特征表读写、按客户端的 DataLoader，以及把「不许全局标准化」从注释变成断言的 `LocalStandardizer` |
| **`src/flsoh/app/`** | Flower 1.41 客户端/服务端 App（`ClientApp` + `@app.train()` + `flwr.serverapp.strategy`） |
| **`scripts/check_continuations.py`** | 用四条独立证据证明批次 2 的 5 颗是批次 1 的延续段。配错一对就会静默删掉正常电池 |
| **`scripts/build_manifest.py`** | **从数据**生成 `cells.txt` / `manifest.json`，让 `P-UNKNOWN` 不可能对着虚构 ID 说 PASS |
| **`scripts/build_features.py`** | 生成 per-cycle 的 ΔQ(V) 特征表，**可断点续跑**——每颗落一块 parquet 分片，中断后重跑同一条命令即可。丢弃的曲线逐条记进 `features.spec.json` |
| **`scripts/check_partition.py`** | 对账划分文件：跨客户端泄漏、测试集污染、空客户端 |
| **`scripts/check_protocol.py`** | 在烧掉 GPU 小时之前先校验实验矩阵 |
| **`scripts/run_local_baseline.py`** | C0（中心化上限）与 C1（本地独立下限）基线——不需要联邦运行时 |
| **`configs/mvp.toml`、`configs/full.toml`** | 机器可读的实验矩阵：MIT-Stanford 135 颗电池、留出 20 颗、k ∈ {2, 5}、α 扫描、DP 扫描 |
| **`tests/`** | 275 条测试。每个检查器都验证过它在坏输入上**确实会失败**——永远只打印 PASS 的检查器比没有检查器更糟 |

## 快速开始

```bash
# Python ≥ 3.11，检查器零第三方依赖
python3 scripts/check_protocol.py --config configs/mvp.toml
python3 scripts/check_partition.py --config configs/mvp.toml
```

预期输出：

```text
PASS configs/mvp.toml  (mvp)  errors=0 warnings=0
  客户端数：k=5→N=23
  协议齐备：划分原子、iid 对照、centralized 基线、seed 数、判据阈值均在场
mvp.toml  cells=135  test=20
  PASS  k=5 iid                    N=23   5颗×23
  PASS  k=5 dirichlet alpha=0.5    N=23   1颗×8 2颗×3 3颗×2 4颗×1 5颗×2 6颗×1 8颗×2 9颗×1 14颗×1 15颗×1 21颗×1

全 PASS：无跨客户端泄漏、无测试集污染、无空客户端、无未覆盖电池。
```

生成一次运行用的划分，再对账**实际用过的那份文件**：

```bash
python3 scripts/check_partition.py --config configs/mvp.toml \
    --cells-file data/mit-stanford/cells.txt \
    --emit results/partition.json --emit-k 5 --emit-scheme iid
python3 scripts/check_partition.py --config configs/mvp.toml \
    --cells-file data/mit-stanford/cells.txt \
    --partition-json results/partition.json
```

生成时 `--cells-file` 不能省：省了它会退回占位 ID（`cell-000`…），
于是对账报出「135 颗电池不在清单内」——那个错指向划分，而真实原因是漏了参数。

跑基线（需要 `pip install -e ".[train]"`）：

```bash
python3 scripts/run_local_baseline.py --config configs/mvp.toml \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition.json --out results/
```

跑一档联邦配置，再把所有 run 汇总成表 1：

```bash
python3 scripts/run_federated.py --config configs/mvp.toml \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition.json \
    --strategy fedavg --rounds 100 --seed 0 --out results/
python3 scripts/summarize_results.py --results results/ --paired
```

`--paired` 按 seed 相减，比较两档配置的配对差。本仓库实测 seed 间波动为 5%–63%，
在这个噪声水平下它是唯一有分辨力的比较方式。

`docs/plan.md` §10.2 的表 1 是**生成区**，不要手改：

```bash
python3 scripts/summarize_results.py --write-plan docs/plan.md  # 重新生成
python3 scripts/summarize_results.py --check-plan docs/plan.md  # 核对（CI 会跑）
```

`--check-plan` 会从 `results/*.json` 重算一遍表，与文档差一个数字就报错。
结果 JSON 与更新后的 `docs/plan.md` 必须同一次提交——没有原始 run 的表无法被验证。

### 判 H2 之前先量标签偏斜

**数量偏斜不等于标签偏斜。** 本数据集上 Dirichlet α=0.5 造出的标签偏斜
反而比 IID 更小（客户端身份解释的标签方差 12.2% vs 17.3%）——因为一个客户端
拿走 21 颗电池，会把电池之间的 SOH 差异平均掉。在这样一份划分上报「Non-IID
没掉点」，等于在报告一个从未被施加的处理。

```bash
python3 scripts/measure_heterogeneity.py \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition-mvp-k5-iid.json \
    --partition-json results/partition-mvp-k5-dirichlet0.5.json \
    --out results/heterogeneity.json
```

然后把它传给判定脚本。**只要某份划分的偏斜不高于 IID 基线，判定就不会判
「推翻」，只会记「未判定（处理未施加）」**：

```bash
python3 scripts/summarize_results.py --judge \
    --hetero results/heterogeneity.json
```

`results/heterogeneity.json` 必须与结果一起提交。CI 会检查它是否覆盖了所有
run 实际用到的划分。

真正施加了处理的划分是 `scripts/check_partition.py --emit-scheme sohrank`
（按电池平均 SOH 排序后切连续分片）：**N、每客户端颗数、总数据量、训练预算
全都不变，只改哪些电池分给谁**，偏斜拉到 47.1%。它用到了标签，所以是压力
测试档，不是可部署场景。

### 关于 `flwr run`

`scripts/run_federated.py` 是**本地联邦循环**，不是 Flower 运行时。它存在的原因很具体：
Flower 1.41 的仿真后端只有 ray，而 ray 的 raylet 在本开发机上启动即崩——主机名是 GBK
编码的中文，protobuf 拒绝这种非 UTF-8 数据。`RAY_raylet_start_wait_time_s`、
`_node_ip_address`、`COMPUTERNAME` 覆盖三种绕法都无效。

官方运行时改在 CI 里验证（`scripts/ci_smoke_flwr.py`，Linux runner，合成数据跑 2 轮）。
两条路径共用 `flsoh/train.py`，所以可能产生差异的只有聚合，而聚合被
`tests/test_train.py` 钉住。完整排查过程见 [`docs/env-limitations.md`](docs/env-limitations.md)。

Windows 上请用 **Python 3.11 或 3.12**：`flwr[simulation]==1.41.0` 硬钉
`ray==2.55.1`，而该版本没有 cp313 的 Windows wheel。

## 每次实验前的七道检查

| 检查码 | 抓什么 | 漏掉的代价 |
|---|---|---|
| `P-LEAK` | 一颗电池被分给两个客户端 | 指标虚高，结论作废 |
| `P-TEST` | 全局测试电池混进客户端 | 同上，且更难发现 |
| `P-EMPTY` | 某客户端 0 颗电池 | 聚合权重异常 |
| `P-COVER` | 有电池既不在客户端也不在测试集 | 数据浪费 |
| `P-COUNT` | 实际客户端数与声明的 N 不符 | 论文里的 N 写错 |
| `P-UNKNOWN` | 出现数据集清单之外的电池 ID | 数据源读错了 |
| `P-CONT` | 同一颗物理电池以两个 ID 出现（跨批次延续） | `P-LEAK` 比的是 ID，结构上抓不到这种 |

## 三个拒绝手抄的数字

`docs/plan.md` 反复引用这几个数，所以由 `src/flsoh/model.py` 算出、
`tests/test_model.py` 钉住：

| 量 | 值 | 来源 |
|---|---|---|
| 参数量 | 136,449 | `expected_param_count(1000, (128, 64))` |
| 单次更新载荷 | 545,796 B | `parameter_bytes()` |
| **单次更新线上字节数** | **546,630 B** | `wire_bytes()`——flwr 的 `ArrayRecord.count_bytes()`，含 139 B/张量的 NumPy 元数据 |
| 每客户端每轮往返 | 1,091,592 B ≈ 1.04 MiB | `round_trip_bytes()` |
| N=23 × 100 轮总计 | 2,510,661,600 B ≈ 2.34 GiB | 派生值；**不是** v1 写的 2.60 GiB |

## 刻意没有的东西

- **没有官方仿真的本机运行。** ray 后端在中文主机名上直接 abort，绕行方案与等价性论证见 `docs/env-limitations.md` 与 `docs/plan.md` §14 坑 14。
- **没有绝对精度目标。** 所有结论以相对中心化基线（C0）的劣化百分比表述，见 `docs/plan.md` §11。
- **不宣称可部署。** 这是仿真。`docs/plan.md` §14 坑 6 讲的就是这件事。
- **重依赖测试不进默认 CI。** `test_model.py` / `test_data.py` / `test_app.py` 需要 torch + flwr，
  放在手动的 `heavy-tests` job 里（Actions → CI → Run workflow）。默认 CI 保持秒级可跑。

## 文档

- **执行方案（v2.1）** — [`docs/plan.md`](./docs/plan.md)，主文档
- **§3.4 API 实测校准** — 对照安装的 `flwr` 1.41.0 逐项核对的结果，含发现的两处差异
- **v1 归档** — [`docs/archive/plan-v1.md`](./docs/archive/plan-v1.md)，保留用于对比
- **Flower 版本说明** — `docs/plan.md` §2：`flwr` 1.41.0 用的是 `ClientApp` + `@app.train()` + `flwr.serverapp.strategy`，不是 `NumPyClient`

## 参与贡献

欢迎贡献，但有两条不可协商：

1. **每一条硬约束都要有可执行的检查。** 往方案里加规则，就要同时加上检查器，
   或加上一条在规则被破坏时会失败的测试。只有「我们应该验证 X」而没有检查器，
   正是 `docs/archive/plan-v1.md` 产生的原因。
2. **没有运行结果就没有数字。** §10.2 的结果表由脚本从 `results/*.json` 重算、CI 逐字核对；没有 run 就没有行，手改一个数字就红。

```bash
python3 -m pytest tests -v        # 纯标准库，CI 里跑的就是这些
pip install -e ".[train,dev]"     # 装 torch/flwr 后才能跑另外 112 条
python3 -m pytest tests -v        # 全部 275 条
```

## 许可证

[MIT](./LICENSE)。数据集遵循各自的许可条款，见 [`data/README.md`](./data/README.md)。
