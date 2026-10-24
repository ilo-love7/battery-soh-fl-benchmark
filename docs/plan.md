# 联邦学习式充电数据众包研究平台 — 执行方案 v2.1

> 基于 v1.0（严谨-审稿专家，2026-10-18）修订 ｜ 2026-10-22，v2.1 修订 2026-10-23
> 面向已有 Python 与模型训练基础的单人执行者。数据口径与既有锂电池项目（MIT-Stanford / NASA PCoE / CALCE）保持一致。
> 本文件是**方案**，不是结果报告。仓库里目前没有任何实验数字，§10 的结果表全部为空，填入时间见 §13。

---

## 相对 v1.0 改了什么（v2.0 起，v2.1 续）

| # | 改动 | 为什么 |
|---|---|---|
| 1 | Flower 版本 1.9.1 → **1.41.0**，API 骨架整体重写 | v1 锁的 1.9.1 是 2024 年的版本。当前 `flwr` 已到 1.41.0，且范式从 `NumPyClient` + `fl.server.strategy` 换成了 `ClientApp` + `@app.train()` + `flwr.serverapp.strategy` + `ArrayRecord/MetricRecord`。照 v1 的代码写，第一天就会卡在 ImportError |
| 2 | 明确 Python **≥3.11** | `flwr` 从 1.31 起要求 `>=3.11,<4.0`。v1 没写，用 3.10 会装不上 |
| 3 | 通信量 0.6 MB → **545,796 B（0.52 MiB）** | v1 是估算值。按 1000→128→64→1 精确算：136,449 参数 × 4 B = 545,796 B。表 2 要拿这个数乘，不能从估算值起算 |
| 4 | 明确 **FedBN 与 Scaffold 都不是内置策略** | v1 只说了 Scaffold 无开箱实现。实际 FedBN 也在 `baselines/` 里（参考实现），内置的只有 FedAvg/FedAvgM/FedProx/FedAdam/FedYogi/FedAdagrad/FedOpt/FedMedian/FedTrimmedAvg/Krum/MultiKrum/Bulyan/qFedAvg/DP-FixedClipping/DP-AdaptiveClipping/FedXgb* |
| 5 | 新增 **§3 代码结构与模块契约** | v1 有代码片段但没有仓库形状。没有模块边界，"复用既有特征工程"就只是口号 |
| 6 | 新增 **§9 实验协议**（机器可读矩阵 + 记录规范） | v1 的 §9 是验收标准，缺"到底跑哪些配置、结果怎么落盘"。协议不落地，3 seeds 会变成 1 seed |
| 7 | 新增 **§11 相对判据** | v1 的验收写"改善量被明确量化"，但没说什么算改善。本项目**不设绝对精度目标**，只认同测试集上的相对差 |
| 8 | 新增 **§12 风险登记册** | v1 的 §8 是"坑清单"（已知错误），风险登记册要的是"概率 × 影响 × 触发信号 × 缓解" |
| 9 | 新增 **§13 里程碑与 DOD** | v1 有 W1-W5 人日，但没有每个里程碑的"完成定义" |
| 10 | §6.3 泄漏检测从规则变成**可执行脚本** | "同一颗电池不得跨客户端"是全文最贵的一条规则，靠人眼核对 124 颗电池不现实 |
| 11 | 新增 **§16 参考文献 / §17 术语表** | 方案里出现的每个方法名都要能追到原文 |
| 12 | 客户端数 **N=25 → 20**，§6.3 补 **P-UNKNOWN** 检查 | 25 是 124÷5 的取整，但 124 不能被 5 整除，而且没扣全局测试集。写 `check_protocol.py` 时才发现：按 §6.2 规则 2 整颗留出 24 颗后只剩 100 颗，k=5 只能得 20 个客户端。报告里 N 写错是硬伤 |
| 13 | 总流量 **2.60 GiB → 2.03 GiB**，补协议开销实测值 | 2.60 是按错的 N=25 算出来的，而且即便按 25 算也是 2.54。§10.1 的派生表抄了错输入。现在表 2 的每个数由 `round_trip_bytes()` 算出并有测试钉住 |
| 14 | 新增 **§3.4 实测校准**：`model.py` / `data.py` 落地后的 API 与数字核对 | v2.0 的方案骨架按 flwr 1.41.0 官方 quickstart 推演，M1 落地后逐项实测，差异记录在案而不是悄悄改掉 |
| 15 | 数据集颗数 **124 → 135**（原始 140，去 5 颗跨批次延续） | 124 是论文**预测 cycle life** 时剔掉「没跑到 80% 容量」电池后的数。本项目标签是逐循环 SOH，那些电池在循环 5–100 区间内是正常数据，不该跟着剔。实测三个文件 46 + 48 + 46 = 140，其中批次 2 有 5 颗是批次 1 的延续 |
| 16 | §6.3 新增 **P-CONT**：同一物理电池跨文件出现的泄漏 | `b2c7` 与 `b1c0` 是同一颗电池的两段。ID 不同、文件也不同，`P-LEAK` 结构上抓不到；不处理的话同一颗电池会同时进两个客户端，指标虚高而曲线看不出来 |
| 17 | 特征从「每颗电池一行 ΔQ₁₀₀₋₁₀」改为 **per-cycle 行** | 按电池划分后每客户端只有 5 颗电池。每颗一行则本地只有 5 个样本，联邦训练不成立。per-cycle 让每客户端约 100 行，H1–H3 才有统计意义 |
| 18 | 新增 **`matr.py`** 与 `build_manifest.py` / `build_features.py` / `check_continuations.py` | 三个 .mat 合计 7.7 GiB，而需要的字节不到 0.1%。整篇下载不可行（单连接实测 17 KB/s），所以按 HDF5 区间随机读；清单必须由数据生成，否则 `P-UNKNOWN` 会对着虚构 ID 说 PASS |
| 20 | `build_features.py` 改为**按颗落分片、可断点续跑** | 135 颗 × 21 循环 × 2 曲线，每次远端往返 1.4 s 到几十秒，这个命令不可能一次跑完；中断一次全部作废，而重来还是跑不完。三处不能省：分片原子落盘（半块 parquet 恰好还能打开，比没有更危险）、规格指纹不匹配就拒绝续跑（换了网格后新旧行的同一列不是同一个物理量，而拼出来的表不报异常）、写完读回来验（cell_id / 行数 / 列数对不上就删掉重抽） |
| 22 | 拆出 **`MatrTransportError`**，传输失败不再记进 `dropped` | 实测被 DNS 失败咬过一次：批次 1 的 46 颗全变成「0 行、丢弃 1 条」，spec 里躺着 46 条「参考曲线读不出」——读起来就是「这些电池没有数据」，而事实是网络断了一次。数据问题该记 `dropped`，传输问题必须抛；`.empty` 也只有不含传输原因时才算「跑完」。**本地全绿是因为那次故障没复现，这类错误只能靠真的断一次网发现** |
| 21 | 纯特征数学拆出 **`flsoh/features.py`**，`data.py` 改为 re-export | `delta_q_features` / `make_v_grid` / `validate_feature_table` 一行 torch 代码都没有，却被 `data.py` 模块层的 `import torch` 挡在 CI 门外——`build_features.py` 要在不装 torch 的 job 里生成特征表，于是红了一次。**能被便宜测试覆盖的代码，不要被昂贵的依赖挡住** |
| 19 | 延续关系的**证据换了一套**：`channel_id` 被证伪，改用 policy 后缀 + 段长 + 交界容量；N=20 → **23**，总流量 2.03 → **2.34 GiB** | 最初拿 `channel_id` 验证，5 对全 FAIL。查下去发现它根本不是身份字段（第 5 位是批次内行号，b1c0 是 47、b2c7 是 56），`barcode` 更糟——140 颗全是同一个常数 3707764736。真正的指纹是 `policy_readable`：延续段存的是原段的后缀（`80%)-3.6C` ← `3.6C(80%)-3.6C`），配合 add_len 段长与交界容量，四条证据全过。颗数确认 135，N 随之为 23 |

---

## 0. 一页速览

| 项 | 内容 |
|---|---|
| 定位 | 用 Flower 做**联邦仿真研究**：多客户端在不上传原始充电数据的前提下联合训练 SOH 估计模型，量化「联邦 vs 中心化」的精度差距与隐私代价 |
| 核心问题 | 不是"能不能跑通联邦"，而是"在按电池划分的 Non-IID 数据上，联邦掉多少点、漂移校正能挽回多少、隐私要花多少精度" |
| MVP | **约 2 周 / 8 人日**：FedAvg/FedProx × IID/Non-IID × 按电池划分，出轮数-精度曲线与对比表 |
| 完整版 | 再加 **2-3 周 / 11 人日**：FedAdam/FedBN 对比、Dirichlet α 扫描、跨化学体系、Opacus DP、App 数据契约原型 |
| 硬件 | 单机笔记本即可。MLP 13.6 万参数，CPU 可跑；有 GPU 更快但不是必需 |
| 主要风险 | ①划分泄漏（同一电池跨客户端）②Non-IID 下 FedAvg 不收敛被误判为代码 bug ③把"仿真能跑"当成"可部署" |
| 明确不做 | 真实 App 部署、安全聚合实现、跨设备异步联邦、LSTM/Transformer 起步 |

---

## 1. 目标与可证伪假设

### 1.1 三个可证伪的问题

1. 在按"电池"划分的 Non-IID 数据下，联邦训练的 SOH 模型比中心化训练差多少？
2. FedProx / Scaffold 等漂移校正策略能挽回多少差距？代价（通信、计算、实现复杂度）是什么？
3. 在模型更新上加本地差分隐私（梯度裁剪 + 噪声）后，精度-隐私的权衡曲线长什么样？

### 1.2 假设与判据

把问题写成可判定的假设。每条都写清"什么结果算支持、什么结果算推翻"——不写判据的假设最后一定会被解释成"看起来还行"。

| 编号 | 假设 | 支持（accept） | 推翻（reject） | 判定依据 | 第一批状态（2026-10-24） |
|---|---|---|---|---|---|
| H1 | IID 划分下，联邦与中心化的差距可以忽略 | 同测试集 MAE 相对劣化 ≤ 5% | > 10% | §11 的 C0 vs I1 | **未判定**（阈值 5% < seed 噪声 18%；等训练量下为 +772%，见 §10.2） |
| H2 | Non-IID（按电池 + Dirichlet α=0.5）下 FedAvg 会显著掉点 | 相对劣化 > 10% | ≤ 5%（说明电池级划分没有想象中异质） | §11 的 C0 vs N1 | **未判定**（α=0.5 在 N=23 时退化，没测到异质性） |
| H3 | FedProx 能挽回一部分 H2 的损失 | 相对 FedAvg 改善 ≥ 5% 且超过 1 个标准差 | 改善 < 1 个标准差，或为负 | §11 的 N1 vs N2 | **推翻**（μ=0.01/0.1 均更差，单调，配对波动 2.7%–38%，见 §11.3.2） |
| H4 | 本地 DP 的精度代价随噪声单调上升，且在 ε ≤ 8 时仍可用 | noise=1.0 处 MAE 劣化 ≤ 30% 且 ε ≤ 8 | 劣化 > 50%，或 ε 报不出来 | §11 的 P 系列 | 未开始（M4） |

**"支持"不等于"证明"**：H2 成立只说明这批电池按此划分确实异质，不能外推成"所有联邦场景都掉点"。报告里必须写成"在 N=23、α=0.5、MIT-Stanford 135 颗的条件下"。

### 1.3 交付物

**MVP（W2 末）**

| 编号 | 交付物 | 验收形式 |
|---|---|---|
| D1 | 可复现仿真代码库 | `flwr run` 一条命令跑通；依赖锁版本；固定 seed |
| D2 | 划分脚本 + 泄漏检测 | `scripts/check_partition.py` 全 PASS（§6.3） |
| D3 | 对比实验 | 中心化 / IID-FedAvg / Non-IID-FedAvg / Non-IID-FedProx 四条配置 |
| D4 | 结果图与表 | 轮数-精度曲线 + 最终精度表，≥3 seeds，均值±标准差 |
| D5 | MVP 结论备忘 | 2-3 页，回答 H1-H3 的 accept/reject |

**完整版（W5 末，MVP + 2-3 周）**

| 编号 | 交付物 | 验收形式 |
|---|---|---|
| D6 | 策略扩展 | FedAdam、FedBN（跨化学体系场景） |
| D7 | α 扫描 | α ∈ {0.1, 0.5, 1.0, IID} 系统实验 |
| D8 | 隐私增强 | Opacus DP-SGD，≥3 个 noise 倍数的隐私-效用曲线，每点附 ε |
| D9 | App 数据入口原型 | Schema v1 + 本地特征提取库 + 模拟数据生成器（§8） |
| D10 | 完整技术报告 | 含威胁模型声明、实验协议、模拟-部署差距讨论 |

---

## 2. 环境与版本锁定

### 2.1 依赖矩阵

```bash
# Python ≥ 3.11（flwr 1.31 起的硬要求）
python -m venv .venv && source .venv/bin/activate

pip install "flwr[simulation]==1.41.0"
pip install "torch==2.10.0" numpy pandas scikit-learn
pip install "opacus==1.6.0"        # 仅完整版 D8 需要
```

| 包 | 锁定版本 | 备注 |
|---|---|---|
| `flwr` | 1.41.0 | 必须带 `[simulation]` extra，否则 `flwr run` 的仿真运行时缺失 |
| Python | ≥3.11, <4.0 | `flwr` 1.31+ 的 `requires-python` |
| `torch` | 2.10.0 | 与官方 quickstart 一致；`opacus` 1.6.0 要求 `torch>=2.6.0` |
| `opacus` | 1.6.0 | 完整版才装。`make_private()` 仍是官方推荐入口 |

⚠️ **不要写成 `pip install "flwr==1.9.1" "flwr[simulation]"`**——v1 是这么写的，两个 specifier 指向同一个包，pip 解析结果不确定。正确形式是带 extra 的单一 pin：`"flwr[simulation]==1.41.0"`。

### 2.2 为什么必须锁版本

Flower 的公开 API 在 1.8 → 1.41 之间变过两次范式：

| 时期 | 客户端基类 | 策略导入 | 参数载体 |
|---|---|---|---|
| ≤1.7 | `fl.client.NumPyClient` | `fl.server.strategy.FedAvg` | `list[np.ndarray]` |
| 1.8-1.9 | `fl.client.NumPyClient` + `ClientApp(client_fn=)` | `fl.server.strategy` / `flwr.serverapp.strategy` 并存 | `list[np.ndarray]` |
| ≥1.10（现行） | `flwr.clientapp.ClientApp` + `@app.train()` | `flwr.serverapp.strategy` | `ArrayRecord` / `MetricRecord` |

三种写法互不兼容。**报告里必须写版本号**，否则别人按文档复现会失败，而你无法解释是版本问题还是结论问题。

### 2.3 可复现性开关

| 项 | 做法 |
|---|---|
| 全局 seed | 每个配置显式给 `seed`，从 §9 的矩阵取，不依赖默认值 |
| Python 哈希随机化 | 训练入口设 `PYTHONHASHSEED=<seed>` |
| torch 确定性 | `torch.use_deterministic_algorithms(True)`；CPU 上代价可接受，GPU 上若报错则记录并降级为"seed 固定但非严格确定" |
| 数据顺序 | DataLoader 的 `shuffle` 用派生自 seed 的 generator，不用全局随机 |
| 结果落盘 | 每个 run 一个 JSON，字段见 §9.3；不允许只存图 |

---

## 3. 代码结构与模块契约

### 3.1 目录结构

```text
battery-soh-fl-benchmark/
├── configs/                  # 机器可读的实验矩阵（TOML，Python 3.11 内置 tomllib）
│   ├── mvp.toml
│   └── full.toml
├── src/flsoh/
│   ├── partition.py          # 电池级划分 + 泄漏检测（纯标准库，可单测）
│   ├── protocol.py           # 配置校验（纯标准库，可单测）
│   ├── matr.py               # MATR .mat 的远端随机读与解析（需 h5py，进 CI）
│   ├── features.py           # ΔQ(V) 特征与特征表契约（只依赖 numpy/pandas，进 CI）
│   ├── data.py               # 特征表 + 本地标准化 + DataLoader（需 torch/pandas）
│   ├── model.py              # MLP 定义
│   └── app/                  # Flower App（需 flwr，不在 CI 里跑）
│       ├── client_app.py
│       └── server_app.py
├── scripts/
│   ├── check_partition.py    # 划分泄漏检测 CLI
│   ├── check_protocol.py     # 实验矩阵校验 CLI
│   ├── check_continuations.py# 跨批次延续验证（policy 后缀 + 段长 + 交界容量）
│   ├── build_manifest.py     # 原始文件 → cells.txt + manifest.json
│   └── build_features.py     # 原始曲线 → features.parquet + spec（按颗落分片，可续跑）
├── tests/                    # 纯标准库 + matr（合成 HDF5，不联网）
└── docs/
```

**分层原则**：`partition.py` 与 `protocol.py` 不依赖 torch/flwr，因此可以在任何机器上跑、可以进 CI。这两个模块承载的是**最容易出错且最贵的逻辑**（划分泄漏、协议缺项），值得脱离深度学习栈单独测试。`matr.py` 只依赖 numpy + h5py，测试用合成 HDF5 与本地 Range 服务器，因此也能进 CI——数据集的 7.7 GiB 不该成为「能不能跑测试」的前提。

`features.py` 是同一原则的第二次应用：`delta_q_features` / `make_v_grid` / `validate_feature_table` 一行 torch 代码都没有，却被 `data.py` 模块层的 `import torch` 挡在 CI 门外——`build_features.py` 要在不装 torch 的 job 里生成特征表，于是红了一次。拆开后 `data.py` 从 `features.py` re-export，`from flsoh.data import CELL_COL` 照旧能用。**能被便宜测试覆盖的代码，不要被昂贵的依赖挡住。** 训练代码依赖重，不进 CI。

### 3.2 模块契约

| 模块 | 输入 | 输出 | 不许做什么 |
|---|---|---|---|
| `partition.py` | 电池 ID 列表、每客户端电池数、划分方案、seed | `{client_id: [cell_id, ...]}` | 不许碰特征、不许读原始数据、不许依赖 pandas |
| `protocol.py` | 一份 TOML 配置 | `[(level, code, message)]` | 不许改配置、不许猜默认值 |
| `data.py` | 电池 ID、本地训练/验证电池集合 | 该客户端的 `DataLoader` 对 | **不许做全局标准化**（见 §14 坑 1） |
| `features.py` | (V, Q) 曲线序列 + 电压网格 | `ΔQ(V)` 特征矩阵；特征表契约校验 | 不许 import torch、不许外延电压网格 |
| `model.py` | 输入维度 | `nn.Module` | 不许内置任何与数据相关的常量 |
| `app/client_app.py` | 一轮的 `Message` | 参数 + 指标 | 不许把原始数据写进 `Message` |

### 3.3 当前 API 骨架（flwr 1.41 范式）

以下写法按 1.41.0 的 `quickstart-pytorch` 实测结构改写，**不是** v1 里的 `NumPyClient` 写法。落地第一天仍应用官方 quickstart 跑一遍再替换数据。

```python
# src/flsoh/app/client_app.py
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flsoh.data import load_client_data
from flsoh.model import MLP

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    model = MLP()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # partition-id / num-partitions 来自仿真运行时，不需要自己传
    cid = str(context.node_config["partition-id"])
    trainloader, _ = load_client_data(cid)

    local_epochs = int(context.run_config["local-epochs"])
    lr = float(msg.content["config"]["lr"])
    train_one(model, trainloader, local_epochs, lr)

    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
        {"train_loss": last_loss, "num-examples": len(trainloader.dataset)}
    )
    return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    model = MLP()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    cid = str(context.node_config["partition-id"])
    _, valloader = load_client_data(cid)

    mae = evaluate_mae(model, valloader)
    metrics = MetricRecord({"mae": mae, "num-examples": len(valloader.dataset)})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
```

```python
# src/flsoh/app/server_app.py
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from flsoh.data import load_global_test
from flsoh.model import MLP

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    mu = float(context.run_config["proximal-mu"])

    strategy = (
        FedProx(proximal_mu=mu) if mu > 0 else FedAvg()
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(MLP().state_dict()),
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = MLP()
    model.load_state_dict(arrays.to_torch_state_dict())
    mae = evaluate_mae(model, load_global_test())
    return MetricRecord({"mae": mae})
```

---

## 3.4 实测校准（M1 落地后核对）

§3.3 的骨架按 flwr 1.41.0 官方 quickstart 推演。装上 `flwr==1.41.0` +
`torch==2.10.0` 后逐项实测，结果如下——**对得上的记"确认"，对不上的记差异**，
不悄悄改前面的章节。

| §3.3 的说法 | 实测 | 结论 |
|---|---|---|
| `from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict` | 全部可导入 | ✅ 确认 |
| `from flwr.clientapp import ClientApp` + `@app.train()` / `@app.evaluate()` | 可导入，装饰器存在 | ✅ 确认 |
| `from flwr.serverapp import Grid, ServerApp` | 可导入 | ✅ 确认 |
| `from flwr.serverapp.strategy import FedAvg, FedProx` | 可导入 | ✅ 确认 |
| `FedProx(proximal_mu=mu)` | `proximal_mu` **确实存在**，但是最后一个参数，排在 `fraction_train` / `min_*` / `weighted_by_key` / `arrayrecord_key` / `configrecord_key` / 两个 `*_aggr_fn` 之后 | ✅ 确认（位置靠后，按关键字传） |
| `strategy.start(grid=..., initial_arrays=..., num_rounds=..., evaluate_fn=...)` | 签名完全一致，另有 `timeout` / `train_config` / `evaluate_config` | ✅ 确认 |
| `ArrayRecord.from_torch_state_dict` / `.to_torch_state_dict` | 两个方法都在 | ✅ 确认 |
| 通信量载荷 545,796 B | `parameter_bytes()` = 545,796 | ✅ 确认 |
| 通信量"即线上的字节数" | `ArrayRecord.count_bytes()` = **546,630** | ⚠️ 差 834 B，见下 |

**两处需要记进报告的差异：**

1. **协议开销 834 B**。`count_bytes()` 比载荷多 834 B = 139 B/张量 × 6 个张量。
   实测多组模型确认这是 NumPy 序列化的 dtype/shape 元数据，**与模型规模无关、
   只与张量数有关**。§10.1 表 2 已补这一项。

2. **聚合权重依赖客户端上报 `num-examples`**。`FedAvg.__init__` 的默认
   `weighted_by_key="num-examples"`——也就是说 §14 坑 3「聚合应按样本数加权」
   这条规则，在 flwr 1.41 里的**执行方式是客户端必须在 `MetricRecord` 里报
   `num-examples`**。漏报不会报错，会静默退化成等权聚合。
   `src/flsoh/app/client_app.py` 因此把 `num-examples` 写进训练与评估两条
   `MetricRecord`，并由 `tests/test_app.py` 钉住。

**另一处与方案不同的实现选择**：§11.2 说 C1（本地独立训练）是下限参考线。
C1 不需要联邦运行时——它就是 N 个单客户端模型各训各的。因此 `server_app.py`
不实现 C1，改由 `scripts/run_local_baseline.py` 直接跑，避免为一条基线
拉起整套 SuperLink。

`pyproject.toml` 的关键字段（版本敏感，照抄官方结构）：

```toml
[tool.flwr.app]
publisher = "<your-github-handle>"
fab-format-version = 1
flwr-version-target = "1.41.0"

[tool.flwr.app.components]
serverapp = "flsoh.app.server_app:app"
clientapp = "flsoh.app.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 100
local-epochs = 1
learning-rate = 0.001
proximal-mu = 0.0
```

---

## 4. 模型与特征

### 4.1 主选：MLP + ΔQ(V) 特征向量

- **特征**：`ΔQ(V) = Qc_k(V) − Qc_ref(V)`，把第 k 个循环的充电容量曲线与参考循环的曲线各自插值到同一个 1000 点电压网格上再逐点相减。电压窗口固定 `[2.0, 3.5] V`，参考循环固定为 1（跳过 formation 循环 0）。**只用充电片段**，这与"众包 App 只采充电数据"的设定一致。
- **行粒度**：**per-cycle**。一行 = (电池, 循环 k)，标签 `SOH_k = QD_k / QD_ref`。取循环 5..100 步长 5，每颗电池 20 行。
- **结构**：1000 → 128 → 64 → 1（SOH 回归）
- **参数量**：`1000×128+128 + 128×64+64 + 64×1+1 = 136,449`
- **单轮上行通信量**：`136,449 × 4 B = 545,796 B ≈ 0.52 MiB`（v1 写 0.6 MB，是估算；表 2 用精确值）
- **优点**：复用既有特征工程与 SOH 标签口径；模型小 → 轮数可以多跑，统计结论更稳

**为什么是 per-cycle 而不是论文的「每颗电池一行」**：按电池划分后每客户端只有 5 颗电池。若每颗电池只出一行，客户端本地只有 5 个样本，联邦训练本身不成立——H1–H3 会退化成噪声比较。per-cycle 让每客户端约 100 行。

**为什么参考循环固定**：若每行拿前一循环作基准，第 k 行与第 k+1 行的尺子不同，模型学到的是尺子变化而不是老化。固定参考循环后，所有行在同一把尺子上。

**为什么电压窗口是常量而不是从数据算**：从数据算出来的窗口会随数据变，两次运行的网格不同，特征就不可比。窗口写死并记录在 `features.spec.json` 里；覆盖不住的曲线**逐条记入 `dropped`**，不静默跳过——静默跳过会让「样本少了 30%」在精度曲线上完全看不出来。

### 4.2 进阶（完整版可选）：1D-CNN

输入固定长度的充电片段三通道序列（V/I/T，重采样到统一长度如 256 点），验证端到端特征的联邦可行性。**MVP 阶段不碰。**

### 4.3 明确不建议起步就上 LSTM/Transformer

通信量大、仿真慢，且模型复杂度会掩盖 Non-IID 问题的本质。真要看大模型，等 §11 的判据在 MLP 上稳定之后再换，否则掉点时分不清是数据异质还是模型没训好。

---

## 5. Non-IID 与策略

### 5.1 三类来源必须分开报告

| 来源 | 含义 | 本项目怎么造 |
|---|---|---|
| 量偏斜 | 各客户端样本数不同 | 每客户端电池数不同（1/2/5 颗） |
| 特征/标签偏斜 | 使用模式不同（快充 vs 慢充、温度带） | 按电池天然带来；不额外注入 |
| 域偏移 | 化学体系/型号不同 | MIT / NASA / CALCE 各成一组客户端 |

**不要混在一张图里**：把量偏斜和域偏移画在同一条曲线上，掉点原因无法归因。

### 5.2 策略矩阵

| 策略 | 适用场景 | 关键参数 | Flower 1.41 是否内置 | 本项目 |
|---|---|---|---|---|
| FedAvg | IID / 轻度 Non-IID 基线 | 无 | ✅ `flwr.serverapp.strategy.FedAvg` | 必做基线 |
| FedProx | 量/特征偏斜，本地 epoch 多 | `proximal_mu` 0.01-0.1 | ✅ `FedProx(proximal_mu=...)` | **首选 remedy** |
| FedAdam / FedYogi | 各客户端梯度尺度差异大 | 服务器端 lr、β | ✅ `FedAdam` / `FedYogi` | 完整版对比 |
| FedBN | 域偏移（跨化学体系） | BN 层不聚合 | ❌ 在 `baselines/` 里，需自行接入 | 完整版必做 |
| Scaffold | 客户端漂移校正 | 双控制变量 | ❌ 在 `baselines/` 里，需自行接入 | 可选，时间紧就砍 |
| DP-FixedClipping | 服务器侧隐私 | 裁剪范数 | ✅ `DP-FixedClipping` | 只做说明（§7 L3） |

**注意**：内置策略只在 `flwr.serverapp.strategy` 下，`flwr.server.strategy` 是旧路径。FedBN/Scaffold 不是内置策略，接进来意味着要自己实现聚合逻辑并自己验证收敛——这就是 §7 里"实现复杂度"那笔账。

### 5.3 调参要点

- **本地 epoch 数与漂移成正比**。Non-IID 下 `local_epochs=5` 常比 `1` 差很多；μ 与 local epochs 必须联合调，不能分开调完再拼。
- 每轮参与比例 < 1 时，客户端抽样本身引入方差，必须报抽样方案的方差，不能只报均值。
- 网格搜索的每一格都要记 seed，否则最优格可能是随机挑出来的。

---

## 6. 客户端划分（本项目最贵的一步）

MIT-Stanford 135 颗电池天然适合按对象划分（原始 140，去掉 5 颗跨批次延续，见 §6.2 规则 3）。

### 6.1 四种方案

| 方案 | 划分方式 | 模拟的真实场景 | 建议 |
|---|---|---|---|
| A 按电池 | 每客户端 = k 颗。留出 20 颗作全局测试后剩 115 颗：k=1/2/5 → 115/57/23 客户端 | 一人一块电池 | **MVP 主方案** |
| B 按用户代理 | 按 batch 分组 + 额定容量档（1.1/1.2/1.5 Ah）聚类 | 同一用户多设备 | 进阶 |
| C 按化学体系/数据源 | MIT / NASA / CALCE 各成一组 | 不同电池厂众包 | 完整版必做（极端域偏移） |
| D Dirichlet 样本分配 | 按 Dirichlet(α) 把电池分给 N 客户端 | 量化任意 Non-IID 程度 | 完整版 α 扫描 |

**算术修正（v2.0）**：v1 写「k=5 → 25 客户端」，那是 124÷5 的取整。但 124 不能被 5 整除，且没扣掉全局测试集。按 §6.2 规则 2 整颗留出测试集后，k=5 得 20 个客户端。`configs/*.toml` 与 `check_protocol.py` 一律按 20 算。

**算术再修正（v2.1）**：上面那个 20 建立在「124 颗、留 24 颗」上，而这两个数都不对。实测三个批次文件共 **140** 颗，去掉 5 颗跨批次延续得 **135** 颗；留出测试集改为 **20** 颗（135 的 15%，比 24 更整齐且仍 ≥ 论文口径），剩 115 颗，k=5 → **23 个客户端**。所以 v2.0 那句「一律按 20 算」现在也是错的，正确值是 23。教训：**N 不是常数，它是「颗数 − 测试集」除以 k 的函数**，颗数一变全链条都要重算。

**Dirichlet 在这里会退化**：`check_partition.py` 的实测输出显示，N=23、α=0.5 时最大的客户端独占 21 颗、8 个客户端各只有 1 颗；α=0.1 时更极端（1 个客户端 44 颗、14 个各 1 颗）。原因是 Dirichlet 分配的是**电池**而不是样本，客户端一多、α 一小，分布自然退化成「一人独占」。这不是 bug，但要写进报告——此时「量偏斜」与「域偏移」已经混在一起，不能再声称只考察了其中一种。

### 6.2 四条硬规则

1. **同一颗电池的相邻循环不得跨客户端**。时间序列截断会让泛化虚高——划分以"电池"为最小原子，客户端内再按时间切 train/val。
2. **全局测试集独立于所有客户端数据**；客户端本地 val 另设，两套指标分开报，不许混。
3. **同一颗物理电池不得以两个 ID 出现**。批次 2 的 `b2c7/8/9/15/16` 是批次 1 的 `b1c0..b1c4` 的延续段。这不是数据错误，是数据集的组织方式——但它让「同一颗电池进两个客户端」这件事在 ID 层面完全合法。处理方式：默认删掉延续段，保留批次 1 的那一段。
4. **客户端数量与轮数写进协议**：典型 N=23（5 颗/客户端，115 颗可用）、`num_rounds=100`；k=2 时 N=58，`fraction_fit` 降到 0.4，否则单轮墙钟翻倍。

### 6.3 泄漏检测（可执行）

规则 1 靠人眼核对 135 颗电池不现实，因此做成脚本：

```bash
python3 scripts/check_partition.py --config configs/mvp.toml
```

它按配置生成划分，然后逐项检查：

| 检查 | 含义 | 失败后果 |
|---|---|---|
| P-LEAK | 同一颗电池出现在两个客户端 | 训练集泄漏，精度虚高，结论作废 |
| P-TEST | 全局测试电池出现在任何客户端 | 同上，且更难发现 |
| P-EMPTY | 某客户端 0 颗电池 | 该轮无梯度，聚合权重异常 |
| P-COVER | 有电池既不在客户端也不在测试集 | 数据浪费；若在测试集则是另一个方向的泄漏 |
| P-COUNT | 实际客户端数与配置声明不符 | 论文/报告里的 N 写错 |
| P-UNKNOWN | 出现不属于该数据集的电池 ID | 多半读错了数据源；所有 ID 都必须能在数据集清单里找到 |
| P-CONT | 同一颗物理电池以两个 ID 出现（跨批次延续） | 规则 3；`P-LEAK` 结构上抓不到，因为它只比较 ID |

**P-CONT 为什么必须单独一条**：`P-LEAK` 的前提是"同一颗电池 = 同一个 ID"。批次 2 的 `b2c7` 与批次 1 的 `b1c0` 是同一颗电池，但 ID 不同、文件也不同，任何基于 ID 集合的检查都会放行。证据只能从数据本身取：

```bash
python3 scripts/check_continuations.py     # 读批次 1 全部 46 颗 + 5 颗延续段，跑四条证据
```

四条证据（脚本逐条打印 OK/FAIL）：

| # | 证据 | 实测 |
|---|---|---|
| 1 | 延续段的 `policy_readable` 是原段 policy 的后缀 | `80%)-3.6C` ← `3.6C(80%)-3.6C`；`80%)-4C` ← `4C(80%)-4C` |
| 2 | 延续段循环数 = 论文 add_len + 1 | 662/981/1060/208/482 ← 661/980/1059/207/481 |
| 3 | 同一条后缀在批次 1 圈出的电池正好是这一组的原段 | 3.6C 组 3↔3，4C 组 2↔2 |
| 4 | 交界处容量落差 < 延续段自身衰减的 1/4 | 落差 0.001–0.015，衰减 0.09–0.14 |

> **两个字段不能用，写下来免得再踩**：`barcode` 在全部 140 颗电池上都是同一个常数 `3707764736`，是占位值；`channel_id` 的第 5 位是批次内行号（b1c0 是 47、b2c7 是 56），跨批次必然不同。最初拿 `channel_id` 当身份，得到「5 对全 FAIL」——那不是延续关系不成立，是字段选错了。**验证失败时先怀疑判据，再怀疑结论。**

**这一步跑在训练之前**。划分错了，后面所有实验都白做。

---

## 7. 隐私保护（分层递进）

默认威胁模型：**半诚实服务器**（服务器按协议执行但好奇，会尝试从更新里推断原始数据）。

| 层 | 做法 | 防什么 | 不防什么 |
|---|---|---|---|
| **L1 联邦本身**（MVP 默认） | 只传模型参数/梯度 | 原始充电曲线直接泄露 | 梯度反演攻击 |
| **L2 本地 DP-SGD**（完整版核心） | 逐样本梯度裁剪 + 高斯噪声 | 从单条更新反推单条样本 | 极低 ε 下的效用崩塌 |
| **L3 服务器侧保护**（只做说明） | Flower 内置 `DP-FixedClipping` / `DP-AdaptiveClipping` | 对客户端更新的保护 | 与 L2 威胁模型不同，不可混谈 |

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, trainloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=trainloader,
    noise_multiplier=1.0,   # 扫描 {0.5, 1.0, 2.0}
    max_grad_norm=1.0,
)
# 训练结束：
epsilon = privacy_engine.get_epsilon(delta=1e-5)
```

**两条常见错误**：

1. 用了 DP 却报不出 ε——没有 RDP 会计器记账，就等于没有隐私保证。
2. 把梯度裁剪当成差分隐私——裁剪只是训练稳定技巧，**没有噪声就没有 DP**。

---

## 8. 与手机 App 的数据入口对接

现阶段没有真实 App 数据流，务实做法是**先定契约、再用模拟数据回填**。

### 8.1 充电会话 Schema v1

```json
{
  "user_id_hash": "sha256-pseudonym",
  "battery_id": "device-battery-uuid",
  "session_id": "uuid",
  "start_ts": 0,
  "end_ts": 0,
  "chemistry_hint": "LiCoO2|unknown",
  "sampling_hint": "variable",
  "records": [{"t": 0, "V": 3.87, "I": -1.2, "T": 29.5, "soc_est": 0.42}]
}
```

### 8.2 本地特征提取库

与联邦仿真**共用同一份代码**。App 侧只执行"片段切分 → 重采样 → **本地**归一化 → 特征向量生成"。特征向量与 SOH 标签之外的一切留在设备/个人网关。这直接决定 §4.1 的 MLP 输入格式。

### 8.3 部署形态（写成设计选项，不做真实部署）

| 形态 | 说明 | 工程成本 |
|---|---|---|
| 本地网关 | 手机 → 用户个人云/家庭设备跑 Flower Client | 最低，最现实 |
| 设备端 | Flower 的 SuperLink/SuperNode 部署模式 | Android 集成成本高 |

### 8.4 模拟数据生成器

从 MIT 充电片段加噪声/漂移生成符合 Schema 的"伪 App 会话"，用于端到端联调：数据入库 → 特征提取 → 本地训练 → 参数上传。这是 D9 的验收物。

---

## 9. 实验协议

### 9.1 为什么要有独立一节

"跑 3 个 seed"这句话在没有矩阵的情况下不会被执行。协议的作用是把"要跑什么"变成文件，让 `scripts/check_protocol.py` 能检查它、让下一个人能照着跑。

### 9.2 配置矩阵

见 `configs/mvp.toml` 与 `configs/full.toml`。每个配置项的含义：

| 键 | 含义 | 校验 |
|---|---|---|
| `data.partition_atom` | 划分原子，必须是 `"cell"` | 不是 cell 直接拒 |
| `partitions.cells_per_client` | 每客户端电池数列表 | 必须整除「电池数 − 测试集」，或显式声明余数策略 |
| `partitions.schemes` | `iid` / `dirichlet` | 至少要有一个 `iid` 作对照 |
| `strategies.include` | 要跑的策略 | **必须含 `centralized`** |
| `seeds.min_required` | 最少 seed 数 | ≥3 |
| `privacy.noise_multiplier` | 噪声倍数扫描点 | 启用时 ≥3 个点 |
| `acceptance.*` | §11 的判据阈值 | 必须在 (0,1) 内 |

### 9.3 结果落盘规范

每个 run 一个 JSON，字段固定：

```json
{
  "run_id": "mvp__fedavg__n25__dirichlet0.5__seed0",
  "config": "configs/mvp.toml",
  "strategy": "fedavg",
  "n_clients": 25,
  "scheme": "dirichlet",
  "alpha": 0.5,
  "seed": 0,
  "flwr_version": "1.41.0",
  "torch_version": "2.10.0",
  "final_mae": null,
  "final_rmse": null,
  "rounds_to_target": null,
  "curve": [{"round": 1, "mae": null}],
  "wall_clock_s": null,
  "notes": ""
}
```

`null` 是占位：**没跑出来就留 null，不许填 0，也不许删字段**。曲线用数组不用图，图由报告阶段生成——图不能进版本库当作数据源。

### 9.4 运行顺序

1. `check_protocol.py` 校验矩阵
2. `check_partition.py` 校验划分
3. 跑 `centralized`（C0）——这是上限，先拿到它
4. 跑 `iid + fedavg`（I1）→ 判 H1
5. 跑 `dirichlet + fedavg`（N1）→ 判 H2
6. 跑 `dirichlet + fedprox`（N2，μ 网格）→ 判 H3
7. 完整版继续 FedAdam / FedBN / α 扫描 / DP

**顺序不能乱**：C0 没跑就不知道联邦掉的是不是"本来就该掉"。

---

## 10. 结果呈现

### 10.1 图表清单（固定协议）

| 编号 | 内容 | 规格 |
|---|---|---|
| 图 1 | 轮数-精度曲线 | x=通信轮数，y=全局模型在统一测试集的 MAE；中心化画水平参考线；按 Non-IID 设置分面板 |
| 表 1 | 最终精度对比 | 所有策略的最终 MAE/RMSE，**≥3 seeds 均值±标准差**，附收敛轮数 |
| 图 2 | Non-IID 程度扫描 | α ∈ {0.1, 0.5, 1.0, IID} 柱状图 |
| 图 3 | 隐私-效用曲线 | noise_multiplier vs MAE，每点标注 ε |
| 表 2 | 通信量估算 | 参数量 × 4 B × 2（上/下行）× 客户端数 × 轮数 |

表 2 用 §4.1 的精确值：`136,449 × 4 × 2 = 1,091,592 B/客户端/轮 ≈ 1.04 MiB`。N=23、100 轮 → **2,510,661,600 B ≈ 2.34 GiB** 总流量（不含协议开销）。

> **协议开销已实测（flwr 1.41.0，2026-10-22）**：`ArrayRecord.count_bytes()` 对上述模型返回 **546,630 B**，比载荷多 834 B。差额恒为 **139 B/张量**，与模型规模无关——那是 NumPy 序列化要写的 dtype 与 shape 元数据。占比 0.15%，本可忽略，但换 1D-CNN/LSTM 时张量数会变，必须重算。算式固化在 `src/flsoh/model.py: wire_bytes()`。
>
> **一个已经犯过两次的错**：v1 写「N=25、100 轮 → 约 2.60 GiB」。N=25 是 124÷5 的取整（正确值 20），而即便按 N=25 算也是 2.54 GiB。v2.0 修成 2.03 GiB，但那个 2.03 仍然建在错颗数上——颗数从 124 改成 135 之后 N=23，正确值是 2.34 GiB。**派生表抄了错的输入，就会一直错下去**；而「输入」本身也会被上一步推翻。所以表 2 的每个数都由 `round_trip_bytes()` 算出，并由 `tests/test_model.py::TestCommunicationBudget` 钉住。

### 10.2 结果表（第一批，2026-10-24）

MIT-Stanford，135 颗电池（115 训练 / 20 测试），k=5 → N=23，100 轮，
Adam 1e-3、`batch_size=64`、标签中心化到 1.0，3 seeds。
除标注 `le=3` 的等训练量对照外，其余联邦配置均为 `local_epochs=10`。
全部数字由 `scripts/summarize_results.py` 从 `results/*.json` 生成，不手抄。

下面这块是**生成区**，由脚本写入、由 CI 逐字核对：

<!-- BEGIN GENERATED: results-table -->
| 编号 | 配置 | 策略 | 客户端数 | 划分 | seeds | MAE (mean±std) | RMSE | 相对 C0 |
|---|---|---|---|---|---|---|---|---|
| C0 | 中心化 | centralized | — | — | 3 | 0.00166 ± 0.00023 | 0.00281 ± 0.00032 | — |
| C1 | 本地独立 | local | 23 | — | 3 | 0.03934 ± 0.01347 | 0.04646 ± 0.01382 | +2270.9% |
| I1@le=10 | FedAvg / IID | fedavg | 23 | iid | 3 | 0.00131 ± 0.00024 | 0.00213 ± 0.00040 | -20.9% |
| I1@le=3 | FedAvg / IID | fedavg | 23 | iid | 3 | 0.01447 ± 0.00907 | 0.01673 ± 0.01005 | +772.0% |
| N1 | FedAvg / Dirichlet | fedavg | 23 | dirichlet α=0.5 | 3 | 0.00098 ± 0.00008 | 0.00149 ± 0.00018 | -40.7% |
| N2@μ=0.01 | FedProx (μ=0.01) / Dirichlet | fedprox | 23 | dirichlet α=0.5 | 3 | 0.00145 ± 0.00017 | 0.00203 ± 0.00021 | -12.3% |
| N2@μ=0.1 | FedProx (μ=0.1) / Dirichlet | fedprox | 23 | dirichlet α=0.5 | 3 | 0.00182 ± 0.00009 | 0.00254 ± 0.00017 | +9.6% |
<!-- END GENERATED: results-table -->

**生成区的规则**：手改任何一个数都会让 CI 的 `--check-plan` 红。要更新就
重跑脚本 `python scripts/summarize_results.py --write-plan docs/plan.md`，
再提交 `results/*.json` 与 `docs/plan.md` 两份改动——**表和原始数据必须同一次
提交进仓库**，否则读者无法验证表里的数是从哪来的。

**这张表不能按字面读。** 五条限制必须先看：

1. **训练量不对等，而且这一条能翻转结论。** 联邦是 100 轮 × 10 本地 epoch
   ≈ 1000 个 epoch，C0 是 300 个——联邦多训 3.3 倍。所以「−20.9%」不是
   「联邦比中心化强」，而是「联邦在 3.3 倍训练量下追平并略超」。
   为此专门补了等训练量对照 **I1@le=3**（100 轮 × 3 = 300 个本地 epoch，
   与 C0 的 300 ep 对齐）：**0.01447，相对劣化 +772%**。
   同一个策略、同一份划分、同一批 seed，只把训练量从 1000 ep 降到 300 ep，
   结论就从「好 21%」变成「差 8.7 倍」。**任何不声明训练量的相对差都没有
   意义**，这不是保守说法，是实测反号。
2. **判据的分辨力不够。** 判据阈值是 5%，而 C0 的 seed 波动是 13.8%、
   IID(le=10) 是 18.0%、IID(le=3) 是 62.7%——**阈值小于噪声**，且在训练量
   不足时噪声会爆炸。所以除 N2 之外的所有相对差都不能据此接受或拒绝假设。
   要判定 H1/H2，seed 数需要从 3 提到量级上的 50 个，或者把判据改成
   「与噪声同量级即视为无差异」（§11.3.1）。
3. **Dirichlet α=0.5 的划分本身退化了。** N=23 时最大客户端独占 21 颗电池，
   8 个客户端各只有 1 颗。加权聚合由大客户端主导，所以 N1 实际上接近
   「中心化 + 少量小客户端」，**比 IID 更好不是 H2 被推翻，而是这个 α 没有
   测到 H2 想测的东西**（§11.3 的预警命中）。要检验 H2 需要 α 扫描
   （0.1 / 0.5 / 1.0）或改成按「化学体系 / 批次」划分。
4. **FedProx 是单调变差，不是「参数没调对」。** μ=0（即 FedAvg）0.00098 →
   μ=0.01 得 0.00145 → μ=0.1 得 0.00182。**μ 越大越差，且两端都差于 μ=0**。
   由于 μ→0 的极限就是 FedAvg，这条曲线在本设置下不存在比 FedAvg 更好的
   内点——**H3 不是「没调好」，是「这个数据集规模下近端项没有可用区间」**。
   机理见 §11.3.2：每客户端只有 80 行训练数据，近端项把参数拴在全局点上。
5. **C1 的 0.039 不是「本地训练很差」，是「23 个模型各测各的再取平均」。**
   它的定义就是各客户端只用自己的数据训、不聚合（§11.2），指标是 23 个
   模型的平均 MAE，不是某个平均模型的 MAE。它证明联邦确实有用（0.039 →
   0.0013），但不能拿来和 C0 比「谁更会训练」。

**H1/H2 的状态是「未判定」，H3 的状态是「否定」。** 这一批结果的真正价值是
把管线跑通、把量级量出来，并暴露判据与 DOD 本身的问题（§11.3、§13）。

### 10.3 报告结构

动机与威胁模型 → 实验协议（数据/划分/策略/超参/seeds）→ 主结果（图 1-3、表 1-2）→ 讨论（模拟 vs 真实差距）→ 局限与后续。

---

## 11. 基线与相对判据

### 11.1 为什么不设绝对精度目标

SOH 估计的绝对 MAE 强依赖数据集、划分方式、标签口径。写"目标 MAE < 5%"会诱导两种作弊：换划分凑指标、换口径凑指标。

**本项目只认同测试集上的相对差**：所有结论写成"相对中心化劣化 X%"，而不是"MAE 达到 Y%"。

### 11.2 基线定义

| 编号 | 基线 | 定义 | 作用 |
|---|---|---|---|
| C0 | 中心化上限 | 把所有客户端的训练数据并起来训同一个 MLP，同超参同轮数（按 epoch 折算） | 上限参考线 |
| C1 | 本地独立训练 | 每个客户端只用自己的数据训，不聚合 | 下限参考线（证明联邦确实有用） |

有了 C0 和 C1，"联邦有没有意义"才有区间可谈。只报 C0 与联邦的差距，读者无法判断联邦是否白做。

### 11.3 判据表

| 判据 | 阈值 | 触发动作 |
|---|---|---|
| IID 相对劣化 | ≤5% 支持 H1；>10% 判为实现问题 | 先查聚合加权与标准化，不要先上策略 |
| Non-IID 相对劣化 | >10% 支持 H2 | 若 ≤5%，说明电池级划分不够异质，需要 α 扫描 |
| FedProx 改善 | ≥5% 且 >1σ 才算支持 H3 | 不达标就如实报"未观察到改善" |
| seed 波动 | 3 次运行最终 MAE 相对标准差 <10% | 超了就加 seed 数，不许挑最好的一次 |
| DP 可用性 | ε ≤ 8 时劣化 ≤30% 支持 H4 | 超了就把结论写成"当前模型规模下 DP 代价过高" |

#### 11.3.1 这张表在 2026-10-24 的第一批结果里失效了

**问题：阈值小于噪声，判据没有分辨力。**

第一批实测的 seed 波动（3 seeds，相对标准差）：

| 配置 | seed 波动 |
|---|---|
| C0 中心化 | 13.8% |
| FedAvg IID（le=10） | 18.0% |
| FedAvg IID（le=3，等训练量） | **62.7%** |
| FedAvg Dirichlet | 8.4% |
| FedProx μ=0.01 | 11.7% |
| FedProx μ=0.1 | 5.0% |

而 H1 的判据阈值是 **5%**。也就是说：**同一个配置换一个 seed 就能让指标动
18%，比要检测的效应还大 3 倍。** 在这种噪声水平下，「相对劣化 3%」和
「相对劣化 8%」无法区分，按 5% 划线等于在噪声里挑结论。

**62.7% 这一格比 18% 更值得注意。** 训练量降到等训练量（le=3）后噪声从
18% 涨到 63%——不是「联邦更不稳定」，而是**欠拟合状态下不同 seed 落在
损失面的不同平底上，模型还没被训练量约束住**。这意味着：训练量不足时
连「跑三次取均值」都不可靠，任何在该工作点上做的比较都要先加 seed。

这不是「结果不好」，是**判据本身需要改**。三个可选做法，按代价排序：

1. **把阈值改成「与噪声同量级」**。例如「|相对差| < 1 个 seed 标准差即视为
   无差异」。代价：几乎必然判「无差异」，但那是诚实的。
2. **提高 seed 数到噪声可接受**。按 CV=18%、想分辨 5%，量级上需要 ~50 个
   seed。单个 run 约 3 分钟，50 seeds × 4 配置 = 10 小时。可行但要专门排期。
3. **改用配对比较**。同一 seed 下比较两个配置（同初始化、同划分），差的
   seed 波动，因为噪声被配对消掉了。这是最有效的做法，也是当前最该做的：
   现有 9 个 run 的 seed 是配对的（seed 0/1/2 跑遍所有配置），可以直接算
   配对差。

**在改判据之前，H1–H3 一律记为「未判定」。** 拿 3 个 seed、18% 噪声去宣布
「H1 成立」，比不报结论更糟。

#### 11.3.2 配对比较：哪一条其实能判定

`scripts/summarize_results.py --paired` 按 seed 相减，消掉 seed 带来的共同
波动。实测（3 seeds，配对）：

| 对比 | 配对差均值 | 相对波动 | 符号一致 | 可判定？ |
|---|---|---|---|---|
| FedAvg IID(le=10) − C0 | −0.00035 | 23.3% | 3/3 为负 | 否（波动 > 阈值） |
| FedAvg IID(le=3) − C0 | +0.01281 | 69.0% | 3/3 为正 | 否（方向稳但量级不可信） |
| FedAvg Dirichlet − C0 | −0.00067 | 37.9% | 3/3 为负 | 否 |
| FedProx μ=0.01 − C0 | −0.00020 | 41.1% | 3/3 为负 | 否 |
| FedProx μ=0.1 − C0 | +0.00016 | 175.0% | 1/3 为负 | 否（完全不可判） |
| FedProx μ=0.01 − FedAvg | +0.00047 | 38.0% | 0/3 为负 | 方向可判，量级不可判 |
| **FedProx μ=0.1 − FedAvg** | **+0.00083** | **2.7%** | **0/3 为负** | **是** |

结论分两类：

- **H3 可以明确否定，而且是被两端夹住的否定。** 两个 μ 点都比同划分的
  FedAvg 差：μ=0.01 差 +0.00047（0/3 方向一致）、μ=0.1 差 +0.00083
  （0/3 方向一致、配对波动仅 2.7%）。而 μ=0 的定义就是 FedAvg
  （`tests/test_train.py::TestFedProx::test_mu_zero_is_bitwise_fedavg` 钉住
  了这一点）。**所以改善只可能出现在 (0, 0.01) 这个区间里，而两端都指向
  更差，单调性又一致——在这个设置下不存在可用的 μ。**
  机理：每客户端仅 80 行训练数据（5 颗电池 × 16 循环），近端项
  `(μ/2)‖w − w_global‖²` 相对数据项过强，把参数拴在全局点上、限制本地拟合。
  **§14 坑 2 给的「μ 取 0.01–0.1」在这套数据规模下是错的**，那条建议来自
  每客户端上千样本的典型 FL 设定。
- **H1 / H2 不能判定。** 方向一致（3/3），但相对波动 23%/38% 远大于 5% 阈值。
  要判定必须加 seed（按 CV=18% 分辨 5%，量级上需 ~50 个）。

顺带两个反直觉但重要的观察：

- **FedProx 的 seed 波动（5.0%）比 FedAvg（8.4%）小，但它更差。**
  稳定性不等于正确性——μ 的强约束让所有 seed 收敛到同一个（更差的）点。
  只看「波动小」就选 FedProx 是错的，而且这个陷阱很容易踩：按 §11.3 的
  「seed 波动 <10%」这一行，FedProx 反而是唯一两个达标的配置之一。
- **等训练量下配对差是 +0.01281，相对 +772%，但配对波动 69%。** 方向三个
  seed 全一致（0/3 为负），所以「等训练量下联邦远差于中心化」这个**方向**
  是可信的；但具体倍数（8.7 倍）不可信，只应报成「数倍」。

另外两条实测约束也要写进任何后续判据：

- **训练量必须对齐**。联邦 100 轮 × 10 ep ≈ 1000 个本地 epoch，C0 是 300 个。
  不声明训练量的「相对劣化」没有意义，而且会反号（见 §10.2 第 1 条）。
- **Dirichlet α=0.5 在 N=23 时退化**（最大客户端 21 颗、8 个客户端各 1 颗），
  它测不到 H2 想测的异质性，需要 α 扫描。

---

## 12. 风险登记册

§14 的"坑清单"是**已知错误**；这里是**不确定性**。

| # | 风险 | 概率 | 影响 | 触发信号 | 缓解 |
|---|---|---|---|---|---|
| R1 | 划分泄漏导致结果虚高 | 中 | **致命**（结论作废） | `check_partition.py` 报 P-LEAK/P-TEST | 划分脚本先行 + 每次改配置重跑检测 |
| R2 | Non-IID 下 FedAvg 不收敛，被误判为代码 bug | 高 | 中（浪费 2-3 天） | 轮数上去 MAE 不降且客户端间差异大 | 先降 `local_epochs` 到 1，再上 FedProx；对照 §14 坑 2 |
| R3 | Flower 版本漂移导致骨架跑不通 | 中 | 中（0.5-1 天） | ImportError / 参数名不匹配 | 锁 1.41.0；第一天先跑官方 quickstart |
| R4 | FedBN/Scaffold 接入成本超预期 | 高 | 中（砍功能） | 自实现聚合后收敛行为与文献不符 | 标为可选；时间紧直接砍，报告里写"未做" |
| R5 | DP 下模型完全学不动（ε 太小） | 中 | 中（H4 变否） | noise=2.0 时 MAE 接近常数预测 | 这就是结论之一；报告写"该规模下不可用"，不硬凑 |
| R6 | 仿真结论被外推成"可部署" | 高 | 高（评审直接否） | 报告里出现"可部署""可用于生产" | §14 坑 6 写进报告讨论节；表 2 给通信量 |
| R7 | 单机 CPU 训练太慢，跑不完矩阵 | 中 | 中（砍 α 扫描） | 单 run 墙钟 > 30 min | MLP 只有 13.6 万参数；先测单 run 时间再定矩阵规模 |
| R8 | 两套 SOH 标签口径 | 低 | 高（结论不可比） | 与既有 MIT 复现项目的 MAE 对不上 | 复用同一预处理管线；口径变更必须在协议里显式声明 |

**概率/影响的判定依据**：基于 v1 已识别的坑位与常见 FL 复现失败模式的主观分级，不是统计结论。

---

## 13. 里程碑与 DOD

人日按**有效人日**计（不含等待与上下文切换）。

| 里程碑 | 内容 | 人日 | DOD（完成定义） |
|---|---|---|---|
| **M1** | 环境 + quickstart + MIT 数据接入 + 划分脚本 | 4 | ①flwr 官方仿真运行时（ray 后端）跑通本仓库的 ClientApp/ServerApp ②换成 MIT 数据后 IID 联邦与中心化差距 <5%（口径见下） ③`check_partition.py` 全 PASS |
| **M2**（MVP 截止） | FedAvg/FedProx × IID/Non-IID，3 seeds，曲线与表 | 4 | ①四条配置各有 3 个 run 的 JSON ②图 1、表 1 出图 ③H1-H3 各有 accept/reject ④2-3 页备忘 |
| **M3** | FedAdam/FedBN + α 扫描 + 跨化学体系 | 4 | ①α 四档各 3 seeds ②FedBN 在跨化学体系场景有量化对比 ③图 2 |
| **M4** | Opacus DP + 隐私-效用曲线 + 通信量表 | 3 | ①≥3 个 noise 点各带 ε ②图 3、表 2 ③H4 有结论 |
| **M5** | Schema v1 + 模拟生成器 + 端到端联调 + 完整报告 | 4 | ①伪会话数据能走完"入库→特征→训练→上传→聚合" ②报告含威胁模型/协议/模拟-部署差距三要素 |

合计 **19 人日 ≈ 4-5 周**（兼职按 1.5 倍放大）。**M2 是 go/no-go 关卡**：MVP 验收不过不进入 M3。

### 13.1 M1 的实际达成情况（2026-10-24）

| DOD | 状态 | 依据 |
|---|---|---|
| ① `flwr run` 跑通官方 MNIST | **达成**（在 CI 上，非本机） | `scripts/ci_smoke_flwr.py` 在 GitHub Actions 的 Linux runner 上跑通 **flwr 1.41.0 + ray 2.55.1** 官方仿真：2 轮、4 客户端、真实 `Aggregated MetricRecord`（`train_loss` 0.00275 → 0.00084，`mae` 0.0377 → 0.0418），输出 `SMOKE OK`。本机因中文主机名让 raylet 的 protobuf 非法 UTF-8 而 abort（§14 坑 14、`docs/env-limitations.md`），这条路径**本机不可替代** |
| ② MIT 数据后 IID 联邦与中心化差距 <5% | **未达成，且该 DOD 的写法本身有缺陷** | 见下 |
| ③ `check_partition.py` 全 PASS | **达成** | IID / Dirichlet α=0.5 两份产物均对账通过，`errors=0`，仅 1 条可接受 WARN |

**①为什么不能写成「`flwr run` 跑通 MNIST」**：原 DOD 是照 quickstart 抄的，
但本仓库的 app 不是 MNIST 分类器，跑官方 MNIST 只能证明「装得上」，不能
证明「本仓库的 ClientApp/ServerApp 能在官方运行时里跑」。所以冒烟改成本
仓库自己的 app + 合成数据：验证的是 client 在独立 ray actor 里各自加载
数据 → `num-examples` 加权聚合 → 全局评估这条链路。

**①仍未覆盖的**：`flwr run` CLI（fab 打包）这条路径。它要求
`[project].dependencies` 含 flwr，而本仓库的 `dependencies = []` 是刻意的
（检查器要能纯标准库跑），所以走的是 `run_simulation` Python API。
「fab 能打包」这件事没被验证，见 `docs/env-limitations.md`。

**②为什么未达成，以及为什么这条 DOD 要改写。**

原写法「IID 联邦与中心化差距 <5%」把两件事压进一句话：**「联邦能追平中心化」**
和**「训练量对齐」**。实测这两件事在本数据上给出**反号**的答案：

| 口径 | 联邦本地 epoch | MAE | 相对 C0 |
|---|---|---|---|
| 等训练量（`le=3`，100 轮） | 300（= C0 的 300 ep） | 0.01447 ± 0.00907 | **+772%** |
| 计划默认（`le=10`，100 轮） | 1000（C0 的 3.3 倍） | 0.00131 ± 0.00024 | **−20.9%** |

所以「<5%」只可能在**联邦拿到 3.3 倍训练量**时出现，而那不是公平比较；
在公平比较下差 8.7 倍。两个数都留着，但都不能拿来当 DOD 通过。

改写后的 DOD ②（M2 起生效）：

> **在声明的训练量口径下，IID 联邦与 C0 的配对差，其相对波动 < 50%（即可判定方向），
> 且该口径必须在协议里写明是「等本地 epoch」还是「等总梯度步」。**

这条是**可判定的**：等训练量口径下配对差 +0.01281、波动 69%（勉强不可判）；
`le=10` 口径下 −0.00035、波动 23.3%（可判方向）。而原 DOD 的「<5%」在
18%–63% 的 seed 噪声下**永远无法判定通过或不通过**——一条不可能被满足的
DOD 和没有 DOD 是一样的。

---

## 14. 常见坑清单

1. **全局标准化 = 隐私泄露**：特征均值/方差若用全量数据算，本身就是信息上传。必须**每客户端本地标准化**，并在报告里说明由此引入的输入分布差异——这恰恰是 Non-IID 的一部分。
2. **Non-IID 下 FedAvg 不收敛**：通常是 `local_epochs` 过多 + 无近端项。先降到 1-2，再上 FedProx 调 μ，**不要先怀疑代码**。
   **⚠️ 这条建议在 2026-10-24 被本仓库的实测同时否掉了两半，照抄会毁掉结果。**
   - 「先降到 1-2」：每客户端只有 80 行训练数据、`batch_size=64`，1 个 epoch 只有 2 步。le=2 的 100 轮共 400 步，欠拟合到 MAE 0.0222，会被误判成「联邦无效」。实测 le=2→0.02220、le=3→0.01447、le=10→0.00131、le=20→0.00120。
   - 「再上 FedProx 调 μ」：本数据规模下 μ 单调变差，不存在可用区间（μ=0→0.00098、0.01→0.00145、0.1→0.00182）。原因见 §11.3.2。
   
   正确的顺序是：**先确认训练量够（看损失还在不在降），再判断是不是 Non-IID 问题**。这两步反了会把欠拟合误诊成异质性。
3. **聚合加权错误**：FedAvg 应按客户端样本数加权平均。手写聚合时等权是高频 bug——用内置策略可规避。
4. **序列泄漏**：同一电池相邻循环被切进不同客户端/测试集 → 结果虚高。以电池为最小原子（§6.2 规则 1）。
5. **测试集污染**：每轮全局评估集必须与所有客户端训练数据不相交；本地 val 与全局测试两套指标不可混报。
6. **模拟规模 ≠ 真实部署**：仿真无网络延迟、掉线、算力异构。至少用 `fraction_fit<1` 模拟部分掉线、用表 2 量化通信量，不要把仿真结论外推成"可部署"。
7. **通信量低估**：开工前先算模型字节数 × 客户端数 × 轮数。MLP 没问题，换 1D-CNN/LSTM 时这张表就是生死线。
8. **随机性掩盖效应**：客户端抽样顺序、初始化都是随机的。结论必须 ≥3 seeds，差异小于标准差的"提升"不予采信。
9. **Flower 版本漂移**：锁版本 + 在 requirements 与报告里写版本号；先跑通官方 quickstart 再改代码。
10. **两套 SOH 标签口径**：与既有 MIT 复现项目统一标签定义与预处理；任何口径变更必须在协议里显式声明。
11. **标签未中心化 = 模型学不动**（2026-10-24 实测，代价 73 倍）。SOH 落在 1.0 附近而 MLP 输出初值在 0 附近，MSE 损失几乎全被「偏置差 1.0」占据，真实信号（std 仅 0.006）被淹没，模型只学会把输出推向 1。C0 中心化基线 300 ep：不中心化 0.11333、中心化到 **1.0** 得 0.00155。中心化基准取 **1.0**（SOH 的物理定义）而不是训练集均值——后者是数据统计量，联邦场景下用它等于上传信息；实测两者差 0.6%，在 seed 波动内。执行点：`CellDataset`，评估方用 `label_offset(loader)` 加回。
12. **`local_epochs` 不是自由超参，它决定总训练量**（同批实测，代价 27 倍）。每客户端只有 5 颗 × 16 循环 = 80 行，`batch_size=64` 时每 epoch 只有 2 步，`local_epochs=2` 的 100 轮共 400 步，而 C0 是 300 ep × 36 batch ≈ 10800 步。IID 实测：le=2 → 0.02220、le=4 → 0.00483、le=10 → 0.00131、le=20 → 0.00120（C0 = 0.0016）。**追平 C0 需要约 1000 个本地 epoch，是它的 3.3 倍**——这是每轮聚合后 Adam 动量归零的真实成本，不是 bug，必须写进报告。反过来，只按 §14 坑 2「先降到 1-2」会在 IID 下把欠拟合误判成「联邦无效」。
13. **客户端内按电池切 train/val 是错的**（注释写对了、代码没做）。`client_app` 原来写 `cells[:split] / cells[split:]`，那是按电池切：只有 1 颗电池的客户端会得到空 val 集，而 5 颗电池的客户端测的是「能不能认出另一颗电池」。§6.2 规则 1 要的是**同一批电池的前 80% 循环训练、后 20% 循环验证**。执行点：`make_client_loaders_by_time`；`tests/test_app.py::TestNumExamplesReported` 断言 16/4（20 循环的 80%/20%）。
14. **官方仿真运行时可能根本跑不起来，而错不在代码**。flwr 1.41 的仿真后端只有 ray；ray 的 raylet 用 ANSI API 取主机名，中文主机名（GBK）写进 protobuf 时不是合法 UTF-8，raylet 直接 abort。`RAY_raylet_start_wait_time_s`、`_node_ip_address`、`COMPUTERNAME` 覆盖均无效。另一条：`ray==2.55.1` 没有 cp313 的 Windows wheel，Windows 上必须用 Python 3.11/3.12。绕行方案与等价性论证见 `docs/env-limitations.md`，官方运行时在 CI 的 Linux runner 上冒烟。
15. **「相对劣化 X%」不声明训练量就会反号**（2026-10-24 实测，代价是差点写出一条错结论）。联邦 `le=10` 时相对 C0 是 **−20.9%**，同一份划分、同一批 seed、只把本地 epoch 从 10 降到 3（与 C0 的 300 ep 对齐），变成 **+772%**。报告里每个相对差都必须带三个限定：本地 epoch 数、轮数、C0 的 epoch 数。缺任何一个，读者无法判断这是「联邦更强」还是「联邦多训了 3.3 倍」。
16. **判据阈值不能小于 seed 噪声，否则那条判据等于没写**。本仓库实测 seed 相对标准差 5%–63%（配置间差 12 倍），而判据阈值写的是 5%。结果是所有配置都「不能判定」，判据既不能接受也不能拒绝任何假设。写判据时必须先量噪声，再定阈值；阈值与噪声同量级时改用**配对比较**（同 seed 相减）——本仓库靠它才判掉了 H3。

---

## 15. 验收标准

**MVP（M2 末）**

- `flwr run` 一键复现；锁版本 + 固定 seed 后 3 次运行最终 MAE 相对波动 <10%
- 中心化 / IID-FedAvg / Non-IID-FedAvg / Non-IID-FedProx 四条配置的曲线与表 1
- FedProx 相对 FedAvg 的 MAE 改善被明确量化（**正负都要报**）
- `check_partition.py` 与 `check_protocol.py` 全 PASS

**完整版（M5 末）**

- α 扫描与跨化学体系实验各 ≥3 seeds；FedBN 增益被量化
- 隐私-效用曲线覆盖 ≥3 个 noise 倍数，每点附 ε
- Schema v1 + 模拟生成器通过端到端联调
- 报告含威胁模型声明、实验协议、模拟-部署差距讨论三要素，缺一退回

---

## 16. 参考文献

1. McMahan et al. **Communication-Efficient Learning of Deep Networks from Decentralized Data**. AISTATS 2017.（FedAvg）
2. Li et al. **Federated Optimization in Heterogeneous Networks**. MLSys 2020.（FedProx）
3. Karimireddy et al. **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**. ICML 2020.
4. Li et al. **FedBN: Federated Learning on Non-IID Features via Local Batch Normalization**. ICLR 2021.
5. Reddi et al. **Adaptive Federated Optimization**. ICLR 2021.（FedAdam / FedYogi）
6. Severson et al. **Data-driven prediction of battery cycle life before capacity degradation**. Nature Energy 4, 383-391, 2019.
7. Beutel et al. **Flower: A Friendly Federated Learning Research Framework**. arXiv:2007.14390.
8. Abadi et al. **Deep Learning with Differential Privacy**. CCS 2016.（DP-SGD）
9. Mironov. **Rényi Differential Privacy**. CSF 2017.（RDP 会计器）
10. Yousefpour et al. **Opacus: User-Friendly Differential Privacy Library in PyTorch**. arXiv:2109.12298.

> 版本信息（2026-10-22 核对）：`flwr` 1.41.0（PyPI），`opacus` 1.6.0（PyPI，要求 torch ≥2.6.0）。撰写时以官方 quickstart 的 `server_app.py` / `client_app.py` 为 API 依据。

---

## 17. 术语表

| 术语 | 含义 |
|---|---|
| SOH | State of Health，健康状态，本项目用容量保持率作标签 |
| Non-IID | 各客户端数据分布不同。分三类：量偏斜、特征/标签偏斜、域偏移 |
| ΔQ(V) | 差分容量-电压曲线，Severson 2019 的特征构造方式 |
| FedAvg | 按样本数加权平均客户端参数 |
| FedProx | FedAvg + 近端项 `μ/2·‖w−w_t‖²`，抑制客户端漂移 |
| FedBN | BN 层不参与聚合，各客户端本地保留 |
| Scaffold | 用控制变量校正客户端漂移 |
| DP-SGD | 逐样本梯度裁剪 + 加噪的随机梯度下降 |
| ε (epsilon) | 差分隐私预算，越小越私密；必须配合 δ 一起报 |
| RDP | Rényi DP，把多步噪声累积换算成 ε 的会计方法 |
| 划分原子 | 划分时不可再分的最小单位。本项目 = 一颗电池 |
| fraction_fit | 每轮参与训练的客户端比例 |
| straggler | 训练慢的客户端，仿真里通常不模拟 |
| 中心化上限 (C0) | 所有数据合起来训一个模型，作为联邦的精度上限参考 |
| 本地独立训练 (C1) | 各客户端只用自己的数据，作为联邦的下限参考 |

---

*v2.0 完。落地第 1 天以 `flwr==1.41.0` 官方 quickstart 实测结果做一次 API 核对；若官方已升版，以实测为准并回写本文件。*
