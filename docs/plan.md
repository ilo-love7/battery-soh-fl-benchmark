# 联邦学习式充电数据众包研究平台 — 执行方案 v2.0

> 基于 v1.0（严谨-审稿专家，2026-10-18）修订 ｜ 2026-10-22
> 面向已有 Python 与模型训练基础的单人执行者。数据口径与既有锂电池项目（MIT-Stanford / NASA PCoE / CALCE）保持一致。
> 本文件是**方案**，不是结果报告。仓库里目前没有任何实验数字，§10 的结果表全部为空，填入时间见 §13。

---

## v2.0 相对 v1.0 改了什么

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

| 编号 | 假设 | 支持（accept） | 推翻（reject） | 判定依据 |
|---|---|---|---|---|
| H1 | IID 划分下，联邦与中心化的差距可以忽略 | 同测试集 MAE 相对劣化 ≤ 5% | > 10% | §11 的 C0 vs I1 |
| H2 | Non-IID（按电池 + Dirichlet α=0.5）下 FedAvg 会显著掉点 | 相对劣化 > 10% | ≤ 5%（说明电池级划分没有想象中异质） | §11 的 C0 vs N1 |
| H3 | FedProx 能挽回一部分 H2 的损失 | 相对 FedAvg 改善 ≥ 5% 且超过 1 个标准差 | 改善 < 1 个标准差，或为负 | §11 的 N1 vs N2 |
| H4 | 本地 DP 的精度代价随噪声单调上升，且在 ε ≤ 8 时仍可用 | noise=1.0 处 MAE 劣化 ≤ 30% 且 ε ≤ 8 | 劣化 > 50%，或 ε 报不出来 | §11 的 P 系列 |

**"支持"不等于"证明"**：H2 成立只说明这批电池按此划分确实异质，不能外推成"所有联邦场景都掉点"。报告里必须写成"在 N=25、α=0.5、MIT 124 颗的条件下"。

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
│   ├── data.py               # MIT 数据加载 + 特征提取（需 torch/pandas）
│   ├── model.py              # MLP 定义
│   └── app/                  # Flower App（需 flwr，不在 CI 里跑）
│       ├── client_app.py
│       └── server_app.py
├── scripts/
│   ├── check_partition.py    # 划分泄漏检测 CLI
│   └── check_protocol.py     # 实验矩阵校验 CLI
├── tests/                    # 只覆盖纯标准库部分
└── docs/
```

**分层原则**：`partition.py` 与 `protocol.py` 不依赖 torch/flwr，因此可以在任何机器上跑、可以进 CI。这两个模块承载的是**最容易出错且最贵的逻辑**（划分泄漏、协议缺项），值得脱离深度学习栈单独测试。训练代码依赖重，不进 CI。

### 3.2 模块契约

| 模块 | 输入 | 输出 | 不许做什么 |
|---|---|---|---|
| `partition.py` | 电池 ID 列表、每客户端电池数、划分方案、seed | `{client_id: [cell_id, ...]}` | 不许碰特征、不许读原始数据、不许依赖 pandas |
| `protocol.py` | 一份 TOML 配置 | `[(level, code, message)]` | 不许改配置、不许猜默认值 |
| `data.py` | 电池 ID、本地训练/验证电池集合 | 该客户端的 `DataLoader` 对 | **不许做全局标准化**（见 §14 坑 1） |
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

- **输入**：部分充电电压曲线差分特征。沿用 Severson et al. 2019 的 ΔQ_{100-10}(V) 思路，取 1000 维（或自选电压区间）。**只用充电片段**，这与"众包 App 只采充电数据"的设定一致。
- **结构**：1000 → 128 → 64 → 1（SOH 回归）
- **参数量**：`1000×128+128 + 128×64+64 + 64×1+1 = 136,449`
- **单轮上行通信量**：`136,449 × 4 B = 545,796 B ≈ 0.52 MiB`（v1 写 0.6 MB，是估算；表 2 用精确值）
- **优点**：复用既有特征工程与 SOH 标签口径；模型小 → 轮数可以多跑，统计结论更稳

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

MIT-Stanford 124 颗电池天然适合按对象划分。

### 6.1 四种方案

| 方案 | 划分方式 | 模拟的真实场景 | 建议 |
|---|---|---|---|
| A 按电池 | 每客户端 = k 颗。留出 24 颗作全局测试后剩 100 颗：k=1/2/5 → 100/50/20 客户端 | 一人一块电池 | **MVP 主方案** |
| B 按用户代理 | 按 batch 分组 + 额定容量档（1.1/1.2/1.5 Ah）聚类 | 同一用户多设备 | 进阶 |
| C 按化学体系/数据源 | MIT / NASA / CALCE 各成一组 | 不同电池厂众包 | 完整版必做（极端域偏移） |
| D Dirichlet 样本分配 | 按 Dirichlet(α) 把电池分给 N 客户端 | 量化任意 Non-IID 程度 | 完整版 α 扫描 |

**算术修正（v2.0）**：v1 写「k=5 → 25 客户端」，那是 124÷5 的取整。但 124 不能被 5 整除，且没扣掉全局测试集。按 §6.2 规则 2 整颗留出 24 颗后可用 100 颗，k=5 得 20 个客户端。`configs/*.toml` 与 `check_protocol.py` 一律按 20 算。

**Dirichlet 在这里会退化**：`check_partition.py` 的实测输出显示，N=20、α=0.5 时有一个客户端拿到 25 颗、7 个客户端各只有 1 颗；α=0.1 时更极端（1 个客户端 65 颗）。原因是 Dirichlet 分配的是**电池**而不是样本，客户端一多、α 一小，分布自然退化成「一人独占」。这不是 bug，但要写进报告——此时「量偏斜」与「域偏移」已经混在一起，不能再声称只考察了其中一种。

### 6.2 三条硬规则

1. **同一颗电池的相邻循环不得跨客户端**。时间序列截断会让泛化虚高——划分以"电池"为最小原子，客户端内再按时间切 train/val。
2. **全局测试集独立于所有客户端数据**；客户端本地 val 另设，两套指标分开报，不许混。
3. **客户端数量与轮数写进协议**：典型 N=20（5 颗/客户端，100 颗可用）、`num_rounds=100`；N=50 时 `fraction_fit` 降到 0.4，否则单轮墙钟翻倍。

### 6.3 泄漏检测（可执行）

规则 1 靠人眼核对 124 颗电池不现实，因此做成脚本：

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

表 2 用 §4.1 的精确值：`136,449 × 4 × 2 = 1,091,592 B/客户端/轮 ≈ 1.04 MiB`。N=25、100 轮 → 约 **2.60 GiB** 总流量（不含协议开销）。

### 10.2 结果表（当前为空）

**本仓库尚无实验结果。** 下表在 W2 末填第一批，W5 末填全。留空不是遗漏，是纪律——方案阶段填数字就是造假。

| 配置 | 策略 | 客户端数 | 划分 | MAE (mean±std) | RMSE | 收敛轮数 |
|---|---|---|---|---|---|---|
| C0 | 中心化 | — | — | — | — | — |
| I1 | FedAvg | 20 | IID | — | — | — |
| N1 | FedAvg | 20 | Dirichlet α=0.5 | — | — | — |
| N2 | FedProx (μ=?) | 20 | Dirichlet α=0.5 | — | — | — |

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
| **M1** | 环境 + quickstart + MIT 数据接入 + 划分脚本 | 4 | ①`flwr run` 跑通官方 MNIST ②换成 MIT 数据后 IID 联邦与中心化差距 <5% ③`check_partition.py` 全 PASS |
| **M2**（MVP 截止） | FedAvg/FedProx × IID/Non-IID，3 seeds，曲线与表 | 4 | ①四条配置各有 3 个 run 的 JSON ②图 1、表 1 出图 ③H1-H3 各有 accept/reject ④2-3 页备忘 |
| **M3** | FedAdam/FedBN + α 扫描 + 跨化学体系 | 4 | ①α 四档各 3 seeds ②FedBN 在跨化学体系场景有量化对比 ③图 2 |
| **M4** | Opacus DP + 隐私-效用曲线 + 通信量表 | 3 | ①≥3 个 noise 点各带 ε ②图 3、表 2 ③H4 有结论 |
| **M5** | Schema v1 + 模拟生成器 + 端到端联调 + 完整报告 | 4 | ①伪会话数据能走完"入库→特征→训练→上传→聚合" ②报告含威胁模型/协议/模拟-部署差距三要素 |

合计 **19 人日 ≈ 4-5 周**（兼职按 1.5 倍放大）。**M2 是 go/no-go 关卡**：MVP 验收不过不进入 M3。

---

## 14. 常见坑清单

1. **全局标准化 = 隐私泄露**：特征均值/方差若用全量数据算，本身就是信息上传。必须**每客户端本地标准化**，并在报告里说明由此引入的输入分布差异——这恰恰是 Non-IID 的一部分。
2. **Non-IID 下 FedAvg 不收敛**：通常是 `local_epochs` 过多 + 无近端项。先降到 1-2，再上 FedProx 调 μ，**不要先怀疑代码**。
3. **聚合加权错误**：FedAvg 应按客户端样本数加权平均。手写聚合时等权是高频 bug——用内置策略可规避。
4. **序列泄漏**：同一电池相邻循环被切进不同客户端/测试集 → 结果虚高。以电池为最小原子（§6.2 规则 1）。
5. **测试集污染**：每轮全局评估集必须与所有客户端训练数据不相交；本地 val 与全局测试两套指标不可混报。
6. **模拟规模 ≠ 真实部署**：仿真无网络延迟、掉线、算力异构。至少用 `fraction_fit<1` 模拟部分掉线、用表 2 量化通信量，不要把仿真结论外推成"可部署"。
7. **通信量低估**：开工前先算模型字节数 × 客户端数 × 轮数。MLP 没问题，换 1D-CNN/LSTM 时这张表就是生死线。
8. **随机性掩盖效应**：客户端抽样顺序、初始化都是随机的。结论必须 ≥3 seeds，差异小于标准差的"提升"不予采信。
9. **Flower 版本漂移**：锁版本 + 在 requirements 与报告里写版本号；先跑通官方 quickstart 再改代码。
10. **两套 SOH 标签口径**：与既有 MIT 复现项目统一标签定义与预处理；任何口径变更必须在协议里显式声明。

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
