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
| H1 | IID 划分下，联邦与中心化的差距可以忽略 | 同测试集 MAE 相对劣化 ≤ 5% | > 10% | §11 的 C0 vs I1 | **推翻**（le=10 时 −23.5%，p=0.0006；le=3 等训练量时 +640.6%，p=0.0077。两个口径都远超 10%） |
| H2 | Non-IID 下 FedAvg 会显著掉点 | 相对**同预算 IID FedAvg** 劣化 > 10% | ≤ 5%（说明电池级划分没有想象中异质） | §11 的 I1 vs N1/P1 | **接受**（偏斜扫描四档里两档可判定且都掉点：k=10 偏斜 39.2% vs IID 4.7% → **+50.9%**；k=5 偏斜 47.1% vs IID 17.3% → **+24.2%**，p_t=0.0013、p_sign=0.0078。k=2 未判定（p_t=0.21，训练量结构下测不出）、k=1 未判定（处理未施加）。Dirichlet α=0.5 记**未判定（处理未施加）**：标签偏斜 12.2% 比 IID 还小。见 §11.3.4） |
| H3 | FedProx 能挽回一部分 H2 的损失 | 相对 FedAvg 改善 ≥ 5% 且超过 1 个标准差 | 改善 < 1 个标准差，或为负 | §11 的 N1 vs N2 | **推翻**（μ=0.01 差 +46.7%、μ=0.1 差 +85.8%，均 p<0.001、0/8 方向一致；两端夹住，见 §11.3.5） |
| H4 | 本地 DP 的精度代价随噪声单调上升，且在 ε ≤ 8 时仍可用 | noise=1.0 处 MAE 劣化 ≤ 30% 且 ε ≤ 8 | 劣化 > 50%，或 ε 报不出来 | §11 的 P 系列 | **一半支持、一半推翻**（单调性支持：MAE 随 nm 严格递增，四档 8/8 同向、p_sign=0.0078。可用性推翻：唯一达标的 nm=8（ε=7.84）劣化 +186.3%，比 C1 还差 34.7%。且「noise=1.0 处 ε ≤ 8」在本协议下结构上不可达——nm=1 的 ε 是 152.31。见 §13.4） |

**"支持"不等于"证明"**：H2 成立只说明这批电池按此划分确实异质，不能外推成"所有联邦场景都掉点"。报告里必须写成"在 MIT-Stanford 135 颗、标签偏斜 39.2%–47.1%、每客户端 5–10 颗电池的条件下"——M3 的扫描进一步显示，偏斜的代价还取决于**基线已经有多偏**（k=10 时 +50.9%，k=5 时 +24.2%），且客户端数本身（N）是独立的掉点来源（§10.1.2）。

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
| D8 | 隐私增强 | Opacus DP-SGD，≥3 个 noise 倍数的隐私-效用曲线，每点附 ε | **已做**（M4）：5 个扫描点、4 个带 ε、图 3 + 表 2 的 `D1@nm=*` 五行、H4 有结论。判定见 §13.4——「该规模下 DP 代价过高」，且 DOD 的字面条款「noise=1.0 处 ε ≤ 8」结构上不可达 |
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
| FedAdam / FedYogi | 各客户端梯度尺度差异大 | 服务器端 lr、β | ✅ `FedAdam` / `FedYogi` | **已做**（M3 ②，§13.3.2） |
| FedBN | 域偏移（跨化学体系） | BN 层不聚合 | ❌ 在 `baselines/` 里，需自行接入 | **未做**（两个前置项，见 §13.3.2） |
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

### 7.1 ε 不是自由变量（M4 落地后补，2026-10-25）

上面那段代码读起来像「nm 取小一点就能既私密又好用」。**这是错的**，而且
是这个仓库里最贵的一个误判，所以单独一节写清楚。

ε 由三个量决定，后两个**由训练协议钉死**，不是可以顺手调的旋钮：

    q     = 1 / 每 epoch 批次数 = 1 / ceil(每客户端行数 / batch_size)
    步数  = num_rounds × local_epochs × 每 epoch 批次数
    ε     = RDP(q, noise_multiplier, 步数) at δ

代入本仓库的真实协议（每客户端 5 颗电池 × 16 训练循环 = 80 行、
`batch_size=64`、100 轮、`local_epochs=3`）：

    q = 1/2 = 0.5        步数 = 100 × 3 × 2 = 600

    noise_multiplier     ε (δ=1e-5)    达到 acceptance.dp_epsilon_max=8？
    ─────────────────────────────────────────────────────────────────────
    0（对照组）           ∞             不是 DP 臂
    1.0                  152.31         否
    2.0                  49.11          否
    4.0                  18.65          否
    8.0                  7.84           **是**

**q=0.5 意味着 subsampling 几乎没有放大作用**（放大小 q 才有用），所以 ε
很快冲到几十几百。要让 ε 降到 8 以内，在这个协议下必须把 nm 抬到 8；而
nm=8 时加在批均值梯度上的噪声 Std = `8 × max_grad_norm / 期望批大小`
= `8 × 1.0 / 40` = **0.2**，与实测信号（批均值梯度范数 0.147–0.223）
**同量级**。

所以 H4 的两半在这里迎面撞上：**要 ε ≤ 8 就要把噪声加到和信号一样大**。
这不是实现问题，是数据规模问题——每客户端只有 80 行，而 DP-SGD 的代价
随 `1/√(每客户端样本数)` 恶化。执行这个测算的命令：

    python scripts/check_dp_budget.py --config configs/dp.toml

它同时是 CI 的一道闸门：声明的扫描点里必须至少有一个达到
`acceptance.dp_epsilon_max`，否则 §11.3 的「ε ≤ 8 时劣化 ≤30%」一条都用
不上，而扫描照跑、图照出。**扫 nm 之前先算预算**，否则可能出现三档全都
ε>8 的点，那时唯一的结论只能写成「该规模下不可用」（§12 的 R5 早就写了
这一条，不硬凑）。

三条降压路径都写在检查器的失败输出里：加大 nm（效用更差）、减步数（减轮
数或减 `local_epochs`，训练量更少）、减 q（减小 `batch_size`——但每 epoch
批次数变多、**总步数反而上升**，所以不一定有用，别凭直觉调）。


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
| 图 2 | 标签偏斜扫描 | x=**标签偏斜**（客户端身份解释的方差占比，不是 α、不是 k）；sohrank 与同 k 的 IID 各一条线，两线在同一 k 处的竖直距离就是偏斜的净效应 |
| 图 3 | 隐私-效用曲线 | noise_multiplier vs MAE，每点标注 ε |
| 表 2 | 通信量估算 | 参数量 × 4 B × 2（上/下行）× 客户端数 × 轮数 |

表 2 用 §4.1 的精确值：`136,449 × 4 × 2 = 1,091,592 B/客户端/轮 ≈ 1.04 MiB`。N=23、100 轮 → **2,510,661,600 B ≈ 2.34 GiB** 总流量（不含协议开销）。

### 10.1.1 图 1 已出（2026-10-24）

文件：`docs/figures/fig1-rounds-mae.png` / `.pdf`，由
`scripts/make_figure1.py` 从 `results/*.json` 的 `curve` 字段生成，不手绘。

左 panel 判 H1（IID 的两种训练量口径 + C0 水平参考），右 panel 判 H2/H3
（Dirichlet 的 FedAvg vs FedProx 两个 μ）。

画法上三个决定，都是为了不让图比数据更自信：

1. **每个 seed 画一条细线，均值画粗线；不画 ±1σ 带。** 第一版画的是
   ±1σ，结果 n=3 时某轮 `sd > mean`，`fill_between` 的下界被夹成一条掉到
   1e-6 的尖刺——图上看起来像「某轮 MAE 近乎为零」，其实只是带宽跨过
   均值时的假象。**seed 少的时候，把每个 seed 都画出来比任何汇总量都诚实。**
2. **C0 画成水平带，不画曲线。** C0 的 `curve` 字段是**空的**——它是一次性
   300 epoch 训练，没有「轮」这个概念。给它编一条 x 轴会让读者以为它和
   联邦是同一个时间尺度。
3. **y 轴取对数。** MAE 从 0.03 降到 0.001，线性轴上前 20 轮就贴底，正好
   把最有信息量的收敛尾部压扁。

图 2（标签偏斜扫描）与图 3（隐私-效用）**尚未出**，分别属 M3 与 M4。

### 10.1.2 图 2 的口径（2026-10-24 改写）

原计划图 2 画「α ∈ {0.1, 0.5, 1.0, IID} 柱状图」。**这个口径是错的**，理由是
§13.3：α 控制的是**数量偏斜**（客户端样本量的离散程度），而 M2 已证明 H2 的
效应来自**标签偏斜**（客户端之间 SOH 分布的差异）。扫 α 会得到一个
「α 越小越不掉点」的假趋势，而它测的是聚合权重的集中度，不是 H2 的机制。

改写后：**横轴是标签偏斜**，纵轴是 MAE。每一档 k 同时跑 sohrank 与 IID 两份
划分——同 k 保证 N 与每客户端数据量相同，只改「哪些电池分给谁」——所以两条线
在**同一 k 处**的竖直距离才是偏斜的净效应。实测矩阵（各 8 seeds）：

| k | N | IID 偏斜 | sohrank 偏斜 | IID MAE | sohrank MAE | 配对差（相对 IID） | 判定 |
|---|---|---|---|---|---|---|---|
| 10 | 12 | 4.7% | 39.2% | 0.00093 ± 0.00005 | 0.00140 ± 0.00005 | **+50.9%**（0/8 为负，p_t<0.0001） | 接受 |
| 5 | 23 | 17.3% | 47.1% | 0.00120 ± 0.00019 | 0.00150 ± 0.00008 | **+24.2%**（8/8 同向，p_t=0.0013） | 接受 |
| 2 | 58 | 32.5% | 57.2% | 0.00372 ± 0.00313 | 0.00350 ± 0.00289 | −6.0%（p_t=0.21） | 未判定（噪声） |
| 1 | 115 | 70.4% | 70.4% | 0.00426 ± 0.00361 | 0.00421 ± 0.00378 | −1.2%（p_t=0.78） | 未判定（处理未施加） |

四条读数：

1. **偏斜确实掉点，幅度取决于「基线已经有多偏」。** k=10 时 IID 几乎不偏
   （4.7%），把偏斜拉到 39.2% 的代价是 +50.9%；k=5 时基线已有 17.3%，
   同样拉偏的边际代价降到 +24.2%。偏斜的「余量」越小，再拉偏越贵。
2. **只看 sohrank 一条线会读错因果。** sohrank 的 MAE 从 k=10 的 0.00140
   涨到 k=1 的 0.00421（3 倍），很容易读成「偏斜 70% 让精度掉 3 倍」。
   对照显示 IID 在偏斜只有 32.5% 时就已经是 0.00372——**同一个 MAE 水平
   在更低偏斜下就达到了**。k≤2 的劣化主因是 N（每客户端数据太少），不是偏斜。
3. **k=2 的偏斜效应不可分辨**：配对差 −6.0%、p_t=0.21。这一档每客户端
   16–32 行、每 epoch 只 1 步，seed 间波动 83%——不是「偏斜无害」，是
   **这一档的训练量结构下任何 20% 量级的效应都测不出来**。
4. **k=1 那一行是天然零效应对照，实测吻合了理论预测。** 一颗电池一个客户端
   时，IID 与 sohrank 的偏斜完全相同（70.4%），两份划分的**分组也完全相同**
   （115/115 个 singleton 重叠）——「谁拿哪颗」只改客户端标签。此时每客户端
   同 seed、同样 16 行、每 epoch 同样只有 1 个 batch，per-cell 更新只取决于
   数据与当前全局参数；聚合是等权平均，按客户端 ID 排序后顺序求和——**同一组
   115 个更新按不同顺序相加**。理论预期是吻合到浮点舍入量级（100 轮混沌训练
   会把它放大，但放不大的系统差异不存在）。实测配对差 **−1.2%**
   （p_t=0.78，逐 seed 差 −0.0005～+0.0011），与预期一致：**这一档是整张图
   的内部校准点，不是处理组**。若它测出显著差异，说明管线里存在未被察觉的、
   依赖客户端身份或顺序的环节，其他档的差异也不再可信。

> **协议开销已实测（flwr 1.41.0，2026-10-22）**：`ArrayRecord.count_bytes()` 对上述模型返回 **546,630 B**，比载荷多 834 B。差额恒为 **139 B/张量**，与模型规模无关——那是 NumPy 序列化要写的 dtype 与 shape 元数据。占比 0.15%，本可忽略，但换 1D-CNN/LSTM 时张量数会变，必须重算。算式固化在 `src/flsoh/model.py: wire_bytes()`。
>
> **一个已经犯过两次的错**：v1 写「N=25、100 轮 → 约 2.60 GiB」。N=25 是 124÷5 的取整（正确值 20），而即便按 N=25 算也是 2.54 GiB。v2.0 修成 2.03 GiB，但那个 2.03 仍然建在错颗数上——颗数从 124 改成 135 之后 N=23，正确值是 2.34 GiB。**派生表抄了错的输入，就会一直错下去**；而「输入」本身也会被上一步推翻。所以表 2 的每个数都由 `round_trip_bytes()` 算出，并由 `tests/test_model.py::TestCommunicationBudget` 钉住。

### 10.1.3 图 3 的口径（2026-10-25，M4）

文件：`docs/figures/fig3-privacy-utility.png` / `.pdf`，由
`scripts/make_figure3.py` 从 `results/*.json` 生成，不手绘。

**横轴有两个，分左右两个面板。** 左面板是 `noise_multiplier`（实验里真正
拧的旋钮，也是复现要照抄的参数），每个点旁边标 ε；右面板是 ε 本身（字面
意义上的隐私-效用曲线）。只画左面板会让读者自己去做 nm→ε 的换算，而那个
换算恰恰是最容易出错的一步——**ε 不是自由变量**，见 §13.4。

三条参考线一条都不能少：

| 线 | 作用 |
|---|---|
| `nm=0` 机制对照 | 只做逐样本裁剪 + Poisson 抽样、不加噪，ε=inf。没有它，曲线左端到非 DP 基线之间的落差会被读成「小噪声很便宜」，而那其实是换批大小 + 裁剪的代价（§7 明确警告过） |
| 非 DP 同预算基线 | DP **一共**付出多少（含机制开销） |
| C0 中心化上限 | 联邦掉的那部分与 DP 掉的那部分要能分开看 |

`nm=0` **不画在横轴上**：它的 ε 是 ∞，对数轴上没有位置；画成水平参考线并
在图例里写明「无隐私保证」。把它画成一个点会让人以为「ε=∞ 也是一个 DP 档
位」——这正是 plan §7 第二条常见错误。

y 轴取对数：nm 从 0 到 8，MAE 跨一个量级以上，线性轴会把小噪声那一段压成
一条平线，而那一带正是判据所在的位置。每个 seed 画一个淡点、均值画实心点
+ 标准差误差棒，与图 1/图 2 同一套约定。

**参考线必须与 DP 臂同策略、同划分、同 N。** 第一版只按 `local_epochs`
过滤，结果「非 DP 同预算」抓到了同 le 的 FedAdam（0.00162）而不是 FedAvg
（0.01165），参考线画低 7 倍而图上毫无异常（坑 24）。现在由
`make_figure3.pick_references()` 选，缺任一条就拒绝出图。右面板另有两块
浅色区域，合起来就是判据真正允许的范围：`ε ≤ 8`（竖条）与
`MAE ≤ 非 DP 同预算 × 1.3`（横条）。**两块交集为空**，即这个协议下一个点
都不满足——这比任何文字都直接。

### 10.2 结果表（第二批，2026-10-24）

MIT-Stanford，135 颗电池（115 训练 / 20 测试），100 轮，
Adam 1e-3、`batch_size=64`、标签中心化到 1.0，**每配置 8 seeds**。
除标注 `le=3` 的等训练量对照外，其余联邦配置均为 `local_epochs=10`；
**M4 的 `D1@nm=*` 五行也是 `le=3`**（要与非 DP 同预算比，见第 9 条）。
全部数字由 `scripts/summarize_results.py` 从 `results/*.json` 生成，不手抄。

表里 `I1@N=*` / `P1@N=*` 是 M3 的**标签偏斜扫描**（§10.1.2）：同一份数据、
同一个训练预算，只改分片粗细 k 与「哪些电池分给谁」。它们与 k=5 那几行
**不是同一组条件的重复**——N 不同，见下面第 7 条限制。`A1@*` 是 M3 ② 的
FedAdam（§13.3.2），其中 seeds=1 的带 `η=`/`τ=` 后缀行是探针，见第 8 条。

下面这块是**生成区**，由脚本写入、由 CI 逐字核对：

<!-- BEGIN GENERATED: results-table -->
| 编号 | 配置 | 策略 | 客户端数 | 划分 | seeds | MAE (mean±std) | RMSE | 相对 C0 |
|---|---|---|---|---|---|---|---|---|
| A1@le=3,η=0.01,τ=0.001 | FedAdam (η=0.01, τ=0.001, β=(0.9,0.99)) / IID | fedadam | 23 | iid | 8 | 0.00162 ± 0.00077 | 0.00229 ± 0.00076 | +2.9% |
| A1@le=10,η=0.001,τ=0.001 | FedAdam (η=0.001, τ=0.001, β=(0.9,0.99)) / IID | fedadam | 23 | iid | 1 | 0.00340 | 0.00413 | +116.4% |
| A1@le=10,η=0.01,τ=1e-06 | FedAdam (η=0.01, τ=1e-06, β=(0.9,0.99)) / IID | fedadam | 23 | iid | 1 | 0.00173 | 0.00312 | +9.8% |
| A1@le=10,η=0.01,τ=1e-05 | FedAdam (η=0.01, τ=1e-05, β=(0.9,0.99)) / IID | fedadam | 23 | iid | 1 | 0.00128 | 0.00177 | -18.7% |
| A1@le=10,η=0.01,τ=0.001 | FedAdam (η=0.01, τ=0.001, β=(0.9,0.99)) / IID | fedadam | 23 | iid | 8 | 0.00123 ± 0.00051 | 0.00175 ± 0.00058 | -21.9% |
| A1@le=10,η=0.1,τ=0.001 | FedAdam (η=0.1, τ=0.001, β=(0.9,0.99)) / IID | fedadam | 23 | iid | 1 | 0.00268 | 0.00365 | +70.3% |
| C0 | 中心化 | centralized | — | — | 8 | 0.00157 ± 0.00021 | 0.00288 ± 0.00030 | — |
| C1 | 本地独立 | local | 23 | — | 8 | 0.03446 ± 0.00956 | 0.04170 ± 0.00994 | +2090.6% |
| D1@nm=0 | FedAvg / IID + DP 机制 (nm=0, C=1, 无保证) | fedavg | 23 | iid | 8 | 0.01621 ± 0.00189 | 0.02057 ± 0.00255 | +930.7% |
| D1@nm=1 | FedAvg / IID + DP (nm=1, C=1, ε=152.31) | fedavg | 23 | iid | 8 | 0.02947 ± 0.00408 | 0.03748 ± 0.00498 | +1773.7% |
| D1@nm=2 | FedAvg / IID + DP (nm=2, C=1, ε=49.11) | fedavg | 23 | iid | 8 | 0.03475 ± 0.00468 | 0.04404 ± 0.00575 | +2109.1% |
| D1@nm=4 | FedAvg / IID + DP (nm=4, C=1, ε=18.65) | fedavg | 23 | iid | 8 | 0.04066 ± 0.00534 | 0.05140 ± 0.00676 | +2484.8% |
| D1@nm=8 | FedAvg / IID + DP (nm=8, C=1, ε=7.84) | fedavg | 23 | iid | 8 | 0.04643 ± 0.00603 | 0.05860 ± 0.00787 | +2851.3% |
| I1@N=12,le=10 | FedAvg / IID | fedavg | 12 | iid | 8 | 0.00093 ± 0.00005 | 0.00138 ± 0.00010 | -40.8% |
| I1@N=23,le=3 | FedAvg / IID | fedavg | 23 | iid | 8 | 0.01165 ± 0.00777 | 0.01345 ± 0.00863 | +640.6% |
| I1@N=23,le=10 | FedAvg / IID | fedavg | 23 | iid | 8 | 0.00120 ± 0.00019 | 0.00198 ± 0.00037 | -23.5% |
| I1@N=58,le=10 | FedAvg / IID | fedavg | 58 | iid | 8 | 0.00372 ± 0.00313 | 0.00486 ± 0.00331 | +136.3% |
| I1@N=115,le=10 | FedAvg / IID | fedavg | 115 | iid | 8 | 0.00426 ± 0.00361 | 0.00557 ± 0.00376 | +171.1% |
| N1 | FedAvg / Dirichlet | fedavg | 23 | dirichlet α=0.5 | 8 | 0.00101 ± 0.00017 | 0.00149 ± 0.00022 | -35.9% |
| N2@μ=0.01 | FedProx (μ=0.01) / Dirichlet | fedprox | 23 | dirichlet α=0.5 | 8 | 0.00148 ± 0.00012 | 0.00211 ± 0.00020 | -6.0% |
| N2@μ=0.1 | FedProx (μ=0.1) / Dirichlet | fedprox | 23 | dirichlet α=0.5 | 8 | 0.00187 ± 0.00011 | 0.00267 ± 0.00021 | +19.0% |
| P1@N=12 | FedAvg / sohrank | fedavg | 12 | sohrank | 8 | 0.00140 ± 0.00005 | 0.00246 ± 0.00030 | -10.7% |
| P1@N=23 | FedAvg / sohrank | fedavg | 23 | sohrank | 8 | 0.00150 ± 0.00008 | 0.00266 ± 0.00020 | -4.9% |
| P1@N=58 | FedAvg / sohrank | fedavg | 58 | sohrank | 8 | 0.00350 ± 0.00289 | 0.00471 ± 0.00309 | +122.2% |
| P1@N=115 | FedAvg / sohrank | fedavg | 115 | sohrank | 8 | 0.00421 ± 0.00378 | 0.00552 ± 0.00389 | +167.8% |
<!-- END GENERATED: results-table -->

**生成区的规则**：手改任何一个数都会让 CI 的 `--check-plan` 红。要更新就
重跑脚本 `python scripts/summarize_results.py --write-plan docs/plan.md`，
再提交 `results/*.json` 与 `docs/plan.md` 两份改动——**表和原始数据必须同一次
提交进仓库**，否则读者无法验证表里的数是从哪来的。

**这张表不能按字面读。** 八条限制必须先看：

1. **训练量不对等，而且这一条能翻转结论。** 联邦是 100 轮 × 10 本地 epoch
   ≈ 1000 个 epoch，C0 是 300 个——联邦多训 3.3 倍。所以「−23.5%」不是
   「联邦比中心化强」，而是「联邦在 3.3 倍训练量下追平并略超」。
   为此专门补了等训练量对照 **I1@le=3**（100 轮 × 3 = 300 个本地 epoch，
   与 C0 的 300 ep 对齐）：**0.01165，相对劣化 +640.6%**。
   同一个策略、同一份划分、同一批 seed，只把训练量从 1000 ep 降到 300 ep，
   结论就从「好 23.5%」变成「差 6.4 倍」。**任何不声明训练量的相对差都没有
   意义**，这不是保守说法，是实测反号。
2. **判据的分辨力不够，但配对比较救回来了。** 判据阈值是 5%，而 seed 波动
   实测：C0 13.3%、IID(le=10) 15.9%、**IID(le=3) 66.7%**、Dirichlet 17.2%、
   SOH 分片 5.2%、FedProx 5.8%–8.1%；M3 的 k 扫描里 **I1@N=58 84.1%、
   I1@N=115 84.8%、P1@N=58 82.6%、P1@N=115 89.7%**。**阈值小于噪声**，
   且在训练量不足时噪声会爆炸。单看 mean±std 只能判出低噪声档；其余靠
   **按 seed 配对**的 t 检验（§11.3.2）。
3. **N1（Dirichlet α=0.5）的标签偏斜比 IID 还小，它没有施加 Non-IID 处理。**
   实测「客户端身份解释的标签方差占比」：IID 17.3%、Dirichlet **12.2%**、
   SOH 排序分片 47.1%（上界 70.4%）。Dirichlet 只造成**数量偏斜**（基尼 0.519，
   最大客户端 21 颗），而把 21 颗电池合成一个客户端会把电池间的 SOH 差异
   **平均掉**。所以「N1 比 IID 好 16.3%」不能读成「Non-IID 没掉点」——
   **处理没施加**，这一格记「未判定」而不是「推翻」（§11.3.4）。
   真正检验 H2 的是 **P1（SOH 排序分片）**：N、k、总量、预算全同，只改
   哪些电池分给谁，偏斜拉到 47.1%，结论是劣化 **+24.2%**（p_t=0.0013）。
4. **FedProx 是单调变差，不是「参数没调对」。** μ=0（即 FedAvg）0.00101 →
   μ=0.01 得 0.00148 → μ=0.1 得 0.00187。**μ 越大越差，且两端都差于 μ=0**。
   由于 μ→0 的极限就是 FedAvg，这条曲线在本设置下不存在比 FedAvg 更好的
   内点——**H3 不是「没调好」，是「这个数据集规模下近端项没有可用区间」**。
   机理见 §11.3.5：每客户端只有 80 行训练数据，近端项把参数拴在全局点上。
5. **C1 的 0.034 不是「本地训练很差」，是「23 个模型各测各的再取平均」。**
   它的定义就是各客户端只用自己的数据训、不聚合（§11.2），指标是 23 个
   模型的平均 MAE，不是某个平均模型的 MAE。它证明联邦确实有用（0.034 →
   0.0012），但不能拿来和 C0 比「谁更会训练」。
6. **P1 用到了标签排序，它不是可部署场景，是压力测试档。**
   `partition_by_soh_rank` 按电池平均 SOH 排序后切片，等于把答案告诉了
   客户端。它回答的问题是「如果划分真的造出标签偏斜，FedAvg 会不会掉点」，
   不是「这种划分在现实中多常见」。
7. **`@N=*` 那些行只能同 N 内比，不能跨 N 比。** k 从 10 降到 1 时，N 从 12
   涨到 115、每客户端从 10 颗电池降到 1 颗——**「客户端更多」本身就会掉点**
   （k≤2 时每客户端 16–32 行 < `batch_size=64`，每 epoch 只 1 步，seed 间
   波动 83%–90%）。所以 P1@N=115 的 0.00421 不能读成「偏斜 70% 的代价」：
   同 N 的 IID 对照在偏斜同样 70.4% 时也是 0.00426。**跨 N 的差主要来自
   训练量结构，不来自偏斜**；偏斜的净效应只能是同 N 下 P1 与 I1 的差
   （§10.1.2、图 2）。
8. **`A1@*` 是 M3 ② 的 FedAdam（§13.3.2）。** 带 `η=`/`τ=` 后缀且 seeds=1 的
   那三行是**探针**（η=0.1 / η=0.001 / τ=1e-5 / τ=1e-6 各一个 seed），它们
   存在的意义是说明 η 与 τ 是怎么选出来的，**不能当结论读**；结论只有
   `A1@le=10,η=0.01,τ=0.001`（8 seeds）与 `A1@le=3,η=0.01,τ=0.001`（8 seeds）
   两行。另外 A1 与 I1 的比较必须是**同 le、同 N**：换 le 就会把训练量的
   差异算到服务端优化器头上（坑 15）。
9. **`D1@nm=*` 是 M4 的 DP 扫描（§13.4），五行必须一起读，单看一行会得出相反
   的结论。** 它们是 `local_epochs=3`（与非 DP 同预算对齐），不是 le=10。
   - `D1@nm=0` **不是 DP 臂**：它只做逐样本裁剪 + Poisson 抽样、不加噪，
     ε=inf，是**机制对照组**。把它算进「DP 的代价」会把换批大小与裁剪的开销
     记到噪声头上（实测机制开销 +39.2% 不显著，噪声在其上再加 +82%～+186%）。
   - 「相对 C0」那一列对 DP 行几乎没有信息量：分母是中心化上限，而 DP 的
     代价要跟**同预算非 DP**（`I1@N=23,le=3` = 0.01165）和**机制对照**
     （`D1@nm=0` = 0.01621）比。五档里最值得记的一条：**nm=8 比 C1
     （0.03446）还差 34.7%**，DP 把模型推到了「不如不聚合」的地步。
   - 五档 DP 的 seed 波动（11.7%–13.9%）小于非 DP 同预算的 66.7%，但那是
     裁剪压住每样本梯度范数的结果，**不是 DP 的优点**（坑 22）。

**H1 与 H3 的状态是「推翻」，H2 的状态是「接受」（依据 P1）**，N1 记
「未判定（处理未施加）」，**H4 是一半支持一半推翻**（单调性支持、ε≤8 可用性
推翻，依据 `D1@nm=*`，见 §13.4）。这一批结果的真正价值是把管线跑通、把量级
量出来，并暴露判据与 DOD 本身的问题（§11.3、§13）。

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
| DP 可用性 | ε ≤ 8 时劣化 ≤30% 支持 H4 | 超了就把结论写成"当前模型规模下 DP 代价过高"。**2026-10-25 已触发**：唯一达标的 nm=8（ε=7.84）劣化 +186.3%，见 §13.4 |

#### 11.3.1 这张表在 2026-10-24 的第一批结果里失效了

**问题：阈值小于噪声，判据没有分辨力。**

8 seeds 实测的 seed 波动（相对标准差）：

| 配置 | seed 波动 |
|---|---|
| C0 中心化 | 13.3% |
| C1 本地独立 | 27.7% |
| FedAvg IID（le=10） | 15.9% |
| FedAvg IID（le=3，等训练量） | **66.7%** |
| FedAvg Dirichlet | 17.2% |
| FedAvg SOH 排序分片 | 5.2% |
| FedProx μ=0.01 | 8.1% |
| FedProx μ=0.1 | 5.8% |

而 H1 的判据阈值是 **5%**。也就是说：**同一个配置换一个 seed 就能让指标动
16%，比要检测的效应还大 3 倍。** 在这种噪声水平下，「相对劣化 3%」和
「相对劣化 8%」无法区分，按 5% 划线等于在噪声里挑结论。

**66.7% 这一格比 16% 更值得注意。** 训练量降到等训练量（le=3）后噪声从
16% 涨到 67%——不是「联邦更不稳定」，而是**欠拟合状态下不同 seed 落在
损失面的不同平底上，模型还没被训练量约束住**。这意味着：训练量不足时
连「跑八次取均值」都不可靠，任何在该工作点上做的比较都要先加 seed。

这不是「结果不好」，是**判据本身需要改**。三个可选做法，按代价排序：

1. **把阈值改成「与噪声同量级」**。例如「|相对差| < 1 个 seed 标准差即视为
   无差异」。代价：几乎必然判「无差异」，但那是诚实的。
2. **提高 seed 数到噪声可接受**。按 CV=16%、想分辨 5%，量级上需要 ~50 个
   seed。单个 run 约 3–6 分钟，50 seeds × 4 配置 = 10–20 小时。可行但要专门排期。
3. **改用配对比较**。同一 seed 下比较两个配置（同初始化、同划分），差的
   seed 波动，因为噪声被配对消掉了。这是最有效的做法，本轮 8 个 seed
   跑遍所有配置，可以直接算配对差。

**在改判据之前，H1–H3 一律记为「未判定」。** 拿 3 个 seed、18% 噪声去宣布
「H1 成立」，比不报结论更糟。

#### 11.3.2 配对比较：哪一条其实能判定

`scripts/summarize_results.py --paired` 按 seed 相减，消掉 seed 带来的共同
波动。实测（8 seeds，配对）：

| 对比 | 配对差均值 | 相对差 | p_t | p_sign | 可判定？ |
|---|---|---|---|---|---|
| FedAvg IID(le=10) − C0 | −0.00037 | −23.5% | 0.0006 | 0.0078 | 是 |
| FedAvg IID(le=3) − C0 | +0.01008 | +640.6% | 0.0077 | 0.0703 | 方向是，量级否 |
| FedAvg Dirichlet − C0 | −0.00057 | −35.9% | 0.0009 | 0.0078 | 是 |
| FedProx μ=0.01 − C0 | −0.00009 | −6.0% | 0.2600 | 1.0000 | 否 |
| FedProx μ=0.1 − C0 | +0.00030 | +19.0% | 0.0148 | 0.2891 | 否 |
| FedAvg SOH 分片 − C0 | −0.00008 | −4.9% | 0.2014 | 0.7266 | 否 |
| FedProx μ=0.01 − FedAvg(Dirichlet) | +0.00047 | +46.7% | 0.0002 | 0.0078 | 是 |
| FedProx μ=0.1 − FedAvg(Dirichlet) | +0.00086 | +85.8% | 0.0000 | 0.0078 | 是 |
| **FedAvg SOH 分片 − FedAvg IID** | **+0.00029** | **+24.2%** | **0.0013** | **0.0078** | **是** |

**符号检验在 n=8 时最小双侧 p 仍是 0.0078**，所以「8/8 方向一致」只够
到 0.01 量级。想判更小的效应就必须加 seed，没有别的路。

#### 11.3.3 H2 的对照组必须是同预算的 IID，不是 C0

原判据把 H2 写成「Non-IID 相对劣化 >10%」，隐含分母是 C0。这是错的：
**H2 问的是「划分方式造成多少损失」，联邦与中心化的差距是 H1 的问题。**
拿 C0 当分母会把两个效应混在一个数里，而它们给出的画面相反：

| 对比 | Dirichlet 的表现 |
|---|---|
| 相对 C0 | **−35.9%**（好很多） |
| 相对同预算 IID | **−16.3%**（还是好，但幅度差一半） |

`summarize_results.py --judge` 现在把 H2 的分母固定为同 `local_epochs` 的
IID FedAvg 组；找不到就报「缺同预算的 IID 对照组」，**不默默退回 C0**。

#### 11.3.4 Manipulation check：Dirichlet α=0.5 其实没施加 Non-IID 处理

这是本轮最该被记住的一条。H2 长期被判不了，不是因为噪声，是因为**那份
"Non-IID 划分"根本没造出标签偏斜**。

`scripts/measure_heterogeneity.py` 给出「客户端身份解释的标签方差占比」
（定义见 `src/flsoh/heterogeneity.py`）：

| 划分 | 标签偏斜 | 样本量基尼 |
|---|---|---|
| IID k=5 | 17.3% | 0.000（每客户端 5 颗） |
| Dirichlet α=0.5 | **12.2%** | 0.519（最大 21 / 最小 1 颗） |
| SOH 排序分片 k=5 | **47.1%** | 0.000（每客户端 5 颗） |
| 上界（一颗电池一个客户端） | 70.4% | — |

**Dirichlet 划分的标签偏斜比 IID 还小。** 原因是它只造成**数量偏斜**：
N=23 时 α=0.5 让一个客户端独占 21 颗电池，而把 21 颗电池合成一个客户端
会把电池之间的 SOH 差异**平均掉**——于是这个划分实际上比 IID 更均匀。
所以「Non-IID 没掉点」这句话在此之前是没有意义的：处理没施加。

解法是 `partition_by_soh_rank`：按电池平均 SOH 排序后切成连续分片，
**N、k、总数据量、训练预算全部不变，只改哪些电池分给谁**。偏斜从 17.3%
拉到 47.1%，而样本量完全均衡（基尼 0）。这是 FL 文献里的 pathological
non-IID，它用到了标签，所以**不是可部署的场景，是压力测试档**。

在它上面 FedAvg 比同预算 IID 差 **+24.2%**（p_t=0.0013，p_sign=0.0078，
8/8 方向一致）——**H2 因此判为接受**：Non-IID 确实掉点，而且掉点幅度与
标签偏斜的量级对得上（偏斜 ×2.7 → MAE +24%）。

配套的硬规则（已写进 `--judge`）：**判 H2 之前必须先看 heterogeneity.json**，
偏斜不高于 IID 基线时一律记「未判定（处理未施加）」，不许判「推翻」。
否则等于拿一个没通电的电路去证明断路器好用。

**M3 的 k 扫描把这条规则用出了第二种形态：偏斜「完全相同」。** k=1 时一颗
电池一个客户端，sohrank 与 IID 的偏斜逐位相同（70.4%）、分组也完全相同
（115/115 个 singleton 重叠），「谁拿哪颗」只改客户端标签。原来的闸门只写
「偏斜 ≤ 基线 → 处理未施加」，等于号会被读成「70.4% ≤ 70.4%」——技术上对、
没有解释力。现在分成两支：**小于**（Dirichlet 那种，合并把差异平均掉）与
**等于**（k=1 那种，排序只改客户端标签，注明这一档是图 2 的内部校准点）。
实测这一档配对差 −1.2%（p_t=0.78），与「同一组 per-cell 更新按不同顺序
求和、100 轮混沌训练放大浮点舍入」的理论预期一致。

同一轮扫描还暴露了规则的一个漏洞：H2 闸门原来只 grep「缺
heterogeneity.json」，而 k 扫描新增四档 N 之后出现了第二种失败——**文件在、
但没有这一档 k 的记录**（重跑测量时漏传一份划分，N1 的判定当场失效）。
文案是「找不到该 N 的划分」，不含旧关键字，闸门会静默放行。现在两个都拦。

#### 11.3.5 H1 / H3 的最终状态

- **H1 推翻（两个口径都是）。** le=10 时联邦比 C0 好 23.5%（p=0.0006），
  le=3（与 C0 等训练量）时差 640.6%（p=0.0077）。两个方向相反、都远超
  10%——所以 H1「差距可以忽略」在两个口径下都不成立。**但要注意好 23.5%
  那一条靠的是 3.3 倍训练量**，见 §10.2 第 1 条。
- **H3 推翻，且被两端夹住。** μ=0.01 差 +46.7%、μ=0.1 差 +85.8%，
  都是 0/8 方向一致、p<0.001。而 μ=0 的定义就是 FedAvg
  （`tests/test_train.py::TestFedProx::test_mu_zero_is_bitwise_fedavg` 钉住）。
  改善只可能出现在 (0, 0.01)，两端都指向更差 → **不存在可用内点**。
  机理：每客户端仅 80 行训练数据（5 颗电池 × 16 循环），近端项
  `(μ/2)‖w − w_global‖²` 相对数据项过强，把参数拴在全局点上、限制本地拟合。
  **§14 坑 2 给的「μ 取 0.01–0.1」在这套数据规模下是错的**，那条建议来自
  每客户端上千样本的典型 FL 设定。

顺带两个反直觉但重要的观察：

- **FedProx 的 seed 波动（5.8%–8.1%）比 FedAvg（15.9%）小，但它更差。**
  稳定性不等于正确性——μ 的强约束让所有 seed 收敛到同一个（更差的）点。
  只看「波动小」就选 FedProx 是错的，而且这个陷阱很容易踩：按 §11.3 的
  「seed 波动 <10%」这一行，FedProx 反而是唯一两个达标的配置之一。
- **等训练量下配对差是 +0.01008（相对 +640.6%），但符号检验 p=0.0703。**
  方向 1/8 为负（即 7 个 seed 联邦更差）——**方向可信，倍数不可信**。
  只应报成「数倍」，不许报「6.4 倍」。

另外两条实测约束也要写进任何后续判据：

- **训练量必须对齐**。联邦 100 轮 × 10 ep ≈ 1000 个本地 epoch，C0 是 300 个。
  不声明训练量的「相对劣化」没有意义，而且会反号（见 §10.2 第 1 条）。
- **判 H2 之前必须量偏斜**。数量偏斜（基尼）和标签偏斜（方差占比）是两件事，
  Dirichlet 只给了前者。

---

## 12. 风险登记册

§14 的"坑清单"是**已知错误**；这里是**不确定性**。

| # | 风险 | 概率 | 影响 | 触发信号 | 缓解 |
|---|---|---|---|---|---|
| R1 | 划分泄漏导致结果虚高 | 中 | **致命**（结论作废） | `check_partition.py` 报 P-LEAK/P-TEST | 划分脚本先行 + 每次改配置重跑检测 |
| R2 | Non-IID 下 FedAvg 不收敛，被误判为代码 bug | 高 | 中（浪费 2-3 天） | 轮数上去 MAE 不降且客户端间差异大 | 先降 `local_epochs` 到 1，再上 FedProx；对照 §14 坑 2 |
| R3 | Flower 版本漂移导致骨架跑不通 | 中 | 中（0.5-1 天） | ImportError / 参数名不匹配 | 锁 1.41.0；第一天先跑官方 quickstart |
| R4 | FedBN/Scaffold 接入成本超预期 | 高 | 中（砍功能） | 自实现聚合后收敛行为与文献不符 | 标为可选；时间紧直接砍，报告里写"未做" |
| R5 | DP 下模型完全学不动（ε 太小） | 中 | 中（H4 变否） | noise=2.0 时 MAE 接近常数预测 | 这就是结论之一；报告写"该规模下不可用"，不硬凑。**2026-10-25 已发生**（§13.4）：ε ≤ 8 的唯一一档 nm=8 比常数预测器差 12.3 倍，H4 第二半推翻 |
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
| **M3** | FedAdam/FedBN + **标签偏斜扫描** + 跨化学体系 | 4 | ①sohrank 分片粗细 k=1/2/5/10 各 ≥3 seeds，每档先量偏斜 ②FedBN 在跨化学体系场景有量化对比 ③图 2 |
| **M4** | Opacus DP + 隐私-效用曲线 + 通信量表 | 3 | ①≥3 个 noise 点各带 ε ②图 3、表 2 ③H4 有结论（实际达成情况见 §13.4；①的字面条款「noise=1.0 处 ε ≤ 8」在本协议下**结构上不可达**，已在该节说明） |
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

### 13.2 M2 的实际达成情况（2026-10-24）

| DOD | 状态 | 依据 |
|---|---|---|
| ① 四条配置各有 3 个 run 的 JSON | **超额达成** | 64 个 run 入库：5 个配置 × 8 seeds + 等训练量对照 |
| ② 图 1、表 1 出图 | **达成** | `docs/figures/fig1-rounds-mae.png`（三个 panel，一个假设一个图）；表 1 由脚本生成，CI 逐字核对 |
| ③ H1–H3 各有 accept/reject | **达成** | H1 推翻、H2 接受、H3 推翻；N1（Dirichlet）记「未判定（处理未施加）」 |
| ④ 2–3 页备忘 | **达成** | `docs/memo-m2.md` |

### 13.3 M3 DOD ① 的改写（2026-10-24）

原写法「α 四档各 3 seeds」扫的是 **Dirichlet 的 α**，而 α 控制的是**数量偏斜**
（客户端样本量的离散程度）。M2 已经证明 H2 的效应来自**标签偏斜**（客户端之间
SOH 分布的差异），这两者在本数据集上甚至可以反向：α=0.5 造出的数量偏斜
（基尼 0.519）远大于 IID（0.000），但标签偏斜反而更小（12.2% vs 17.3%），
因为把 21 颗电池合成一个客户端会把电池间的 SOH 差异平均掉。

所以扫 α 会得到一个「α 越小越不掉点」的假趋势，而它测的是聚合权重的集中度，
不是 H2 的机制。

改写后的 DOD ①：

> **sohrank 分片粗细 k=1/2/5/10 各 ≥3 seeds，每档在跑之前必须先量出标签偏斜
> 并写进 `results/heterogeneity.json`；图 2 的横轴是标签偏斜，不是 k。**

这样图 2 直接回答「偏斜多大才开始掉点、掉多少」，而且横轴是有物理含义的量
（客户端身份解释的方差占比），不是实现参数。

### 13.3.1 M3 DOD ① 的实际达成情况（2026-10-24）

| DOD | 状态 | 依据 |
|---|---|---|
| ① sohrank k=1/2/5/10 各 ≥3 seeds，每档先量偏斜 | **超额达成** | 四档各 **8 seeds**（32 个 run）+ 同 k IID 对照各 8 seeds（24 个 run），56 个 run 全部入库；偏斜在跑之前量完并写进 `results/heterogeneity.json`（9 份划分） |
| ③ 图 2 | **达成** | `docs/figures/fig2-skew-mae.png`（横轴=标签偏斜，sohrank 与同 k IID 两条线，竖直距离=偏斜净效应） |

结果（§10.1.2 的矩阵）：**k=10 与 k=5 两档 H2 接受**（+50.9% / +24.2%），
k=2 未判定（训练量结构下测不出 20% 量级效应），k=1 未判定（处理未施加，
内部校准点）。②（FedBN 跨化学体系）尚未开始。

### 13.3.2 M3 DOD ②：FedBN 做不了，改做 FedAdam（2026-10-24）

原 DOD ② 是「FedBN 在跨化学体系场景有量化对比」。它被**两个前置项**挡住，
而且都不是"再跑几个实验"能绕过的：

1. **模型里没有 BatchNorm，而且是刻意的**。`src/flsoh/model.py` 的类注释
   写明：「不用 BatchNorm/Dropout。BN 会让 FedBN 变成必选项（§5.2），
   MVP 阶段先把问题收窄到"数据异质"这一件事上。」加 BN 是**架构变更**，
   参数量、通信量表（表 2）、所有历史结果的对照基准都会变。
2. **本地没有第二个化学体系的 ΔQ(V) 特征**。本仓库的特征管线吃的是
   MIT-Stanford 的 ΔQ(V) 1000 维向量；CALCE/NASA 只有容量曲线，
   ΔQ(V) 要另做提取与对齐。

在补这两件事之前，M3 ② 改成做 **FedAdam**——它没有前置阻塞（flwr 内置），
而且直接检验本仓库目前最硬的一条机理发现：

> H3 被推翻时定位到的机理：FedAvg/FedProx 只聚合**参数**，不聚合优化器
> 状态，所以客户端 Adam 的动量每轮归零（§14 坑 12）。`local_epochs`
> 因此不是自由超参——追平 C0 要 3.3 倍训练量。

FedAdam 把动量挪到服务端：服务端动量跨轮存活，客户端本地动量仍每轮重建。
**如果动量归零是绑定约束，FedAdam 应当在更少的本地 epoch 下追平 FedAvg。**

实现按仓库既有约定：`flsoh/train.py::ServerAdam` 与 flwr 1.41 内置
`FedAdam` 逐字对应（读过安装版源码核对），等价性由
`tests/test_app.py::TestFedAdamEquivalence` 逐轮钉住；本地循环
（`run_federated.py --strategy fedadam`）与官方路径（`server_app.py` 读
`strategy="fedadam"`）共享同一份实现。

**η 与 τ 是实测选出来的，不是默认值捡的**（seed 0，IID k=5，100 轮，le=10）：

| η（τ=1e-3） | 最终 MAE | 相对 FedAvg 同 seed（0.00131） |
|---|---|---|
| 0.1 | 0.00268 | +105% |
| **0.01** | **0.00101** | **−23%** |
| 0.001 | 0.00340 | +160% |

| τ（η=0.01） | 最终 MAE |
|---|---|
| **1e-3（flwr 默认）** | **0.00101** |
| 1e-5 | 0.00128 |
| 1e-6 | 0.00173 |

两条读数：

- **η 就是服务端每轮的步长**。闭式解：伪梯度恒定时每轮位移恰为 η
  （偏置矫正把 β1^t、β2^t 约掉），与 |Δ| 无关。实测每轮伪梯度的
  逐坐标 mean |Δ| ≈ 7×10⁻⁴、中位 6×10⁻⁴、max 1×10⁻²——η=0.01 落在
  这个分布的上沿，η=0.001 落在中位以下，所以一个太快一个太慢。
- **τ 在这里不是数值保护，是噪声地板**。稳态步长 = η/(1+τ/|Δ|)，τ=1e-3
  与 mean |Δ| 同量级，把 |Δ|≲1e-3 的坐标阻尼掉、只放行真实位移；
  τ 再小等于给全部 13.6 万个坐标都开归一化步长，噪声坐标一起动，
  结果反而差（0.00173）。

所以正式实验用 flwr 的默认 β1=0.9 / β2=0.99 / τ=1e-3，只把 η 定在 0.01。

**正式结果**（IID k=5、N=23、100 轮、8 seeds，与 I1 完全同条件）：

| 配置 | 本地 epoch | 最终 MAE | 相对同预算 FedAvg |
|---|---|---|---|
| FedAvg I1@le=10 | 10 | 0.00120 ± 0.00019 | — |
| FedAdam A1@le=10 | 10 | 0.00123 ± 0.00051 | +2.0%（p_t=0.91 → 无差异） |
| FedAvg I1@le=3 | 3 | 0.01165 ± 0.00777 | — |
| **FedAdam A1@le=3** | 3 | **0.00162 ± 0.00077** | **−86.1%**（8/8 同向，p_t=0.0056、p_sign=0.0078） |

判定（`summarize_results.py --judge` 的 M3 ② 段）：**机理被证实，但只证实了一半**。

- 本地训练量不够时，服务端动量补上绝大部分差距：le=3 下 FedAdam 比
  FedAvg 好 86.1%，且与 C0 不可区分（+2.9%，p_t=0.88）——300 个总梯度步
  达到中心化 300 epoch 的水平。
- 但补不满：A1@le=3 相对 I1@le=10 仍差 +34.5%（p_t=0.14，样本量分辨不出
  这 30% 是真是噪），所以「追平」未判定，「大幅缩小」判定。
- 本地训练量已经够时没有额外收益（le=10 下 +2.0%，p_t=0.91）。这与坑 12
  自洽：le=10 时客户端动量归零的代价已被 10 个本地 epoch 摊薄。

完整分析见 `docs/memo-m3.md` §5。②（FedBN）保持未做，理由见本节开头。

### 13.4 M4 DOD 的实际达成情况（2026-10-25）

| DOD | 状态 | 依据 |
|---|---|---|
| ① ≥3 个 noise 点各带 ε | **超额达成** | 5 个扫描点，其中 **4 个是 DP 臂、每个都由 RDP 会计器算出 ε**（152.31 / 49.11 / 18.65 / 7.84），第 5 个 `nm=0` 是**机制对照组**（ε=inf，plan §7：没有噪声就没有 DP）。5 × 8 seeds = 40 个 run 全部入库 |
| ② 图 3、表 2 | **达成** | `docs/figures/fig3-privacy-utility.png`（双面板，左 nm / 右 ε）；表 2 即 §10.2 的 `D1@nm=*` 五行，由脚本生成、CI 逐字核对 |
| ③ H4 有结论 | **达成，且是「一半支持一半推翻」** | 见下 |

#### 13.4.1 预算测算先行：先算 ε，再决定扫哪些 nm

扫 nm **之前**先跑 `scripts/check_dp_budget.py --config configs/dp.toml`。
这不是仪式——如果随手扫 {0.5, 1, 2}，三档的 ε 分别是 300+ / 152 / 49，
**没有一个达到 §11.3 的 ε ≤ 8**，那么「ε ≤ 8 时劣化 ≤ 30%」这条判据一条都用
不上，扫描照跑、图照出、结论只能写成「该规模下不可用」。扫描点是照着预算表
选的，不是照着习惯选的。

| noise_multiplier | ε (δ=1e-5) | 达到 ε ≤ 8？ | 噪声 Std（批均值梯度上） |
|---|---|---|---|
| 0（对照组） | ∞ | 不是 DP 臂 | 0（不加噪） |
| 1 | 152.31 | 否 | 0.0250 |
| 2 | 49.11 | 否 | 0.0500 |
| 4 | 18.65 | 否 | 0.1000 |
| **8** | **7.84** | **是** | **0.2000** |

推导见 §7.1：q=0.5、步数 600，**只有 nm=8 达标**。而 nm=8 的噪声 Std
`8 × 1.0 / 40 = 0.2` 与实测信号（批均值梯度范数 0.147–0.223）**同量级**——
H4 的两半在这里迎面撞上，这不是实现问题，是数据规模问题。

#### 13.4.2 五档结果（IID k=5、N=23、100 轮、le=3、batch 64、Adam 1e-3、8 seeds）

| nm | MAE (mean±std) | seed 波动 | ε | vs `nm=0` 机制对照（配对） | vs 非 DP 同预算 |
|---|---|---|---|---|---|
| 0（机制对照） | 0.01621 ± 0.00189 | 11.7% | ∞ | — | +39.2%（p_t=0.0945，不显著） |
| 1 | 0.02947 ± 0.00408 | 13.9% | 152.31 | **+81.8%**（8/8，p_t≈0，p_sign=0.0078） | +153.0% |
| 2 | 0.03475 ± 0.00468 | 13.5% | 49.11 | **+114.3%**（8/8） | +198.3% |
| 4 | 0.04066 ± 0.00534 | 13.1% | 18.65 | **+150.8%**（8/8） | +249.0% |
| **8** | 0.04643 ± 0.00603 | 13.0% | **7.84** | **+186.3%**（8/8） | **+298.5%** |

参照：非 DP FedAvg le=3 = 0.01165 ± 0.00777；C0 = 0.00157；C1 本地独立 =
0.03446。每档 round1 / round100 / best 单调下降且 best≈final（如 nm=8 为
0.08026 / 0.04643 / 0.04630），**没有过训回落**，曲线是训练不足的形状。

#### 13.4.3 H4 的判定：单调性支持，可用性推翻

H4 原文是「精度代价随噪声单调上升，**且**在 ε ≤ 8 时仍可用」——两个半句
必须分开判，因为它们在这一批数据上给出**不同**的答案：

- **单调性：支持。** MAE 随 nm 严格递增（0.01621 → 0.02947 → 0.03475 →
  0.04066 → 0.04643），四档 DP 臂相对机制对照的劣化 8/8 同向且全部显著
  （p_sign=0.0078 是 n=8 的下限）。噪声确实按预期在伤精度。
- **ε ≤ 8 时仍可用：推翻。** 唯一达标的 nm=8（ε=7.84）相对机制对照劣化
  **+186.3%**、相对非 DP 同预算劣化 **+298.5%**，远超 30% 阈值。
  还有一条比百分比更直观的读数：**nm=8 比 C1（完全不联邦，0.03446）还差
  34.7%**——DP 把模型推到了「不如不聚合」的地步。而常数预测器（输出中心化
  后的 1.0）在测试集 MAE = 0.00350，五档 DP 全部远差于它，说明模型不是
  「学得慢」而是「学不动」。

**DOD ① 的字面条款「noise=1.0 处 ε ≤ 8」在本协议下结构上不可达**：nm=1
对应 ε=152.31（§13.4.1）。这条 DOD 写的时候假设了「nm=1 差不多就是 ε 个
位数」，而 q=0.5 让 subsampling 几乎没有放大作用。判据本身没错（ε ≤ 8 时
劣化 ≤ 30%），错的是把「哪个 nm 能达到 ε ≤ 8」当成了可以顺手填的参数。

#### 13.4.4 机制开销与噪声开销必须分开报

`nm=0` 那一行只做逐样本裁剪 + Poisson 抽样、**不加噪**，ε=inf，它不是 DP
臂。它相对非 DP 同预算差 +39.2%（p_t=0.0945，**不显著**），而 nm=1 在它之上
再加 +81.8%（显著）。合起来报成「DP 一共劣化 +153%」会把两件事混成一个数：

- 换批大小（Poisson 让期望批从 64 变 40）+ 裁剪：+39.2%，测不出来；
- 噪声：在其之上再加 +82%～+186%，测得很稳。

把机制开销算到噪声头上，会高估 DP 的价格；反过来只报机制开销，会让读者
以为「DP 很便宜」。**nm=0 是这一节里最容易被删、删了最贵的一行。**

#### 13.4.5 一条不能当优点报的附带发现

DP 各档的 seed 波动（11.7%–13.9%）明显小于非 DP 同预算的 **66.7%**。看起来
像「DP 让训练更稳定」，但按 §11.3.5 记下的 FedProx 教训（**稳定性 ≠ 正确性**），
这只是逐样本裁剪把每样本梯度范数压住的结果，而它同时就是代价的来源之一。
**这一格只作为现象记录，不进结论。**

---

### 13.5 M5 DOD 的实际达成情况（2026-10-25）

M5 是最后一个里程碑：Schema v1 契约 + 本地特征提取库 + 模拟数据生成器 +
端到端联调 + 完整报告。三项 DOD 的达成情况如下。

#### 13.5.1 DOD ①：Schema v1 与校验器 — 达成

`src/flsoh/schema.py`（纯标准库）定契约，`scripts/check_schema.py` 做闸门，
`tests/test_schema.py` 43 条钉住边界。两条设计决定值得记下来：

- **版本不匹配直接报错**，不尝试兼容。v2 的语义可能改过（比如 `I` 的符号约定
  翻转），按 v1 读会得到一份**结构合法、语义相反**的数据——校验器全绿、特征
  全错。这是唯一一类「越校验越危险」的错误。
- **未知字段只警告**。将来的 v2 会加字段，现在报错等于宣告「这个仓库只认 v1」。

指纹 `session_fingerprint` 覆盖**全部内容**而不是只覆盖 ID：重装 App 后
`session_id` 可能被复用而内容不同，按 ID 去重会丢掉一个真实的循环。

#### 13.5.2 DOD ②：本地特征库 + 模拟数据 — 达成，但**查出一个贯穿 M1–M4 的数据缺陷**

`src/flsoh/segments.py` 走 App 路径（原始记录 → 放电段 → (V, Q) → ΔQ），
`src/flsoh/simulate.py` 生成带真值的模拟会话，`src/flsoh/ingest.py` 做校验 /
去重 / 组装。三者的测试共 69 条。

**在这一步查出的缺陷**：`matr.charge_curve()` 返回的是**整条循环（充电段 +
放电段）**，而 `delta_q_curve` 内部按电压 `argsort` 后插值——放电段被排进
充电段的电压区间，而放电段的 `Qc` 是常数。于是低电压区出现一片「容量 = 满容量」
的平台，而那里本该是充电曲线的陡坡。

用 b1c0 的真实循环 1 与 5 量化（1000 点网格）：

| 喂给 `delta_q_curve` | TV | std | 与正确值相关 |
|---|---|---|---|
| 整条循环（M1–M4 用的） | 269.60 | 0.3918 | **−0.019** |
| 仅放电段（已修） | 0.21 | 0.0026 | — |

TV 差 1280 倍，相关系数 −0.02。**也就是说 M1–M4 用的「ΔQ 特征」和真正的 ΔQ
特征基本是两套不相干的数据，而它的形状、长度、数值范围全都正常，指标照样
算得出来**（C0 的 MAE 0.0016 就是这么来的）。全量特征表上 TV/幅度 的中位数
是 122.4，正确口径实测 5.9。

三件事必须同时做，缺一条这个错还会再活四个里程碑：

1. **改代码**：`matr.discharge_curve()` 按电流符号取 `I < 0` 的段，用 `Qd`
   而不是 `Qc`；`build_features.py` 改调它。**为什么必须是放电**：恒压充电段里
   V 几乎不变而容量猛涨，dQ/dV → ∞，均匀电压网格在那里会把 0.1 Ah 压进一两个
   网格点（充电段最大相邻跳变 1.118 Ah，放电段 0.013 Ah）。
2. **加检查器**：`scripts/check_features.py` 算每行的 TV/幅度。判据是**中位数
   20 + 单行硬上限 60**（最初是逐行卡 20，2026-10-25 改为双层，理由见坑 32）。
   另有第三道**老化信号**：每颗电池的「特征 vs 循环号」中位 Spearman 相关必须
   低于 −0.7、且 ρ ≤ −0.8 的电池占比不低于 80%（错口径 −0.168 / 0%，正确
   口径 −0.979 / 91%）。理由见坑 34——光滑性结构上抓不到「特征与标签无关」。
   `build_features.py` 跑完自动调它，不通过就非零退出。
3. **修指纹**：`FEATURE_SPEC_VERSION` 进 `state_fingerprint`。原指纹只描述
   「要算什么」（循环区间、参考循环、网格），不描述「怎么算」——修完算法重跑
   同一条命令，旧分片会被当成有效数据**全部跳过**，表里一行新数据都没有，
   而脚本打印「全部有分片」、退出码 0。这是指纹机制本来要防的那类错误，只是
   方向换了一下。

**这条缺陷对已发布数字的影响**：§10.2 的表、图 1–3、§13.1–13.4 的每个相对差，
全部是在错误的特征表上算出来的。它们**不是**「精度差一点」，而是「换了一套
不相干的特征」。因此：

- 特征表已按新口径**重建**（`--restart`，135 颗 × 20 循环）。本次提交时重建
  仍在进行（约 50 s/颗，总计约 2 小时）。
- 重建前 15 颗的抽查已经证明修复生效：b1c0 分片的 TV/幅度 中位 **5.68**、最大
  7.51、超限 **0/20**（旧口径同表 122.4）。幅度 0.0139–0.0310 Ah，与手工
  核对循环 1/5 得到的 0.018 Ah 同量级。
- 重建完成后必须**重跑全部实验**并重生成表与图，然后逐条复核 H1–H4 的判定。
  在那之前，§10.2 与 §13.1–13.4 的**数字一律标注为待复核**，但其中的
  **方法论结论**（噪声量级、训练量口径、ε 的推导、稳定性 ≠ 正确性）与特征
  口径无关，仍然成立。

#### 13.5.3 DOD ③：端到端联调 — 达成

`scripts/run_e2e.py` 一条命令跑通「模拟会话 → 入库去重 → 特征 → 光滑性闸门 →
划分 → 联邦训练 → 结果 JSON」。五个闸门各自非零退出（见脚本 docstring 的表）。

首次跑通的读数（24 颗 × 20 循环 = 480 份会话，4 客户端，60 轮 × le=5）：

| 指标 | 数值 |
|---|---|
| 入库 | 480 份 → 接收 480，拒收 0，重复 0 |
| 特征 | 456 行 × 1000 维，跳过 0 份 |
| TV/幅度 | 中位 11.88，最大 45.55（阈值 60） |
| 常数基线 MAE | 0.02638 |
| 联邦 MAE | **0.02289** |

**只比常数基线好 13%，这个数字必须带着解释出现**，否则读者会以为管道有问题：

- 同一批特征上，线性回归（不标准化）MAE **0.0112**、只用「ΔQ 幅度」一维
  **0.0041**——**信号是明确存在的**。
- 但 `LocalStandardizer` 逐客户端标准化会把这个「幅度」按客户端归一化掉，
  而模拟数据把 SOH 几乎全部编码在幅度上。逐电池标准化后线性回归退化到
  **0.0397**，比常数基线还差。
- 结论：**这是模拟器的局限，不是管道的缺陷**。真实 LFP 的 ΔQ 曲线随老化
  同时改变形状，模拟器只改了幅度（内阻增长与 γ 漂移合计只带来二阶变化）。

因此 e2e 的 MAE **不能与 §10.2 的数放在一起看**，它的用途只有两个：证明管道
接通、以及给出「模拟 vs 部署」差距的一个下界。

#### 13.5.4 M5 遗留

1. 特征表重建完成后重跑实验、重生成表与图（阻塞项，见 §13.5.2）。
2. 模拟器补上「形状随老化变化」的机制（低 SOC 端可用窗口收缩），否则 e2e 的
   精度永远被标准化压住。
3. `configs/full.toml` 的 k=2 会让 q 退化到 1.0，跑前必须重算预算（M4 遗留）。

#### 13.5.5 重跑协议（2026-10-25 定）

172 个 run 全部要重跑。**手抄命令行不可行**——172 条命令里少一个
`--server-tau`、把 `--mu 0.1` 写成 `0.01`、漏 `--noise-multiplier 0`，结果
都照样写出来，只是换了个实验，而文件名靠 `--tag` 固定，从外面完全看不出来。

`scripts/rerun_all.py` 因此**从结果 JSON 反推命令行**：JSON 里记了什么就重放
什么，记不住的项报错退出，不填默认值。预检五道，全部是硬闸门：

| 闸门 | 拦的是什么 |
|---|---|
| 分片比汇总表新 | 重建还没汇总就开跑 → 又跑一遍旧表，退出码 0 |
| `feature_version` 不符 | 同上，按口径字符串判 |
| 解释器 torch 版本不符 | 换版本会改数值（下表） |
| `torch_num_threads` 不符 | 同上 |
| 请求的线程数没被 torch 采纳（请求 24 实际 16） | 钉不住的值 → 开跑前拒绝，而不是跑完才发现 |
| 结果目录新旧混杂（部分 run 缺 `torch_num_threads`） | 重跑中断后续跑，半新半旧。带 `--skip-existing` 时降级为逐产物核对，否则续跑会被自己的闸门拒死 |
| `--skip-existing` 按产物自报口径判 | 只判文件存在 → 跳过上一版的结果（坑 31） |
| 反推不出参数 / 划分文件缺失 / C1 客户端数不符 | 悄悄换了个实验 |
| 提升时少拷 run / 拷进口径不符的 run（`promote_results.py`） | `results/` 内部新旧混杂，而 `--check-plan` 照样绿 |

**两道运行时闸门的实测依据**（同一份特征表、同一 seed、同一仓库、C0 seed0）：

| 变量 | 值 | C0 final_mae | 结论 |
|---|---|---|---|
| torch | `2.10.0+cpu` | `0.0015666428208351135` | 与历史记录**逐位相同** |
| torch | `2.14.1+cpu` | `0.0015434211492538453` | 差 1.5%，不是噪声 |
| 线程数 | 24（默认） | `0.0015666428208351135` | |
| 线程数 | 1 | `0.0015997225046157837` | 差 2.1% |

所以重跑统一 `--threads 1`：默认线程数意味着**结果随机器核数变化**，换台
机器就复现不了。产物里同时记 `torch_num_threads` 与 `python_version`，让这条
约束以后可核对。

**而且「钉 24 线程」在这台机器上根本钉不住**：`pinned_env` 设
`OMP_NUM_THREADS=24`，`torch.get_num_threads()` 只返回 **16**（torch 自己钳）。
所以 `rerun_all.py` 加了一道预检 `check_probed_threads`——开跑前用
`pinned_env` 起子进程问一次实际线程数，与请求值不符直接拒绝。实测现在
`--threads 24` 被拒、`--threads 1` 打印「线程数钉住了」。

代价是时间：1 线程不是慢 9%，而是约 **1.8 倍**（同一 run 33.1 s → 61.0 s）。
11.91 小时的旧墙钟在 1 线程下约 21 小时，`--jobs 12` 约 1.8 小时。

**旧结果的归档与对比方式**：旧口径的 172 个 JSON 已归档到
`results/fullcycle-bug/`（含一份说明为什么作废的 README）。它们跑在 24 线程
下，**不能**与新数字直接相减——线程数这一项就贡献 2.1%。要量「特征修复到底
改了多少」，必须在**同一线程数**下把旧表重跑一遍：

```bash
# 旧表重跑（隔离特征效应；只跑有配对意义的 6 个配置 × 8 seeds）
python scripts/rerun_all.py --execute --jobs 8 --threads 1 \
  --features data/mit-stanford/features.fullcycle-bug.parquet \
  --out results/fullcycle-bug-rerun --only 'c0__|c1__|iid__le10|iid__seed|dirichlet0.5|sohrank__seed'

# 全量重跑（新表）：写到独立目录，不在 results/ 上原地覆盖
# --compare 指向同线程数的旧口径消融（56 run 重叠），这样差值只剩特征本身
python scripts/rerun_all.py --execute --jobs 12 --threads 1 \
  --out results/discharge-v2 --compare results/ablation-old-features \
  --expect-feature-version discharge-v2 --skip-existing
```

`--out` 特意不是 `results/`：那里装的正是**作废的旧结果**，原地覆盖会让
「中断后重跑」与「跑完了吗」两个问题都变成不可判定的。写到独立目录后，
`results/discharge-v2/` 里出现的东西只可能来自这一次重跑。`--skip-existing`
配合它才是安全的续跑——产物逐个过 `artifact_is_current`，口径对不上就重新排队。

`--jobs` 不写死：先单独跑一个任务量出峰值 RSS，再取
`min(20, 0.7 × 可用内存 / RSS)`。本机 24 核 / 16 GB（实际可用 5.9 GB），
按单个 torch 进程约 350 MB 估，12 个并发是安全上限；直接开到 20 会换页，
而换页期间所有任务一起变慢，看不出是哪一步出的问题。

**重跑完成后的落地顺序**（顺序有依赖，不能跳）：

1. 核对 `results/discharge-v2/` 的 157 个任务全齐，且每个 JSON 的
   `feature_version` 都是 `discharge-v2`、`torch_num_threads` 都是 1。
2. 把这批 JSON **覆盖进 `results/`**——用 `scripts/promote_results.py`，不要手写
   `cp`：

   ```bash
   python scripts/promote_results.py --from results/discharge-v2 --to results \
       --expect-feature-version discharge-v2 --threads 1 --execute
   ```

   旧 172 个已在 `results/fullcycle-bug/` 完整归档（逐 run 核对过：差集为空），
   所以覆盖不丢东西；分区文件（`partition-*.json`、`heterogeneity.json`）与
   特征口径无关，脚本按「有没有 `final_mae`」区分，原地不动。
3. `summarize_results.py --write-plan docs/plan.md` 重生成 §10.2 的表，
   `--judge --hetero results/heterogeneity.json` 重出 H1–H4 判定。
4. 重生成图 1–3，`--compare results/ablation-old-features` 出新旧逐 run 对比。
5. 把新数字写进 §13.5.6 的判定对照表，新增一列「新表（1 线程）」。

第 2 步必须做而不是让 CI 改指 `results/discharge-v2`：CI 的
`no-fabricated-results` job 跑的是 `--check-plan docs/plan.md`（默认
`--results results`），不覆盖就会红；而把 CI 改成指另一个目录，等于让仓库的
默认路径指向一个临时名字。

**为什么第 2 步要一个脚本**：`cp` 有两种半途而废的方式，两种都看不出来——
少拷几个（某个任务失败了，旧的留在原地）、拷进口径不同的几个（源目录里混了
上一版）。后果是同一条：`--write-plan` 把**混杂的表**写进文档，而 CI 的
`--check-plan` 会**通过**，因为它只核对「文档与 `results/` 是否一致」，不核对
「`results/` 内部是否同源」。`promote_results.py` 的四道检查就针对这个：
源口径逐个核对、目标里的 run 必须全部出现在源里、拷完重扫目标目录、任一条
不过就**一个文件都不拷**（「提升一半」比「不提升」更糟）。

两条命令的 `--threads`、解释器、划分文件完全相同，所以两者的差就是特征口径
的差。

#### 13.5.6 旧表消融已完成（2026-10-25，56 run）

上一条的第一条命令已经跑完：**旧特征表 + 1 线程 + 7 配置 × 8 seeds = 56 run**，
落在 `results/ablation-old-features/`。它的作用是给出「同线程数下的旧口径基准」，
让重建完成后的差异只剩特征本身。

| 配置 | 均值 MAE | 相对 C0 |
|---|---|---|
| C0 集中式 le=10 | 0.001516 | — |
| C1 本地各训 le=10 | 0.034487 | +2175.3% |
| FedAvg iid le=10 | 0.001170 | −22.8% |
| FedAvg iid le=3 | 0.011499 | +658.6% |
| FedAvg sohrank le=10 | 0.001515 | −0.1% |
| FedAvg dirichlet le=10 | 0.000957 | −36.8% |
| FedProx dirichlet le=10 | 0.001891 | +24.8% |

**线程数单独造成的差异被量化了**（同一份旧表，24 线程 → 1 线程，56 对）：

- |相对差| 中位 **1.86%**、最大 **27.37%**
- 方向**对称**：变好 28 个 / 变差 28 个

也就是说线程数不是一个「微小的数值扰动」，而是一个**与 seed 同量级的独立随机源**
（seed 相对标准差实测 5%–63%）。之前只看了 C0 seed0 的 2.1% 就以为影响很小，
那是单个样本的运气——56 对里最大到 27%。这条反过来加强了「必须钉死线程数」：
判据阈值是 5%，而线程数能单独造成 27%，跨机器比较必然失效。

**好消息是它不翻转任何判定。** 消融集跑完整条判定链，结论与归档一致：

| 判定 | 归档（24 线程） | 消融（1 线程） | 一致？ |
|---|---|---|---|
| H1 le=10 | −23.5% → 推翻 | −22.8% → 推翻 | ✓ |
| H1 le=3（等训练量） | +640.6% → 推翻 | +658.6% → 推翻 | ✓ |
| H2 sohrank（P1） | 接受 | +29.5% → 接受（p_t=0.0003） | ✓ |
| H2 dirichlet（N1） | 未判定 | −18.1% → 未判定（处理未施加） | ✓ |
| H3 FedProx | 推翻 | 改善 −97.5% → 推翻 | ✓ |

**这张表的覆盖范围要说清楚**：7 个配置里**没有 FedAdam**。补测两条 FedAdam
（同表、同 torch、1 线程 vs 16 线程）后：

| 配置 | 1 线程 / 16 线程 |
|---|---|
| `fedadam eta=0.01 le=3` | **×1.80** |
| `fedadam eta=0.001 le=10` | ×1.017 |

所以不是「FedAdam 敏感」，而是 `server-lr 0.01 + le=3` 这条**激进配置**把轨迹
推到近似分岔点，每轮的小差异被指数放大；另一条 FedAdam 只差 1.7%，与那 56 对
的中位 1.86% 一样。正确的说法是：**线程数敏感度依配置而变，尾部能到 80%**，
而那 56 对的最大值 27.37% 之所以偏小，是因为 7 个配置里没有一条踩在分岔点上。

这对 M3 ② 那条「FedAdam 替代 FedBN：le=3 好 86.1%，le=10 无收益」意味着什么：
**不是**要推翻它（两个臂在同一环境下测的，内部可比），而是说它的**幅度不可
移植**——86.1% 与同一旋钮能造出的 80.3% 同量级，换台机器可能变号。重跑在
1 线程下重出这两个数，如果幅度明显变了，就说明这条结论对环境太脆，不能当
结论报。

**manipulation check 也独立复核了一次**（这次是重新跑的
`measure_heterogeneity.py`，与 `results/heterogeneity.json` 逐位一致）：
Dirichlet α=0.5 的客户端身份解释方差 **12.2%**，比同 N 的 IID **17.3%** 还低，
而数量基尼是 **0.519**（最大 21 颗 / 最小 1 颗，21×）。所以「Dirichlet 比 IID
好 36.8%」不是发现，是两个混淆叠出来的：标签偏斜没施加（甚至更均匀）+ 数量
偏斜把 FedAvg 拉向部分中心化。判定器如实输出「未判定（处理未施加）」。

这条也是「先量标签偏斜」这条硬约束的第二次实战命中——第一次是 M3，这次是
重跑前复核。


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
17. **FedAdam 的 η 不是"学习率"，是服务端每轮的步长；τ 不是数值保护，是噪声地板**（2026-10-24 实测）。两个想当然都会毁掉这组实验：
    - **η=0 不等于 FedAvg**。更新式是 `w + eta_norm·m/(√v+τ)`，η=0 时全局参数**冻结**；FedAvg 是服务端 η=1 的普通梯度步。把"η=0 退化成 FedAvg"写进文档会让人以为它能当对照用。
    - **η 的量级由伪梯度决定，而伪梯度可以量**。本数据集实测每轮 mean |Δ| ≈ 7×10⁻⁴、max 1×10⁻²，η 探针在同一 seed 上差 3.4 倍（0.1→0.00268、0.01→0.00101、0.001→0.00340）。恒定伪梯度下每轮位移恰为 η（闭式解，偏置矫正把 β^t 约掉），所以 η 选错不是"收敛慢"，是"每轮走多远"直接定错。
    - **τ 与 |Δ| 同量级时它在筛选坐标**。稳态步长 = η/(1+τ/|Δ|)，τ=1e-3 把 |Δ|≲1e-3 的坐标阻尼掉。实测 τ 从 1e-3 降到 1e-6，MAE 从 0.00101 退到 0.00173——默认值恰好匹配本数据的伪梯度尺度，但这**是巧合不是定律**，换数据集必须重量 |Δ| 再定 τ。
    - **`server_lr` 必须进结果表的分组键**（`summarize_results._key` 第七维）。少了它，η 探针的三档会被平均成一组，那张表的 A1 行描述的是任何一个都没跑过的配置——和当年 `n_clients` 没进键是同一类事故。
18. **ε 不是自由变量，nm 才是**（2026-10-25 实测，M4 的结论建立在这个推导上）。ε 由 q、步数、nm 共同决定，而 q 与步数**由训练协议钉死**：本仓库每客户端 80 行、batch 64 → 每 epoch 2 批 → **q=0.5**；100 轮 × le=3 → **600 步**。于是 nm=1 的 ε 是 **152.31**，不是「个位数」。要把 ε 压到 8 以内必须把 nm 抬到 8，而那时加在批均值梯度上的噪声 Std = `nm × C / 期望批大小` = `8 × 1.0 / 40` = **0.2**，与实测信号（批均值梯度范数 0.147–0.223）**同量级**。**扫 nm 之前先用 `scripts/check_dp_budget.py` 算预算**，否则可能出现三档全都 ε>8 的点，那时 §11.3 的判据一条都用不上。三条降压路径（加大 nm / 减步数 / 减 q）都写在检查器的失败输出里，其中「减小 batch_size」会**增加**每 epoch 批次数从而增加总步数，不一定有用，别凭直觉调。
19. **DP 会计器每客户端一个，且绝不乘 N**（同一个 bug 会让 ε 虚高 23 倍，而输出看起来完全正常）。联合机制是 N 个机制的乘积，「改变客户端 i 的一条记录」只影响客户端 i 自己的更新（其余 N−1 个逐位不变），所以 `ε_joint = max_i ε_i`。写成 `N × ε_i` 会让报告里的隐私保证凭空紧 23 倍——**虚高的 ε 看起来很"严谨"，所以这个错特别难被发现**。执行点：`src/flsoh/dp.py`，会计器必须**跨轮存活**（每轮新建会让 ε 少算 100 倍），`tests/test_dp.py` 钉住「每客户端独立且不求和」。
20. **`noise_multiplier=0` 是对照组，不是 DP 臂**。它仍走逐样本裁剪 + Poisson 抽样的完整 Opacus 管道（梯度范数确实被 C 压住），只是不加噪，ε 报 `inf`。没有它，「DP 一共付出多少」与「噪声单独付出多少」分不开——实测机制开销单独是 +39.2%（不显著），噪声在其上再加 +82%～+186%。合起来报会把机制开销算到噪声头上。plan §7 第二条常见错误（把裁剪当成 DP）说的就是这件事的另一面。
21. **Opacus 默认 Poisson 抽样，它把有效批大小从 64 变成 40**。`sample_rate = 1/len(loader)`，期望批大小 = 行数 × q = 80 × 0.5 = **40** 而不是 64。这不是可以顺手关掉的装饰：关掉只是**近似**（Opacus 自己的 docstring 就这么写），而本仓库选择保住保证。代价是 loss 必须按 `x.shape[0]` 加权（批大小逐批不同），以及 DP 臂的基线不能直接与非 DP 臂比——后者吃的是满批。
22. **DP 让 seed 波动变小，但那不是优点**。实测非 DP 同预算（le=3）的 seed 波动是 66.7%，五档 DP 臂全部降到 11.7%–13.9%。看起来像「DP 让训练更稳定」，但按坑 16 与 §11.3.5 记下的 FedProx 教训，**稳定性 ≠ 正确性**：这只是逐样本裁剪把每样本梯度范数压住的结果，而裁剪同时就是代价的来源。这一格只作为现象记录，不许写进结论。
23. **`axhline(..., label=...)` 配对数轴会把图例排乱，而图的其他部分完全正常**（2026-10-25 修图 3 时发现，图 1 从生成那天起就是这个状态）。matplotlib 3.11.2 实测：条目顺序错、句柄画成横贯整个坐标区的长线、几行文字叠在一起。曲线、坐标轴、标题全对，**「看一眼」发现不了**。`axhline`/`axvline` 用的是混合变换（一个轴分数、一个轴数据），坐标轴取对数后图例的布局就算错了。安全写法是 `ax.plot([0, 1], [y, y], transform=ax.get_xaxis_transform(), label=...)`：横向仍然满宽，图例里的句柄是正常的一小段。`tests/test_figure_style.py` 用 `ast` 扫所有 `scripts/make_figure*.py`，带 `label=` 的 `axhline`/`axvline` 一律判红（含负控：一个脚本都没扫到就报错）。
24. **图 3 的「非 DP 同预算」参考线差点画成了 FedAdam**（同一次修图）。第一版写 `plain = [k for k in sel if k[4] is None]`，而 `sel` 只按 `local_epochs` 过滤——同 le=3 的组里有 FedAvg（0.01165）也有 FedAdam（0.00162），`plain[0]` 抓到哪个取决于字典顺序。参考线因此画低了 7 倍，而图上没有任何异常。**「同 le」只对齐训练量，策略/划分/N 不同就不是同一个实验**。现在是 `make_figure3.pick_references()`，要求三者全同，缺一条就拒绝出图；`tests/test_figure3.py::TestPickReferences` 用 FedAdam / sohrank / 不同 N 三个干扰项钉住它。
25. **argparse 的 help 文本里一个半角 `%` 就是定时炸弹，而它在 Python ≤3.13 上完全不响**（2026-10-25 收尾 M4 时发现）。argparse 会拿 help 字符串做 `%` 格式化；老版本只在出现 `%(` 时才格式化，所以 `"抓取耗时占墙钟 98.2%，看着像"` 一直显示正常。Python 3.14 换了行为，同一行直接 `ValueError: unsupported format character '?' (0xff0c)`——`--help` 崩、参数解析全崩，而堆栈指向 argparse 内部，看不出是帮助文本写错了。**写成 `%%` 只在 3.14 上对**（≤3.13 不格式化，会显示成两个百分号）；本仓库统一用**全角 ％（U+FF05）**，任何版本都原样显示，且中文排版里全角百分号本来就更合适。执行点：`tests/test_cli_help.py` 用 `ast` 扫所有脚本的 `add_argument(help=...)`，见半角 `%` 即红（含负控）。
26. **「插值链把不相干的点配到一起」是唯一一类不抛异常、不改形状、还能训出好指标的错误**（2026-10-25 查出，贯穿 M1–M4）。MATR 每个循环的数组是「充电 + 放电」拼起来的，而 `charge_curve()` 返回整条循环；`delta_q_curve` 内部 `argsort` 把放电段排进充电段的电压区间，而放电段的 `Qc` 是常数。结果：TV 269.60 vs 正确的 0.21（1280 倍），与正确值的**相关系数 −0.019**，而 C0 照样训出 MAE 0.0016。判据是**总变差 / 曲线幅度**：正确数据 5.9、错配数据 122.4。执行点：`matr.discharge_curve()`（按 `I < 0` 取段、用 `Qd`）+ `scripts/check_features.py`（阈值 20，`build_features.py` 跑完自动调）。**为什么必须用放电段**：恒压充电段 dQ/dV → ∞，均匀电压网格在那里会把 0.1 Ah 压进一两个网格点（充电段最大相邻跳变 1.118 Ah，放电段 0.013 Ah）。
27. **规格指纹只描述「要算什么」，不描述「怎么算」**（同上，这个错会让修完的算法跑不出新数据）。`state_fingerprint` 原来只有循环区间 / 参考循环 / 电压网格。修完特征口径后重跑同一条命令，旧分片全部被判「规格一致」而跳过，表里一行新数据都没有，脚本打印「全部有分片」、退出码 0。**这类错误比参数漂移更隐蔽**：参数漂移会当场拒绝续跑，算法漂移会安静地复用旧数据。执行点：`flsoh.features.FEATURE_SPEC_VERSION` 进指纹，缺该键的旧目录一律拒绝（`tests/test_features.py::TestStateFingerprint` 三条钉住）。
28. **TV/幅度 的噪声本底依赖「采样密度 × 电压噪声」，阈值不能跨数据源搬**（2026-10-25 实测）。同一张 1000 点网格上：实验室 MATR（约 0.2 mV、5 s）5.9；模拟 App（2 mV、10 s）中位 12.8、尾部 45.2；错配数据 122.4。阈值 20 在实验室路径上有 20 倍余量，在 App 路径上**连噪声本底都压不住**。两个方向都错：不改阈值 → 每份合法 App 数据都被判红，所有人学会加 `--no-check`；悄悄抬到 200 → 拦不住错配。正确做法是**按数据源分别定阈值并写明余量**（`run_e2e.py` 用 60，并在 docstring 里写明「对 App 路径这道闸门只有 2.7 倍余量」）。附带发现：**最大跳变/幅度** 对噪声不敏感（错配 0.78、实验室 0.11、App 0.19），是更稳的判别量，但 0.55 与 0.78 的间隔不够宽，只能作为诊断量报出，不设闸门。
29. **端到端只看 MAE 等于没看**。模拟数据的标签 std 只有 0.03 量级，一个「看起来正常」的 0.067 比常数基线（0.027）还差 2.5 倍。`run_e2e.py` 因此**必须**同时输出 `baseline_mae` 与 `beats_baseline`，跑不赢基线就非零退出。同理，任何「模型有效」的陈述都要带一个同数据上的平凡基线——否则那个数字只说明「损失函数算得出来」。
30. **解释器与线程数会改数值，而且改了看不出来**（2026-10-25 实测，同一份特征表、同一 seed、同一仓库、C0 seed0）：
    - torch `2.10.0+cpu` → `0.0015666428208351135`；torch `2.14.1+cpu` → `0.0015434211492538453`。**用 2.10.0 重跑能逐位复现历史记录，2.14.1 不能**——所以这不是随机性，是换了 BLAS 内核与算子实现。
    - `OMP_NUM_THREADS=1` → `0.0015997225046157837`；默认 24 线程 → `0.0015666428208351135`，差 2.1%。线程数改的是浮点累加顺序，300 个 epoch 会放大它。
    - **默认线程数的真正问题是结果随机器核数变化**：4 核机器与 24 核机器跑同一个 run 得到不同数字，而两边都"复现了同一条命令"。所以重跑统一钉 `--threads 1`（比 24 线程慢约 9%，但唯一与硬件无关），产物记 `torch_num_threads` 与 `python_version`。
    - 执行点：`scripts/rerun_all.py` 的 `check_torch_version` / `check_thread_count` / `pinned_env`，torch 版本不符可直接拒绝，线程数不符没有绕过开关。
    - 补记（同日，56 对配对后）：上表的 2.1% 是单个样本。56 对实测 |相对差| 中位 1.86%、**最大 27.37%**、方向 28 好 / 28 差完全对称——它是与 seed 同量级的随机源，不是小扰动。详见 §13.5.6。
    - **再补记（同日，重跑冒烟时撞到，比上面两条都严重）**：那个「慢约 9%」是错的，而且「钉 24 线程」在这台机器上**根本钉不住**。同一 run、同一份表、同一个 torch：

      | 配置 | 1 线程 | 16 线程（`--threads 24` 实际拿到的） | 相对 |
      |---|---|---|---|
      | `fedadam eta=0.01 le=3` | `0.0021398472785949707` | `0.0011869078874588012` | **×1.80** |
      | `fedadam eta=0.001 le=10` | `0.003461749404668808` | `0.0034043075144290923` | ×1.017 |

      1 线程跑两遍逐位相同、16 线程与归档值逐位相同（两个配置都是），所以这些倍数是**线程数单独造成的**，不是噪声。
    - **别急着归因给「FedAdam 有动量」**：另一条 FedAdam 只差 1.7%，与那 56 对的中位 1.86% 一样。真正的差别是 `server-lr 0.01 + le=3` 这条**激进配置**把轨迹推到了近似分岔点，每轮的小差异被指数放大。所以正确的说法是「线程数敏感度**依配置而变，尾部能到 80%**」，而不是「FedAdam 敏感」——56 对的最大值 27.37% 之所以偏小，是因为那 7 个配置里没有一条踩在分岔点上。
    - 两个后果：① `--threads 24` 是个**钉不住的值**——`pinned_env` 设 24，torch 只给 16，产物如实记 16，然后全部跑完之后 `check_thread_count` 才报不一致；② 判据阈值 5% 在尾部完全失效，而且**你没法预先知道哪个臂在尾部**——只能同一环境重跑。
    - 加了一道预检 `probe_thread_count` + `check_probed_threads`：在 `pinned_env(threads)` 下起子进程问 torch 实际拿到几个线程，与请求值不符就**在开跑前**拒绝。实测现在 `--threads 24` 直接被拒（附 1.80 倍的证据），`--threads 1` 打印「线程数钉住了」。
    - 顺带修正时间估算：1 线程不是慢 9%，而是约 **1.8 倍**（同一 run 单独跑 33.1 s → 61.0 s；并发时不可比——16 线程那次 250 s 是争用造成的）。所以 11.91 小时的墙钟在 1 线程下约 21 小时，`--jobs 12` 约 1.8 小时而不是 1 小时。

31. **`--skip-existing` 只判「文件在不在」，会把上一版的结果当成已完成跳过**（2026-10-25 修）。`--out` 默认是 `results/`，而那里装的正是上一次跑的结果。于是「重跑中断 → 带 `--skip-existing` 续跑」会跳过全部任务、打印「剩 0 个」、退出码 0，接着 `summarize_results` 汇总出一张**新旧混杂的表**——不抛异常、行数也对、seeds 也齐，只有一部分数字是上一版的。这与坑 26/27/30 是同一个形状：错的不是数值计算，是「什么算已完成」。
    - 更隐蔽的是配套的 `check_thread_count`：它只看「带 `torch_num_threads` 的那些 run」，混杂目录因此得到 `{1}` 并判定一致。**「全都没有」是合法旧目录，「部分有」才是混杂**——两者必须分开判。
    - 修法三件：① 产物记 `feature_version`（`src/flsoh/featurespec.py`，纯标准库，两个训练脚本各加一个 `features=` 参数）；② `artifact_is_current()` 按「线程数 + torch 版本 + 特征口径」判能否跳过，读不出的期望值不拿来下结论；③ `check_thread_count` 增加混杂检测。
    - **混杂检测不能连续跑一起禁掉**：重跑中断后结果目录**必然**半新半旧，一律判失败等于逼人删掉已完成的一半重来，而删数据比混数据更贵。所以 `allow_mixed` 只跟随 `--skip-existing` 打开——那时每个产物都过 `artifact_is_current` 逐个核对，混杂不再是静默的。**线程数取值不符仍然硬拒**（`--threads 8` 打错成 `1` 不是续跑能解释的）。
    - 实测：修前对 172 个旧结果跑 `--skip-existing` 是「跳过 157 个，剩 0 个」，修后是「跳过 0 个，剩 157 个，其中 157 个已有产物但不是本次设置的产物」。
    - **教训的形状**：这一条和坑 26/27 是同一类——错的东西不抛异常、不改形状、指标还很好看。区别是 26/27 错在数据，这条错在**环境**，而环境从来不在产物里，除非你主动记。

32. **TV/幅度 被「1/幅度」污染，逐行卡阈值等于在测「这一行的幅度够不够大」**（2026-10-25 修）。修完特征口径后，1080 行实测按幅度四分组：

    | 幅度分组 | 幅度中位 | TV/幅度 中位 | 最大 |
    |---|---|---|---|
    | 最小 25% | 0.0137 | 8.13 | 21.75 |
    | 25–50% | 0.0197 | 5.84 | 12.75 |
    | 50–75% | 0.0282 | 4.43 | 9.72 |
    | 最大 25% | 0.0430 | 3.45 | 6.74 |

    单调下降，`corr(TV/幅度, 1/幅度) = 0.774`。机制很直白：噪声不随幅度缩放，`TV ≈ TV_信号 + TV_噪声`，幅度越小噪声占比越大。于是原来的「逐行卡 20」会把**最小的那一档幅度的正常行**判成坏行——实测 2/1080 行越线（21.75 与 20.18），而它们只是幅度 0.010 Ah 的早期循环。
    - **为什么必须改而不是调大阈值**：逐行判据的失败概率**随表规模增长**。多加几十颗电池，尾部的低幅度行迟早踩线，而失败原因与数据质量无关。一道会因为「数据变多」而失败的闸门，测的不是数据质量。把它抬到 30 只是把踩线推迟。
    - **改法：两道闸门各管一件事。** 中位数 20 管**系统性错位**（那个 bug 让所有行一起坏，中位 122.41，6 倍余量；正确数据中位 5.3，3.7 倍余量，单行噪声影响不了中位数）；单行硬上限 60 管**个别坏行**（NaN、整行常数、某颗电池错配；错配逐行 122–172，正确数据最大 21.75，两边各约 2.7 倍余量）。
    - 判定规则收敛到 `check_features.report_is_ok()` 一个函数，三个调用点（CLI、`build_features.py`、`run_e2e.py`）共用——否则改判据时漏掉一处，就会出现「一个入口说通过、另一个说失败」，而两条路径的输出都正常。
    - **负控**：旧错表仍判 FAIL（中位 122.41 > 20），新表判 PASS（中位 5.29、越硬上限 0 行、2 行落在两线之间作诊断）。回归测试 `test_a_tail_of_low_amplitude_rows_does_not_fail` 直接把「76 行正常 + 4 行低幅度噪声行」钉成必须通过。
    - 这条的元教训：**闸门本身也要有「为什么是这个阈值」的实测依据**。原来的 20 是拿中位数 5.9 估的「3 倍余量」，却当成了逐行上限用——把分布统计量当逐行判据，是这次返工的全部原因。

33. **重名定义会静默吞掉测试：pytest 全绿，而你以为写了的那些根本没跑**（2026-10-25 加闸门）。Python 允许重名，后一个定义**静默覆盖**前一个；对测试来说就是「粘贴时多贴一遍整个类」→ pytest 只收集最后一份，退出码 0、没有 warning、收集数只少几条，而少的那几条你正好以为在跑。这是坑 26/27/30/31 之外的第四种「输出正常但错了」：前三种错在数字、数据、环境，这一种错在**测试本身消失了**。
    - 不是假想的：`find_shadowed_definitions()` 第一次运行就抓到 `tests/test_rerun_all.py` 里 `TestMixedDirectory` 与 `TestSkipExistingFreshness` 各被贴了两遍——122 行死代码、**11 个测试函数从来没跑过**。而就在同一次会话里，`TestMixedDirectoryResume` 又被贴了两遍（`pytest tests/test_rerun_all.py` 照样全绿）。
    - 暴露它的是**徽章**：静态清点会把两份都数上，于是徽章数（748）与「实际新增 3 条」的预期对不上。也就是说，这个洞是被「另一条看似无关的闸门」抓住的——闸门之间互相校对的价值就在这里。
    - 闸门 `tests/test_readme_badge.py::TestNoShadowedTests`：仓库自身必须干净 + 重复类必须被抓 + 重复方法必须被抓 + **名字不同不许误报**（正控，否则这条闸门只是在拒绝一切）。
    - 顺带一条工程习惯：粘贴大段代码后不要只看「测试全绿」，要看**收集数变化是不是你预期的那个数**。

34. **「光滑」与「有信息」是两件事，光滑性闸门结构上抓不到「特征与标签无关」**（2026-10-25 加第三道闸门）。一条常数曲线的 TV/幅度 是 **0**——完美光滑。所以中位数闸门与单行硬上限都不可能发现「这 1000 维特征其实没在变」。2026-10-25 那个错口径恰好两者都坏（TV 122.4 且 ρ −0.168），于是修完光滑性容易产生一种错觉：闸门绿了就是对了。
    - **加的是什么**：每颗电池算「1000 个电压点各自的 ΔQ(V) 与循环号 k 的 Spearman 相关」，取中位数，再跨电池汇总。物理依据是硬约束——SOH 从 1.0 掉到 0.8 时，同一电压点的放电容量只会变小，所以 ρ 必须显著为负。
    - **实测分辨力比 TV/幅度 强得多**（同一把尺子量两张表）：

      | 表 | ρ 中位 | ρ ≤ −0.8 的电池 | TV/幅度 中位 |
      |---|---|---|---|
      | 错口径（旧） | **−0.1677** | 0/135 | 122.41 |
      | 正确（新） | **−0.9789** | 61/67 | 4.96 |

      阈值 −0.7 与 80% 两边都不贴线（正确 −0.979 / 91%，错口径 −0.168 / 0%）。模拟 App 路径上 ρ 中位 **−1.000**、100% 达标——模拟器把 SOH 几乎全编码在幅度上，所以整条曲线按容量因子缩放，每个电压点都单调下降。
    - **三个必须分开的状态**：通过 / 失败 / **未判定**。模拟数据将来若改成「形状不随老化变化」，ρ 全是 NaN，那是未判定而不是通过——把「未判定」写成「通过」正是坑 7 里「算法免疫 vs 处理未施加」的同一个形状。`report_is_ok()` 不把未判定当失败（模拟路径需要），但 `build_features.py` **单独把未判定判失败**，因为真实电池必然老化，量不出相关说明特征里没有容量信息。
    - **两个实现坑**（都当场踩到）：① 秩相关会把「所有值相同」变成「秩 0,1,2,…」，于是常数列看起来是完美正相关 +1——必须显式把常数列判成 NaN，否则「整块常数」会被误判成「信号反向」；② 判常数用 `np.ptp` 不要用 `np.std`，后者对完全相同的值会因均值误差给出 **6e-19**，`== 0` 判不出来（实测：3 颗电池的常数表被判成 ρ = +1、`undetermined=False`）。
    - **负控就是这道闸门的理由**：造一张「幅度随循环不变 + 1e-6 独立噪声」的表，TV/幅度 中位 **1.0**（两道光滑性闸门全放行），而 ρ 中位 **0.013** → 判失败。这条测试 `test_smooth_but_uninformative_table_fails` 直接把这个场景钉住；另有反向趋势（ρ = +1）、整表常数（未判定）两条。

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
