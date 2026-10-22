# 联邦学习式充电数据众包研究平台 — 详细执行方案 v1.0

> 撰写：严谨-审稿专家 ｜ 2026-10-18 ｜ 面向已有 Python 与模型训练基础的用户
> 数据口径与既有锂电池项目（NASA/CALCE/MIT）保持一致，特征工程可直接复用 MIT 复现管线。

---

## 0. 一页速览

| 项 | 内容 |
|---|---|
| 定位 | 用 Flower 框架做**联邦仿真研究**：多客户端在不上传原始充电数据前提下联合训练 SOH 估计模型，量化"联邦 vs 中心化"精度差距与隐私代价 |
| 最小可行版本 MVP | 约 **2 周**：FedAvg/FedProx × IID/Non-IID × MIT 数据按电池划分，出轮数-精度曲线与对比表 |
| 完整版 | 再加 **2-3 周**：Scaffold/FedAdam/FedBN 对比、跨化学体系极端 Non-IID、本地差分隐私（DP-SGD）与隐私-效用曲线、App 数据入口 Schema 原型 |
| 硬件要求 | 单机笔记本即可（CPU 也能跑，MLP 模型很小；有 GPU 更快） |
| 关键风险 | Flower API 版本变动快（须锁版本）；Non-IID 下 FedAvg 不收敛；全局标准化统计量 = 隐私泄露（详见 §8） |

---

## 1. 项目目标与最终交付物

### 1.1 项目目标（可证伪的三问）
1. 在按"用户/电池"划分的 Non-IID 数据下，联邦训练的 SOH 模型比中心化训练差多少（MAE/RMSE 差距量化）？
2. FedProx/Scaffold 等漂移校正策略能挽回多少差距？代价（额外通信/计算/实现复杂度）是什么？
3. 在模型更新上加本地差分隐私（梯度裁剪+噪声）后，精度-隐私的权衡曲线长什么样？

### 1.2 交付物划分

**MVP（2 周）**
- D1 可复现仿真代码库：`flwr run` 一条命令跑通，`requirements.txt` 锁版本、固定 seed
- D2 数据划分脚本：MIT 数据集 → 10/20/50 个虚拟客户端（按电池划分），含 IID 与 Dirichlet Non-IID 两种模式
- D3 对比实验：中心化上限 vs IID-FedAvg vs Non-IID-FedAvg vs Non-IID-FedProx
- D4 结果图：通信轮数-精度曲线 + 最终精度对比表（≥3 seeds，均值±标准差）
- D5 一份 2-3 页的 MVP 结论备忘

**完整版（MVP + 2-3 周）**
- D6 策略扩展：FedAdam（或 FedYogi）、跨化学体系场景下的 FedBN 对比
- D7 Non-IID 程度扫描：Dirichlet α ∈ {0.1, 0.5, 1.0, ∞} 的系统实验
- D8 隐私增强：Opacus 本地 DP-SGD，隐私-效用曲线（噪声倍数 vs MAE）
- D9 App 数据入口原型：充电会话 JSON Schema v1 + 本地特征提取函数库 + 模拟数据生成器（§5）
- D10 完整技术报告（含威胁模型声明与实验协议）

---

## 2. Flower 框架快速上手路径

### 2.1 环境（第 1 天）
```bash
python -m venv .venv && source .venv/bin/activate
pip install "flwr==1.9.1" "flwr[simulation]" flwr-datasets torch numpy pandas
```
⚠️ **必须锁版本**。Flower 从 1.8 起改用 `ClientApp/ServerApp` 范式，之后各版本策略类的导入路径仍在变动（如 FedProx 在不同版本位于 `flwr.server.strategy` 或 `flwr.serverapp.strategy`）。本文代码以 flwr 1.9.x 范式为准；装好后第一件事是跑通官方 PyTorch quickstart，再逐步替换为自己的数据与模型。

### 2.2 Server/Client 结构与核心 API

Flower 仿真的三个组成：
1. **ClientApp**：定义单个客户端的行为（训练、评估）
2. **ServerApp**：定义聚合策略（Strategy）与通信轮数
3. **pyproject.toml**：声明两个 App 与运行配置，`flwr run` 启动仿真引擎（VirtualClientEngine 在本进程内模拟 N 个客户端，不需要真实网络）

核心 API 速查：

| API | 作用 |
|---|---|
| `fl.client.NumPyClient` | 客户端基类，需实现 4 个方法：`get_parameters` / `set_parameters` / `fit`（本地训练）/ `evaluate`（本地评估） |
| `fl.client.ClientApp(client_fn=...)` | 按 `cid`（客户端 ID）创建客户端实例 |
| `fl.server.strategy.FedAvg(...)` | 聚合策略；关键参数 `fraction_fit`、`min_fit_clients`、`min_available_clients` |
| `fl.server.strategy.FedProx(proximal_mu=0.01)` | FedProx 策略（其余用法同 FedAvg） |
| `fl.server.ServerConfig(num_rounds=100)` | 通信轮数 |
| `fl.serverapp.ServerApp(config=..., strategy=...)` | 服务端封装 |
| Partitioner（flwr-datasets） | 内置 IID/Dirichlet/Pathological 划分器（按样本切分场景用；按电池划分自己写更可控） |

### 2.3 代码骨架（可直接改造）

```python
# client_app.py
import flwr as fl
import torch
from model import MLP
from data import load_client_data   # 自己实现：按 cid 返回该客户端的 DataLoader

class SohClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net, self.trainloader, self.valloader = net, trainloader, valloader
    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.net.state_dict().items()]
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        self.net.load_state_dict({k: torch.tensor(v) for k, v in params_dict})
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=config["local_epochs"])  # 本地训练，原始数据不出本地
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, mae = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"mae": mae}

def client_fn(cid: str):
    net = MLP()
    trainloader, valloader = load_client_data(cid)   # ← 数据划分的核心入口
    return SohClient(net, trainloader, valloader).to_client()

app = fl.client.ClientApp(client_fn=client_fn)
```

```python
# server_app.py
import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,            # 每轮参与比例（大客户端数时可降到 0.2~0.5）
    min_fit_clients=10,
    min_available_clients=10,
)
app = fl.serverapp.ServerApp(
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
)
```
（FedProx 只需把策略换成 `FedProx(proximal_mu=0.01)`；**注意** FedProx 要求客户端在 `fit` 中支持返回"未完成的本地步数"，用官方 FedProx 示例的 fit 写法即可。）

### 2.4 三步上手节奏
1. **Day 1-2**：跑通官方 PyTorch quickstart（MNIST，10 客户端仿真）
2. **Day 3-4**：把 MNIST 换成 MIT 充电数据的 DataLoader + MLP，先跑 IID 划分，确认联邦结果与中心化接近
3. **Day 5**：换成按电池的 Non-IID 划分，观察 FedAvg 掉点 → 进入 §3 的策略设计

---

## 3. 联邦学习具体设计

### 3.1 模型选择

**主选：MLP + ΔQ(V) 特征向量**（与既有 MIT 复现基线同构）
- 输入：部分充电电压曲线差分特征（如 Severson 2019 的 ΔQ_{100-10}(V) 1000 维，或自选电压区间），只用**充电片段**即满足"众包 App 只采充电数据"的设定
- 结构：1000 → 128 → 64 → 1（SOH 回归），参数量 < 15 万 → 单轮通信量约 0.6 MB/客户端，仿真快
- 优点：复用现有特征工程与 SOH 标签口径；模型小 → 通信轮数可以多跑、统计结论更稳

**进阶（完整版可选）：1D-CNN**，输入固定长度的充电片段三通道序列（V/I/T，重采样到统一长度如 256 点），验证端到端特征的联邦可行性。MVP 阶段不碰。

**明确不建议起步就上 LSTM/Transformer**：通信量大、仿真慢，且模型复杂度会掩盖 Non-IID 问题的本质。

### 3.2 Non-IID 问题与策略选择

先明确 Non-IID 的三种来源（实验时要分开报告，不要混在一张图里）：
- **量偏斜（quantity skew）**：各客户端循环数不同（真实用户必然如此）
- **特征/标签偏斜**：各客户端使用模式不同（快充 vs 慢充、温度带）
- **域偏移（domain shift）**：化学体系/电池型号不同（最极端）

| 策略 | 适用场景 | 关键参数/代价 | 本项目建议 |
|---|---|---|---|
| FedAvg | IID 或轻度 Non-IID 基线 | 无额外参数；Non-IID 下易发散 | 必做基线 |
| FedProx | 量/特征偏斜，本地 epoch 多时漂移大 | `proximal_mu` 0.01–0.1 网格搜；实现零成本（Flower 内置） | **首选 remedy** |
| FedAdam/FedYogi | 各客户端梯度尺度差异大 | 服务器端学习率/β 需调 | 完整版对比 |
| FedBN | **跨化学体系/域偏移** | BN 层不聚合、各客户端本地保留；改聚合函数即可 | 强烈建议在 MIT/NASA/CALCE 场景对比 |
| Scaffold | 客户端漂移校正理论上最干净 | Flower 无开箱策略，需自实现（客户端+服务器各维护一份控制变量，显存/通信翻倍） | 可选；时间紧就砍 |

调参要点：**本地 epoch 数与漂移成正比**。Non-IID 下 local_epochs=5 常比 1 差很多；μ 与 local epochs 联合调。每轮参与比例低于 1 时需报客户端抽样带来的方差。

### 3.3 模拟多客户端的划分方式

MIT 数据集 124 颗电池，天然适合按对象划分：

| 方案 | 划分方式 | 模拟的真实场景 | 建议 |
|---|---|---|---|
| A 按电池 | 每客户端 = k 颗电池（k=1,2,5 → 124/62/25 个客户端） | 一人一块电池 | **MVP 主方案** |
| B 按用户代理 | 按 MIT 的 batch 分组 + 额定容量档（1.1/1.2/1.5 Ah）聚类成"用户" | 同一用户多设备 | 进阶 |
| C 按化学体系/数据源 | MIT / NASA / CALCE 各为一个（或少数几个）大客户端 | 不同电池厂众包 | 完整版必做（极端 domain shift） |
| D Dirichlet 样本分配 | 把循环片段按 Dirichlet(α) 随机分给 N 客户端 | 量化任意 Non-IID 程度 | 完整版 α 扫描 |

**三条硬规则（审稿视角）：**
1. **同一颗电池的相邻循环不得跨客户端**（时间序列截断泄漏会让泛化虚高）——划分以"电池"为最小原子，客户端内再按时间切 train/val
2. 全局测试集（每轮评估用）须独立于所有客户端数据；客户端本地 val 集另设，两套指标分开报
3. 客户端数量与轮数写进实验协议：典型配置 N=25（按 5 颗/客户端）、num_rounds=100；N=124 时 fraction_fit 降到 0.2

---

## 4. 隐私保护的可选增强（入门做法）

分层递进，每层都写清威胁模型（本文默认：**半诚实服务器**，即服务器不作恶但好奇）：

- **L1 联邦本身（MVP 默认）**：只传模型参数/梯度，原始充电曲线不出本地。注意这只防"数据直接泄露"，不防梯度反演。
- **L2 本地 DP-SGD（完整版核心）**：用 Opacus 给本地训练加"逐样本梯度裁剪 + 噪声"
  ```python
  from opacus import PrivacyEngine
  privacy_engine = PrivacyEngine()
  net, optimizer, trainloader = privacy_engine.make_private(
      module=net, optimizer=optimizer, data_loader=trainloader,
      noise_multiplier=1.0,      # 扫描：{0.5, 1.0, 2.0}
      max_grad_norm=1.0,         # L2 裁剪范数
  )
  # 训练结束报告 privacy_engine.get_epsilon(delta=1e-5)
  ```
  产出：噪声倍数 × SOH MAE 的**隐私-效用曲线**，并报告对应 ε（RDP 会计器计算）。
- **L3 服务器侧保护（了解即可）**：Flower 内置服务器端差分隐私（固定/自适应裁剪），属于对客户端更新的保护，与 L2 的威胁模型不同，不要混淆。安全聚合（SecAgg）在仿真里可只做说明不做实现。

**常见错误**：用了 DP 却报不出 ε；或把梯度裁剪（只是训练稳定技巧）当成差分隐私。没有噪声就没有 DP，报告中必须分清。

---

## 5. 与手机充电数据 App 的数据入口对接

现阶段没有真实 App 数据流，务实做法是**先定契约、再用模拟数据回填**：

1. **定义充电会话 JSON Schema v1**（与清单第 4 项 App 共用同一契约）：
```json
{
  "user_id_hash": "sha256-pseudonym",
  "battery_id": "device-battery-uuid",
  "session_id": "...", "start_ts": 0, "end_ts": 0,
  "chemistry_hint": "LiCoO2|unknown",
  "sampling_hint": "variable",
  "records": [{"t": 0, "V": 3.87, "I": -1.2, "T": 29.5, "soc_est": 0.42}]
}
```
2. **本地特征提取函数库**（与联邦仿真共用同一份代码）：App 侧只执行"片段切分 + 重采样 + **本地**归一化 + 特征向量生成"，特征向量与 SOH 标签之外的一切留在设备/个人网关。这直接决定 §3.1 的 MLP 输入格式。
3. **部署形态两条路**（报告里写成设计选项即可，不做真实部署）：
   - 本地网关：手机 → 用户个人云/家庭设备上跑 Flower Client（工程上最现实）
   - 设备端：Flower 的 SuperLink/SuperNode 部署模式，Android 集成成本高
4. **模拟数据生成器**：从 MIT 充电片段加噪声/漂移生成符合 Schema 的"伪 App 会话"，用于端到端联调（数据入库 → 特征提取 → 本地训练 → 上传参数）。这是完整版 D9 的验收物。

**与既有项目的接口**：特征工程函数、SOH 标签口径、MIT 预处理管线全部复用（两套口径 = 两套结论 = 审稿直接否）。

---

## 6. 结果呈现形式

**图表清单（固定协议，避免"看着像有效果"的自由发挥）：**

1. **图 1 轮数-精度曲线**：x=通信轮数，y=全局模型在统一测试集上的 SOH MAE；中心化训练画水平参考线；FedAvg/FedProx/(Scaffold) 各一条；按 Non-IID 设置分面板（IID ｜ Dirichlet α=0.5 ｜ α=0.1 ｜ 跨化学体系）
2. **表 1 最终精度对比**：所有策略最终 MAE/RMSE，**≥3 seeds 报均值±标准差**，附收敛所需轮数
3. **图 2 Non-IID 程度扫描**：α ∈ {0.1, 0.5, 1.0, ∞(IID)} 各策略的柱状图
4. **图 3 隐私-效用曲线**：noise_multiplier vs MAE（标注各点 ε）
5. **表 2 通信量估算**：参数量 × 4 bytes × 2（上/下行）× 客户端数 × 轮数，让读者自行判断真实部署可行性

**报告结构模板**：动机与威胁模型 → 实验协议（数据/划分/策略/超参/seeds）→ 主结果（图 1-3、表 1-2）→ 讨论（模拟 vs 真实差距）→ 局限与后续。

---

## 7. 工作量与时间预估（按有效人日）

| 阶段 | 内容 | 人日 |
|---|---|---|
| W1 | Flower 环境+官方 quickstart 跑通；MIT 数据接入与按电池划分脚本；MLP 本地训练管线 | 4 |
| W2 | MVP 实验：FedAvg/FedProx × IID/Non-IID，3 seeds；曲线与表格；MVP 备忘 | 4 |
| W3 | 策略扩展（FedAdam/FedBN）；Dirichlet α 扫描；跨化学体系场景 | 4 |
| W4 | Opacus DP 集成与隐私-效用曲线；通信量估算 | 3 |
| W5 | Schema v1 + 模拟数据生成器 + 端到端联调；完整报告 | 4 |

合计约 **19 人日 ≈ 4-5 周**（兼职节奏按 1.5 倍放大）。MVP 截止点设在 W2 末，验收不过不进入 W3。

---

## 8. 常见坑清单（审稿人重点核查项）

1. **全局标准化 = 隐私泄露**：特征均值/方差若用全量数据计算，本身就是信息上传。必须**每客户端本地标准化**（并在报告中说明由此引入的各客户端输入分布差异——这恰恰是 Non-IID 的一部分）。
2. **Non-IID 下 FedAvg 不收敛**：通常是 local_epochs 过多 + 无近端项。先降 local epochs 到 1-2，再上 FedProx 调 μ，而不是先怀疑代码。
3. **聚合加权错误**：FedAvg 应按客户端样本数加权平均；自己手写聚合时等权加权是高频 bug（用 Flower 内置策略可规避）。
4. **序列泄漏**：同一电池相邻循环被切进不同客户端/测试集 → 结果虚高。划分以电池为最小原子（§3.3 规则 1）。
5. **测试集污染**：每轮全局评估集必须与所有客户端训练数据不相交；"客户端本地 val"与"全局测试"两套指标不可混报。
6. **模拟规模 ≠ 真实部署**：仿真无网络延迟/掉线/算力异构（straggler）；至少在讨论一节量化说明（如 fraction_fit<1 模拟部分掉线、报告通信量表 2），不要把仿真结论直接外推成"可部署"。
7. **通信量低估**：开工前先算清楚模型字节数 × 客户端数 × 轮数；MLP 没问题，换 1D-CNN/LSTM 时这个表就是生死线。
8. **随机性掩盖效应**：客户端抽样顺序、初始化都是随机的；结论必须 ≥3 seeds，差异小于标准差的"提升"不予采信。
9. **Flower 版本漂移**：锁版本 + 在 requirements 与报告中写明版本号；跑通官方 quickstart 再改代码，不裸写。
10. **两套 SOH 标签口径**：与既有 MIT 复现项目统一标签定义与预处理；任何口径变更需在报告"实验协议"中显式声明。

---

## 9. 量化验收标准

**MVP（W2 末）**
- `flwr run` 一键复现，锁定版本与 seed 后 3 次运行最终 MAE 相对波动 < 10%
- 完成中心化 / IID-FedAvg / Non-IID-FedAvg / Non-IID-FedProx 四条配置的轮数-精度曲线与表 1
- Non-IID 下 FedProx 相对 FedAvg 的 MAE 改善量被明确量化（正/负都要报）

**完整版（W5 末）**
- α 扫描与跨化学体系实验各至少 3 seeds；FedBN 在跨化学体系场景的增益被量化
- 隐私-效用曲线覆盖 ≥3 个 noise 倍数，每点附 ε 报告
- Schema v1 + 模拟生成器通过端到端联调（伪会话数据 → 本地训练 → 参数上传 → 聚合）
- 报告含威胁模型声明、实验协议、模拟-部署差距讨论三要素，缺一退回

---

*v1.0 完。本方案中 Flower API 写法以 flwr==1.9.1 为准，实施第 1 天以官方 quickstart 实测为准做一次核对。*
