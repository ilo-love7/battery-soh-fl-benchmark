# M3 备忘：标签偏斜扫到哪里掉点，以及把动量挪到服务端

**日期**：2026-10-24
**数据**：MIT-Stanford，135 颗电池（115 训练 / 20 测试），2700 行 × 1003 列特征
**设置**：100 轮、`local_epochs=10`、Adam 1e-3、`batch_size=64`、标签中心化到 1.0
**规模**：sohrank k=1/2/5/10 各 8 seeds + 同 k IID 对照各 8 seeds（56 个 run），
FedAdam 19 个 run（8 × le=10 + 8 × le=3 + 3 个 η/τ 探针）

---

## 1. 结论摘要

M3 DOD ① 问的是「偏斜多大才开始掉点、掉多少」。做法：固定总数据量与训练预算，
只改**哪些电池分给谁**（sohrank）与**分片粗细**（k=1/2/5/10），每一档配一个
**同 k 的 IID 对照**。

| k | N | IID 偏斜 | sohrank 偏斜 | IID MAE | sohrank MAE | 配对差（相对 IID） |
|---|---|---|---|---|---|---|
| 10 | 12 | 4.7% | 39.2% | 0.00093 ± 0.00005 | 0.00140 ± 0.00005 | **+50.9%**（0/8 为负，p_t<0.0001） |
| 5 | 23 | 17.3% | 47.1% | 0.00120 ± 0.00019 | 0.00150 ± 0.00008 | **+24.2%**（8/8 同向，p_t=0.0013） |
| 2 | 58 | 32.5% | 57.2% | 0.00372 ± 0.00313 | 0.00350 ± 0.00289 | −6.0%（4/8 为负，p_t=0.21 → 未判定） |
| 1 | 115 | 70.4% | 70.4% | 0.00426 ± 0.00361 | 0.00421 ± 0.00378 | **−1.2%**（p_t=0.78 → 未判定，处理未施加） |

四条读数：

1. **偏斜确实掉点，幅度取决于「基线已经有多偏」。** k=10 时 IID 几乎不偏
   （4.7%），把偏斜拉到 39.2% 的代价是 **+50.9%**；k=5 时基线已有 17.3%，
   同样拉偏的边际代价降到 **+24.2%**。偏斜的「余量」越小，再拉偏越贵。
2. **k=1 是内部校准点，不是处理组。** 一颗电池一个客户端时，sohrank 与 IID
   的偏斜**逐位相同**（70.4%），两份划分的分组也**完全相同**（115/115 个
   客户端重叠）——「谁拿哪颗」只改客户端标签。实测配对差 **−1.2%**
   （p_t=0.78，逐 seed 差 −0.0005～+0.0011），判定机械地产出「未判定
   （处理未施加）」。
3. **k≤2 时训练量本身不够，偏斜效应在这一档不可分辨。** 每客户端 16–32 行
   < `batch_size=64`，每 epoch 只 1 步，seed 间 CV 爆炸到 83%–90%。对照
   显示 IID 与 sohrank 同样地坏（0.00372 vs 0.00350）——**k=2 的劣化是
   「客户端太多、每客户端数据太少」的 N 效应，不是偏斜**。
4. **只看 sohrank 一条线会读错因果。** sohrank 的 MAE 从 k=10 的 0.00140
   涨到 k=1 的 0.00421（3 倍），很容易读成「偏斜 70% 让精度掉 3 倍」。
   对照显示 IID 在偏斜只有 32.5% 时就已经是 0.00372——**同一个 MAE 水平
   在更低偏斜下就达到了**。掉点的主因是 N，偏斜是在 N 之上的额外一层。

---

## 2. 为什么横轴是偏斜，不是 k

M3 DOD ① 原文写「α 四档各 3 seeds」。**这个口径是错的**：α 控制的是**数量
偏斜**（客户端样本量的离散程度），而 M2 已证明 H2 的效应来自**标签偏斜**
（客户端之间 SOH 分布的差异）。M2 的实测把这件事钉死了：Dirichlet α=0.5
的样本量基尼是 0.519（看着很异质），标签偏斜却只有 12.2%，**比 IID 的
17.3% 还小**——把 21 颗电池合成一个客户端会把电池间的 SOH 差异平均掉。
扫 α 会得到一个「α 越小越不掉点」的假趋势，而它测的是聚合权重的集中度。

改写后横轴取 `between_client_variance_fraction`（客户端身份解释的标签方差
占比，`src/flsoh/heterogeneity.py`）。这是有物理含义的量：它度量「如果每个
客户端只按自己的数据拟合，能解释全局标签方差的多少」。本数据集上它的上界
是 70.4%——因为电池内 SOH std 只有 0.00066，而电池间均值 std 是 0.00486
（7 倍），SOH 在单颗电池上几乎是个常数。

**但换横轴只解决了一半问题。** sohrank 的 k 从 10 降到 1 时，偏斜从 39.2%
涨到 70.4%，**同时 N 从 12 涨到 115**——每客户端从 10 颗电池降到 1 颗。
曲线因此有两个解释：「偏斜导致掉点」和「客户端越多掉点越多」。分离它们的
唯一办法是每一档都配一个**同 k 的 IID**：同 k 保证 N 与每客户端数据量相同，
只改哪些电池分给谁。所以图 2 画两条线，**同一 k 处的竖直距离才是偏斜的
净效应**。

四份划分对（iid vs sohrank）已验证是**同一组电池、同一套客户端规模、零
重叠分组**：`Counter(所有电池)` 相同、规模序列相同、`frozenset` 分组交集
为 0/58、0/23、0/12。k=1 例外——那一档分组完全相同（见下节）。

---

## 3. k=1：一个不用设计就存在的零效应对照

k=1 时每颗电池自成一个客户端，**「哪些电池分给谁」这个问题不存在了**：
把 115 颗电池随机打散和按 SOH 排序，得到的是同一份划分（115/115 个
singleton 分组重叠），只有客户端名字到电池的映射不同。于是：

- 偏斜逐位相同（70.4% = 70.4%），因为客户端均值就是电池均值，怎么编号
  都不改变这组均值的分布；
- 训练完全确定：每客户端同 seed、同样 16 行、每 epoch 同样只有 1 个 batch
  （16 < 64），local 10 步。per-cell 的更新只取决于数据和当前全局参数；
- 聚合是等权平均（每客户端 16 行 → 权重相同），`weighted_average` 按客户端
  ID 排序后顺序求和——**同一组 115 个更新按不同顺序相加**。

所以 k=1 这一对的理论预期是：**吻合到浮点舍入量级**（100 轮训练是混沌的，
1e-7 的扰动的会被放大，但放不出一个系统性的差）。实测配对差 **−1.2%**
（−0.00005 ± 0.00050，p_t=0.78），逐 seed 差 −0.0005～+0.0011，无方向性
——与预期一致。

它是整张图的内部校准点：**如果 k=1 都测出显著差异，说明管线里存在未被
察觉的、依赖客户端身份或顺序的环节，那其他档的差异也不再可信。**判定器把
这种情况机械地判成「未判定（处理未施加：偏斜与 IID 同 N 划分完全相同）」，
不需要人记得去排除它——这正是 M2 那条「判 H2 前先量偏斜」的规则在等于号
上的形态。

---

## 4. 工程改动

### 4.1 分组键加 `n_clients`（commit 401aeb8）

`summarize_results.py` 的 `_key` 原来是 (strategy, scheme, alpha, mu,
local_epochs)。sohrank 四档共用这五个字段——**它们会塌成表里的一行**，
均值 0.0026 描述的是任何一个都不存在的实验，而表看起来照样正常。实测四档
MAE 是 0.0014 / 0.0015 / 0.0035 / 0.0042，差 3 倍。

配套改动：

- `_codes` 在碰撞时加 `N=` 后缀（P1@N=12 … P1@N=115）；
- 新增 `_sort_key`/`_sorted_groups`：字典序会把 `"I1@N=115"` 排到
  `"I1@N=12"` 前面（`"115" < "12"`），表里 115 会跑到 12 前面去；
- H2 对照组必须**同 N**：k 扫描里「客户端更多」本身就会掉点（每客户端数据
  更少、梯度步更少），拿不同 N 的 IID 当分母会把 N 的效应算到偏斜头上；
- `_skew_of` 按 (scheme, N) 取偏斜：四份 sohrank 划分的偏斜从 39.2% 到
  70.4%，不给 N 就会永远返回文件里第一份匹配项，manipulation check 随之
  失效而输出看起来完全正常。

### 4.2 manipulation check 的「偏斜完全相同」分支

原来的闸门只判「偏斜 ≤ IID 基线 → 处理未施加」。k=1 是**等于**而不是小于，
文案会读成「70.4% ≤ 70.4%」，技术上对但没有解释力。现在分成两支：小于
（Dirichlet 那种，合并把差异平均掉）与等于（k=1 那种，排序只改客户端标签），
后者注明这一档是图 2 的内部校准点。

### 4.3 CI 闸门补第二个失败文案

H2 闸门原来只 grep「缺 heterogeneity.json」。k 扫描新增四档 N 之后出现了
第二种失败：**文件在、但没有这一档 k 的记录**（重跑测量时漏传一份划分）。
文案是「找不到该 N 的划分」，不含旧关键字 → 闸门会静默放行。现在两个都拦。

这不是假想：本轮重跑 k 扫描时就把 `partition-mvp-k5-dirichlet0.5.json`
落下了，N1 的 manipulation check 当场失效（闸门行为正确——拒绝判定；
但记录不能缺，已补测）。

### 4.4 `make_figure2.py` 的 strategy 维度

图 2 的 `_key` 原来不含 strategy，dirichlet 划分上的 16 个 FedProx run
（两个 μ × 8 seeds）被并进「FedAvg / dirichlet」组——图不画那条线所以图
没事，但汇总把 n=8 报成 n=24，读者会以为那是 FedAvg 的结果。新增
`tests/test_figure2.py` 7 条钉住分组（纯函数，进默认 CI job）。

---

## 5. FedAdam：把动量挪到服务端（M3 ②）

M3 DOD ② 原本是 FedBN 跨化学体系，被两个前置项挡住：模型**故意**不含
BatchNorm（`model.py` 类注释写明「BN 会让 FedBN 成为必选项，MVP 先把问题
收窄到数据异质」），而本地没有第二个化学体系的 ΔQ(V) 特征。所以改成做
FedAdam——它没有前置阻塞，而且直接检验 M2 留下的最硬一条机理：

> FedAvg/FedProx 只聚合参数、不聚合优化器状态，客户端 Adam 动量每轮归零，
> 追平 C0 要 3.3 倍训练量（坑 12）。

### 5.1 结果

IID k=5（N=23）、100 轮、8 seeds，与 I1 完全同条件，只换服务端优化器：

| 配置 | 本地 epoch | 总梯度步量 | 最终 MAE | CV | 相对同预算 FedAvg |
|---|---|---|---|---|---|
| FedAvg（I1@le=10） | 10 | 1000 | 0.00120 ± 0.00019 | 15.9% | — |
| **FedAdam（A1@le=10）** | 10 | 1000 | 0.00123 ± 0.00051 | 41.3% | +2.0%（p_t=0.91 → 无差异） |
| FedAvg（I1@le=3） | 3 | 300 | 0.01165 ± 0.00777 | 66.7% | — |
| **FedAdam（A1@le=3）** | 3 | 300 | 0.00162 ± 0.00077 | 47.6% | **−86.1%**（8/8 同向，p_t=0.0056、p_sign=0.0078） |

三条读数，方向一致：

1. **本地训练量不够时，服务端动量能补上绝大部分**。I1 从 le=3 到 le=10
   把 MAE 压低 8.7 倍；A1 只把 le 从 3 加到 10 几乎没有变化
   （0.00162 → 0.00123，+32% 相对、p_t=0.24 不显著）。换句话说，
   **服务端动量把「训练量」这个瓶颈从客户端挪走了**——这正是 H3 调查
   定位到的机理（每轮聚合后客户端动量归零）被直接修补的证据。
2. **但它不能全补**。A1@le=3 相对 I1@le=10 仍差 +34.5%（p_t=0.14，
   样本量分辨不出这 30% 是真是噪）。所以「追平」未被判定，
   「大幅缩小差距」被判定。
3. **本地训练量已经够时，服务端动量没有额外收益**（+2.0%，
   p_t=0.91）。这与坑 12 的结论自洽：le=10 时客户端动量归零的代价
   已经被 10 个本地 epoch 摊薄，服务端那一步不再是瓶颈。

附带一条对报告有用的：**A1@le=3 与 C0 不可区分**（+2.9%，p_t=0.88）——
300 个总梯度步达到中心化 300 epoch 的水平，而 I1@le=10 要用 3.3 倍的
训练量才做到 −23.5%。CV 也从 66.7% 降到 47.6%。

### 5.2 η 与 τ 是量出来的，不是默认值捡的

η 探针（seed 0，其余同正式条件）：

| η（τ=1e-3） | 最终 MAE | 相对 FedAvg 同 seed（0.00131） |
|---|---|---|
| 0.1 | 0.00268 | +105% |
| **0.01** | **0.00101** | **−23%** |
| 0.001 | 0.00340 | +160% |

τ 探针（η=0.01）：τ=1e-3 → 0.00101、τ=1e-5 → 0.00128、τ=1e-6 → 0.00173。

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
**默认值恰好匹配本数据的伪梯度尺度是巧合，不是定律**——换数据集必须
重量 |Δ| 再定这两个数（坑 17）。

### 5.3 工程改动

- `flsoh/train.py::ServerAdam`：与 flwr 1.41 内置 `FedAdam` 逐字对应
  （读过安装版 `flwr/serverapp/strategy/fedadam.py` 与 `fedopt.py` 核对），
  末尾加有限性检查——τ=0 且伪梯度为 0 时 `0/(√0+0)` 出 NaN，而 NaN 不会
  让训练崩，只会静默污染之后每一轮。
- `run_federated.py`：`--strategy fedadam` + `--server-lr/--server-beta1/
  --server-beta2/--server-tau`；η 进结果文件名与 JSON（扫 η 时两个只差
  服务端学习率的 run 会撞名，撞名的表现是后一个静默覆盖前一个）。
- `server_app.py`：run_config 的 `strategy="fedadam"` 走内置 `FedAdam`，
  默认行为不变（pyproject 里没有这个键，CI 冒烟跑的仍是 FedAvg 路径）。
- `summarize_results.py`：服务端四元组 (η, β1, β2, τ) 整体进分组键
  （第七维），`A1` 编号，judge 增加 M3 ② 段（对照必须是同 le、同 N 的
  FedAvg）。
- 测试：`tests/test_train.py::TestServerAdam`（13 条，含闭式解与动量跨轮
  累积）、`tests/test_app.py::TestFedAdamEquivalence`（与 flwr 逐轮相等 +
  负面对照）、`tests/test_summarize.py::TestServerLrGrouping` /
  `TestFedAdamVerdict`（10 条）。

**一个当场被抓到的 bug，值得单独说**：分组键第一版只放了 `server_lr`。
跑完探针后重新汇总，A1@le=10 对 I1@le=10 的配对差是 +0.00006；而我拿
JSON 手算是 +0.00002。差额来自 τ=1e-5 的探针 run 与正式 run **只差 τ**，
两组塌成一行后 `_by_seed` 用后一个文件静默覆盖了 seed 0（0.00101 →
0.00128），而 seeds 列还写着 8。**漏一个维度的表现不是「少一行」，
是某一行的数被换掉**——比少一行更难发现，因为行数、seeds 列、均值
全都看着正常。修法是把服务端四元组整体放进键，并补了
`test_tau_probe_does_not_merge_with_formal_runs` 钉住。

## 6. 复现

```bash
PY=/d/项目/venvs/flsoh-py312/Scripts/python.exe
F=data/mit-stanford/features.parquet

# 划分产物（k=1/2/5/10 × iid/sohrank；sohrank 走 check_partition 的「模式 0」）
for k in 1 2 5 10; do
  python scripts/check_partition.py --config configs/mvp.toml \
      --cells-file data/mit-stanford/cells.txt \
      --emit results/partition-mvp-k$k-iid.json --emit-k $k --emit-scheme iid
  python scripts/check_partition.py --config configs/mvp.toml \
      --cells-file data/mit-stanford/cells.txt \
      --emit results/partition-mvp-k$k-sohrank.json --emit-k $k \
      --emit-scheme sohrank --cell-soh-json data/mit-stanford/cell_soh.json
done

# 标签偏斜（九份划分一起量，别漏）
$PY scripts/measure_heterogeneity.py --features $F \
    --out results/heterogeneity.json \
    $(for k in 1 2 5 10; do echo --partition-json results/partition-mvp-k$k-iid.json \
      --partition-json results/partition-mvp-k$k-sohrank.json; done) \
    --partition-json results/partition-mvp-k5-dirichlet0.5.json

# k 扫描与同 k IID 对照（各 8 seeds；串行跑，别并行——wall_clock_s 会失真）
for k in 1 2 5 10; do
  for s in 0 1 2 3 4 5 6 7; do
    $PY scripts/run_federated.py --config configs/mvp.toml --features $F \
        --partition-json results/partition-mvp-k$k-sohrank.json \
        --strategy fedavg --seed $s --tag "fedavg__sohrank_k${k}__seed$s"
  done
done
for k in 1 2 5 10; do
  for s in 0 1 2 3 4 5 6 7; do
    $PY scripts/run_federated.py --config configs/mvp.toml --features $F \
        --partition-json results/partition-mvp-k$k-iid.json \
        --strategy fedavg --seed $s --tag "fedavg__iid_k${k}__seed$s"
  done
done

# 汇总、判定、出图
$PY scripts/summarize_results.py --paired --judge --hetero results/heterogeneity.json
$PY scripts/summarize_results.py --write-plan docs/plan.md
$PY scripts/make_figure2.py
```

k=5 那一档的 sohrank run 复用 M2 的 `fedavg__sohrank__seed*`（同一份
`partition-mvp-k5-sohrank.json`），IID 对照复用 M2 的 `fedavg__iid__seed*`。

### 6.1 FedAdam（M3 ②）

```bash
PY=/d/项目/venvs/flsoh-py312/Scripts/python.exe
F=data/mit-stanford/features.parquet

# η / τ 探针（seed 0；正式实验只保留 η=0.01、τ=1e-3）
for eta in 0.1 0.01 0.001; do
  $PY scripts/run_federated.py --config configs/mvp.toml --features $F \
      --partition-json results/partition-mvp-k5-iid.json \
      --strategy fedadam --server-lr $eta --seed 0 \
      --tag "fedadam__iid__eta${eta}__seed0"
done
for tau in 0.00001 0.000001; do
  $PY scripts/run_federated.py --config configs/mvp.toml --features $F \
      --partition-json results/partition-mvp-k5-iid.json \
      --strategy fedadam --server-lr 0.01 --server-tau $tau --seed 0 \
      --tag "fedadam__iid__eta0.01__tau${tau}__seed0"
done

# 正式实验：8 seeds × (le=10 同预算对照 / le=3 训练量追问)
for s in 0 1 2 3 4 5 6 7; do
  $PY scripts/run_federated.py --config configs/mvp.toml --features $F \
      --partition-json results/partition-mvp-k5-iid.json \
      --strategy fedadam --server-lr 0.01 --seed $s \
      --tag "fedadam__iid__eta0.01__seed${s}"
done
for s in 0 1 2 3 4 5 6 7; do
  $PY scripts/run_federated.py --config configs/mvp.toml --features $F \
      --partition-json results/partition-mvp-k5-iid.json \
      --strategy fedadam --server-lr 0.01 --local-epochs 3 --seed $s \
      --tag "fedadam__iid__eta0.01__le3__seed${s}"
done
```

每轮伪梯度的逐坐标量级（用来定 η 与 τ，不是用来汇报的）：拿 FedAvg 的
循环跑 5 轮，每轮算 `weighted_average − global` 的 |Δ| 分布。实测
mean ≈ 7×10⁻⁴、中位 ≈ 6×10⁻⁴、max ≈ 1×10⁻²。
