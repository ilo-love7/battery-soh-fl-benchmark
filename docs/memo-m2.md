# M2 备忘：电池 SOH 联邦学习的三个假设，与两条被实测逼出来的方法论修正

**日期**：2026-10-24
**数据**：MIT-Stanford，135 颗电池（115 训练 / 20 测试），2700 行 × 1003 列特征
**设置**：k=5 → N=23 客户端，100 轮，`local_epochs=10`，Adam 1e-3，`batch_size=64`，标签中心化到 1.0
**规模**：5 个配置 × 8 seeds + 等训练量对照，共 64 个 run，全部 `results/*.json` 入库

---

## 1. 结论摘要

| 假设 | 判定 | 依据 |
|---|---|---|
| **H1** IID 划分下联邦与中心化的差距可以忽略 | **推翻** | le=10 时联邦比 C0 好 23.5%（p=0.0006）；le=3 等训练量时差 640.6%（p=0.0077）。两个口径都远超 10% 阈值 |
| **H2** Non-IID 下 FedAvg 会显著掉点 | **接受** | SOH 排序分片（偏斜 47.1%）相对同预算 IID（偏斜 17.3%）劣化 **+24.2%**，p_t=0.0013、p_sign=0.0078、8/8 方向一致 |
| **H3** FedProx 能挽回一部分 H2 的损失 | **推翻** | μ=0.01 差 +46.7%、μ=0.1 差 +85.8%，均 p<0.001、0/8 方向一致。两端夹住，不存在可用内点 |

**Dirichlet α=0.5 那一档记「未判定（处理未施加）」**，理由是第 2 节——这是本轮最该被记住的一条。

---

## 2. 方法论修正一：H2 长期判不了，是因为处理从没被施加

### 2.1 现象

在 8 seeds 之前，H2 是「未判定」。当时的解释是「seed 噪声 18% 大于阈值 5%，
分辨力不够」。补到 8 seeds 之后发现：**即使噪声解决了，Dirichlet α=0.5 这一档
仍然判不了**——因为它比 IID 还好 16.3%。

如果只看 MAE 表，最自然的结论是「电池级 Non-IID 划分没有想象中异质」，
即推翻 H2。这个结论是错的。

### 2.2 为什么错

`scripts/measure_heterogeneity.py` 量出「客户端身份解释的标签方差占比」：

| 划分 | 标签偏斜 | 样本量基尼 |
|---|---|---|
| IID k=5 | 17.3% | 0.000（每客户端 5 颗） |
| Dirichlet α=0.5 | **12.2%** | 0.519（最大 21 / 最小 1 颗） |
| SOH 排序分片 k=5 | **47.1%** | 0.000（每客户端 5 颗） |
| 上界（一颗电池一个客户端） | 70.4% | — |

**Dirichlet 的标签偏斜比 IID 还小。** 它只造成**数量偏斜**：N=23 时 α=0.5 让
一个客户端独占 21 颗电池，而把 21 颗电池合成一个客户端会把电池之间的 SOH 差异
**平均掉**——于是这个划分实际上比 IID 更均匀。

所以「Non-IID 没掉点」这句话在此之前是没有意义的：**处理没施加**。
「算法免疫」和「处理没施加」这两种解释在 MAE 表上长得一模一样，而结论相反。

### 2.3 数据层面为什么只有这么点偏斜可造

每颗电池的 SOH 内部标准差（中位）只有 **0.00066**，而电池之间均值的标准差是
**0.00486**——后者是前者的 7 倍。也就是说 SOH 在单颗电池上几乎是个常数，
模型主要要学的是「这颗电池的水平是多少」。而全局 std 是 0.0058，
`(0.00486/0.0058)² ≈ 70%` 就是划分能造出的标签偏斜上界。

按批次（b1/b2/b3）也造不出偏斜：三个批次的 SOH 均值是 1.0035 / 1.0020 / 1.0017，
彼此只差 0.002，除以全局 std 后只剩约 3% 的方差。

### 2.4 解法

`partition_by_soh_rank`：按电池平均 SOH 排序后切成连续分片。
**N、每客户端颗数、总数据量、训练预算全部不变，只改哪些电池分给谁。**
偏斜从 17.3% 拉到 47.1%，样本量完全均衡（基尼 0）。这是 FL 文献里的
pathological non-IID。

它用到了标签，所以**不是可部署的场景，是压力测试档**。它回答的问题是
「如果划分真的造出标签偏斜，FedAvg 会不会掉点」，不是「这种划分多常见」。
这一点写进了 `partition.py` 的 docstring、`configs/mvp.toml` 的注释和 README。

### 2.5 落成的闸门

`summarize_results.py --judge` 判 H2 之前必须查 `results/heterogeneity.json`：

- 偏斜不高于 IID 基线 → 一律记「未判定（处理未施加）」，**不许判「推翻」**
- 缺文件 → 同样不许判，不静默当成「异质性没问题」
- CI 检查 `heterogeneity.json` 是否覆盖了所有 run 实际用到的划分
  （负控：删一条记录 → 退出码 1）

否则等于拿一个没通电的电路去证明断路器好用。

---

## 3. 方法论修正二：H2 的对照组必须是同预算的 IID，不是 C0

原判据把 H2 写成「Non-IID 相对劣化 > 10%」，隐含分母是 C0。这是错的：
**H2 问的是「划分方式造成多少损失」，联邦与中心化的差距是 H1 的问题。**
拿 C0 当分母会把两个效应混在一个数里，而它们给出的画面相反：

| 对比 | Dirichlet 的表现 |
|---|---|
| 相对 C0 | **−35.9%**（好很多） |
| 相对同预算 IID | **−16.3%**（还是好，但幅度差一半） |

现在 `--judge` 把 H2 的分母固定为同 `local_epochs` 的 IID FedAvg 组；
找不到就报「缺同预算的 IID 对照组」，**不默默退回 C0**。

---

## 4. 训练量：一个能翻转结论的混杂因素

| 口径 | 联邦本地 epoch | MAE | 相对 C0 |
|---|---|---|---|
| 等训练量 le=3 | 300（= C0） | 0.01165 | **+640.6%** |
| 默认 le=10 | 1000（C0 的 3.3 倍） | 0.00120 | **−23.5%** |

同一个策略、同一份划分、同一批 seed，只把训练量从 1000 ep 降到 300 ep，
结论就从「好 23.5%」变成「差 6.4 倍」。**任何不声明训练量的相对差都没有意义。**

配套：`check_protocol.py` 的 `C-TRAIN` 警告现在判**训练量比值**
`num_rounds × local_epochs / baseline_epochs` 是否落在 [0.8, 1.25]，
不再判「`local_epochs` 大不大」。理由是同一条 le=10，在每客户端 80 行下是
**必要的**，在每客户端上万样本下才是漂移风险——只有比值有意义。

---

## 5. 关于 seed 数

8 seeds 之后，配对 t 检验让三条比较变得可判定（p<0.01），但其余仍然不行。
符号检验在 n=8 时的最小双侧 p 是 0.0078——**「8/8 方向一致」只够到 0.01 量级**，
想判更小的效应就必须加 seed。

seed 波动实测（相对标准差）：

| 配置 | 波动 |
|---|---|
| C0 中心化 | 13.3% |
| C1 本地独立 | 27.7% |
| FedAvg IID（le=10） | 15.9% |
| FedAvg IID（le=3） | **66.7%** |
| FedAvg Dirichlet | 17.2% |
| FedAvg SOH 分片 | 5.2% |
| FedProx μ=0.01 / 0.1 | 8.1% / 5.8% |

**66.7% 那一格值得单独说**：训练量降到等训练量后噪声从 16% 涨到 67%。
不是「联邦更不稳定」，而是欠拟合状态下不同 seed 落在损失面的不同平底上，
模型还没被训练量约束住。训练量不足时连「跑八次取均值」都不可靠。

按 CV=16% 分辨 5%，量级上需要 ~50 个 seed。本轮没有做到，M3 若要报更精细的
结论需要专门排期。

---

## 6. FedProx 为什么在这个规模下不可用

μ=0（即 FedAvg）0.00101 → μ=0.01 得 0.00148 → μ=0.1 得 0.00187。
**单调变差，两端都差于 μ=0。** 而 μ→0 的极限就是 FedAvg
（`tests/test_train.py::TestFedProx::test_mu_zero_is_bitwise_fedavg` 钉住），
所以改善只可能出现在 (0, 0.01)，两端都指向更差 → **不存在可用内点**。

机理：每客户端仅 80 行训练数据（5 颗电池 × 16 循环），近端项
`(μ/2)‖w − w_global‖²` 相对数据项过强，把参数拴在全局点上、限制本地拟合。
**§14 坑 2 给的「μ 取 0.01–0.1」在这套数据规模下是错的**——那条建议来自
每客户端上千样本的典型 FL 设定。

顺带一个反直觉观察：**FedProx 的 seed 波动（5.8%–8.1%）比 FedAvg（15.9%）
小，但它更差。** 稳定性不等于正确性——μ 的强约束让所有 seed 收敛到同一个
（更差的）点。按 §11.3 的「seed 波动 <10%」这一行，FedProx 反而是唯一两个
达标的配置之一。只看稳定性会选错。

---

## 7. 对 M3 / M4 的影响

1. **M3 要报 H2 必须带上偏斜数字**。任何新的 Non-IID 划分都要先跑
   `measure_heterogeneity.py`，CI 已经把它变成硬闸门。
2. **α 扫描（0.1 / 0.5 / 1.0）价值有限**。它扫的是数量偏斜，而 H2 的效应
   来自标签偏斜。要测「偏斜多大才开始掉点」，应该扫 sohrank 的**分片粗细**
   （k=1/2/5/10），而不是 Dirichlet 的 α。
3. **M4（DP）的判据要按本轮修正后的写法重写**：声明训练量口径、声明对照组
   是谁、先做显著性再分档。原判据的「<5%」在 16% 噪声下永远无法判定。
4. **C1 的 0.034 不要读成「本地训练很差」**。它是 23 个模型各测各的再取平均，
   定义就是不聚合。它证明联邦有用（0.034 → 0.0012），但不能和 C0 比训练能力。

---

## 8. 复现

```bash
# 划分产物
python scripts/check_partition.py --config configs/mvp.toml \
    --cells-file data/mit-stanford/cells.txt \
    --emit results/partition-mvp-k5-iid.json --emit-k 5 --emit-scheme iid
python scripts/check_partition.py --config configs/mvp.toml \
    --cells-file data/mit-stanford/cells.txt \
    --emit results/partition-mvp-k5-dirichlet0.5.json --emit-k 5 \
    --emit-scheme dirichlet --emit-alpha 0.5
python scripts/check_partition.py --config configs/mvp.toml \
    --cells-file data/mit-stanford/cells.txt \
    --emit results/partition-mvp-k5-sohrank.json --emit-k 5 \
    --emit-scheme sohrank --cell-soh-json data/mit-stanford/cell_soh.json

# 标签偏斜
python scripts/measure_heterogeneity.py \
    --features data/mit-stanford/features.parquet \
    --partition-json results/partition-mvp-k5-iid.json \
    --partition-json results/partition-mvp-k5-dirichlet0.5.json \
    --partition-json results/partition-mvp-k5-sohrank.json \
    --out results/heterogeneity.json

# 汇总与判定
python scripts/summarize_results.py --paired --judge \
    --hetero results/heterogeneity.json
python scripts/summarize_results.py --write-plan docs/plan.md
python scripts/make_figure1.py
```

`results/heterogeneity.json` 与所有 `results/*.json` 均已入库；表 1 由脚本生成，
CI 从原始 run 逐字重算核对（改一个数字即退出码 3）。
