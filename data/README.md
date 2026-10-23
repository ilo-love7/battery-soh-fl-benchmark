# `data/` 放置说明

本目录**不进版本库**（见 `.gitignore` 的 `data/` 规则），只保留这份说明。
但 `cells.txt` / `manifest.json` / `features.parquet` 是**生成物**，删掉可以重跑脚本再生成。

## 数据集

| 数据集 | canonical 来源 | 许可要点 |
|---|---|---|
| MIT-Stanford（Severson et al. 2019） | https://data.matr.io/1/ | 学术用途免费，商用需联系 |
| NASA PCoE | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | 美国政府作品，公有领域 |
| CALCE UMD | http://www.calce.umd.edu/battery-data.html | 学术用途，引用出处 |

本项目目前只用 MIT-Stanford 的前三个批次（`2017-05-12` / `2017-06-30` / `2018-04-12`）。
第四批（`2019-01-24`）来自 Attia 的闭环快充论文，不属于 Severson 的电池集，没有纳入。

## 为什么不做「整篇下载」

三个 `.mat` 合计约 **7.7 GiB**，而本项目每颗电池真正需要的只有：

- `summary` 里的 8 条逐循环曲线（约 76 KB/颗）
- 循环 1 与 5..100 的 `V` / `Qc`（约 170 KB/颗）

135 颗合计约 **33 MB**，占数据集体积的 0.4%。为 0.4% 的字节付 7.7 GiB 的带宽没有道理。
MATLAB v7.3 就是 HDF5，HDF5 允许按字节区间随机读，所以 `src/flsoh/matr.py` 实现了
一个「HTTP Range 支撑的可 seek 文件对象」，交给 h5py 的 `fileobj` 驱动。

实测：单连接约 17 KB/s，6 个并发约 131 KB/s——瓶颈在**每连接限速**，不在总带宽。
所以代码里到处是「并行补洞」和「减少请求数」：远端随机读的成本几乎全在往返上（每次约 1.4 s），
不在字节上。

## 实测的颗数（与论文口径不同）

| 批次 | 文件 | 原始颗数 |
|---|---|---|
| 1 | `2017-05-12_...mat` | 46 |
| 2 | `2017-06-30_...mat` | 48 |
| 3 | `2018-04-12_...mat` | 46 |
| | **合计** | **140** |

其中批次 2 有 **5 颗是批次 1 的延续**（`b2c7/8/9/15/16` ↔ `b1c0..b1c4`）：同一颗电池，
第一个文件跑完一段、第二个文件接着跑。去重后 **135 颗**。

这个对应关系有四条独立证据，`scripts/check_continuations.py` 逐条打印：

| # | 证据 | 实测 |
|---|---|---|
| 1 | 延续段的 `policy_readable` 是原段 policy 的后缀 | `80%)-3.6C` ← `3.6C(80%)-3.6C`；`80%)-4C` ← `4C(80%)-4C` |
| 2 | 延续段循环数 = 论文 `add_len` + 1 | 662/981/1060/208/482 ← 661/980/1059/207/481 |
| 3 | 同一条后缀在批次 1 圈出的电池正好是这一组的原段 | 3.6C 组 3↔3，4C 组 2↔2 |
| 4 | 交界处容量落差 < 延续段自身衰减的 1/4 | 落差 0.001–0.015，衰减 0.09–0.14 |

> **两个字段不能用，写在这里免得再踩**：`barcode` 在全部 140 颗电池上都是同一个常数
> `3707764736`（占位值）；`channel_id` 的第 5 位是批次内行号（b1c0 是 47、b2c7 是 56），
> 跨批次必然不同。最初拿 `channel_id` 当身份做验证，得到「5 对全 FAIL」——
> 那不是延续关系不成立，是字段选错了。**验证失败时先怀疑判据，再怀疑结论。**

论文里的 **124** 是另一件事：它为「预测 cycle life」剔掉了没跑到 80% 容量的电池。
本项目的标签是逐循环 SOH，那些电池在循环 5–100 区间内是正常数据，所以不跟着剔。

## 期望的目录形状

```text
data/
├── README.md              # 本文件（进版本库）
└── mit-stanford/
    ├── cells.txt          # 135 行，一行一个 cell_id（生成物）
    ├── manifest.json      # 批次、颗数、来源、去重记录（生成物）
    ├── features.parquet   # per-cycle 特征表（生成物）
    ├── features.spec.json # 网格、循环区间、丢弃统计（生成物）
    └── raw/               # 可选：本地已下载的 .mat
```

## 怎么生成

```bash
# 1) 清单：只读元数据，约 1 分钟
python scripts/build_manifest.py --out data/mit-stanford

# 2) 跨批次延续的四条证据（读批次 1 全部 46 颗 + 5 颗延续段）
python scripts/check_continuations.py

# 3) 特征表：读曲线，耗时取决于网络
python scripts/build_features.py \
    --manifest data/mit-stanford/manifest.json \
    --out data/mit-stanford/features.parquet
```

第 3 步**不可能一次跑完**：135 颗 × 21 个循环 × 2 条曲线，每次远端往返 1.4 s 到
几十秒。所以它按颗落分片（`features.parquet.parts/`），中断后**重跑同一条命令**
即可，已有分片自动跳过。

「中断」是关键前提：**跑完的那一轮会清掉分片目录**，所以重复执行一条已经跑完的
命令是全新一轮，不是续跑。想留着分片（比如要接着换参数做对照）就加 `--keep-parts`。

```bash
# 中断后接着跑（同一条命令）
python scripts/build_features.py \
    --manifest data/mit-stanford/manifest.json \
    --out data/mit-stanford/features.parquet

# 换了网格 / 循环区间：拒绝续跑，确认要重来才加 --restart
python scripts/build_features.py --manifest ... --out ... --points 500 --restart
```

拒绝续跑不是麻烦，是必须：网格从 1000 点改成 500 点，新旧分片的第 i 列就不是
同一个电压，拼出来的表**不报任何异常**。规格指纹（循环区间、参考循环、电压
网格）写在分片目录的 `state.json` 里，任何一项变了都要显式 `--restart`。

`cells.txt` 是**生成物，不是手写清单**。手写清单一旦和数据脱节，
`check_partition.py` 就会对着一份虚构的 ID 清单说 PASS——`P-UNKNOWN` 防的是
「ID 不在数据集里」，它防不住「清单本身就是假的」。

## 为什么 `cells.txt` 是独立的

`scripts/check_partition.py --cells-file data/mit-stanford/cells.txt` 要用它。

占位 ID（`cell-000`…）只能验证「划分逻辑没有泄漏」，不能验证「真实的那 135 颗没有泄漏」。
真实电池 ID 的拼写、数量、是否与数据一致，只能靠清单对账——这就是 `P-UNKNOWN`
检查码存在的理由。

## 不要把数据提交上来

`data/` 在 `.gitignore` 里。如果发现 `git status` 出现了 `.mat` 或 `features.parquet`，
先确认许可条款允许再分发，再手动 `git add -f`。生成物（`cells.txt` / `manifest.json`）
不含原始曲线，可以提交，但本项目默认不提交——它们依赖远端数据的当前状态。
