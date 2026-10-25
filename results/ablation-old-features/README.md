# 旧特征表消融（1 线程）— 重跑前的对照基准

**这 56 个 run 用的是一份已知有缺陷的特征表，不能当作结果引用。**
它们存在的唯一目的是：当新特征表重跑完成后，能在一个**除特征口径外完全相同**
的设置下回答「特征修复到底改了多少数字」。

## 为什么需要这一批

旧口径的 172 个 run 归档在 `results/fullcycle-bug/`，但它们跑在**默认线程数
（24）**下。而线程数本身会改数值：

| 变量 | 值 | C0 seed0 的 `final_mae` |
|---|---|---|
| 线程数 | 24（默认） | `0.0015666428208351135` |
| 线程数 | 1 | `0.0015997225046157837` |

直接拿新数字减归档数字，会把「特征口径」和「线程数」两件事混在一起。所以
必须先把旧表在同一线程数下重跑一遍，作为对照。

## 怎么跑出来的

```bash
./.venv/Scripts/python.exe scripts/rerun_all.py \
  --features data/mit-stanford/features.fullcycle-bug.parquet \
  --out results/ablation-old-features --threads 1 --jobs 8 --execute \
  --log-dir data/mit-stanford/ablation-logs \
  --only '^(c0__|c1__|fedavg__iid__seed|fedavg-le3-iid-seed|fedavg__dirichlet0\.5__seed|fedavg__sohrank__seed|fedprox__dirichlet0\.5__seed|fedavg__iid__le3__nm8__seed)'
```

产物里带 `torch_num_threads=1` 与 `python_version=3.13.14`，可直接核对。

## 线程数的影响有多大（56 对配对）

| 统计量 | 值 |
|---|---|
| \|相对差\| 中位 | 1.86% |
| \|相对差\| 最大 | 27.37% |
| 方向 | 变好 28 / 变差 28（对称） |

方向对称说明它是**随机源**而非系统性偏移。量级与 seed 噪声（相对标准差
5%–63%）同阶，所以线程数实际上等于「又抽了一次 seed」。判据阈值是 5%，
而它能单独造成 27%——跨机器比较必然失效。

## 判定结论没有翻转

消融集跑完整条判定链，与归档（24 线程）逐条一致：H1 两个档位都推翻、
H2 sohrank 接受 / dirichlet 未判定（处理未施加）、H3 FedProx 推翻。
详见 `docs/plan.md` §13.5.6。
