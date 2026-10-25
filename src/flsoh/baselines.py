#!/usr/bin/env python3
"""平凡基线阶梯：用最简单的模型量「这份特征表里到底有多少信号」。

## 为什么需要它

端到端只报一个 ``final_mae`` 等于没报。本仓库已经吃过一次亏：模拟数据上
``final_mae = 0.067`` 看着像个正常数字，而「全预测成训练集均值」的 MAE 是
``0.027``——模型比一条水平线还差 2.5 倍。那次之后 ``run_e2e.py`` 加上了常数
基线。

但常数基线是**最弱**的平凡基线，过了它不代表模型有用。真正的问题是：
**1000 维神经网络，比「只拿一个标量（ΔQ 的幅度）」好多少？**

模拟数据上的实测（24 颗 × 20 循环，留出 6 颗整颗测试，
``results/e2e/e2e__sim__seed0.json``）::

    常数基线                     MAE 0.02638
    只用幅度（1 维岭回归）       MAE 0.00350   ← 最强平凡基线
    全部 1000 维（岭回归）       MAE 0.00605
    只用形状（归一化后 1000 维） MAE 0.07209   ← 比水平线还差
    神经网络（run_e2e 默认设置） MAE 0.02289

读法：**信号几乎全在幅度里**。1000 维模型打不过 1 维线性模型，而且归一化后
（幅度信息被除掉）连水平线都不如——说明另外 999 维在模拟数据上主要是噪声。

这不是「神经网络不行」，是**模拟数据的任务太单薄**：容量衰减在
:mod:`flsoh.simulate` 里只按 ``Q(c) = Q0·(1 − fade·√(c/C))`` 缩放整条曲线，
形状随老化的变化（γ 随容量损失下降 15%）只有约 1.8% 的量级。所以这条阶梯
的作用是**把 e2e 的指标放进可解释的位置**：它证明的是管道接对了，不是方法
有效；而当 ``final_mae`` 明显大于 ``amplitude`` 那一格时，该查的是模型与训练
设置，不是继续调数据。

**真实 MATR 表上的结论是反的**，而且反得比「缺形状机制」更彻底：那里
``amplitude`` 比常数基线还差（5 个划分种子下 1.09–1.32 倍），1000 维全量好
2.4–3.4 倍。机制是幅度的跨电池离散度（33.8%）远大于标签的（0.5%）——幅度
主要编码「这是哪颗电池」。所以别把这里的「1 维就够」当成数据的性质。

## 为什么是岭回归而不是别的

闭式解、无超参搜索、逐位可复现、不依赖 torch——CI 的纯标准库 job 之外也能跑。
标准化只用**训练集**的均值与标准差（``alpha`` 加在标准化后的正规方程上），
所以测试集信息不会漏进来。
"""

from __future__ import annotations

import numpy as np

#: 岭回归的正则强度。标准化之后各列量纲一致，1.0 是「温和收缩」的量级：
#: 它不会把真实信号压掉，但能让共线的 1000 列不至于把解炸掉。
DEFAULT_ALPHA = 1.0


def amplitude(features: np.ndarray) -> np.ndarray:
    """每行的幅度 ``max_j |ΔQ_j|``，返回 ``(n, 1)``。

    「幅度」是 ΔQ(V) 曲线最朴素的一个标量摘要。它不含任何形状信息，所以
    拿它当基线，等于问：**除了「这条曲线整体多大」，特征还提供了什么？**

    **它不是一个「万能的」标量**：真实 MATR 数据上，``max|ΔQ|`` 的判别力比
    常数基线还差（1.17 倍）。:func:`scalar_summaries` 把常见的几种摘要都算
    出来——如果**每一种**都不如常数基线，那结论是「SOH 不在任何标量幅度里」，
    而不是「我选的这个标量不好」。
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"features 必须是二维，收到 shape={X.shape}")
    return np.abs(X).max(axis=1, keepdims=True)


def scalar_summaries(features: np.ndarray) -> dict[str, np.ndarray]:
    """一组「一维摘要」候选，每个返回 ``(n, 1)``。

    目的是把「幅度有没有用」这个问题与「我选的幅度统计量对不对」分开。
    只用 ``max|ΔQ|`` 一个数就下结论，等于把两件事混在一起；真实数据上
    它们全都比常数基线差（1.14–1.31 倍），所以那个结论是稳的。
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"features 必须是二维，收到 shape={X.shape}")
    absX = np.abs(X)
    return {
        "max|ΔQ|": absX.max(axis=1, keepdims=True),
        "median|ΔQ|": np.median(absX, axis=1, keepdims=True),
        "mean|ΔQ|": absX.mean(axis=1, keepdims=True),
        "L2 范数": np.sqrt((X**2).sum(axis=1, keepdims=True)),
        "峰峰值": (X.max(axis=1) - X.min(axis=1)).reshape(-1, 1),
        "中点电压处的值": X[:, X.shape[1] // 2].reshape(-1, 1),
    }


def ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """标准化 + 岭回归的闭式解。

    标准化统计量**只从训练集算**。用全体数据算会让测试集的信息漏进来，而
    漏进来的方式是「MAE 好看一点」，从外面完全看不出来——这正是本项目要防的
    那类错误。
    """
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("X_train / X_test 必须是二维")
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(
            f"训练与测试的特征维数不同：{Xtr.shape[1]} vs {Xte.shape[1]}"
        )
    if len(Xtr) != len(ytr):
        raise ValueError(f"X_train 有 {len(Xtr)} 行而 y_train 有 {len(ytr)} 个")

    mean = Xtr.mean(axis=0)
    scale = Xtr.std(axis=0)
    # 常数列（含整列相同）除以 0 会得到 inf/nan，进而让整个解变成 nan。
    # 模拟数据里整列相同是常态：网格两端的点在裁剪后没有变化。
    scale = np.where(scale > 0, scale, 1.0)

    A = np.c_[(Xtr - mean) / scale, np.ones(len(Xtr))]
    B = np.c_[(Xte - mean) / scale, np.ones(len(Xte))]
    # 截距项不惩罚：它吸收标签的均值，惩罚它等于把预测整体拉向 0。
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    return B @ w


def _mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)).mean())


def feature_ladder(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """把「信号在哪」量成一张阶梯表。

    四格：

    - ``constant``：全预测成训练集均值（最弱基线，必须过）
    - ``amplitude``：只用 ``max_j |ΔQ_j|`` 一个标量（**最强平凡基线**）
    - ``all_features``：全部特征维
    - ``shape_only``：每行除以自身幅度后再回归，幅度信息被除掉

    ``amplitude`` 是这张表的锚点：``all_features`` 明显不如它，说明多余维度
    在贡献噪声；``shape_only`` 明显不如 ``constant``，说明形状里没有标签信息
    （它拟合的是噪声，所以比一条水平线还差）。
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).ravel()
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if X.ndim != 2:
        raise ValueError(f"features 必须是二维，收到 shape={X.shape}")
    if len(X) != len(y):
        raise ValueError(f"features 有 {len(X)} 行而 labels 有 {len(y)} 个")
    if len(train_mask) != len(X) or len(test_mask) != len(X):
        raise ValueError("掩码长度必须与特征行数一致")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("训练集或测试集为空，阶梯表没有意义")
    if (train_mask & test_mask).any():
        # 同一行既训练又测试，MAE 会好得毫无道理。宁可报错也不给数。
        raise ValueError("训练掩码与测试掩码有重叠：同一行不能既训练又测试")

    y_train, y_test = y[train_mask], y[test_mask]
    X_train, X_test = X[train_mask], X[test_mask]

    constant = _mae(np.full(len(y_test), y_train.mean()), y_test)

    amp = amplitude(X)
    amp_mae = _mae(ridge_predict(amp[train_mask], y_train, amp[test_mask], alpha=alpha),
                   y_test)

    all_mae = _mae(ridge_predict(X_train, y_train, X_test, alpha=alpha), y_test)

    row_amp = np.abs(X).max(axis=1, keepdims=True)
    safe = np.where(row_amp > 0, row_amp, 1.0)
    shape = X / safe
    shape_mae = _mae(
        ridge_predict(shape[train_mask], y_train, shape[test_mask], alpha=alpha), y_test
    )

    return {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": int(X.shape[1]),
        "label_std_test": float(y_test.std()),
        "constant": constant,
        "amplitude": amp_mae,
        "all_features": all_mae,
        "shape_only": shape_mae,
        "alpha": float(alpha),
    }


def label_spread(labels: np.ndarray, groups: np.ndarray) -> dict:
    """标签的**组间 / 组内**离散度。``groups`` 一般是 ``cell_id``。

    为什么阶梯表必须配这一格：``all_features`` 比常数基线好 3 倍，这句话
    本身不区分两种完全不同的能力——

    - 估出**这颗电池**衰减得快不快（组间：同一颗的标签水平与全局均值之差）
    - 跟住**这颗电池随循环**的变化（组内：同一颗内部随循环的移动）

    真实 MATR 表（135 颗 × 循环 5..100 步长 5）上两者差一个数量级：组间 std
    0.00486，组内 std 中位只有 0.00066。``var_fraction_between`` 实测 **70.4%**
    ——**七成标签方差在组间**，所以「好 3 倍」基本只说明第一件事。不拆开就写
    「模型能跟踪老化」，是把没测过的事说成测过了。

    ``var_fraction_between`` 用 ANOVA 的 ``SS_between / SS_total``（不等长分组
    也成立）。注意它**恰好等于** ``between_std² / 全局方差``——因为
    ``between_std`` 定义为 ``sqrt(SS_between / n)``。容易混的是「组均值的
    **无加权** std」：那个量在不等长分组下会偏（小组被放大），所以这里不用它。

    **组内 std 有两个口径，不能混用**（2026-10-26 实测，差 4.8 倍）：

    - ``within_std_median``：每颗电池各算一个 std，再取中位。回答「**典型**的
      一颗电池，标签在它内部动多少」。
    - ``within_std_pooled``：所有电池去均值后合并算一个 std（``sqrt(mean(r²))``，
      ``r`` 为去组均值后的残差）。回答「**整批行**平均动多少」。

    两个数在真实 MATR 表上是 0.00066 vs 0.00316（**4.8 倍**）：少数早期就掉得
    厉害的电池把池化口径拉高了，而中位口径里它们只占一票。所以
    :func:`error_decomposition` 的 ``within_rmse``（池化）**只能**与
    ``within_std_pooled`` 比；拿它跟 ``within_std_median`` 比会凭空得出
    「模型没跟住组内变化」——2026-10-25 就这么误判过一次，见 plan §13.5.4。
    """
    y = np.asarray(labels, dtype=float).ravel()
    g = np.asarray(groups)
    if len(y) != len(g):
        raise ValueError(f"labels 有 {len(y)} 个而 groups 有 {len(g)} 个")
    if len(y) == 0:
        raise ValueError("labels 为空")
    uniq = sorted(set(g.tolist()))
    if len(uniq) < 2:
        raise ValueError(f"至少需要两个分组才能谈组间/组内，收到 {len(uniq)} 个")

    grand = y.mean()
    ss_between = 0.0
    ss_total = float(((y - grand) ** 2).sum())
    ss_within = 0.0
    within_stds = []
    for key in uniq:
        mask = g == key
        block = y[mask]
        ss_between += float(len(block) * (block.mean() - grand) ** 2)
        if len(block) > 1:
            # ddof=0：这是描述这批数据的离散度，不是估计总体参数
            within_stds.append(float(block.std()))
            ss_within += float(((block - block.mean()) ** 2).sum())
    if not within_stds:
        raise ValueError("每个分组都只有一行，组内离散度无定义")

    return {
        "n_groups": len(uniq),
        "n_rows": int(len(y)),
        "between_std": float(np.sqrt(ss_between / len(y))),
        "within_std_median": float(np.median(within_stds)),
        "within_std_pooled": float(np.sqrt(ss_within / len(y))),
        "var_fraction_between": float(ss_between / ss_total) if ss_total > 0 else 0.0,
    }


def error_decomposition(pred: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    """把预测误差拆成组间与组内两部分。

    ``between_rmse``：每颗电池的**均值**估得准不准（同一颗内部的偏差先平均掉）
    ``within_rmse``：同一颗内部**随循环的移动**估得准不准（每颗的偏差先减掉均值）

    **``within_rmse`` 是池化的，它的下界是 ``within_std_pooled``，不是
    ``within_std_median``。** 这两个口径在真实 MATR 表上差 **4.8 倍**
    （0.00316 vs 0.00066，见 :func:`label_spread`），所以本函数把同口径的
    基准一起算出来，放在 ``within_std`` / ``within_ratio`` 里——调用方不需要
    再去挑一个「看起来差不多」的标签统计量来当分母。

    为什么这不是吹毛求疵：2026-10-25 我用 ``within_rmse``（0.00093）去比
    ``within_std_median``（0.00076），得出「新口径丢掉了组内跟踪能力」，
    还据此在 plan §13.5.4 开了一条阻塞项。换成同口径后那个比值是
    **0.00093 / 0.00316 = 0.29**，模型跟得相当好。**两个都「正常」的数相除
    也能得出反的结论**，而这一次错误活过了一整轮复核。

    ``within_ratio < 1`` 表示比「每颗各自猜一个常数」更会跟循环变化；
    ``= 1`` 说明一点没跟住；``> 1`` 说明预测在组内比常数还差（在放大噪声）。
    """
    p = np.asarray(pred, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    g = np.asarray(groups)
    if not (len(p) == len(y) == len(g)):
        raise ValueError(
            f"长度必须一致，收到 pred={len(p)} labels={len(y)} groups={len(g)}"
        )
    if len(y) == 0:
        raise ValueError("输入为空")
    uniq = sorted(set(g.tolist()))
    if len(uniq) < 2:
        raise ValueError(f"至少需要两个分组，收到 {len(uniq)} 个")

    between_sq, within_sq, n_within = [], [], 0
    ss_within_labels = 0.0
    for key in uniq:
        mask = g == key
        between_sq.append((y[mask].mean() - p[mask].mean()) ** 2)
        residual = (y[mask] - p[mask]) - (y[mask] - p[mask]).mean()
        within_sq.append(float((residual**2).sum()))
        # 同口径的标签基准：标签自己去掉组均值后的平方和
        ss_within_labels += float(((y[mask] - y[mask].mean()) ** 2).sum())
        n_within += int(mask.sum())

    within_rmse = float(np.sqrt(sum(within_sq) / n_within))
    within_std = float(np.sqrt(ss_within_labels / n_within))
    return {
        "mae": _mae(p, y),
        "between_rmse": float(np.sqrt(np.mean(between_sq))),
        # 用全部行数做分母（每颗内部的残差均值本来就为 0，不影响结果）
        "within_rmse": within_rmse,
        # 这两个是「同口径」的配套量，别换成 label_spread 的 within_std_median
        "within_std": within_std,
        "within_ratio": within_rmse / within_std if within_std > 0 else float("nan"),
    }


def ladder_verdict(ladder: dict) -> str:
    """把阶梯表读成一句话。**四种情形必须分开**，不能合并成「有没有信号」。

    上一版把 ``shape_only > constant`` 直接写成「形状里没有标签信息（幅度是
    唯一载体）」——在真实数据上这是**错的**：那里 ``amplitude`` 也比常数基线
    差（五个划分种子下 1.09–1.32 倍），所以幅度不是载体；信息在幅度与形状的
    **组合**里（全量 0.30–0.42 倍）。两件事混成一句，就把一个反直觉的发现读成
    了「符合预期」。
    """
    amp_ok = ladder["amplitude"] < ladder["constant"]
    shape_ok = ladder["shape_only"] < ladder["constant"]
    multi_ok = ladder["all_features"] < ladder["amplitude"]

    if amp_ok and shape_ok:
        head = "幅度与形状各自单独就带信息"
    elif amp_ok and not shape_ok:
        head = "信息在幅度里，形状是噪声"
    elif shape_ok and not amp_ok:
        head = "信息在形状里，标量幅度是噪声"
    elif multi_ok:
        head = "任何一个一维摘要都不够，信息在幅度与形状的组合里"
    else:
        head = "特征里没有可用信号（全部特征也不如只用幅度）"

    if multi_ok:
        gain = ladder["amplitude"] / ladder["all_features"]
        return f"{head}；全部特征比只用幅度好 {gain:.2f} 倍"
    return head


def ladder_note(ladder: dict, neural_mae: float | None = None) -> str:
    """把阶梯表翻译成一句话：信号在哪，以及模型有没有超过最强平凡基线。"""
    amp = ladder["amplitude"]
    if neural_mae is None:
        return (
            f"最强平凡基线（只用幅度）MAE {amp:.5f}，"
            f"全部特征 {ladder['all_features']:.5f}"
        )
    ratio = neural_mae / amp if amp else float("inf")
    if ratio <= 1.0:
        return (
            f"模型 MAE {neural_mae:.5f} ≤ 最强平凡基线 {amp:.5f}"
            f"（×{ratio:.2f}）：1000 维模型确实用上了幅度之外的信号"
        )
    return (
        f"模型 MAE {neural_mae:.5f} 比最强平凡基线 {amp:.5f} 差 "
        f"×{ratio:.2f}：**1 维线性模型就够了**，多余维度在贡献噪声。"
        "这不是「方法有效」的证据，是模拟数据的任务太单薄"
    )


#: 主成分阶梯要看的维数。到 50 为止：再往上每一步只降 0.01，而「有效自由度是
#: 几」这个问题在前 10 维就答完了。
DEFAULT_PC_KS = (1, 2, 3, 5, 10, 20, 50)


def pc_ladder(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    ks: tuple[int, ...] = DEFAULT_PC_KS,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """主成分阶梯：**「1000 维」到底有几个有效自由度**。

    阶梯表报的是「全部特征比常数基线好 3 倍」，但没说清这 3 倍是从几个方向
    来的。1000 列里可能只有 3 列在起作用，其余 997 列在贡献噪声——两者的工程
    含义完全不同（前者说明该降维，后者说明该换特征）。

    实测（真实 MATR 表，135 颗，留出 23 颗）：

    - **第 1 主成分占 91% 的方差，判别力 1.22 倍**（比水平线还差）。它与
      ``max|ΔQ|`` 的相关是 0.66——就是「这是哪颗电池」那个方向。
    - **第 2–3 主成分（合计 7.5% 的方差）把 MAE 从 1.15 倍拉到 0.43 倍。**
      信号藏在高方差方向的正交补里。
    - 再加到 50 维只从 0.43 降到 0.34，收益递减。

    所以「1000 维」是个假象：有效自由度是 3，而且真正有用的那 3 维只占总方差
    的 7.5%。这也解释了为什么 seed 之间的波动会大——模型要在 91% 的干扰项下面
    找出 7.5% 的信号。

    **主成分只在训练集上拟合**，测试集用同一组载荷投影。否则 PCA 本身就看过
    测试集，而这一步的「判别力」会好得没有道理。

    ``first_useful_k`` 是第一个比常数基线好的维数；``saturated_k`` 是第一个
    达到最优判别力 90% 增益的维数（即再往后加维度不值得）。两个都给，因为
    「从几维开始有用」与「加到几维就够」不是同一个数。
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).ravel()
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if X.ndim != 2:
        raise ValueError(f"features 必须是二维，收到 shape={X.shape}")
    if len(X) != len(y):
        raise ValueError(f"features 有 {len(X)} 行而 labels 有 {len(y)} 个")
    if len(train_mask) != len(X) or len(test_mask) != len(X):
        raise ValueError("掩码长度必须与特征行数一致")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("训练集或测试集为空，主成分阶梯没有意义")
    if (train_mask & test_mask).any():
        raise ValueError("训练掩码与测试掩码有重叠：同一行不能既训练又测试")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    constant = _mae(np.full(len(y_test), y_train.mean()), y_test)

    mean = X_train.mean(axis=0)
    centred = X_train - mean
    # 只需要前 max(ks) 个方向；full_matrices=False 已经是最省的写法
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    total = float((s**2).sum())
    explained = (s**2 / total) if total > 0 else np.zeros_like(s)
    n_available = int(len(s))

    rows = []
    for k in ks:
        k_eff = min(int(k), n_available)
        if k_eff < 1:
            continue
        loadings = vt[:k_eff].T
        train_scores = centred @ loadings
        test_scores = (X_test - mean) @ loadings
        mae = _mae(ridge_predict(train_scores, y_train, test_scores, alpha=alpha), y_test)
        rows.append({
            "k": int(k),
            "k_used": k_eff,
            "explained_variance": float(explained[:k_eff].sum()),
            "mae": mae,
            "ratio": mae / constant if constant else float("inf"),
        })
    if not rows:
        raise ValueError("ks 里没有一个可用的维数")

    useful = [r for r in rows if r["ratio"] < 1.0]
    best = min(r["ratio"] for r in rows)
    saturated = next(
        (r for r in rows if r["ratio"] <= best + 0.10 * (1.0 - best)), rows[-1]
    )
    return {
        "constant": constant,
        "n_components_available": n_available,
        "explained_first": float(explained[0]) if n_available else 0.0,
        "rows": rows,
        "first_useful_k": useful[0]["k"] if useful else None,
        "saturated_k": saturated["k"],
        "best_ratio": best,
    }


def pc_verdict(ladder: dict) -> str:
    """把主成分阶梯读成一句话：几个方向在起作用，它们占多少方差。"""
    first = ladder["rows"][0]
    sat = next(r for r in ladder["rows"] if r["k"] == ladder["saturated_k"])
    if ladder["first_useful_k"] is None:
        return (
            f"主成分方向上一个都不如常数基线（最好 {ladder['best_ratio']:.2f} 倍）："
            "特征里没有可线性提取的信号"
        )
    return (
        f"有效自由度约 {ladder['saturated_k']}（占方差 {sat['explained_variance']:.1%}）；"
        f"第 1 主成分占 {first['explained_variance']:.1%} 的方差却只有 "
        f"{first['ratio']:.2f} 倍判别力"
    )
