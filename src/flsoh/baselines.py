#!/usr/bin/env python3
"""平凡基线阶梯：用最简单的模型量「这份特征表里到底有多少信号」。

## 为什么需要它

端到端只报一个 ``final_mae`` 等于没报。本仓库已经吃过一次亏：模拟数据上
``final_mae = 0.067`` 看着像个正常数字，而「全预测成训练集均值」的 MAE 是
``0.027``——模型比一条水平线还差 2.5 倍。那次之后 ``run_e2e.py`` 加上了常数
基线。

但常数基线是**最弱**的平凡基线，过了它不代表模型有用。真正的问题是：
**1000 维神经网络，比「只拿一个标量（ΔQ 的幅度）」好多少？**

模拟数据上的实测（24 颗 × 20 循环，留出 6 颗整颗测试）::

    常数基线                     MAE 0.02703
    只用幅度（1 维岭回归）       MAE 0.00430   ← 最强平凡基线
    全部 1000 维（岭回归）       MAE 0.00568
    只用形状（归一化后 1000 维） MAE 0.05411   ← 比水平线还差
    神经网络（run_e2e 默认设置） MAE 0.02289

读法：**信号几乎全在幅度里**。1000 维模型打不过 1 维线性模型，而且归一化后
（幅度信息被除掉）连水平线都不如——说明另外 999 维在模拟数据上主要是噪声。

这不是「神经网络不行」，是**模拟数据的任务太单薄**：容量衰减在
:mod:`flsoh.simulate` 里只按 ``Q(c) = Q0·(1 − fade·√(c/C))`` 缩放整条曲线，
形状随老化的变化（γ 随容量损失下降 15%）只有约 1.8% 的量级。所以这条阶梯
的作用是**把 e2e 的指标放进可解释的位置**：它证明的是管道接对了，不是方法
有效；而当 ``final_mae`` 明显大于 ``amplitude`` 那一格时，该查的是模型与训练
设置，不是继续调数据。

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


def ladder_verdict(ladder: dict) -> str:
    """把阶梯表读成一句话。**四种情形必须分开**，不能合并成「有没有信号」。

    上一版把 ``shape_only > constant`` 直接写成「形状里没有标签信息（幅度是
    唯一载体）」——在真实数据上这是**错的**：那里 ``amplitude`` 也比常数基线
    差（1.17 倍），所以幅度不是载体；信息在幅度与形状的**组合**里（全量
    0.36 倍）。两件事混成一句，就把一个反直觉的发现读成了「符合预期」。
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
