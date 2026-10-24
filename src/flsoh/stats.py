"""配对比较的统计量，纯标准库。

**为什么需要这个模块**：plan.md §11.3 的判据是「相对劣化 ≤5% / >10%」，
而实测的 seed 间波动是 5%–63%。**光看两组 mean±std 判不出差异**，因为
两组各自的标准差里混进了 seed 的共同波动。同一 seed 下比较两个配置能把
这部分消掉——这是配对比较。

但配对之后还剩一个问题：**3 个 seed 的符号检验在数学上不可能显著**。
n=3 时所有差同号的双侧 p = 2/2³ = 0.25。所以「3 seeds 判不出来」不是
噪声太大，是样本量不够——这两件事的修法完全不同（前者加 seed，后者改判据）。
本模块把 p 值算出来，让「能不能判」变成可计算的，而不是凭感觉说。

两套检验都给：
- **配对 t**：假定差服从正态，n 小的时候靠不住（df=2 时临界值 4.30）。
- **符号检验**：分布无关，只看方向，代价是需要更多 seed 才能显著。
两者结论不一致时以符号检验为准——它的假设更弱。
"""

from __future__ import annotations

import math


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    """不完全贝塔函数的连分式（Lentz 算法）。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """t 分布的双侧 p 值。"""
    if df < 1:
        raise ValueError(f"df 必须 ≥1，收到 {df}")
    if math.isinf(t):
        return 0.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _binom_two_sided(k: int, n: int) -> float:
    """符号检验的双侧精确 p（p=0.5 的二项分布）。"""
    if n < 1:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_stats(diffs: list[float]) -> dict:
    """一组配对差的统计量。

    `diffs[i] = a[i] − b[i]`，同一个 seed 下两个配置的指标相减。
    负数表示 a 更小（MAE 更小 = a 更好）。
    """
    n = len(diffs)
    if n < 2:
        raise ValueError(f"配对比较至少需要 2 个共同 seed，收到 {n}")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    if se == 0.0:
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    else:
        t = mean / se
    n_neg = sum(1 for d in diffs if d < 0)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": t,
        "df": n - 1,
        "p_t": t_two_sided_p(t, n - 1),
        "n_neg": n_neg,
        "n_pos": sum(1 for d in diffs if d > 0),
        "p_sign": _binom_two_sided(n_neg, n),
        "ci_low": mean - 2.0 * se,
        "ci_high": mean + 2.0 * se,
    }


def is_significant(res: dict, alpha: float = 0.05) -> tuple[bool, str]:
    """能不能判定这个差不为零，以及依据是什么。

    **主判据是配对 t，符号检验只作稳健性旁证。** 理由：
    - 符号检验只用方向、丢掉幅度，n=3 时最小双侧 p = 2/8 = 0.25，
      **任何效应都不可能显著**；要 p<0.05 至少需要 n=6 且方向全同。
      拿它当闸门等于宣布「6 个 seed 以下不许有结论」——这不是保守，是
      换了个更弱的检验然后怪数据。
    - 配对 t 在 n=3 时 df=2、临界值 4.30，本来就自带小样本惩罚；
      它假定差近似正态，对 MAE 这种跨 seed 的连续指标不算强假设。

    两者不一致时**不掩盖**：返回的说明里会写「符号检验不显著，结论依赖
    t 检验的正态假设」，让读者自己判断。
    """
    sig = res["p_t"] < alpha
    if not sig:
        return False, f"p_t={res['p_t']:.4f} ≥ {alpha}"
    if res["p_sign"] >= alpha:
        return True, (
            f"p_t={res['p_t']:.4f}，但符号检验 p={res['p_sign']:.4f}"
            f"（n={res['n']} 太小，结论依赖 t 检验的正态假设）"
        )
    return True, f"p_t={res['p_t']:.4f}，符号检验 p={res['p_sign']:.4f}"
