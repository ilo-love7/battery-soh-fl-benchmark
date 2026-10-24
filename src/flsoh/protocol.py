"""实验协议（TOML 配置）校验。

设计约束（docs/plan.md §3.2 模块契约）：

- 只读配置，不改配置，不猜默认值。缺键就报缺键，不替用户填。
- 不依赖 torch / flwr，纯标准库 + ``tomllib``（Python 3.11 内置）。

为什么值得单独一个模块：「跑 3 个 seed」这句话在没有机器可读矩阵的情况
下不会被执行。协议校验器的作用是把「要跑什么」变成可检查的文件，让
"忘了 centralized 基线" 或 "只跑 1 个 seed" 在开跑前就红，而不是在写
报告时才发现。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

LEVEL_ERROR = "ERROR"
LEVEL_WARN = "WARN"

KNOWN_ATOMS = ("cell",)
KNOWN_SCHEMES = ("iid", "dirichlet")
KNOWN_REMAINDERS = ("spread", "drop")
# 与 docs/plan.md §5.2 的策略矩阵一致。fedbn / scaffold 标注为「需自行接入」，
# 允许写进配置（这是计划），但不等于已经实现。
KNOWN_STRATEGIES = (
    "centralized",
    "fedavg",
    "fedprox",
    "fedadam",
    "fedyogi",
    "fedbn",
    "scaffold",
)
# 需要自己实现聚合逻辑的策略：写进配置时必须显式声明，避免报告里变成"已对比"
NON_BUILTIN_STRATEGIES = ("fedbn", "scaffold")

TOP_LEVEL_SECTIONS = (
    "meta",
    "data",
    "partitions",
    "strategies",
    "train",
    "seeds",
    "privacy",
    "acceptance",
)

# 判据阈值：全部是相对量，必须落在 (0, 1)
ACCEPTANCE_FRACTIONS = (
    "iid_degradation_max",
    "non_iid_degradation_min",
    "fedprox_improvement_min",
    "seed_cv_max",
    "dp_degradation_max",
)

MIN_SEEDS = 3
MIN_NOISE_POINTS = 3


class ProtocolError(ValueError):
    """配置文件本身读不出来（不是内容不合规）。"""


def load_config(path: str | Path) -> dict:
    """读一份 TOML。语法错误抛 ProtocolError，内容问题交给 validate_config。"""
    path = Path(path)
    if not path.exists():
        raise ProtocolError(f"配置文件不存在：{path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ProtocolError(f"{path} 不是合法 TOML：{exc}") from exc


def _get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(cfg: dict) -> list[tuple[str, str, str]]:
    """返回 ``[(level, code, message)]``。有 ERROR 就不该开跑。"""
    issues: list[tuple[str, str, str]] = []

    def err(code: str, message: str) -> None:
        issues.append((LEVEL_ERROR, code, message))

    def warn(code: str, message: str) -> None:
        issues.append((LEVEL_WARN, code, message))

    for section in cfg:
        if section not in TOP_LEVEL_SECTIONS:
            warn("C-UNKNOWN", f"未知配置节 [{section}]，检查器不认识，确认不是拼错")

    # --- 数据与划分原子 ---------------------------------------------
    atom = _get(cfg, "data.partition_atom")
    if atom is None:
        err("C-ATOM", "缺少 data.partition_atom")
    elif atom not in KNOWN_ATOMS:
        err(
            "C-ATOM",
            f"data.partition_atom={atom!r}，本项目只允许 {KNOWN_ATOMS}："
            "以循环为原子会让同一颗电池跨客户端，时间序列截断使泛化虚高",
        )

    n_cells = _get(cfg, "data.n_cells")
    if not _is_int(n_cells) or n_cells <= 0:
        err("C-DATASET", f"data.n_cells 必须是正整数，收到 {n_cells!r}")

    dataset = _get(cfg, "data.dataset")
    if not dataset:
        err("C-DATASET", "缺少 data.dataset（结果 JSON 要记数据源）")

    # --- 划分 -------------------------------------------------------
    test_cells = _get(cfg, "partitions.test_cells")
    if not _is_int(test_cells) or test_cells < 0:
        err("C-TEST", f"partitions.test_cells 必须是非负整数，收到 {test_cells!r}")
        test_cells = 0
    elif _is_int(n_cells) and test_cells >= n_cells:
        err(
            "C-TEST",
            f"test_cells={test_cells} >= n_cells={n_cells}，客户端将无数据可用",
        )

    split_seed = _get(cfg, "partitions.split_seed")
    if not _is_int(split_seed):
        err(
            "C-SPLIT-SEED",
            "缺少 partitions.split_seed：留出哪些测试电池必须与训练 seed 解耦，"
            "否则换 seed 就换测试集，曲线之间不可比",
        )

    remainder = _get(cfg, "partitions.remainder", "spread")
    if remainder not in KNOWN_REMAINDERS:
        err("C-REMAINDER", f"partitions.remainder={remainder!r}，只允许 {KNOWN_REMAINDERS}")

    k_list = _get(cfg, "partitions.cells_per_client")
    if not isinstance(k_list, list) or not k_list:
        err("C-CPC", "partitions.cells_per_client 必须是非空列表，如 [5]")
        k_list = []
    for k in k_list:
        if not _is_int(k) or k <= 0:
            err("C-CPC", f"cells_per_client 元素必须是正整数，收到 {k!r}")
            continue
        if not _is_int(n_cells) or not _is_int(test_cells):
            continue
        available = n_cells - test_cells
        if available <= 0:
            continue
        if available % k != 0 and remainder == "drop":
            warn(
                "C-DIVIDE",
                f"{available} 颗电池不能被 k={k} 整除，remainder=drop 会丢弃 "
                f"{available % k} 颗电池（§6.3 的 P-COVER 不会报警，因为丢弃发生在划分之前）",
            )
        if available // k == 0:
            err("C-DIVIDE", f"{available} 颗电池分不出 k={k} 的客户端")

    schemes = _get(cfg, "partitions.schemes")
    if not isinstance(schemes, list) or not schemes:
        err("C-SCHEME", "partitions.schemes 必须是非空列表")
        schemes = []
    for scheme in schemes:
        if scheme not in KNOWN_SCHEMES:
            err("C-SCHEME", f"未知划分方案 {scheme!r}，只支持 {KNOWN_SCHEMES}")
    if schemes and "iid" not in schemes:
        err(
            "C-SCHEME",
            "partitions.schemes 缺少 iid：没有 IID 对照就无法区分"
            "「Non-IID 掉点」与「代码有 bug」（§11.3）",
        )

    if "dirichlet" in schemes:
        alphas = _get(cfg, "partitions.dirichlet_alpha")
        if not isinstance(alphas, list) or not alphas:
            err("C-ALPHA", "schemes 含 dirichlet 但 dirichlet_alpha 为空")
        else:
            for alpha in alphas:
                if not _is_number(alpha) or alpha <= 0:
                    err("C-ALPHA", f"dirichlet_alpha 元素必须为正数，收到 {alpha!r}")

    # --- 策略 -------------------------------------------------------
    strategies = _get(cfg, "strategies.include")
    if not isinstance(strategies, list) or not strategies:
        err("C-STRATEGY", "strategies.include 必须是非空列表")
        strategies = []
    for name in strategies:
        if name not in KNOWN_STRATEGIES:
            err("C-STRATEGY", f"未知策略 {name!r}，已知：{KNOWN_STRATEGIES}")
    if strategies and "centralized" not in strategies:
        err(
            "C-STRATEGY",
            "strategies.include 缺少 centralized（C0 上限基线）："
            "没有上限就不知道联邦掉的是不是本来就该掉（§9.4）",
        )

    if "fedprox" in strategies:
        mus = _get(cfg, "strategies.proximal_mu")
        if not isinstance(mus, list) or not mus:
            err("C-PROX", "strategies 含 fedprox 但 proximal_mu 为空")
        elif not any(_is_number(m) and m > 0 for m in mus):
            warn(
                "C-PROX",
                "proximal_mu 全为 0，此时 FedProx 退化为 FedAvg，"
                "网格里会白跑一遍同名配置",
            )

    for name in strategies:
        if name in NON_BUILTIN_STRATEGIES:
            warn(
                "C-NONBUILTIN",
                f"{name} 不是 flwr 内置策略（在 baselines/ 里），"
                "接入前先确认能否在时间预算内验证收敛，接不进来就从 include 里删掉",
            )

    # --- 训练超参 ---------------------------------------------------
    num_rounds = _get(cfg, "train.num_rounds")
    if not _is_int(num_rounds) or num_rounds < 1:
        err("C-TRAIN", f"train.num_rounds 必须是正整数，收到 {num_rounds!r}")
    local_epochs = _get(cfg, "train.local_epochs")
    if not _is_int(local_epochs) or local_epochs < 1:
        err("C-TRAIN", f"train.local_epochs 必须是正整数，收到 {local_epochs!r}")
    elif _is_int(num_rounds) and num_rounds >= 1:
        # 判据不是「local_epochs 大不大」，而是「联邦的总本地 epoch 与中心化
        # 基线对不对等」。同一条 le=10，在每客户端 80 行数据下是**必要的**
        # （le=2 只有 400 步、欠拟合到 0.0222），在每客户端上万样本下才是漂移
        # 风险。所以只有拿总训练量比才有意义。
        #
        # 这条不是风格建议：本仓库实测同一份划分、同一批 seed，只把
        # local_epochs 从 10 降到 3（与基线的 300 ep 对齐），相对差就从
        # −20.9% 变成 +772%（§14 坑 15）。不声明训练量的相对差会反号。
        base_epochs = _get(cfg, "train.baseline_epochs")
        if _is_int(base_epochs) and base_epochs >= 1:
            fed_epochs = num_rounds * local_epochs
            ratio = fed_epochs / base_epochs
            if ratio > 1.25 or ratio < 0.8:
                warn(
                    "C-TRAIN",
                    f"训练量不对等：联邦 {num_rounds} 轮 × {local_epochs} ep = "
                    f"{fed_epochs} 个本地 epoch，中心化基线 {base_epochs} ep，"
                    f"比值 {ratio:.2f}。相对差会随这个比值反号（§14 坑 15），"
                    "报结果时必须写明该比值，或另跑一组等训练量对照",
                )
    fraction_fit = _get(cfg, "train.fraction_fit")
    if not _is_number(fraction_fit) or not (0 < fraction_fit <= 1):
        err("C-TRAIN", f"train.fraction_fit 必须落在 (0,1]，收到 {fraction_fit!r}")

    # --- seed -------------------------------------------------------
    seeds = _get(cfg, "seeds.values")
    if not isinstance(seeds, list) or not seeds:
        err("C-SEEDS", "seeds.values 必须是非空列表")
        seeds = []
    else:
        if len(set(seeds)) != len(seeds):
            err("C-SEEDS", f"seeds.values 有重复：{seeds}")
        for seed in seeds:
            if not _is_int(seed) or seed < 0:
                err("C-SEEDS", f"seed 必须是非负整数，收到 {seed!r}")
    min_required = _get(cfg, "seeds.min_required")
    if not _is_int(min_required) or min_required < MIN_SEEDS:
        err(
            "C-SEEDS",
            f"seeds.min_required 必须 >= {MIN_SEEDS}，收到 {min_required!r}："
            "1 个 seed 的结论无法与随机波动区分（§14 坑 8）",
        )
    elif isinstance(seeds, list) and len(seeds) < min_required:
        err(
            "C-SEEDS",
            f"seeds.values 只有 {len(seeds)} 个，少于声明的 min_required={min_required}",
        )

    # --- 隐私 -------------------------------------------------------
    dp_enabled = _get(cfg, "privacy.enabled")
    if not isinstance(dp_enabled, bool):
        err("C-PRIVACY", "缺少 privacy.enabled（true/false 必须显式写）")
        dp_enabled = False
    noise = _get(cfg, "privacy.noise_multiplier")
    if dp_enabled:
        if not isinstance(noise, list) or not noise:
            err(
                "C-PRIVACY",
                f"启用 DP 时 noise_multiplier 必须是非空列表，收到 {noise!r}",
            )
            noise = []
        # nm=0 是**对照组**（plan §7：没有噪声就没有 DP），不是 DP 臂。
        # 它必须被允许写进矩阵，但不算扫描点——否则"3 个点"里有 1 个是
        # 对照，隐私-效用曲线实际只有 2 个点，而表里写着 3。
        zeros = [s for s in noise if _is_number(s) and s == 0]
        positive = [s for s in noise if _is_number(s) and s > 0]
        if zeros:
            warn(
                "C-PRIVACY",
                f"noise_multiplier 含 0（{len(zeros)} 个）：这是**对照组**，"
                "只做逐样本裁剪与 Poisson 抽样、不加噪，ε=inf。"
                "它不计入扫描点数，报告里也不能算作 DP 臂",
            )
        if len(positive) < MIN_NOISE_POINTS:
            err(
                "C-PRIVACY",
                f"启用 DP 时**正的** noise_multiplier 至少 {MIN_NOISE_POINTS} 个"
                f"扫描点，实际 {len(positive)} 个（{positive!r}）："
                "少于 3 点画不出隐私-效用曲线",
            )
        for sigma in noise:
            if not _is_number(sigma) or sigma < 0:
                err("C-PRIVACY", f"noise_multiplier 元素必须是非负数，收到 {sigma!r}")
        max_grad_norm = _get(cfg, "privacy.max_grad_norm")
        if not _is_number(max_grad_norm) or max_grad_norm <= 0:
            err(
                "C-PRIVACY",
                f"privacy.max_grad_norm 必须为正数，收到 {max_grad_norm!r}："
                "没有裁剪范数就没有敏感度上界，ε 报不出来",
            )
        delta = _get(cfg, "privacy.target_delta")
        if not _is_number(delta) or not (0 < delta < 1):
            err("C-PRIVACY", f"privacy.target_delta 必须落在 (0,1)，收到 {delta!r}")

    # --- 判据 -------------------------------------------------------
    for key in ACCEPTANCE_FRACTIONS:
        value = _get(cfg, f"acceptance.{key}")
        if value is None:
            err("C-ACCEPT", f"缺少 acceptance.{key}")
        elif not _is_number(value) or not (0 < value < 1):
            err("C-ACCEPT", f"acceptance.{key} 必须落在 (0,1)，收到 {value!r}")
    eps = _get(cfg, "acceptance.dp_epsilon_max")
    if eps is None:
        err("C-ACCEPT", "缺少 acceptance.dp_epsilon_max")
    elif not _is_number(eps) or eps <= 0:
        err("C-ACCEPT", f"acceptance.dp_epsilon_max 必须为正数，收到 {eps!r}")

    return issues


def has_errors(issues: list[tuple[str, str, str]]) -> bool:
    return any(level == LEVEL_ERROR for level, _, _ in issues)


def expected_clients_for(cfg: dict, cells_per_client: int) -> int:
    """按配置算某档 k 应产生的客户端数，供 check_partition 的 P-COUNT 对账。"""
    from flsoh.partition import expected_n_clients

    return expected_n_clients(
        int(_get(cfg, "data.n_cells", 0)),
        int(_get(cfg, "partitions.test_cells", 0)),
        int(cells_per_client),
        str(_get(cfg, "partitions.remainder", "spread")),
    )


# pyproject 的 [tool.flwr.app.config] 与协议配置里同义键的对照表。
# 键名在两边不同（flwr 用连字符、TOML 用下划线嵌套），所以必须显式映射。
_APP_CONFIG_AGREEMENT = {
    "num-server-rounds": ("train", "num_rounds"),
    "local-epochs": ("train", "local_epochs"),
    "learning-rate": ("train", "learning_rate"),
    "batch-size": ("train", "batch_size"),
}


def check_app_config_agreement(cfg: dict, pyproject: str | Path) -> list[tuple[str, str, str]]:
    """检查 pyproject 的 run_config 是否与协议配置一致。

    为什么必须查：**官方运行时读 pyproject，本地联邦循环读 configs/*.toml。**
    这两份是分开维护的，改了一份忘了另一份时两边都不会报错——只是跑出来的
    轮数与本地 epoch 数不一样，而结果 JSON 里记的是本地那份。

    2026-10-24 实际发生过：`configs/mvp.toml` 已经是 100 轮 × 10 ep，
    `pyproject.toml` 还停在 2 轮 × 2 ep。没人发现，因为 heavy-tests 是手动
    触发的，而那一步的检查又是坏的（ModuleNotFoundError）。
    """
    issues: list[tuple[str, str, str]] = []
    path = Path(pyproject)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [(LEVEL_WARN, "C-APP", f"读不出 {path}：{exc}")]
    except tomllib.TOMLDecodeError as exc:
        return [(LEVEL_ERROR, "C-APP", f"{path} 不是合法 TOML：{exc}")]

    app_cfg = (
        data.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
    )
    if not app_cfg:
        return [(LEVEL_WARN, "C-APP", f"{path} 里没有 [tool.flwr.app.config]")]

    for app_key, (section, key) in _APP_CONFIG_AGREEMENT.items():
        dotted = f"{section}.{key}"
        expected = _get(cfg, dotted)
        if expected is None:
            continue  # 协议里没这一项，无从对照
        if app_key not in app_cfg:
            issues.append(
                (LEVEL_WARN, "C-APP", f"{path.name} 缺 {app_key}，官方运行时会用默认值")
            )
            continue
        got = app_cfg[app_key]
        if _is_number(got) and _is_number(expected):
            same = abs(float(got) - float(expected)) < 1e-12
        else:
            same = got == expected
        if not same:
            issues.append(
                (
                    LEVEL_ERROR,
                    "C-APP",
                    f"{path.name} 的 {app_key}={got!r} 与配置的 {dotted}={expected!r} "
                    "不一致：官方运行时读 pyproject、本地循环读 TOML，"
                    "两份会跑出不同的训练量",
                )
            )
    return issues
