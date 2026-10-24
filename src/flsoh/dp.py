"""本地 DP-SGD（plan.md §7 的 L2）：Opacus 薄封装 + 跨轮存活的会计器。

## 为什么值得单独一个模块

Opacus 的 API 是**一次性**的：``make_private`` 把模型、优化器、数据加载器
打包成一套 DP 对象，会计器从零开始记账。联邦循环里每个客户端每轮都要训练，
如果每轮新建一套，会计器每轮归零——ε 会被少算 100 倍，而训练照跑、曲线
照出，只是结果里那个 ε 是假的。

这正是本仓库反复踩的那类静默失败（``flsoh/train.py`` 里 ``ServerAdam`` 的
docstring 列过三个同型的）。所以这里把「一套 DP 对象活过一个客户端的全部
轮次」做成类的唯一职责：**构造一次 → 每轮 ``load_global`` 换参数 →
``epsilon()`` 读账**。

## 会计器为什么每客户端一个，而且**不乘 N**

plan §7 的威胁模型是半诚实服务器，它每轮看到全部 N 个客户端的更新。
直觉上应该把 ``N × 轮数 × 步数`` 一起记账——**那是错的**。各客户端的噪声
相互独立，且「改变客户端 i 数据集里的一条记录」只影响客户端 i 自己的更新，
其余 N−1 个更新逐位不变。写成乘积形式：联合机制是 N 个机制的乘积，相邻
数据集只在一个因子上不同，隐私损失取各因子的上确界：

    ε_joint = max_i ε_i        （不是 N · ε_i）

乘 N 会让 ε 大 23 倍，而它看起来更"保守"、更像严谨，所以特别容易骗过
评审——这一点必须在代码里写明，否则下一个人"修"的时候会把它改回去。

## Poisson 采样不是可选装饰

``make_private`` 默认把 DataLoader 换成 Poisson 抽样，抽样率
``q = 1 / 批次数``（本仓库 80 行 / batch 64 → 2 批 → q = 0.5）。这是
DP-SGD  subsampling 分析的前提；关掉它（``poisson_sampling=False``）只是
近似，Opacus 自己的 docstring 就这么写。本仓库选择保住保证，代价是
**有效批大小从 64 变成期望 40**——所以 DP 组与非 DP 组之间多出一个纯机制
差异，它由 ``noise_multiplier=0`` 的对照组隔离（见 docs/plan.md §13.4）。

## ``noise_multiplier=0`` 是什么：不是 DP，是对照组

Opacus 接受 0：逐样本裁剪与 Poisson 抽样照做，只是不加噪，
``get_epsilon`` 返回 ``inf``。plan §7 明确警告「把梯度裁剪当成差分隐私——
没有噪声就没有 DP」，所以这一组**不能算 DP 臂**。它的作用是把「DP 机制的
开销」（Poisson + 裁剪）与「噪声的开销」分开：没有它，隐私-效用曲线上
nm→0 的外推点与非 DP 基线之间的落差会被误读成"小噪声很便宜"。

## 实测的梯度尺度（2026-10-25，IID k=5、client-0、首轮、未训练）

============= ========== ========== ==========
量              min       中位数      max
============= ========== ========== ==========
逐样本梯度范数   0.0062     0.3962     2.3273
============= ========== ========== ==========

批均值梯度范数 0.147 / 0.223。所以 ``max_grad_norm=1.0`` 对中位数样本不
起作用、只裁掉头部约四分之一——裁剪这一侧是**活跃的**，对照组因此有意义。
噪声 Std（加在批均值梯度上）= ``noise_multiplier × max_grad_norm / 批大小``
= ``nm × 1.0 / 40``，与信号 0.2 同量级，这就是曲线随 nm 劣化的来源。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from flsoh.train import make_optimizer

if TYPE_CHECKING:  # 只为类型标注；运行期不引入 opacus
    from flsoh.model import MLP

#: 会计器默认用 RDP（Rényi 差分隐私）。这是 Opacus 的默认，也是
#: DP-SGD 论文（Abadi et al. 2016）之后最常用的紧界。换 PRV/GDP 会得到
#: 不同的 ε，所以把它钉成常量并写进结果 JSON，复现时不用猜。
ACCOUNTANT = "rdp"


class DPTrainer:
    """一个客户端的 DP-SGD 训练器，寿命横跨该客户端的全部通信轮。

    ## 为什么不是一个函数

    ``make_private`` 返回的会计器必须跨轮累积，否则 ε 少算 100 倍。
    跨轮累积这件事只能靠一个活过整个联邦循环的对象来记住——所以是类。

    ## 与 ``flsoh.train.train_epochs`` 的分工

    本类只负责 Opacus 的管道（包装模型/优化器/加载器、记账、读 ε）；
    训练循环本身在 :func:`flsoh.train.train_epochs` 里，与非 DP 路径共用
    同一份代码（``train_epochs(..., dp=self)``）。这样"训练只有一份实现"
    的模块契约（plan.md §3.3）才守得住。

    ## 参数

    ``noise_multiplier`` 为 ``0.0`` 时是**对照组**：裁剪与 Poisson 抽样
    照做，不加噪，``epsilon()`` 返回 ``inf``。要"完全不开 DP"就不要构造
    本类（``run_federated`` 的 ``noise_multiplier=None``）。

    ``secure_mode=False``（Opacus 默认）用普通 RNG 生成噪声，比 secure_mode
    快。这是一个 benchmark，不做生产部署；Opacus 的告警原文也这么说。
    """

    def __init__(
        self,
        model: "MLP",
        loader: DataLoader,
        *,
        noise_multiplier: float,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        lr: float = 1e-3,
        target_delta: float = 1e-5,
    ) -> None:
        try:
            from opacus import PrivacyEngine
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "DP 需要 opacus：pip install '.[privacy]'（或 pip install "
                "opacus==1.6.0）。注意它是可选依赖，默认 CI job 不装，"
                "所以这里必须在运行期导入而不是模块顶部。"
            ) from exc

        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier 不能为负：{noise_multiplier}")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm 必须为正：{max_grad_norm}")
        if not 0 < target_delta < 1:
            raise ValueError(f"target_delta 必须在 (0,1)：{target_delta}")

        self.model = model
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.target_delta = float(target_delta)
        self.optimizer_name = optimizer
        self.lr = float(lr)

        # 抽样率与批次数在包装前量好。Opacus 用 `1/len(data_loader)`，
        # 而包装后的 DPDataLoader 的 __len__ 不保证与原来一致——自己记一份，
        # 顺便让它进结果 JSON（复现时不用再推）。
        self.n_batches = len(loader)
        self.sample_rate = 1.0 / self.n_batches

        base_optimizer = make_optimizer(model, optimizer, lr)
        self.engine = PrivacyEngine(accountant=ACCOUNTANT, secure_mode=False)
        self.wrapped, self.optimizer, self.loader = self.engine.make_private(
            module=model,
            optimizer=base_optimizer,
            data_loader=loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        if len(self.loader) != self.n_batches:
            # DPDataLoader 的批次数变了，sample_rate 就对不上账。这种情况
            # 训练照跑、ε 照报，只是报的不是实际发生的那个机制的 ε。
            raise RuntimeError(
                f"DP 包装后批次数从 {self.n_batches} 变成 {len(self.loader)}，"
                "抽样率与会计器不再匹配"
            )
        self.steps = 0
        #: 期望批大小（Poisson 抽样的均值）。噪声 Std = nm × max_grad_norm /
        #: expected_batch_size，所以这个数决定信噪比，必须落盘。
        self.expected_batch_size = int(self.optimizer.expected_batch_size)

    # ------------------------------------------------------------------ 状态

    def load_global(self, state_dict: dict[str, torch.Tensor]) -> None:
        """把本轮全局参数灌进来。

        ``make_private`` 包装出来的 ``GradSampleModule`` 把原模型存在
        ``_module`` 子模块里，且 ``parameters()`` 返回的是**同一批张量**；
        ``load_state_dict`` 是原地拷贝，所以灌原模型就等于灌包装模型，
        ``DPOptimizer`` 持有的参数引用不会失效。

        这里刻意不碰 ``wrapped._module``（私有属性）：灌 ``self.model``
        就够了，而 ``self.model`` 就是被包装的那个对象。
        """
        self.model.load_state_dict(state_dict)

    def state(self) -> dict[str, torch.Tensor]:
        """训练后的参数（深拷贝，供聚合用）。"""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------ 记账

    def epsilon(self, delta: float | None = None) -> float:
        """到目前为止花费的隐私预算，``δ`` 默认取构造时的 ``target_delta``。

        ``noise_multiplier=0`` 时返回 ``inf``——不是"ε 很小"，是"没有保证"。
        这两个数在图和表里长得完全不一样，混起来就会把对照组讲成 DP 臂。
        """
        if self.noise_multiplier == 0.0:
            return float("inf")
        if self.steps == 0:
            # ε=0 在数学上没错（没有释放就没有损失），但在这个仓库里
            # "跑过 DP 却报 ε=0" 只可能是记账没接上。宁可炸掉。
            raise ValueError("还没训练过就问 ε：会计器里没有任何一步")
        d = self.target_delta if delta is None else float(delta)
        if not 0 < d < 1:
            raise ValueError(f"delta 必须在 (0,1)：{d}")
        return float(self.engine.get_epsilon(delta=d))

    def privacy_record(self) -> dict:
        """结果 JSON 里的隐私字段。

        ``epsilon`` 取该客户端自己的账；调用方（``run_federation``）再对
        所有客户端取 max——按模块 docstring 的论证，联合机制的 ε 是各客户
        端 ε 的上确界，不是和。
        """
        return {
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "target_delta": self.target_delta,
            "epsilon": self.epsilon(),
            "steps": self.steps,
            "sample_rate": self.sample_rate,
            "expected_batch_size": self.expected_batch_size,
            "n_batches_per_epoch": self.n_batches,
            "accountant": ACCOUNTANT,
            "poisson_sampling": True,
            "optimizer": self.optimizer_name,
            "learning_rate": self.lr,
        }
