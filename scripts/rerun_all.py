#!/usr/bin/env python3
"""按 `results/*.json` 里记录的参数重跑全部实验（特征口径修复后的一次性动作）。

## 为什么要有这个脚本

`docs/plan.md` §13.5.2 查出 M1–M4 的特征表是错的（`matr.charge_curve()` 返回
整条循环，放电段被按电压 argsort 打散进充电段的电压区间）。修复后旧表训出的
172 个 run 全部作废，必须重跑。

重跑不能手抄命令行。172 条命令里，错法都不会自己暴露：

- 少一个 `--server-tau`，FedAdam 退回 flwr 默认的 1e-3；而 M3 ② 恰好扫过
  τ=1e-6 / 1e-5，于是"退回默认"把那一档悄悄换成了另一档；
- `--mu 0.1` 写成 `--mu 0.01`，FedProx 从"差 +85.8%"变成"差 +46.7%"，
  方向不变、数字全错，而文件名靠 `--tag` 固定，从外面完全看不出来；
- 漏 `--noise-multiplier 0`，DP 对照组会变成"完全不开 DP"：ε 从 inf 变成
  字段不存在，而 M4 的结论正是拿 nm=0 当机制对照的。

所以命令行从结果 JSON 反推：**JSON 里记了什么就重放什么**。反推不出来的项
报错退出，不猜——"猜一个默认值"与"用错的超参跑一遍"是同一件事。

## 用法

    python3 scripts/rerun_all.py                        # 只打印命令（默认）
    python3 scripts/rerun_all.py --execute --jobs 8     # 真跑，8 路并行
    python3 scripts/rerun_all.py --execute --skip-existing --jobs 8
    python3 scripts/rerun_all.py --only 'fedavg__iid'   # 只重跑匹配的 run

`--jobs` 并行是安全的：每个 run 是独立进程，只读特征表与划分文件，各写各的
JSON，文件名由 `--tag` 固定。并行前必须把 OMP/MKL 线程数压到 1（脚本自动
设），否则 N 个进程各开满核会互相抢，比串行还慢。

## 三道预检（都不许静默跳过）

1. **分片比汇总表新 → 拒绝**。`features.parquet.parts/` 里任何一个分片比
   `features.parquet` 新，说明重建还没汇总，这时开跑等于又跑一遍旧表——
   而这正是最危险的失败模式：命令、日志、退出码全都正常。
2. **`feature_version` 对不上 → 拒绝**（给了 `--expect-feature-version` 时）。
3. **反推不出参数 → 拒绝**，并列出缺哪一项。不跳过、不降级。

退出码：0 = 全部成功；1 = 有 run 失败或产物校验不过；2 = 预检失败。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from flsoh.featurespec import read_feature_version  # noqa: E402

FEDERATED_STRATEGIES = ("fedavg", "fedprox", "fedadam")
BASELINE_STRATEGIES = ("centralized", "local")

# 反推命令行时必需的字段。少一个就报错，不填默认值——见模块 docstring。
REQUIRED_FEDERATED = (
    "strategy", "rounds", "local_epochs", "learning_rate", "optimizer",
    "seed", "partition_json",
)
REQUIRED_BASELINE = ("strategy", "seed", "epochs", "learning_rate")
# FedAdam 的服务端超参必须齐：默认值随 flwr 版本漂移，缺一项就复现不了。
REQUIRED_SERVER = ("server_lr", "server_beta_1", "server_beta_2", "server_tau")
# DP 三件套必须齐：只给 noise_multiplier 会退回默认裁剪范数与 δ。
REQUIRED_DP = ("noise_multiplier", "max_grad_norm", "target_delta")


def _num(value) -> str:
    """把数字写成能精确回读的字符串。

    不用 `f"{v:g}"`：`:g` 只保留 6 位有效数字，0.0010000001 会变成
    0.001，而 `--lr` 差一位就是另一个实验。`repr(float)` 是往返精确的。
    """
    return repr(float(value))


def load_runs(results_dir: Path) -> dict[str, dict]:
    """读 `results/*.json` 里真正的 run（有 `final_mae` 的那些）。

    同一目录下还躺着 `partition-*.json` 与 `heterogeneity.json`，它们没有
    `final_mae`。按"有没有 final_mae"筛，而不是按文件名模式——文件名模式
    会随实验命名习惯漂移（M2 用 `-` 分隔，M3 起用 `__`）。
    """
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "final_mae" in payload:
            runs[payload.get("run_id") or path.stem] = payload
    return runs


def build_federated_command(
    run: dict, *, config: str, features: str, out: str,
    python: str | None = None,
) -> list[str]:
    """把一个联邦 run 的 JSON 还原成命令行。

    `--tag` 用 run_id 原样传入：文件名必须逐字对上，否则重跑产物与
    `docs/plan.md` §10.2 的表、图 1–3 的分组键全部错位。
    """
    argv = [
        python or sys.executable, "scripts/run_federated.py",
        "--config", config,
        "--features", features,
        "--partition-json", str(run["partition_json"]),
        "--strategy", str(run["strategy"]),
        "--rounds", str(int(run["rounds"])),
        "--local-epochs", str(int(run["local_epochs"])),
        "--lr", _num(run["learning_rate"]),
        "--optimizer", str(run["optimizer"]),
        "--seed", str(int(run["seed"])),
        "--out", out,
        "--tag", str(run["run_id"]),
    ]
    if run["strategy"] == "fedprox":
        # 必须显式传 μ：不传的话脚本会退回配置里 proximal_mu 的最后一档，
        # 而 M2 扫过 0.01 与 0.1 两档，退回哪一档都不对。
        argv += ["--mu", _num(run["mu"])]
    if run["strategy"] == "fedadam":
        argv += [
            "--server-lr", _num(run["server_lr"]),
            "--server-beta1", _num(run["server_beta_1"]),
            "--server-beta2", _num(run["server_beta_2"]),
            "--server-tau", _num(run["server_tau"]),
        ]
    if "noise_multiplier" in run:
        argv += [
            "--noise-multiplier", _num(run["noise_multiplier"]),
            "--max-grad-norm", _num(run["max_grad_norm"]),
            "--target-delta", _num(run["target_delta"]),
        ]
    return argv


def _baseline_optimizer(run: dict) -> str:
    """C0/C1 的 optimizer 只记在 notes 里（早期字段没平铺到顶层）。

    读不出来就返回空串，由调用方判成"反推不出"，不假设是 adam。
    """
    match = re.search(r"optimizer=(\w+)", str(run.get("notes") or ""))
    return match.group(1) if match else ""


def baseline_params(runs: list[dict]) -> tuple[int, float, str, list[int]]:
    """从一组基线 run 里取出 (epochs, lr, optimizer, seeds)。

    一条 `run_local_baseline.py` 命令同时产出 C0 与 C1 的全部 seed，所以这
    四项必须全组一致。不一致说明这批基线不是同一次跑出来的（比如中途换过
    lr），那种情况不能合成一条命令——返回不一致由调用方报错，不挑一个代表。
    """
    epochs = {int(r["epochs"]) for r in runs}
    lrs = {float(r["learning_rate"]) for r in runs}
    opts = {_baseline_optimizer(r) for r in runs}
    if len(epochs) != 1 or len(lrs) != 1 or len(opts) != 1 or "" in opts:
        raise ValueError(
            f"基线的 epochs/lr/optimizer 不唯一：{sorted(epochs)} {sorted(lrs)} {sorted(opts)}"
        )
    seeds = sorted({int(r["seed"]) for r in runs})
    return epochs.pop(), lrs.pop(), opts.pop(), seeds


def build_baseline_command(
    runs: list[dict], *, config: str, features: str, partition: str, out: str,
    python: str | None = None,
) -> list[str]:
    """把 C0/C1 基线还原成一条命令。"""
    epochs, lr, optimizer, seeds = baseline_params(runs)
    return [
        python or sys.executable, "scripts/run_local_baseline.py",
        "--config", config,
        "--features", features,
        "--partition-json", partition,
        "--seeds", ",".join(str(s) for s in seeds),
        "--epochs", str(epochs),
        "--lr", _num(lr),
        "--optimizer", optimizer,
        "--out", out,
    ]


def preflight(
    runs: dict[str, dict], *, features: Path, baseline_partition: Path
) -> tuple[list[dict], list[str]]:
    """逐 run 检查能否反推命令行。返回 (任务列表, 问题列表)。

    任务列表里每个元素是 `{"run_ids": [...], "argv": [...]}`；基线因为一条
    命令产出多个 run，`run_ids` 是列表。
    """
    problems: list[str] = []
    jobs: list[dict] = []

    if not features.exists():
        problems.append(f"特征表不存在：{features}")
    if not baseline_partition.exists():
        problems.append(f"基线划分不存在：{baseline_partition}")
    if problems:
        return jobs, problems

    baselines: list[dict] = []
    for run_id, run in sorted(runs.items()):
        strategy = run.get("strategy")
        if strategy in BASELINE_STRATEGIES:
            baselines.append(run)
            continue
        if strategy not in FEDERATED_STRATEGIES:
            problems.append(f"{run_id}: 未知策略 {strategy!r}")
            continue
        missing = [f for f in REQUIRED_FEDERATED if f not in run]
        if missing:
            problems.append(f"{run_id}: 缺字段 {missing}")
            continue
        if strategy == "fedprox" and "mu" not in run:
            problems.append(f"{run_id}: fedprox 缺 mu")
            continue
        if strategy == "fedadam":
            missing = [f for f in REQUIRED_SERVER if f not in run]
            if missing:
                problems.append(f"{run_id}: fedadam 缺服务端超参 {missing}")
                continue
        if "noise_multiplier" in run:
            missing = [f for f in REQUIRED_DP if f not in run]
            if missing:
                problems.append(f"{run_id}: 开了 DP 但缺 {missing}")
                continue
        if not Path(run["partition_json"]).exists():
            problems.append(f"{run_id}: 划分文件不存在 {run['partition_json']}")
            continue
        jobs.append({"run_ids": [run_id], "kind": "federated", "run": run})

    # ---- 基线合成一条命令 ----
    if baselines:
        baseline_problems: list[str] = []
        for run in baselines:
            missing = [f for f in REQUIRED_BASELINE if f not in run]
            if missing:
                baseline_problems.append(f"{run['run_id']}: 基线缺字段 {missing}")
        try:
            baseline_params(baselines)
        except ValueError as exc:
            baseline_problems.append(str(exc))
        payload = json.loads(baseline_partition.read_text(encoding="utf-8"))
        n_partition = len(payload.get("partition") or {})
        # C1 的结果是"每个客户端各训各的、指标取均值"，客户端数不同就是另一个
        # 实验。这条检查专抓"基线划分文件选错了"——那种错误在结果里只表现为
        # 数字变了，文件名、行数、seeds 全正常。
        for run in baselines:
            if run["strategy"] == "local" and run.get("n_clients") not in (
                None, n_partition
            ):
                baseline_problems.append(
                    f"{run['run_id']}: n_clients={run['n_clients']} 与基线划分的 "
                    f"{n_partition} 个客户端不一致"
                )
        problems.extend(baseline_problems)
        # 有问题的组不进任务列表：带着问题还产出命令，等于给"跑一遍错的"开口子。
        if not baseline_problems:
            jobs.append({"run_ids": [r["run_id"] for r in baselines],
                         "kind": "baseline", "runs": baselines})

    return jobs, problems


def check_features_freshness(features: Path, *, allow_stale: bool) -> list[str]:
    """分片比汇总表新 = 重建还没汇总，这时开跑等于又跑一遍旧表。

    这是本轮最危险的失败模式：命令正常、日志正常、退出码 0，而表里全是旧
    数字。所以做成硬闸门而不是提醒。
    """
    problems: list[str] = []
    parts_dir = features.with_name(features.name + ".parts")
    if not parts_dir.is_dir():
        return problems
    newest = None
    for part in parts_dir.glob("*.parquet"):
        mtime = part.stat().st_mtime
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return problems
    if features.exists() and newest > features.stat().st_mtime:
        msg = (
            f"{parts_dir.name}/ 里有分片比 {features.name} 新：重建还没汇总，"
            "现在跑实验用的仍是旧表"
        )
        if not allow_stale:
            problems.append(msg)
        else:
            print(f"WARN {msg}（--allow-stale-features 已显式放行）")
    return problems


def check_feature_version(features: Path, expected: str | None) -> list[str]:
    """核对汇总表的 `feature_version`。不给 expected 就只打印。"""
    spec_path = features.with_name(features.stem + ".spec.json")
    if not spec_path.exists():
        if expected:
            return [f"{spec_path.name} 不存在，无法核对 feature_version"]
        return []
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    actual = spec.get("feature_version")
    print(f"特征表口径：feature_version={actual!r}")
    if expected and actual != expected:
        return [f"feature_version={actual!r}，期望 {expected!r}"]
    return []


def probe_torch_version(python: str) -> str | None:
    """问目标解释器的 torch 版本。解释器不存在或导不进 torch 就返回 None。"""
    try:
        proc = subprocess.run(
            [python, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        # 路径写错时 subprocess 抛 FileNotFoundError 而不是返回非零码。
        # 不接住的话预检会以一个无关的 traceback 结束，看起来像脚本坏了。
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def check_torch_version(runs: dict[str, dict], python: str) -> list[str]:
    """目标解释器的 torch 版本必须与结果记录里的一致。

    实测（2026-10-25，同一份特征表、同一 seed、同一个仓库、默认线程数）：

    ============== =======================
    torch          C0 seed0 最终 MAE
    ============== =======================
    2.10.0+cpu     0.0015666428208351135
    2.14.1+cpu     0.0015434211492538453
    ============== =======================

    差 1.5%，**不是噪声**：用 2.10.0 重跑能逐位复现记录值，用 2.14.1 不能。
    换了 torch 就换了 BLAS 内核与算子实现，累加顺序随之改变，300 个 epoch 的
    训练会把这点差异放大。不核对这一项，"特征修复前后"的对比里就混进了一个
    版本变量，而且从数字上完全看不出来。
    """
    recorded = {
        str(r.get("torch_version")) for r in runs.values() if r.get("torch_version")
    }
    actual = probe_torch_version(python)
    if actual is None:
        return [f"{python} 里 import torch 失败，无法核对版本"]
    print(f"目标解释器 torch={actual}；结果记录里是 {sorted(recorded) or '（无记录）'}")
    bad = sorted(v for v in recorded if v != actual)
    if bad:
        return [
            f"torch 版本不一致：目标解释器 {actual}，记录里 {bad}。"
            "换版本会改数值（实测 2.10.0→2.14.1 让 C0 MAE 从 0.0015666 变 "
            "0.0015434），重跑出的差异就分不清是特征修复还是版本升级"
        ]
    return []


def check_thread_count(runs: dict[str, dict], threads: int) -> list[str]:
    """记录里带了 `torch_num_threads` 的 run，线程数必须与 `--threads` 一致。

    早期结果没有这个字段（它是 2026-10-25 才加进产物的），那时不检查——
    也不假装检查过。字段一旦存在就当成硬约束：实测 C0 在 24 线程与 1 线程下
    差 2.1%（0.0015666428208351135 vs 0.0015997225046157837），而默认线程数
    还意味着结果随机器核数变化。

    **「部分有、部分没有」是另一种错误**：那说明新旧结果混在同一个目录里
    （典型成因是重跑中断后用 `--skip-existing` 续跑）。只看「有字段的那些」
    会得到 `{1}` 而判定一致，于是混杂目录被放行——这比线程数本身更危险。
    """
    recorded = {
        int(r["torch_num_threads"]) for r in runs.values() if "torch_num_threads" in r
    }
    if not recorded:
        return []
    problems: list[str] = []
    missing = sorted(
        rid for rid, r in runs.items() if "torch_num_threads" not in r
    )
    if missing:
        problems.append(
            f"{len(missing)}/{len(runs)} 个 run 没有 torch_num_threads 而其余有："
            f"新旧结果混在同一个目录里（例如重跑中断后用 --skip-existing 续跑）。"
            f"例：{missing[:3]}"
        )
    if recorded != {int(threads)}:
        problems.append(
            f"线程数不一致：--threads {threads}，记录里是 {sorted(recorded)}。"
            "线程数改浮点累加顺序，实测 C0 在 24 与 1 线程下差 2.1%"
        )
    return problems


def artifact_is_current(
    path: Path,
    *,
    torch_version: str | None,
    threads: int,
    feature_version: str | None,
) -> bool:
    """这个产物是不是**本次重跑**的产物——决定 `--skip-existing` 能不能跳。

    原来只判「文件在不在」。`--out` 默认是 `results/`，而那里装的可能是上一次
    的结果，于是重跑中断后带 `--skip-existing` 续跑会跳过全部任务、打印
    「剩 0 个」、退出码 0，接着汇总出一张新旧混杂的表。这类错误不抛异常、
    行数也对，只有数字是旧的。

    判据取「产物自己记了什么」：线程数必须一致；torch 版本与特征口径在能读到
    期望值时也必须一致。读不出（None）就不拿它下结论——拿一份不报口径的表去
    证明一致，等于没查。
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(d, dict) or "final_mae" not in d:
        return False
    if d.get("torch_num_threads") != threads:
        return False
    if torch_version is not None and d.get("torch_version") != torch_version:
        return False
    if feature_version is not None and d.get("feature_version") != feature_version:
        return False
    return True



def pinned_env(threads: int) -> dict[str, str]:
    """把线程数钉死的环境变量。

    并行靠 `--jobs` 而不是让每个进程各开满核：N 个进程 × 24 线程只会互相抢，
    而且线程数本身会影响浮点累加顺序（见 `--threads` 的说明）。
    """
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    # 前插而不是 setdefault：调用方 shell 里可能已经有一个不含 src 的
    # PYTHONPATH，setdefault 就不会补，而训练脚本虽然自己插了 sys.path，
    # 依赖它"碰巧也做了"不是配置该有的样子。
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" + (os.pathsep + existing if existing else "")
    return env


def _run_job(job: dict, *, config: str, features: str, out: str,
             baseline_partition: str, log_dir: Path | None,
             python: str, threads: int) -> dict:
    """跑一个任务，返回 {run_ids, ok, seconds, output}。"""
    if job["kind"] == "baseline":
        argv = build_baseline_command(
            job["runs"], config=config, features=features,
            partition=baseline_partition, out=out, python=python,
        )
    else:
        argv = build_federated_command(
            job["run"], config=config, features=features, out=out, python=python
        )
    env = pinned_env(threads)
    t0 = time.time()
    proc = subprocess.run(
        argv, cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    elapsed = round(time.time() - t0, 1)
    output = (proc.stdout or "") + (proc.stderr or "")
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        name = job["run_ids"][0] if len(job["run_ids"]) == 1 else "baselines"
        (log_dir / f"{name}.log").write_text(output, encoding="utf-8")
    return {
        "run_ids": job["run_ids"], "ok": proc.returncode == 0,
        "seconds": elapsed, "returncode": proc.returncode, "output": output,
    }


def verify_outputs(runs: dict[str, dict], out_dir: Path) -> list[str]:
    """重跑后逐个核对产物：文件在、有 final_mae、run_id 与文件名一致。

    不做这一步的话，"少写了一个文件"和"跑成功了"在汇总表上长得一样——都是
    少了那几行，而 `summarize_results` 会拿剩下的一批算出看起来正常的均值。
    """
    problems: list[str] = []
    for run_id in sorted(runs):
        path = out_dir / f"{run_id}.json"
        if not path.exists():
            problems.append(f"缺少产物：{path.name}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "final_mae" not in payload:
            problems.append(f"{path.name} 里没有 final_mae")
        elif payload.get("run_id") != run_id:
            problems.append(
                f"{path.name} 里的 run_id={payload.get('run_id')!r} 与文件名不一致"
            )
    return problems


def compare(old_dir: Path, new_dir: Path, runs: dict[str, dict]) -> None:
    """新旧逐 run 对比。只打印，不判定——判定在 summarize_results 里。"""
    rows = []
    for run_id in sorted(runs):
        old_path = old_dir / f"{run_id}.json"
        new_path = new_dir / f"{run_id}.json"
        if not old_path.exists() or not new_path.exists():
            continue
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
        o, n = old["final_mae"], new["final_mae"]
        ratio = (n / o) if o else float("inf")
        rows.append((run_id, o, n, ratio))
    if not rows:
        print("\n没有可对比的 run（新旧目录没有同名产物）。")
        return
    print(f"\n新旧对比（{len(rows)} 个 run，ratio = 新/旧）：")
    print(f"  {'run_id':44s} {'旧 MAE':>10s} {'新 MAE':>10s} {'ratio':>8s}")
    for run_id, o, n, ratio in rows:
        print(f"  {run_id:44s} {o:10.5f} {n:10.5f} {ratio:8.2f}")
    better = sum(1 for _, _, _, r in rows if r < 1.0)
    print(f"\n  新表更好 {better}/{len(rows)}，更差 {len(rows) - better}/{len(rows)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按 results/*.json 重跑全部实验")
    ap.add_argument("--results", default="results", help="读旧结果的目录")
    ap.add_argument("--config", default="configs/mvp.toml")
    ap.add_argument("--features", default="data/mit-stanford/features.parquet")
    ap.add_argument("--baseline-partition", default="results/partition-mvp-k5-iid.json",
                    help="C0/C1 用的划分（早期结果 JSON 没记这一项）")
    ap.add_argument("--out", default="results", help="重跑产物写到哪")
    ap.add_argument("--only", default=None, help="只处理 run_id 匹配该正则的 run")
    ap.add_argument("--execute", action="store_true", help="真跑；不给则只打印命令")
    ap.add_argument("--jobs", type=int, default=6, help="并行进程数（默认 6）")
    ap.add_argument("--python", default=None,
                    help="跑实验用的解释器，默认当前解释器。必须与结果记录里的 "
                         "torch 版本一致，否则预检拒绝")
    ap.add_argument("--threads", type=int, default=1,
                    help="每个进程的 OMP/MKL 线程数（默认 1）。并行靠 --jobs，"
                         "不靠让每个进程开满核")
    ap.add_argument("--allow-torch-mismatch", action="store_true",
                    help="放行 torch 版本不一致的检查（只在做版本对比时用）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="产物已存在就跳过（断点续跑）")
    ap.add_argument("--log-dir", default=None, help="把每个 run 的输出写进该目录")
    ap.add_argument("--compare", default=None,
                    help="跑完与旧结果目录逐 run 对比（如 results/fullcycle-bug）")
    ap.add_argument("--expect-feature-version", default=None,
                    help="要求汇总表的 feature_version 等于该值")
    ap.add_argument("--allow-stale-features", action="store_true",
                    help="放行「分片比汇总表新」的检查（只在确认汇总已完成时用）")
    ap.add_argument("--quiet", action="store_true", help="不逐条打印命令")
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    features = Path(args.features)

    runs = load_runs(results_dir)
    if not runs:
        print(f"FAIL {results_dir} 里没有可识别的 run（没有 final_mae 的 JSON）")
        return 2
    if args.only:
        pattern = re.compile(args.only)
        runs = {k: v for k, v in runs.items() if pattern.search(k)}
        if not runs:
            print(f"FAIL --only {args.only!r} 没匹配到任何 run")
            return 2
    print(f"读到 {len(runs)} 个 run（{results_dir}）")

    problems = check_features_freshness(features, allow_stale=args.allow_stale_features)
    problems += check_feature_version(features, args.expect_feature_version)
    python = args.python or sys.executable
    torch_problems = check_torch_version(runs, python)
    if args.allow_torch_mismatch:
        for p in torch_problems:
            print(f"WARN {p}（--allow-torch-mismatch 已显式放行）")
    else:
        problems += torch_problems
    problems += check_thread_count(runs, args.threads)
    jobs, job_problems = preflight(
        runs, features=features, baseline_partition=Path(args.baseline_partition)
    )
    problems += job_problems
    if problems:
        print("FAIL 预检未通过：")
        for p in problems:
            print(f"  - {p}")
        return 2

    if args.skip_existing:
        target_torch = probe_torch_version(python)
        want_version = read_feature_version(features)
        if target_torch is None or want_version is None:
            print(
                "WARN 读不出目标 torch 版本或特征口径（"
                f"torch={target_torch!r}，feature_version={want_version!r}），"
                "跳过判定只能靠线程数——这时 --skip-existing 的风险更高，"
                "确认输出目录本来就是空的更稳妥"
            )
        kept = []
        skipped = 0
        stale = 0
        for job in jobs:
            fresh = all(
                artifact_is_current(
                    out_dir / f"{r}.json",
                    torch_version=target_torch,
                    threads=args.threads,
                    feature_version=want_version,
                )
                for r in job["run_ids"]
            )
            if fresh:
                skipped += 1
            else:
                if any((out_dir / f"{r}.json").exists() for r in job["run_ids"]):
                    stale += 1
                kept.append(job)
        jobs = kept
        print(
            f"--skip-existing：跳过 {skipped} 个已完成的任务，剩 {len(jobs)} 个"
        )
        if stale:
            print(
                f"  其中 {stale} 个任务目录里已有产物，但**不是本次设置的产物**"
                "（线程数/torch 版本/特征口径对不上），已重新排队"
            )

    est = sum(
        float(job["run"].get("wall_clock_s") or 0) if job["kind"] == "federated"
        else sum(float(r.get("wall_clock_s") or 0) for r in job["runs"])
        for job in jobs
    )
    print(f"待跑 {len(jobs)} 个任务，旧墙钟合计 {est / 3600:.2f} 小时，"
          f"--jobs {args.jobs} 预计 {est / 3600 / max(args.jobs, 1):.2f} 小时"
          f"（解释器 {python}，每进程 {args.threads} 线程）")

    if not args.execute:
        for job in jobs:
            if job["kind"] == "baseline":
                cmd = build_baseline_command(
                    job["runs"], config=args.config, features=args.features,
                    partition=args.baseline_partition, out=args.out, python=python,
                )
            else:
                cmd = build_federated_command(
                    job["run"], config=args.config, features=args.features,
                    out=args.out, python=python,
                )
            if not args.quiet:
                print("  " + " ".join(cmd[1:]))
        print("\n（未加 --execute，以上只是命令预览）")
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else None
    failures: list[dict] = []
    done = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _run_job, job, config=args.config, features=args.features,
                out=args.out, baseline_partition=args.baseline_partition,
                log_dir=log_dir, python=python, threads=args.threads,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            done += 1
            label = ",".join(res["run_ids"])
            if res["ok"]:
                print(f"[{done}/{len(jobs)}] OK   {label}  {res['seconds']}s", flush=True)
            else:
                failures.append(res)
                tail = "\n".join(res["output"].strip().splitlines()[-15:])
                print(f"[{done}/{len(jobs)}] FAIL {label}  {res['seconds']}s\n{tail}",
                      flush=True)
    print(f"\n重跑结束，墙钟 {round(time.time() - t0, 1)}s，失败 {len(failures)} 个")

    missing = verify_outputs(runs, out_dir)
    if missing:
        print(f"FAIL 产物校验：{len(missing)} 处")
        for m in missing[:20]:
            print(f"  - {m}")
    if args.compare:
        compare(Path(args.compare), out_dir, runs)
    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
