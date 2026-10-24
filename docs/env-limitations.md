# 环境限制与绕行方案

记录**为什么某些命令在这台机器上跑不了**，以及替代方案是什么、等价性靠什么保证。

写这份文档的理由：这些限制会让人以为「代码没写完」。实际是环境不支持，而排查
过程花了几小时、试了五六条路，不写下来下次还要重走。

---

## 1. flwr 官方仿真运行时在本机不可用

### 现象

`flwr run .` 或 `flwr.simulation.run_simulation(...)` 都会失败：

```
Exception: The current node timed out during startup. This could happen
because some of the raylet failed to startup or the GCS has become overloaded.
```

`RAY_raylet_start_wait_time_s=60` 也无效。真正的错误在
`%TEMP%\ray\session_*\logs\raylet.err` 的第一行：

```
Unhandled exception: class nlohmann::detail::type_error.
what(): [json.exception.type_error.316] invalid UTF-8 byte at index 0: 0xBA
```

以及它上面一行：

```
String field 'ray.rpc.GcsNodeInfo.node_manager_hostname' contains invalid
UTF-8 data when serializing a protocol buffer.
```

### 根因

**主机名是 GBK 编码的中文**（本机 `寒不忧雪`）。

- Python 层 `socket.gethostname()` 返回正确的 `str`（Python 3 走 Unicode API），
  所以 `h.encode('utf-8')` 没问题，看起来一切正常；
- 但 raylet 是 **C++ 程序**，用 ANSI API 取主机名，拿到 GBK 字节
  （「寒」= `0xBA 0xAE`，首字节 `0xBA` 不是合法 UTF-8 起始字节）；
- 它把这串字节塞进 protobuf 的 `node_manager_hostname`，序列化时崩。

所以崩的是 ray 自己的进程，不是我们的代码。

### 试过但无效的绕法

| 尝试 | 结果 |
|---|---|
| `RAY_raylet_start_wait_time_s=60` | 仍然超时（不是慢，是崩） |
| `ray.init(_node_ip_address="127.0.0.1")` | 仍然超时（hostname 照样上报） |
| `COMPUTERNAME=rayhost` 环境变量 | 仍然超时（raylet 直接调 Windows API，不读环境变量） |
| `--node-name` | 那是 ray 节点名，不是 hostname |

唯一彻底的办法是**改 Windows 主机名为 ASCII 并重启**，这超出项目范围。

### 另一条硬约束：Python 版本

`flwr[simulation]==1.41.0` 硬钉 `ray==2.55.1`，而该版本的 wheel 覆盖范围是：

```
ray==2.55.1 ; python_full_version >= '3.11' and python_full_version < '3.13'
ray==2.55.1 ; python_full_version >= '3.13' and < '3.15' and sys_platform != 'win32'
```

也就是说 **Windows + Python 3.13 没有 wheel**。实测：

```
$ pip install ray==2.55.1 --dry-run
ERROR: Could not find a version that satisfies the requirement ray==2.55.1
(from versions: 2.58.0, 2.58.1, 2.59.0)
```

用 `pip download --python-version {3.11,3.12,3.13} --platform win_amd64` 逐档验证：
3.11 和 3.12 能下，3.13 不能。所以本项目在 Windows 上需要 **Python 3.11/3.12**。

本机的 venv 因此建在 `D:\项目\venvs\flsoh-py312`（Python 3.12.10），仓库里那个
3.13 的 `.venv` 只够跑不依赖 ray 的部分。

---

## 2. 替代方案：本地联邦循环

`scripts/run_federated.py` 实现了一个不依赖 ray 的联邦循环：

- **客户端数据**：`make_client_loaders_by_time`，客户端内按时间切 train/val
  （plan.md §6.2 规则 1）；
- **本地训练**：`flsoh.train.train_epochs`，支持 FedProx 近端项；
- **聚合**：`flsoh.train.weighted_average`，按 `num-examples` 加权
  （plan.md §14 坑 3）；
- **评估**：`flsoh.train.evaluate`，在 SOH 原始尺度上算 MAE/RMSE。

### 等价性靠什么保证

1. **训练与评估是同一份代码**。`app/client_app.py` 也调用 `flsoh/train.py`，
   不存在「flwr 路径一份、本地路径一份」。所以两条路径的差异只可能在聚合。
2. **聚合被单独测了**。`tests/test_train.py` 断言：加权平均与手算逐位相等、
   加权与等权不同、非法权重抛错、μ=0 时 FedProx 与 FedAvg 逐位相同、
   μ>0 时参数确实被改变。
3. **单客户端对照**。把 115 颗训练电池全给一个客户端（等价于中心化），
   走联邦代码路径得 MAE 0.00142，与 C0 中心化基线 0.0016 同量级——
   说明循环本身没有引入系统误差。

### 不等价的部分（必须写进报告）

- 本地循环**没有**模拟网络延迟、掉线、算力异构（plan.md §14 坑 6）；
- `fraction_fit < 1` 的客户端抽样没实现（MVP 用 1.0，不影响当前结论）；
- flwr 的 `FedAvg` 还支持 `min_train_nodes` / `min_evaluate_nodes` 等容错参数，
  本地循环没有对应物。

所以**不能**把本地产出的数字说成「flwr 跑出来的」。

---

## 3. M1 DOD ① 的验证点移到 CI

M1 的完成定义①是「`flwr run` 跑通官方 MNIST」。本机做不到，所以：

- `.github/workflows/ci.yml` 的 `heavy-tests` job 里加了
  `python scripts/ci_smoke_flwr.py`；
- 该脚本用**合成数据**（20 颗电池 / 200 行）跑 2 轮真实 ray 仿真，只验证
  链路能通（客户端在独立 actor 里各自加载数据 → 聚合 → 全局评估），
  不验证精度；
- GitHub Actions 的 runner 是 Linux、主机名 ASCII，ray 可用。

`heavy-tests` 是 `workflow_dispatch` 手动触发的，所以这条不会每次 push 都跑。
改了 app 层代码时手动点一次。

### 为什么用 Python API 而不是 `flwr run`

除了 ray 的问题，`flwr run` 的 CLI 还有两条 fab 打包校验，与本仓库的设计冲突：

1. `fab-format-version = 1` 要求 `[project].license` **引用根级文件**，
   不能是 `license = { text = "MIT" }`（已改为 `{ file = "LICENSE" }`）；
2. 要求 `[project].dependencies` 里**含 flwr**，而本仓库的 `dependencies = []`
   是刻意为之——`check_protocol.py` / `check_partition.py` 只用标准库，
   装不装 torch 都能跑。把 flwr 放进主依赖会让 `pip install -e .` 拉进
   numpy/grpcio/protobuf，破坏这条性质。

`flwr.simulation.run_simulation(...)` 走 Python API，不经过 fab 校验，
所以不受这两条限制（它仍需要 ray，所以在 CI 跑）。


### 第三个坑：`run_simulation` 根本拿不到 run_config（2026-10-24 实测）

这条与 ray 无关，是 flwr 1.41.0 的 API 事实：

```python
inspect.signature(flwr.simulation.run_simulation)
# (server_app, client_app, num_supernodes, backend_name='ray',
#  backend_config=None, enable_tf_gpu_growth=False, verbose_logging=False)
```

**没有 `run_config` 参数。** `context.run_config` 只在 `flwr run`（fab）路径下
由 `pyproject.toml` 的 `[tool.flwr.app.config]` 填充；直接调 `run_simulation`
时它是空的，于是 `context.run_config["num-server-rounds"]` 直接 KeyError。

第一次真跑 CI 就是这么红的：

```
File "src/flsoh/app/server_app.py", line 165, in main
    num_rounds = int(context.run_config["num-server-rounds"])
KeyError: 'num-server-rounds'
```

**修法**：`flsoh/app/runconfig.py` 的 `app_run_config(context)`，优先级
`context.run_config` → `FLSOH_RUN_CONFIG` 指向的 TOML → `pyproject.toml`。
第 1 与第 3 读的是同一张表，所以不会出现「官方跑 100 轮、本地跑 2 轮」。

**为什么环境变量排第二而不是第一**：`flwr run` 传进来的是显式、与本次运行
绑定的值；环境变量会留在 shell 里跨命令生效，一个忘了清掉的
`FLSOH_RUN_CONFIG` 能悄悄改掉一次正式实验的轮数，而日志里看不出来。

**冒烟脚本因此不再改 `pyproject.toml`。** 之前的做法是「临时把
`num-server-rounds = 100` 改成 2、`finally` 里改回」，但：

- `finally` 不覆盖 SIGKILL，进程被杀就留下假配置——**这件事真的发生过**：
  `pyproject.toml` 停在 `num-server-rounds = 2 / local-epochs = 2`，而
  `configs/mvp.toml` 已经是 100 / 10。官方运行时读前者、本地循环读后者，
  两边都不报错，只是跑的不是同一个实验。
- 冒烟用 16 维合成数据，`input-dim` 也必须跟着改（真实表是 1000 维），
  要改的键不止一个。

现在冒烟写一个临时 TOML 并通过 `FLSOH_RUN_CONFIG` 传进去，**被追踪的文件
一个字节都不动**。CI 里还加了一步 `git diff --exit-code` 兜底，万一以后有人
改回去会红。`pyproject.toml` 与 `configs/mvp.toml` 的漂移则由
`check_protocol.py` 的 `C-APP` 在每次 push 上检查。
