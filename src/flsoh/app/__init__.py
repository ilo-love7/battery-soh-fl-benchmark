"""flsoh.app —— Flower App（flwr 1.41 范式）。

对应 docs/plan.md §3.3 的骨架。这一层**只在本地跑，不进 CI**：
它依赖 flwr + torch，装不动就是装不动，没有"跳过就好了"的中间态。

两个模块的分工：

- ``client_app.py``：一轮训练/评估。把 `num-examples` 写进 MetricRecord
  是**硬要求**——`FedAvg` 默认 `weighted_by_key="num-examples"`，
  漏报不会报错，会静默退化成等权聚合（plan.md §14 坑 3）。
- ``server_app.py``：选策略、起跑、全局评估。
"""
