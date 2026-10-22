# `data/` 放置说明

本目录**不进版本库**（见 `.gitignore` 的 `data/` 规则），只保留这份说明。

原始数据集体积大、有各自的许可条款，需要自行获取：

| 数据集 | 来源 | 许可要点 |
|---|---|---|
| MIT-Stanford（Severson et al. 2019） | https://data.matr.io/ | 学术用途免费，商用需联系 |
| NASA PCoE | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | 美国政府作品，公有领域 |
| CALCE UMD | http://www.calce.umd.edu/battery-data.html | 学术用途，引用出处 |

## 期望的目录形状

```text
data/
├── README.md              # 本文件
└── mit-stanford/
    ├── cells.txt          # 一行一个电池 ID，与 src/flsoh/partition.py 的占位 ID 对应
    ├── raw/               # 原始 .mat / .csv，按电池一个文件
    └── features/          # 由 src/flsoh/data.py 生成的 ΔQ(V) 特征缓存
```

## 为什么 `cells.txt` 是独立的

`scripts/check_partition.py --cells-file data/mit-stanford/cells.txt` 要用它。

占位 ID（`cell-000`…`cell-123`）只能验证"划分逻辑没有泄漏"，不能验证
"真实的那 124 颗电池没有泄漏"。真实电池 ID 的拼写、数量、是否与论文一致，
只能靠清单对账——这就是 `P-UNKNOWN` 检查码存在的理由。

## 不要把数据提交上来

`data/` 在 `.gitignore` 里。如果发现 `git status` 出现了数据文件，先确认
许可条款允许再分发，再手动 `git add -f`。
