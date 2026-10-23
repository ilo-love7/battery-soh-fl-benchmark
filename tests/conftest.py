"""把 src/ 与 scripts/ 放进 sys.path，让测试无需安装即可 import。

刻意不依赖 `pip install -e .`：CI 里只想跑纯标准库的测试，
不想为了跑测试去解析 torch / flwr 的依赖树。

``scripts/`` 也要进去：``build_manifest.py`` 里的延续关系判据是纯函数，
值得被测试钉住。脚本不是「只给命令行用的壳」——判据住在脚本里，
测试就必须能 import 到脚本。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
