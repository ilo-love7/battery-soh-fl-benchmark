"""把 src/ 放进 sys.path，让测试无需安装即可 import flsoh。

刻意不依赖 `pip install -e .`：CI 里只想跑纯标准库的测试，
不想为了跑测试去解析 torch / flwr 的依赖树。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
