#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阻止绕过正式表示链和 BindingStore 的直接语义 JSON 回写。

正式 v6.5 流程只允许三级冲突消解产生新的表示表，并通过追加式
UpdateEvent 记录更新。本门禁始终失败关闭，避免覆盖操作破坏
不可变绑定记录、证据来源与审计链。
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    """输出机器可读的拒绝原因，并以状态码 2 终止。"""

    payload = {
        "schema": "task2_direct_semantic_writeback_gate_v6_5",
        "allowed": False,
        "reason": (
            "Formal v6.5 does not permit direct semantic-JSON writeback."
        ),
        "replacement": (
            "Run 13_resolve_hierarchical_constraints.py, consume the new "
            "embedding table, and append BindingStore UpdateEvents."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
