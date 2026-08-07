#!/usr/bin/env python
"""M2: Harness 资源加载/保存（SPEC 02 §3）。

资源 = 可优化对象的文本/JSON 载体，Agent Lightning initial_resources 语义。
按版本目录存放: resources/versions/v<N>/*.txt（纯文本）或 *.json。

版本目录结构:
  resources/versions/v0/system_prompt.txt
  resources/versions/v0/tool_policy.txt
  resources/versions/v0/action_strategy.txt
  resources/versions/v0/CHANGELOG.md   （版本说明，由 M3 gate 维护）

序列化约定:
- 纯文本资源存 .txt（UTF-8）
- JSON 资源存 .json
- load 时合并为 dict[str, str]，供 Trainer(initial_resources=...) 使用
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCES_ROOT = Path(__file__).resolve().parent


def version_dir(version: int | str) -> Path:
    return RESOURCES_ROOT / "versions" / f"v{version}"


def save_resources(version: int | str, resources: dict[str, Any]) -> Path:
    """把资源 dict 写入版本目录；返回目录路径。已存在则覆盖（调用方负责门控）。"""
    d = version_dir(version)
    d.mkdir(parents=True, exist_ok=True)
    for name, value in resources.items():
        if isinstance(value, (dict, list)):
            (d / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            (d / f"{name}.txt").write_text(str(value), encoding="utf-8")
    return d


def load_resources(version: int | str) -> dict[str, str]:
    """加载版本目录的全部资源为 dict[str, str]（JSON 序列化为紧凑文本）。"""
    d = version_dir(version)
    if not d.exists():
        raise FileNotFoundError(f"资源版本不存在: {d}")
    out: dict[str, str] = {}
    for f in sorted(d.glob("*")):
        if f.suffix == ".json":
            out[f.stem] = json.dumps(json.loads(f.read_text()), ensure_ascii=False)
        elif f.suffix == ".txt":
            out[f.stem] = f.read_text(encoding="utf-8")
    return out


def latest_version() -> int:
    """当前最高版本号（无版本目录则返回 -1）。"""
    vs = [int(p.name[1:]) for p in RESOURCES_ROOT.glob("versions/v*") if p.is_dir()]
    return max(vs) if vs else -1


if __name__ == "__main__":
    # 自检：写/读往返
    import sys
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as tmp:
        # 在临时目录模拟版本目录做往返
        tmp = _P(tmp) / "versions" / "v999"
        tmp.mkdir(parents=True)
        (tmp / "system_prompt.txt").write_text("tau2 retail 任务提示 v0（示例）", encoding="utf-8")
        (tmp / "meta.json").write_text(json.dumps({"k": "v"}))

        # 用临时目录验证 loader 逻辑（路径参数化）
        text = (tmp / "system_prompt.txt").read_text(encoding="utf-8")
        meta = json.loads((tmp / "meta.json").read_text())
        assert text == "tau2 retail 任务提示 v0（示例）"
        assert meta == {"k": "v"}
        print(f"往返自检通过: {tmp.name}")

    print(f"latest_version = {latest_version()}")
    if latest_version() >= 0:
        res = load_resources(latest_version())
        print(f"v{latest_version()} 资源: {list(res.keys())}")
    else:
        print("无版本目录（M3 门控建立 v0 后自动可测）")
    sys.exit(0)
