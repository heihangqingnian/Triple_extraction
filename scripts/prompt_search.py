#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
免训练 Prompt 寻优入口（薄 CLI）。

在不加载任何 LoRA 的前提下，用基座模型对验证集小样本运行各 Prompt 变体，按 F1 选最优 Prompt。
**绝不使用测试集**（核心逻辑见 methods/llm/prompt_search.py，含数据隔离断言）。

Usage::

    python scripts/prompt_search.py --config configs/llm_prompt_search.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.llm.prompt_search import run_prompt_search  # noqa: E402
from utils.common import load_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="免训练 Prompt 寻优（Zero-shot/Few-shot）")
    parser.add_argument(
        "--config", default="configs/llm_prompt_search.yaml",
        help="寻优配置文件（默认 configs/llm_prompt_search.yaml）",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"错误：配置文件不存在: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_yaml(str(cfg_path))
    run_prompt_search(cfg)


if __name__ == "__main__":
    main()
