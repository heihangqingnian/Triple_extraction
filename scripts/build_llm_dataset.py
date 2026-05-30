#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 微调/测试数据构造（在免训练寻优选定「最优 Prompt」之后运行）。

数据流::

    data/processed/llm/{train,test}_raw.jsonl   (preprocess.py 产出，Prompt 无关)
            │  --prompt <best>
            ▼
    data/processed/llm/train.json   →  LLaMA-Factory 微调（外部仓库）
    data/processed/llm/test.json    →  最终测试推理（main.py --method llm --mode predict）
    data/processed/llm/dataset_info.json  →  LLaMA-Factory 数据集注册片段

固定指令（最优 Prompt）放入 Alpaca 记录的 ``system`` 字段，不在每条数据里重复。
``instruction`` 字段承载句子文本（``### 文本\\n...``），``output`` 为三元组列表字符串。

训练侧（LLaMA-Factory chatglm2 模板 + system）与推理侧（methods/llm/infer.py 复现同一模板 + system）
必须逐字一致，二者共用 methods/llm/prompt_templates.py 的同一份定义。

Usage::

    # 假设寻优选出 schema 最优
    python scripts/build_llm_dataset.py --prompt schema --split all

    # 只构造训练集 / 只构造测试集
    python scripts/build_llm_dataset.py --prompt schema --split train
    python scripts/build_llm_dataset.py --prompt schema --split test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.llm.prompt_templates import (  # noqa: E402
    PROMPT_TYPES,
    build_input_field,
    build_instruction,
)
from utils.common import get_logger  # noqa: E402
from utils.io_utils import load_jsonl, save_json  # noqa: E402

logger = get_logger("build_llm_dataset")

# LLaMA-Factory 数据集名（写入 dataset_info.json 的键）
TRAIN_DATASET_NAME = "duie_lora_train"


def _triples_to_output(triples: List) -> str:
    """把 [[s,p,o], ...] 序列化为模型输出字符串 [("s", "p", "o"), ...]。"""
    parts = []
    for t in triples:
        s, p, o = t[0], t[1], t[2]
        if s and p and o:
            parts.append(f'("{s}", "{p}", "{o}")')
    return f"[{', '.join(parts)}]"


def _raw_to_alpaca(rec: Dict, prompt_type: str) -> Dict:
    """把 {text, triples} 转为 Alpaca 记录（固定指令置于 system 字段）。"""
    text = rec.get("text", "")
    triples = rec.get("triples", [])
    return {
        "system": build_instruction(prompt_type),   # 固定指令 = system，所有样本一致
        "instruction": build_input_field(text),     # query：### 文本\n{text}
        "input": "",
        "output": _triples_to_output(triples),
    }


def build_split(input_dir: str, output_dir: str, split: str, prompt_type: str) -> str:
    """构造单个 split 的 Alpaca 文件，返回输出路径。"""
    raw_path = os.path.join(input_dir, f"{split}_raw.jsonl")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"raw 文件不存在: {raw_path}\n"
            "请先运行：python scripts/preprocess.py --input data/raw/DuIE2.0 --method llm"
        )
    records = load_jsonl(raw_path)
    alpaca = [_raw_to_alpaca(r, prompt_type) for r in records]
    out_path = os.path.join(output_dir, f"{split}.json")
    save_json(alpaca, out_path)
    logger.info(f"[{split}] {raw_path} → {out_path}（{len(alpaca)} 条，prompt={prompt_type}）")
    return out_path


def write_dataset_info(output_dir: str, train_file: str, prompt_type: str) -> str:
    """生成 LLaMA-Factory 的 dataset_info.json 片段（Alpaca + system 列映射）。"""
    info = {
        TRAIN_DATASET_NAME: {
            "file_name": os.path.basename(train_file),
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    }
    out_path = os.path.join(output_dir, "dataset_info.json")
    save_json(info, out_path)
    logger.info(f"LLaMA-Factory dataset_info 片段已保存: {out_path}")
    return out_path


def print_llamafactory_hint(output_dir: str, prompt_type: str) -> None:
    """打印对齐的 LLaMA-Factory 训练命令提示（确保模板 = chatglm2，与推理侧一致）。"""
    print("\n" + "=" * 70)
    print("  LLaMA-Factory 微调提示（请在另一个仓库执行）")
    print("=" * 70)
    print(f"  最优 Prompt: {prompt_type}（固定指令已写入数据集 system 字段）")
    print(f"  dataset_info.json: {os.path.join(output_dir, 'dataset_info.json')}")
    print(f"  数据集名: {TRAIN_DATASET_NAME}")
    print("  关键：模板必须用 chatglm2，且训练/推理 system 一致，否则掉点。\n")
    print("  llamafactory-cli train \\")
    print("      --stage sft --do_train \\")
    print("      --model_name_or_path ZhipuAI/chatglm2-6b \\")
    print("      --finetuning_type lora --template chatglm2 \\")
    print(f"      --dataset {TRAIN_DATASET_NAME} \\")
    print(f"      --dataset_dir {output_dir} \\")
    print("      --output_dir models/chatglm2-lora \\")
    print("      --num_train_epochs 3 --per_device_train_batch_size 4 \\")
    print("      --learning_rate 2e-4 --lora_rank 8 --cutoff_len 1024")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="选定最优 Prompt 后，构造 LoRA 微调/测试 Alpaca 数据",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--prompt", required=True, choices=PROMPT_TYPES,
        help="免训练寻优选出的最优 Prompt 变体",
    )
    parser.add_argument(
        "--split", default="all", choices=["train", "test", "dev", "all"],
        help="构造哪个 split（默认 all，仅 train+test；dev 可单独指定）",
    )
    parser.add_argument(
        "--input_dir", default="data/processed/llm",
        help="raw 文件目录（含 {split}_raw.jsonl）",
    )
    parser.add_argument(
        "--output_dir", default="data/processed/llm",
        help="Alpaca 输出目录",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split == "all":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    train_file = None
    for split in splits:
        out = build_split(args.input_dir, args.output_dir, split, args.prompt)
        if split == "train":
            train_file = out

    # 训练集存在时，附带生成 LLaMA-Factory dataset_info 与命令提示
    if train_file is None:
        train_path = os.path.join(args.output_dir, "train.json")
        train_file = train_path if os.path.exists(train_path) else None
    if train_file is not None:
        write_dataset_info(args.output_dir, train_file, args.prompt)
        print_llamafactory_hint(args.output_dir, args.prompt)

    logger.info("LLM 数据构造完成")


if __name__ == "__main__":
    main()
