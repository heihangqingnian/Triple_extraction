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

    # 零样本：写出 train.json / dev.json / test.json
    python scripts/build_llm_dataset.py --prompt schema --split all

    # 固定前缀 few-shot：写出 train_fs2.json / dev_fs2.json / test_fs2.json（不覆盖零样本产物）
    # 全数据集共用同一组 K 个示例（取自 train_raw.jsonl，自动剔除与示例文本重合的样本）
    python scripts/build_llm_dataset.py --prompt schema --split all --few_shot 2

    # 单独构造某个 split
    python scripts/build_llm_dataset.py --prompt schema --split train
    python scripts/build_llm_dataset.py --prompt schema --split dev
    python scripts/build_llm_dataset.py --prompt schema --split test
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.llm.prompt_templates import (  # noqa: E402
    PROMPT_TYPES,
    build_fewshot_block,
    build_instruction,
    build_query,
)
from utils.common import get_logger  # noqa: E402
from utils.io_utils import load_jsonl, save_json  # noqa: E402

logger = get_logger("build_llm_dataset")

# LLaMA-Factory 数据集名（写入 dataset_info.json 的键）
TRAIN_DATASET_NAME = "duie_lora_train"
EVAL_DATASET_NAME = "duie_lora_eval"


def _triples_to_output(triples: List) -> str:
    """把 [[s,p,o], ...] 序列化为模型输出字符串 [("s", "p", "o"), ...]。"""
    parts = []
    for t in triples:
        s, p, o = t[0], t[1], t[2]
        if s and p and o:
            parts.append(f'("{s}", "{p}", "{o}")')
    return f"[{', '.join(parts)}]"


def _sample_fixed_fewshot(
    source_path: str,
    k: int,
    seed: int,
) -> List[Tuple[str, List]]:
    """从 train_raw 采样 K 条作为「固定前缀」few-shot 示例，全数据集共用。

    - 仅采样含非空 text 和非空 triples 的样本
    - 用 ``seed`` 固定 RNG，保证可复现
    - 返回 [(text, triples), ...]，K=0 时返回空列表
    """
    if k <= 0:
        return []
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"few-shot 源文件不存在: {source_path}\n"
            "请先运行：python scripts/preprocess.py --input data/raw/DuIE2.0 --method llm"
        )
    pool = [
        r for r in load_jsonl(source_path)
        if r.get("text") and r.get("triples")
    ]
    if k > len(pool):
        raise ValueError(f"few-shot K={k} 超过可用样本数 {len(pool)}（source={source_path}）")
    rng = random.Random(seed)
    chosen = rng.sample(pool, k)
    return [(r["text"], r["triples"]) for r in chosen]


def _raw_to_alpaca(
    rec: Dict,
    prompt_type: str,
    fewshot_examples: List[Tuple[str, List]],
) -> Dict:
    """把 {text, triples} 转为 Alpaca 记录。

    - ``system`` = 固定指令（base/schema/cot 之一）
    - ``instruction`` = （可选 few-shot 块 + ）``### 文本\\n{text}``，由 ``build_query`` 拼接
    - few-shot 为空时退化到零样本（与 K=0 时旧行为一致）
    """
    text = rec.get("text", "")
    triples = rec.get("triples", [])
    return {
        "system": build_instruction(prompt_type),
        "instruction": build_query(text, fewshot_examples or None),
        "input": "",
        "output": _triples_to_output(triples),
    }


def build_split(
    input_dir: str,
    output_dir: str,
    split: str,
    prompt_type: str,
    fewshot_examples: List[Tuple[str, List]],
    file_suffix: str,
) -> str:
    """构造单个 split 的 Alpaca 文件，返回输出路径。

    若启用 few-shot，会自动剔除"text 与某个 few-shot 示例完全相同"的样本，
    避免训练时模型在 prompt 里直接看到自己的金标输出。
    """
    raw_path = os.path.join(input_dir, f"{split}_raw.jsonl")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"raw 文件不存在: {raw_path}\n"
            "请先运行：python scripts/preprocess.py --input data/raw/DuIE2.0 --method llm"
        )
    records = load_jsonl(raw_path)

    if fewshot_examples:
        fs_texts = {t for t, _ in fewshot_examples}
        before = len(records)
        records = [r for r in records if r.get("text") not in fs_texts]
        dropped = before - len(records)
        if dropped > 0:
            logger.info(f"[{split}] 排除了 {dropped} 条与 few-shot 示例 text 重合的样本")

    alpaca = [_raw_to_alpaca(r, prompt_type, fewshot_examples) for r in records]
    out_path = os.path.join(output_dir, f"{split}{file_suffix}.json")
    save_json(alpaca, out_path)
    logger.info(
        f"[{split}] {raw_path} → {out_path}（{len(alpaca)} 条，"
        f"prompt={prompt_type}，few_shot={len(fewshot_examples)}）"
    )
    return out_path


def write_dataset_info(
    output_dir: str,
    train_file: str,
    dev_file: str,
    prompt_type: str,
    train_dataset_name: str,
    eval_dataset_name: str,
) -> str:
    """生成 LLaMA-Factory 的 dataset_info.json 片段（Alpaca + system 列映射）。

    若 ``dev_file`` 存在，会同时注册 eval 数据集（供 LLaMA-Factory ``--val_dataset`` 使用）。
    ``prompt_type`` 仅用于日志（实际指令已写入每条记录的 system 字段）。
    """
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
    }
    info: Dict[str, Dict] = {
        train_dataset_name: {
            "file_name": os.path.basename(train_file),
            "columns": columns,
        }
    }
    if dev_file and os.path.exists(dev_file):
        info[eval_dataset_name] = {
            "file_name": os.path.basename(dev_file),
            "columns": columns,
        }

    out_path = os.path.join(output_dir, "dataset_info.json")
    save_json(info, out_path)
    registered = list(info.keys())
    logger.info(
        f"LLaMA-Factory dataset_info 片段已保存: {out_path}"
        f"（已注册 {registered}，prompt={prompt_type}）"
    )
    return out_path


def print_llamafactory_hint(
    output_dir: str,
    prompt_type: str,
    train_dataset_name: str,
    eval_dataset_name: str,
    has_dev: bool,
    fewshot_k: int,
) -> None:
    """打印对齐的 LLaMA-Factory 训练命令提示（确保模板 = chatglm2，与推理侧一致）。"""
    print("\n" + "=" * 70)
    print("  LLaMA-Factory 微调提示（请在另一个仓库执行）")
    print("=" * 70)
    fs_note = f"（含 {fewshot_k} 个固定前缀 few-shot 示例）" if fewshot_k > 0 else "（零样本）"
    print(f"  最优 Prompt: {prompt_type} {fs_note}")
    print(f"  dataset_info.json: {os.path.join(output_dir, 'dataset_info.json')}")
    print(f"  训练集名: {train_dataset_name}")
    if has_dev:
        print(f"  验证集名: {eval_dataset_name}（可作为 --val_dataset 监控过拟合）")
    print("  关键：模板必须用 chatglm2，且训练/推理 system + few-shot 块完全一致，否则掉点。\n")
    print("  llamafactory-cli train \\")
    print("      --stage sft --do_train \\")
    print("      --model_name_or_path ZhipuAI/chatglm2-6b \\")
    print("      --finetuning_type lora --template chatglm2 \\")
    print(f"      --dataset {train_dataset_name} \\")
    if has_dev:
        print(f"      --val_dataset {eval_dataset_name} \\")
    print(f"      --dataset_dir {output_dir} \\")
    print("      --output_dir models/chatglm2-lora \\")
    print("      --num_train_epochs 3 --per_device_train_batch_size 4 \\")
    print("      --learning_rate 2e-4 --lora_rank 8 --cutoff_len 1024")
    print("=" * 70)


def _print_fewshot_preview(
    examples: List[Tuple[str, List]],
    source: str,
    seed: int,
) -> None:
    """启动时打印固定前缀里采到的 few-shot 示例，便于人工核对。"""
    if not examples:
        print("[few-shot] 未启用（K=0），零样本数据集")
        return
    print("\n" + "=" * 70)
    print(f"  固定前缀 few-shot 示例（K={len(examples)}，source={source}，seed={seed}）")
    print("=" * 70)
    block = build_fewshot_block(examples)
    print(block)
    print("=" * 70 + "\n")


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
        help="构造哪个 split（默认 all，含 train+dev+test；也可单独指定）",
    )
    parser.add_argument(
        "--input_dir", default="data/processed/llm",
        help="raw 文件目录（含 {split}_raw.jsonl）",
    )
    parser.add_argument(
        "--output_dir", default="data/processed/llm",
        help="Alpaca 输出目录",
    )
    parser.add_argument(
        "--few_shot", type=int, default=0, metavar="K",
        help="固定前缀 few-shot 示例数（默认 0 = 零样本；K>0 时全数据集共用同一组示例）",
    )
    parser.add_argument(
        "--fewshot_source", default="data/processed/llm/train_raw.jsonl",
        help="few-shot 示例采样来源（默认 train_raw.jsonl，仅在 --few_shot K>0 时生效）",
    )
    parser.add_argument(
        "--fewshot_seed", type=int, default=42,
        help="few-shot 采样的随机种子（默认 42，保证可复现）",
    )
    parser.add_argument(
        "--suffix", default=None,
        help="输出文件 / 数据集名后缀（默认 K=0 无后缀，K>0 自动为 _fs{K}；显式指定可覆盖）",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── few-shot 固定前缀（K>0 时全数据集共用同一组示例）────────────────────
    fewshot_examples = _sample_fixed_fewshot(
        args.fewshot_source, args.few_shot, args.fewshot_seed,
    )
    _print_fewshot_preview(fewshot_examples, args.fewshot_source, args.fewshot_seed)

    # 输出文件后缀：自动 _fs{K}，避免覆盖零样本产物；可通过 --suffix 显式覆盖
    if args.suffix is not None:
        file_suffix = args.suffix
    elif args.few_shot > 0:
        file_suffix = f"_fs{args.few_shot}"
    else:
        file_suffix = ""

    train_dataset_name = f"{TRAIN_DATASET_NAME}{file_suffix}"
    eval_dataset_name = f"{EVAL_DATASET_NAME}{file_suffix}"

    if args.split == "all":
        splits = ["train", "dev", "test"]
    else:
        splits = [args.split]

    train_file: str | None = None
    dev_file: str | None = None
    for split in splits:
        out = build_split(
            args.input_dir, args.output_dir, split,
            args.prompt, fewshot_examples, file_suffix,
        )
        if split == "train":
            train_file = out
        elif split == "dev":
            dev_file = out

    # 训练集存在时（本次产出或磁盘上已有）才生成 dataset_info 与命令提示
    if train_file is None:
        candidate = os.path.join(args.output_dir, f"train{file_suffix}.json")
        train_file = candidate if os.path.exists(candidate) else None
    if dev_file is None:
        candidate = os.path.join(args.output_dir, f"dev{file_suffix}.json")
        dev_file = candidate if os.path.exists(candidate) else None

    if train_file is not None:
        write_dataset_info(
            args.output_dir, train_file, dev_file,
            args.prompt, train_dataset_name, eval_dataset_name,
        )
        print_llamafactory_hint(
            args.output_dir, args.prompt,
            train_dataset_name, eval_dataset_name,
            has_dev=bool(dev_file),
            fewshot_k=args.few_shot,
        )

    logger.info("LLM 数据构造完成")


if __name__ == "__main__":
    main()
