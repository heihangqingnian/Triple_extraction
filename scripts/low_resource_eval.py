# -*- coding: utf-8 -*-
"""
低资源（Low-resource）微调对比实验

随机抽取 DUIE 训练集的 1%、5%、10% 数据，分别训练 Pipeline / Joint 模型，
评估并打印三方法 × 三比例的 F1 对比表。

LLM（ChatGLM2）训练通过外部 LlamaFactory 完成，此脚本仅生成对应采样数据文件
并打印 LlamaFactory 调用指引。

采样数据写入 data/processed/low_resource/{ratio}/{method}/，
结果写入 results/low_resource/{ratio}/{method}/metrics.json。

Usage::

    # 全量运行（pipeline + joint，1%/5%/10%）
    python scripts/low_resource_eval.py

    # 仅 pipeline，仅 1%
    python scripts/low_resource_eval.py --methods pipeline --ratios 0.01

    # 仅生成数据，不训练（用于查看采样结果或交给 LlamaFactory）
    python scripts/low_resource_eval.py --data_only
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.common import get_logger, set_seed

SEED = 42
RATIO_LABELS = {0.01: "1pct", 0.05: "5pct", 0.10: "10pct"}

logger = get_logger("low_resource")


# ── 数据采样 ──────────────────────────────────────────────────────────────

def _sample_jsonl(src: str, dst: str, ratio: float) -> int:
    """从 JSONL 文件（每行一个 JSON 对象）随机采样 ratio 比例的行"""
    with open(src, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    k = max(1, int(len(lines) * ratio))
    sampled = random.sample(lines, k)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(sampled)
    logger.info(f"采样 JSONL: {src} → {dst}  ({k}/{len(lines)} 条, ratio={ratio})")
    return k


def _sample_json_array(src: str, dst: str, ratio: float) -> int:
    """从 JSON 数组文件随机采样 ratio 比例的元素"""
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组格式: {src}")
    k = max(1, int(len(data) * ratio))
    sampled = random.sample(data, k)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    logger.info(f"采样 JSON: {src} → {dst}  ({k}/{len(data)} 条, ratio={ratio})")
    return k


def _sample_bio(src: str, dst: str, ratio: float) -> int:
    """从 BIO 标注文件（空行分隔句子）随机采样句子"""
    with open(src, "r", encoding="utf-8") as f:
        content = f.read()

    # 按空行分割为句子块
    sentences = [s.strip() for s in content.strip().split("\n\n") if s.strip()]
    k = max(1, int(len(sentences) * ratio))
    sampled = random.sample(sentences, k)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sampled) + "\n")
    logger.info(f"采样 BIO: {src} → {dst}  ({k}/{len(sentences)} 句, ratio={ratio})")
    return k


def _sample_tsv(src: str, dst: str, ratio: float) -> int:
    """从 TSV/纯文本文件随机采样行"""
    with open(src, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    k = max(1, int(len(lines) * ratio))
    sampled = random.sample(lines, k)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(sampled)
    logger.info(f"采样 TSV: {src} → {dst}  ({k}/{len(lines)} 条, ratio={ratio})")
    return k


# ── Pipeline 数据采样 ─────────────────────────────────────────────────────

def prepare_pipeline_data(ratio: float, label: str) -> str:
    """
    为 Pipeline 方法生成低资源训练数据。
    返回新的 processed_dir 路径（供临时 config 使用）。
    """
    src_base = Path("data/processed/pipeline")
    dst_base = Path(f"data/processed/low_resource/{label}/pipeline")

    # NER 训练数据（BIO 格式）
    ner_src = src_base / "train" / "train_ner.txt"
    ner_dst = dst_base / "train" / "train_ner.txt"
    if ner_src.exists():
        _sample_bio(str(ner_src), str(ner_dst), ratio)
    else:
        logger.warning(f"Pipeline NER 训练文件不存在: {ner_src}")

    # RE 训练数据
    re_src = src_base / "train" / "train_re.txt"
    re_dst = dst_base / "train" / "train_re.txt"
    if re_src.exists():
        _sample_tsv(str(re_src), str(re_dst), ratio)
    else:
        logger.warning(f"Pipeline RE 训练文件不存在: {re_src}")

    # 复制 dev/test 数据（保持评估集不变）
    for split in ("dev", "test"):
        for fname in (f"{split}_ner.txt", f"{split}_re.txt", "pipeline_test.jsonl"):
            s = src_base / split / fname
            d = dst_base / split / fname
            if s.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(str(s), str(d))

    return str(dst_base)


# ── Joint 数据采样 ────────────────────────────────────────────────────────

def prepare_joint_data(ratio: float, label: str) -> str:
    """
    为 Joint 方法生成低资源训练数据。
    返回新的 processed_dir 路径。
    """
    src_base = Path("data/processed/joint")
    dst_base = Path(f"data/processed/low_resource/{label}/joint")

    train_src = src_base / "train" / "train.json"
    train_dst = dst_base / "train" / "train.json"
    if train_src.exists():
        _sample_jsonl(str(train_src), str(train_dst), ratio)
    else:
        logger.warning(f"Joint 训练文件不存在: {train_src}")

    # 复制 dev/test
    import shutil
    for split in ("dev", "test"):
        s = src_base / split / f"{split}.json"
        d = dst_base / split / f"{split}.json"
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(s), str(d))

    return str(dst_base)


# ── LLM 数据采样（仅生成文件，不训练）────────────────────────────────────

def prepare_llm_data(ratio: float, label: str, prompt: str = "base") -> str:
    """
    为 LLM 方法生成低资源训练数据（单 Prompt 的 Alpaca JSON）。

    从 train_raw.jsonl 采样后，用选定的最优 Prompt 构造 train.json（system 字段承载固定指令），
    供 LLaMA-Factory 微调。返回新的数据目录路径。
    """
    src_base = Path("data/processed/llm")
    dst_base = Path(f"data/processed/low_resource/{label}/llm")

    raw_src = src_base / "train_raw.jsonl"
    if not raw_src.exists():
        logger.warning(f"LLM raw 训练文件不存在: {raw_src}")
        return str(dst_base)

    # 采样 raw 行
    raw_dst = dst_base / "train_raw.jsonl"
    _sample_jsonl(str(raw_src), str(raw_dst), ratio)

    # 用选定 Prompt 构造单一 train.json（复用 build_llm_dataset 的逻辑）
    from scripts.build_llm_dataset import _raw_to_alpaca
    from utils.io_utils import load_jsonl, save_json
    alpaca = [_raw_to_alpaca(r, prompt) for r in load_jsonl(str(raw_dst))]
    save_json(alpaca, str(dst_base / "train.json"))
    logger.info(f"LLM 低资源训练数据已生成: {dst_base / 'train.json'}（prompt={prompt}，{len(alpaca)} 条）")

    return str(dst_base)


# ── 临时 Config 生成 ──────────────────────────────────────────────────────

def _make_pipeline_config(label: str, processed_dir: str) -> dict:
    from utils.common import load_yaml
    cfg = load_yaml("configs/pipeline.yaml")
    cfg["data"]["processed_dir"] = processed_dir
    cfg["ner"]["checkpoint"] = f"results/low_resource/{label}/pipeline/ner_best.pt"
    cfg["re"]["checkpoint"]  = f"results/low_resource/{label}/pipeline/re_best.pt"
    cfg["output"]["dir"]         = f"results/low_resource/{label}/pipeline"
    cfg["output"]["predictions"] = f"results/low_resource/{label}/pipeline/predictions.jsonl"
    cfg["output"]["metrics"]     = f"results/low_resource/{label}/pipeline/metrics.json"
    cfg["output"]["error_report"]= f"results/low_resource/{label}/pipeline/error_report.txt"
    cfg["output"]["log"]         = f"results/low_resource/{label}/pipeline/train.log"
    return cfg


def _make_joint_config(label: str, processed_dir: str) -> dict:
    from utils.common import load_yaml
    cfg = load_yaml("configs/joint.yaml")
    cfg["data"]["processed_dir"] = processed_dir
    cfg["model"]["checkpoint"]   = f"results/low_resource/{label}/joint/best.pt"
    cfg["output"]["dir"]         = f"results/low_resource/{label}/joint"
    cfg["output"]["predictions"] = f"results/low_resource/{label}/joint/predictions.jsonl"
    cfg["output"]["metrics"]     = f"results/low_resource/{label}/joint/metrics.json"
    cfg["output"]["error_report"]= f"results/low_resource/{label}/joint/error_report.txt"
    cfg["output"]["log"]         = f"results/low_resource/{label}/joint/train.log"
    return cfg


# ── 训练与评估 ────────────────────────────────────────────────────────────

def run_pipeline_experiment(cfg: dict) -> Optional[Dict]:
    try:
        from methods.pipeline.pipeline import run
        run(cfg, "train")
        run(cfg, "evaluate")
        metrics_path = Path(cfg["output"]["metrics"])
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Pipeline 训练/评估失败: {e}")
    return None


def run_joint_experiment(cfg: dict) -> Optional[Dict]:
    try:
        from methods.joint.trainer import run
        run(cfg, "train")
        run(cfg, "evaluate")
        metrics_path = Path(cfg["output"]["metrics"])
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Joint 训练/评估失败: {e}")
    return None


# ── 结果打印 ──────────────────────────────────────────────────────────────

def print_summary(all_results: Dict[str, Dict[str, Optional[Dict]]]) -> None:
    """打印 比例 × 方法 的 F1 对比表"""
    methods = list(all_results.keys())
    ratios  = sorted({r for m in all_results.values() for r in m.keys()})

    col_w = 14
    header = f"{'比例':>6}"
    for m in methods:
        header += f"  {m[:col_w]:>{col_w}}"
    sep = "=" * len(header)

    print(f"\n{'低资源微调 F1 对比':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for ratio in ratios:
        label = RATIO_LABELS.get(ratio, str(ratio))
        row = f"{label:>6}"
        for m in methods:
            r = all_results[m].get(ratio)
            if r and "f1" in r:
                row += f"  {r['f1']:>{col_w}.4f}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)

    print(sep)


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="低资源微调对比实验")
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.01, 0.05, 0.10],
        help="训练集采样比例列表（默认: 0.01 0.05 0.10）",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["pipeline", "joint"],
        choices=["pipeline", "joint", "llm"],
        help="参与实验的方法（默认: pipeline joint）",
    )
    parser.add_argument(
        "--data_only", action="store_true",
        help="仅生成采样数据文件，不执行训练",
    )
    parser.add_argument(
        "--prompt", default="base", choices=["base", "schema", "cot"],
        help="LLM 低资源数据使用的 Prompt（应为免训练寻优选出的最优 Prompt，默认 base）",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    all_results: Dict[str, Dict[float, Optional[Dict]]] = {m: {} for m in args.methods}

    for ratio in args.ratios:
        label = RATIO_LABELS.get(ratio, f"{int(ratio*100)}pct")
        print(f"\n{'─'*50}")
        print(f"比例: {ratio*100:.0f}%  ({label})")
        print(f"{'─'*50}")

        if "pipeline" in args.methods:
            processed_dir = prepare_pipeline_data(ratio, label)
            if not args.data_only:
                cfg = _make_pipeline_config(label, processed_dir)
                os.makedirs(cfg["output"]["dir"], exist_ok=True)
                result = run_pipeline_experiment(cfg)
                all_results["pipeline"][ratio] = result
                f1 = result.get("f1", "N/A") if result else "N/A"
                print(f"  Pipeline F1: {f1}")

        if "joint" in args.methods:
            processed_dir = prepare_joint_data(ratio, label)
            if not args.data_only:
                cfg = _make_joint_config(label, processed_dir)
                os.makedirs(cfg["output"]["dir"], exist_ok=True)
                result = run_joint_experiment(cfg)
                all_results["joint"][ratio] = result
                f1 = result.get("f1", "N/A") if result else "N/A"
                print(f"  Joint   F1: {f1}")

        if "llm" in args.methods:
            llm_dir = prepare_llm_data(ratio, label, prompt=args.prompt)
            print(f"  LLM 数据已生成: {llm_dir}")
            print(f"  LLM 训练请使用 LlamaFactory，训练数据路径: {llm_dir}/train.json（prompt={args.prompt}）")

    if args.data_only:
        labels = [RATIO_LABELS.get(r, f"{int(r*100)}pct") for r in args.ratios]
        print(f"\n数据采样完成，跳过训练（--data_only 模式）")
        print(f"已生成比例: {', '.join(labels)}")
        print("数据根目录: data/processed/low_resource/<ratio>/{pipeline,joint,llm}/")
        return

    # 打印汇总表
    if any(all_results[m] for m in args.methods):
        print_summary(all_results)


if __name__ == "__main__":
    main()
