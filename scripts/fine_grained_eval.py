# -*- coding: utf-8 -*-
"""
细粒度重叠类型评测：Normal / EPO / SEO

直接读取各方法已生成的预测文件（results/*/predictions*.jsonl），
在内存中对每条样本按金标三元组分类，计算各子集的 F1，不生成子集文件。

Usage::

    python scripts/fine_grained_eval.py
    python scripts/fine_grained_eval.py --pipeline results/pipeline/predictions.jsonl \\
                                         --joint   results/joint/predictions.jsonl \\
                                         --llm_dir results/llm
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import load_jsonl
from utils.metrics import TripleMetrics, parse_triple_string

OVERLAP_TYPES = ["Normal", "SEO", "EPO"]
LLM_PROMPT_TYPES = ["base", "schema", "cot"]


# ── 重叠类型分类 ──────────────────────────────────────────────────────────

def classify_overlap(gold_triples: List[Dict]) -> str:
    """
    按优先级判断样本重叠类型：EPO > SEO > Normal

    EPO (EntityPairOverlap):   同一实体对 (subj, obj) 出现在多个不同谓词的三元组中
    SEO (SingleEntityOverlap): 某个实体（主体或客体）出现在多个三元组，且无 EPO
    Normal:                    无实体共享
    """
    pairs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    subjects: List[str] = []
    objects: List[str] = []

    for t in gold_triples:
        s = t.get("subject", "")
        obj = t.get("object", "")
        o = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
        p = t.get("predicate", "")
        pairs[(s, o)].add(p)
        subjects.append(s)
        objects.append(o)

    # EPO：同一实体对有多个不同谓词
    if any(len(ps) > 1 for ps in pairs.values()):
        return "EPO"

    # SEO：同一主体或客体出现在多个三元组中
    if len(subjects) != len(set(subjects)) or len(objects) != len(set(objects)):
        return "SEO"

    # SEO：某三元组的主体恰好是另一三元组的客体（嵌套实体）
    if set(subjects) & set(objects):
        return "SEO"

    return "Normal"


# ── 预测文件加载 ──────────────────────────────────────────────────────────

def _parse_gold_from_record(rec: Dict) -> List[Dict]:
    """从预测记录中提取金标三元组（兼容 pipeline/joint 和 LLM 两种格式）"""
    # pipeline / joint 格式：gold_triples 已是 list[dict]
    if "gold_triples" in rec:
        return rec["gold_triples"]
    # LLM 格式：label 是三元组字符串
    label_str = rec.get("label", "[]")
    return [
        {"subject": s, "predicate": p, "object": {"@value": o}}
        for s, p, o in parse_triple_string(label_str)
    ]


def _parse_pred_from_record(rec: Dict) -> List[Dict]:
    """从预测记录中提取预测三元组（兼容两种格式）"""
    if "pred_triples" in rec:
        return rec["pred_triples"]
    pred_str = rec.get("predict", "[]")
    return [
        {"subject": s, "predicate": p, "object": {"@value": o}}
        for s, p, o in parse_triple_string(pred_str)
    ]


def load_predictions(path: str) -> Dict[str, Dict]:
    """
    加载预测文件，返回 text → {pred_triples, gold_triples} 映射。
    同一文本取首条（用于 LLM 同 text 出现多 prompt_type 时）。
    """
    records = load_jsonl(path)
    result: Dict[str, Dict] = {}
    for rec in records:
        text = rec.get("text", "")
        if text and text not in result:
            result[text] = {
                "pred_triples": _parse_pred_from_record(rec),
                "gold_triples": _parse_gold_from_record(rec),
            }
    return result


# ── 子集评测 ─────────────────────────────────────────────────────────────

def evaluate_by_overlap(preds: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    按重叠类型分组，计算各子集的 P/R/F1。

    Returns:
        {overlap_type: {"count": int, "precision": float, "recall": float, "f1": float}}
    """
    subset_metrics: Dict[str, TripleMetrics] = {t: TripleMetrics() for t in OVERLAP_TYPES}
    subset_counts: Dict[str, int] = defaultdict(int)

    for text, data in preds.items():
        gold = data["gold_triples"]
        pred = data["pred_triples"]
        overlap_type = classify_overlap(gold)
        subset_metrics[overlap_type].update(pred, gold)
        subset_counts[overlap_type] += 1

    return {
        ot: {"count": subset_counts[ot], **subset_metrics[ot].compute()}
        for ot in OVERLAP_TYPES
    }


# ── 结果打印 ──────────────────────────────────────────────────────────────

def _fmt(val) -> str:
    return f"{val:.4f}" if isinstance(val, float) else str(val)


def print_results(method_results: Dict[str, Dict[str, Dict]]) -> None:
    """打印各方法 × 各重叠类型的 F1 对比表"""
    methods = list(method_results.keys())
    col_w = 14

    # 表头
    header = f"{'子集':<8} {'样本数':>6}"
    for m in methods:
        header += f"  {m:>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for ot in OVERLAP_TYPES:
        counts = [method_results[m][ot]["count"] for m in methods]
        count_str = f"{counts[0]:>6}" if counts else "     0"
        row = f"{ot:<8} {count_str}"
        for m in methods:
            r = method_results[m].get(ot, {})
            f1_str = _fmt(r.get("f1", 0.0))
            row += f"  {f1_str:>{col_w}}"
        print(row)

    print("=" * len(header))

    # 详细 P/R/F1
    print("\n详细 Precision / Recall / F1：")
    for ot in OVERLAP_TYPES:
        print(f"\n  [{ot}]")
        for m in methods:
            r = method_results[m].get(ot, {})
            print(f"    {m:<20} P={_fmt(r.get('precision', 0))}  "
                  f"R={_fmt(r.get('recall', 0))}  F1={_fmt(r.get('f1', 0))}  "
                  f"(n={r.get('count', 0)})")


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="细粒度重叠类型评测（Normal/EPO/SEO）")
    parser.add_argument("--pipeline", default="results/pipeline/predictions.jsonl",
                        help="Pipeline 预测文件路径")
    parser.add_argument("--joint", default="results/joint/predictions.jsonl",
                        help="Joint 预测文件路径")
    parser.add_argument("--llm_dir", default="results/llm",
                        help="LLM 预测文件所在目录（含 predictions_base/schema/cot.jsonl）")
    args = parser.parse_args()

    method_results: Dict[str, Dict[str, Dict]] = {}

    # Pipeline
    if Path(args.pipeline).exists():
        print(f"加载 Pipeline 预测: {args.pipeline}")
        method_results["Pipeline"] = evaluate_by_overlap(load_predictions(args.pipeline))
    else:
        print(f"[跳过] Pipeline 预测文件不存在: {args.pipeline}")

    # Joint
    if Path(args.joint).exists():
        print(f"加载 Joint 预测: {args.joint}")
        method_results["Joint"] = evaluate_by_overlap(load_predictions(args.joint))
    else:
        print(f"[跳过] Joint 预测文件不存在: {args.joint}")

    # LLM（每种 prompt 类型独立评估）
    for pt in LLM_PROMPT_TYPES:
        llm_pred = Path(args.llm_dir) / f"predictions_{pt}.jsonl"
        if llm_pred.exists():
            print(f"加载 LLM-{pt} 预测: {llm_pred}")
            method_results[f"LLM-{pt}"] = evaluate_by_overlap(load_predictions(str(llm_pred)))
        else:
            print(f"[跳过] LLM-{pt} 预测文件不存在: {llm_pred}")

    if not method_results:
        print("\n未找到任何预测文件，请先运行各方法的评估：")
        print("  python main.py --method pipeline --mode evaluate")
        print("  python main.py --method joint    --mode evaluate")
        print("  python main.py --method llm      --mode predict && python main.py --method llm --mode evaluate")
        return

    print_results(method_results)


if __name__ == "__main__":
    main()
