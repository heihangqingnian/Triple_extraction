# -*- coding: utf-8 -*-
"""
三元组密度分组评测 + LLM 截断/遗忘分析

按每句话包含的金标三元组数量分桶，统计各方法在不同密度下的 F1，
并专项分析 LLM 在 5 个以上三元组长句中是否出现"遗忘/截断"现象。

分桶规则：
  Bucket-1 : 金标三元组数 = 1
  Bucket-2 : 金标三元组数 = 2-3
  Bucket-3 : 金标三元组数 >= 4
  5+ 子集  : 金标三元组数 >= 5（LLM 专项分析）

不生成任何子集文件，所有统计在内存中完成。

Usage::

    python scripts/density_eval.py
    python scripts/density_eval.py --pipeline results/pipeline/predictions.jsonl \\
                                    --joint   results/joint/predictions.jsonl \\
                                    --llm     results/llm/predictions.jsonl
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import load_jsonl
from utils.metrics import TripleMetrics, parse_triple_string

BUCKETS: List[Tuple[str, int, Optional[int]]] = [
    ("1",   1, 1),
    ("2-3", 2, 3),
    ("4+",  4, None),
]
LLM_LARGE_THRESHOLD = 5  # 专项分析阈值


# ── 预测文件加载（与 fine_grained_eval.py 共用逻辑）───────────────────────

def _parse_gold_from_record(rec: Dict) -> List[Dict]:
    if "gold_triples" in rec:
        return rec["gold_triples"]
    return [
        {"subject": s, "predicate": p, "object": {"@value": o}}
        for s, p, o in parse_triple_string(rec.get("label", "[]"))
    ]


def _parse_pred_from_record(rec: Dict) -> List[Dict]:
    if "pred_triples" in rec:
        return rec["pred_triples"]
    return [
        {"subject": s, "predicate": p, "object": {"@value": o}}
        for s, p, o in parse_triple_string(rec.get("predict", "[]"))
    ]


def load_predictions(path: str) -> List[Dict]:
    """加载预测文件，返回标准化样本列表（每条含 text/pred/gold/gold_count）"""
    records = load_jsonl(path)
    seen = set()
    samples = []
    for rec in records:
        text = rec.get("text", "")
        if text in seen:
            continue
        seen.add(text)
        gold = _parse_gold_from_record(rec)
        pred = _parse_pred_from_record(rec)
        samples.append({
            "text": text,
            "pred_triples": pred,
            "gold_triples": gold,
            "gold_count": len(gold),
            "pred_count": len(pred),
        })
    return samples


# ── 密度分组评测 ──────────────────────────────────────────────────────────

def _in_bucket(count: int, lo: int, hi: Optional[int]) -> bool:
    return count >= lo and (hi is None or count <= hi)


def evaluate_by_density(samples: List[Dict]) -> Dict[str, Dict]:
    """
    按三元组密度分桶计算 F1。

    Returns:
        {bucket_label: {"count": int, "precision": float, "recall": float, "f1": float}}
    """
    bucket_metrics = {label: TripleMetrics() for label, _, _ in BUCKETS}
    bucket_counts  = defaultdict(int)

    for s in samples:
        gc = s["gold_count"]
        for label, lo, hi in BUCKETS:
            if _in_bucket(gc, lo, hi):
                bucket_metrics[label].update(s["pred_triples"], s["gold_triples"])
                bucket_counts[label] += 1
                break  # 每条样本只属于一个桶

    return {
        label: {"count": bucket_counts[label], **bucket_metrics[label].compute()}
        for label, _, _ in BUCKETS
    }


def analyze_llm_large(samples: List[Dict], threshold: int = LLM_LARGE_THRESHOLD) -> Dict:
    """
    专项分析 LLM 在高密度句（>=threshold 个三元组）上的表现。

    指标：
      - recall：召回率（越低说明遗忘越严重）
      - precision：精确率
      - f1：F1
      - avg_gold_count：平均金标三元组数
      - avg_pred_count：平均预测三元组数
      - truncation_ratio：预测数 < 金标数的样本比例（截断信号）
    """
    large = [s for s in samples if s["gold_count"] >= threshold]
    if not large:
        return {"count": 0, "note": f"无 {threshold}+ 三元组样本"}

    metrics = TripleMetrics()
    total_gold = 0
    total_pred = 0
    truncated  = 0

    for s in large:
        metrics.update(s["pred_triples"], s["gold_triples"])
        total_gold += s["gold_count"]
        total_pred += s["pred_count"]
        if s["pred_count"] < s["gold_count"]:
            truncated += 1

    result = metrics.compute()
    result.update({
        "count":           len(large),
        "avg_gold_count":  round(total_gold / len(large), 2),
        "avg_pred_count":  round(total_pred / len(large), 2),
        "truncation_ratio": round(truncated / len(large), 4),
        "truncation_signal": "明显" if truncated / len(large) > 0.4 else "不明显",
    })
    return result


# ── 结果打印 ──────────────────────────────────────────────────────────────

def _fmt(val) -> str:
    return f"{val:.4f}" if isinstance(val, float) else str(val)


def print_density_table(method_results: Dict[str, Dict[str, Dict]]) -> None:
    methods = list(method_results.keys())
    bucket_labels = [b[0] for b in BUCKETS]
    col_w = 13

    header = f"{'密度':>6} {'样本数':>6}"
    for m in methods:
        header += f"  {m[:col_w]:>{col_w}}"
    sep = "=" * len(header)

    print(f"\n{'三元组密度分组 F1 对比':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for label in bucket_labels:
        counts = [method_results[m].get(label, {}).get("count", 0) for m in methods]
        row = f"{label:>6} {counts[0]:>6}"
        for m in methods:
            r = method_results[m].get(label, {})
            row += f"  {_fmt(r.get('f1', 0.0)):>{col_w}}"
        print(row)

    print(sep)


def print_llm_truncation(llm_large_results: Dict[str, Dict]) -> None:
    print(f"\n{'LLM 高密度句（5+三元组）截断/遗忘分析':^60}")
    print("=" * 60)
    print(f"{'方法':<16} {'n':>4} {'Recall':>8} {'Precision':>10} {'F1':>8} "
          f"{'avg_gold':>9} {'avg_pred':>9} {'截断比例':>9} {'信号':>6}")
    print("-" * 60)
    for name, r in llm_large_results.items():
        if r.get("count", 0) == 0:
            print(f"{name:<16}  {'(无样本)':>50}")
            continue
        print(f"{name:<16} {r['count']:>4} "
              f"{r.get('recall', 0):>8.4f} "
              f"{r.get('precision', 0):>10.4f} "
              f"{r.get('f1', 0):>8.4f} "
              f"{r.get('avg_gold_count', 0):>9.2f} "
              f"{r.get('avg_pred_count', 0):>9.2f} "
              f"{r.get('truncation_ratio', 0):>9.4f} "
              f"{r.get('truncation_signal', '?'):>6}")
    print("=" * 60)
    print('  截断信号说明：预测三元组数 < 金标数的样本比例 > 40% 视为「明显」')


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="三元组密度分组评测 + LLM 截断分析")
    parser.add_argument("--pipeline", default="results/pipeline/predictions.jsonl")
    parser.add_argument("--joint",    default="results/joint/predictions.jsonl")
    parser.add_argument("--llm",      default="results/llm/predictions.jsonl")
    args = parser.parse_args()

    method_results:     Dict[str, Dict[str, Dict]] = {}
    llm_large_results:  Dict[str, Dict]            = {}

    # Pipeline
    if Path(args.pipeline).exists():
        print(f"加载 Pipeline: {args.pipeline}")
        s = load_predictions(args.pipeline)
        method_results["Pipeline"] = evaluate_by_density(s)
    else:
        print(f"[跳过] {args.pipeline}")

    # Joint
    if Path(args.joint).exists():
        print(f"加载 Joint: {args.joint}")
        s = load_predictions(args.joint)
        method_results["Joint"] = evaluate_by_density(s)
    else:
        print(f"[跳过] {args.joint}")

    # LLM（单 Prompt/单 LoRA 的最终测试预测）
    if Path(args.llm).exists():
        print(f"加载 LLM: {args.llm}")
        s = load_predictions(args.llm)
        method_results["LLM"] = evaluate_by_density(s)
        llm_large_results["LLM"] = analyze_llm_large(s)
    else:
        print(f"[跳过] {args.llm}")

    if not method_results:
        print("\n未找到任何预测文件，请先运行评估。")
        return

    print_density_table(method_results)

    if llm_large_results:
        print_llm_truncation(llm_large_results)
    else:
        print("\n[注] 未找到 LLM 预测文件，跳过截断分析。")


if __name__ == "__main__":
    main()
