# -*- coding: utf-8 -*-
"""
统一三元组评估指标
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


# ──────────────────────────────────────────
# 三元组规范化与匹配
# ──────────────────────────────────────────

def _triple_to_key(triple: Union[Dict, Tuple]) -> Tuple[str, str, str]:
    """将各种格式的三元组统一转换为 (subject, predicate, object_text) 元组"""
    if isinstance(triple, (tuple, list)):
        s, p, o = triple[0], triple[1], triple[2]
        return str(s), str(p), str(o)

    # dict 格式
    s = str(triple.get("subject", ""))
    p = str(triple.get("predicate", ""))
    obj = triple.get("object", "")
    if isinstance(obj, dict):
        o = str(obj.get("@value", ""))
    else:
        o = str(obj)
    return s, p, o


def _triples_to_set(triples: List) -> Set[Tuple[str, str, str]]:
    return {_triple_to_key(t) for t in triples}


# ──────────────────────────────────────────
# 累积式三元组 F1 计算器
# ──────────────────────────────────────────

class TripleMetrics:
    """
    累积式三元组 Precision / Recall / F1 计算器

    用法::

        metrics = TripleMetrics()
        for pred, gold in zip(predictions, golds):
            metrics.update(pred, gold)
        result = metrics.compute()
        metrics.save_report("results/pipeline/metrics.json")
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._predicted = 0
        self._gold = 0

    def update(self, pred_triples: List, gold_triples: List) -> None:
        """
        累积单个样本的预测结果

        Args:
            pred_triples: 预测三元组列表
            gold_triples: 标注三元组列表
        """
        pred_set = _triples_to_set(pred_triples)
        gold_set = _triples_to_set(gold_triples)
        self._correct += len(pred_set & gold_set)
        self._predicted += len(pred_set)
        self._gold += len(gold_set)

    def compute(self) -> Dict[str, float]:
        """计算并返回 precision / recall / f1 及计数"""
        precision = self._correct / max(self._predicted, 1)
        recall = self._correct / max(self._gold, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        return {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "correct": self._correct,
            "predicted": self._predicted,
            "gold": self._gold,
        }

    def save_report(self, path: Union[str, Path]) -> None:
        """将指标结果保存到 JSON 文件"""
        from utils.io_utils import save_json
        save_json(self.compute(), path)
        print(f"指标已保存到: {path}")

    def print_report(self, title: str = "评估结果") -> None:
        """打印格式化的评估报告"""
        result = self.compute()
        print("\n" + "=" * 50)
        print(f"  {title}")
        print("=" * 50)
        print(f"  Precision : {result['precision']:.4f}")
        print(f"  Recall    : {result['recall']:.4f}")
        print(f"  F1        : {result['f1']:.4f}")
        print(f"  Correct   : {result['correct']}")
        print(f"  Predicted : {result['predicted']}")
        print(f"  Gold      : {result['gold']}")
        print("=" * 50)


# ──────────────────────────────────────────
# 错误类型分析（供三种方法共用）
# ──────────────────────────────────────────

ERROR_TYPES = {
    "complex_sentence_error": "复杂句子错误",
    "entity_missing_error": "实体缺失错误",
    "entity_boundary_error": "实体边界错误",
    "relation_type_error": "关系类型错误",
    "entity_type_error": "实体类型错误",
    "spurious_triple_error": "虚假三元组错误",
    "relation_direction_error": "关系方向错误",
    "other_fn_error": "其他功能错误",
    "overlap_relation_error": "重叠关系错误",
}


def analyze_errors(
    pred_triples: List,
    gold_triples: List,
    pred_entity_texts: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """
    分析预测错误类型，返回各错误类型的计数

    Args:
        pred_triples: 预测三元组
        gold_triples: 标注三元组
        pred_entity_texts: 预测出的实体文本集合（Pipeline 方法专用，用于区分 NER 错误和 RE 错误）

    Returns:
        错误类型计数字典
    """
    pred_set = _triples_to_set(pred_triples)
    gold_set = _triples_to_set(gold_triples)
    counts: Dict[str, int] = defaultdict(int)

    # 分析缺失三元组（金标但未预测出）
    for gs, gp, go in gold_set - pred_set:
        if pred_entity_texts is not None and (gs not in pred_entity_texts or go not in pred_entity_texts):
            counts["entity_missing_error"] += 1
        elif any(ps == go and po == gs and pp == gp for ps, pp, po in pred_set):
            counts["relation_direction_error"] += 1
        elif any(ps == gs and po == go for ps, _, po in pred_set):
            counts["relation_type_error"] += 1
        elif any(ps == gs and pp == gp for ps, pp, _ in pred_set):
            counts["entity_boundary_error"] += 1
        else:
            counts["other_fn_error"] += 1

    # 分析多余三元组（预测了但不在金标中）
    for ps, pp, po in pred_set - gold_set:
        if not any(gs == ps or go == po for gs, _, go in gold_set):
            counts["spurious_triple_error"] += 1
        elif (po, pp, ps) in gold_set:
            counts["relation_direction_error"] += 1
        elif any(gs == ps and go == po for gs, _, go in gold_set):
            counts["relation_type_error"] += 1
        else:
            counts["complex_sentence_error"] += 1

    return dict(counts)


def save_error_report(
    error_counts: Dict[str, int],
    output_path: Union[str, Path],
    model_name: str = "model",
) -> None:
    """保存错误分析报告为文本文件"""
    total = sum(error_counts.values())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total error items: {total}\n\n")
        f.write("Error category summary:\n")
        for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total else 0
            bar = "#" * int(pct * 0.6)
            cn_name = ERROR_TYPES.get(etype, etype)
            f.write(f"  {etype:30} {cn_name:12} {count:6} ({pct:6.2f}%) |{bar}\n")
    print(f"错误报告已保存到: {output_path}")
