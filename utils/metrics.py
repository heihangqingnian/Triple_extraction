# -*- coding: utf-8 -*-
"""
统一三元组评估指标
"""

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
    # 实体层面错误
    "entity_boundary_error":      "实体边界切分错误（定位到但边界不准）",
    "entity_fn_error":            "实体漏抽（实体完全未被捕获）",
    "entity_fp_error":            "实体无中生有（幻觉实体）",
    # 关系层面错误
    "relation_misclassification": "关系分类错误（实体正确，关系类型错）",
    "relation_direction_error":   "关系方向反转（主宾角色颠倒）",
    # 组合配对层面错误
    "wrong_pairing_error":        "张冠李戴（无关实体被错误配对）",
    "triplet_fn_error":           "三元组完全漏报（整体缺失）",
    # 复杂场景重叠遗漏（Joint 模型特有场景）
    "seo_failure":                "一主多客遗漏 SEO（共主语多宾语场景部分遗漏）",
    "epo_failure":                "实体对多关系遗漏 EPO（同实体对多关系场景部分遗漏）",
    # 其他多余预测兜底
    "complex_sentence_error":     "复杂句子混淆（其他多余预测）",
}


def _has_boundary_overlap(entity_text: str, entity_set: Set[str]) -> bool:
    """判断 entity_text 是否与 entity_set 中某实体有子串重叠（用于边界错误检测）"""
    if not entity_text:
        return False
    for e in entity_set:
        if e and e != entity_text and (entity_text in e or e in entity_text):
            return True
    return False


def analyze_errors(
    pred_triples: List,
    gold_triples: List,
    pred_entity_texts: Optional[Set[str]] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    分析预测错误类型，返回各错误类型的计数。

    错误分类遵循以下四层体系：
      1. 实体层面：边界错误 / 漏抽 / 无中生有
      2. 关系层面：分类错误 / 方向反转
      3. 配对层面：张冠李戴 / 三元组完全漏报
      4. 复杂场景：SEO 遗漏 / EPO 遗漏

    Args:
        pred_triples: 预测三元组列表
        gold_triples: 标注三元组列表
        pred_entity_texts: 预测实体文本集合；Pipeline 传 NER 输出实体，
                           Joint 不传（自动从预测三元组中提取）
        verbose: True 时在终端打印每条错误的分类细节（用于调试验证）

    Returns:
        错误类型计数字典
    """
    pred_set = _triples_to_set(pred_triples)
    gold_set = _triples_to_set(gold_triples)
    counts: Dict[str, int] = defaultdict(int)

    # 构建所有金标实体文本集合
    gold_entities: Set[str] = {e for gs, _, go in gold_set for e in (gs, go)}

    # 若未从外部传入预测实体集（Joint 场景），从预测三元组中自动提取
    if pred_entity_texts is None:
        pred_entity_texts = {e for ps, _, po in pred_set for e in (ps, po)}

    if verbose:
        print(f"\n[analyze_errors] pred={len(pred_set)} gold={len(gold_set)} "
              f"pred_entities={len(pred_entity_texts)} gold_entities={len(gold_entities)}")

    # ── 分析缺失三元组（gold_set - pred_set）─────────────────────────────
    # 每条缺失三元组按优先级分类，归入最具体的错误类型
    for gs, gp, go in gold_set - pred_set:
        label = ""

        # 优先级 1：关系方向反转 —— 存在 (go, gp, gs) 在预测集中
        if (go, gp, gs) in pred_set:
            counts["relation_direction_error"] += 1
            label = "relation_direction_error"

        # 优先级 2：EPO 遗漏 —— 同实体对出现但关系类型错，
        #   且该实体对 (gs, go) 在 gold 中另有关系已被正确预测
        elif any(ps == gs and po == go and pp != gp for ps, pp, po in pred_set):
            sibling_predicted = any(
                (gs, gp2, go) in pred_set
                for gs2, gp2, go2 in gold_set
                if gs2 == gs and go2 == go and gp2 != gp
            )
            if sibling_predicted:
                counts["epo_failure"] += 1
                label = "epo_failure"
            else:
                counts["relation_misclassification"] += 1
                label = "relation_misclassification"

        # 优先级 3：SEO 遗漏 —— 同(主语, 关系)对出现但宾语不同，
        #   且该(gs, gp)对在 gold 中另有宾语已被正确预测
        elif any(ps == gs and pp == gp and po != go for ps, pp, po in pred_set):
            sibling_predicted = any(
                (gs, gp, go2) in pred_set
                for gs2, gp2, go2 in gold_set
                if gs2 == gs and gp2 == gp and go2 != go
            )
            if sibling_predicted:
                counts["seo_failure"] += 1
                label = "seo_failure"
            else:
                counts["triplet_fn_error"] += 1
                label = "triplet_fn_error"

        # 优先级 4：实体边界错误 —— 实体大体正确但边界不准（子串/超串匹配）
        elif _has_boundary_overlap(gs, pred_entity_texts) or \
                _has_boundary_overlap(go, pred_entity_texts):
            counts["entity_boundary_error"] += 1
            label = "entity_boundary_error"

        # 优先级 5：实体漏抽 —— 主语或宾语完全不在预测实体集中
        elif gs not in pred_entity_texts or go not in pred_entity_texts:
            counts["entity_fn_error"] += 1
            label = "entity_fn_error"

        # 优先级 6：兜底 —— 三元组整体漏报（实体存在，关系无对应）
        else:
            counts["triplet_fn_error"] += 1
            label = "triplet_fn_error"

        if verbose:
            print(f"  [MISS] ({gs}, {gp}, {go})  → {label}")

    # ── 分析多余三元组（pred_set - gold_set）─────────────────────────────
    for ps, pp, po in pred_set - gold_set:
        label = ""

        # 优先级 1：关系方向反转 —— (po, pp, ps) 在金标中
        if (po, pp, ps) in gold_set:
            counts["relation_direction_error"] += 1
            label = "relation_direction_error"

        # 优先级 2：关系分类错误 —— 实体对 (ps, po) 在金标中有对应（但关系类型错）
        elif any(gs == ps and go == po for gs, _, go in gold_set):
            counts["relation_misclassification"] += 1
            label = "relation_misclassification"

        # 优先级 3：实体无中生有 —— 主宾语均不在任何金标实体中
        elif ps not in gold_entities and po not in gold_entities:
            counts["entity_fp_error"] += 1
            label = "entity_fp_error"

        # 优先级 4：张冠李戴 —— 实体各自存在于金标，但这对组合无任何金标关系
        elif ps in gold_entities and po in gold_entities:
            counts["wrong_pairing_error"] += 1
            label = "wrong_pairing_error"

        # 优先级 5：兜底 —— 复杂句子中的其他混淆
        else:
            counts["complex_sentence_error"] += 1
            label = "complex_sentence_error"

        if verbose:
            print(f"  [EXTRA] ({ps}, {pp}, {po})  → {label}")

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
            f.write(f"  {etype:35} {cn_name:40} {count:6} ({pct:6.2f}%) |{bar}\n")
    print(f"错误报告已保存到: {output_path}")
