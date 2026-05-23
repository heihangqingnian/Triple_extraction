# -*- coding: utf-8 -*-
"""
统一三元组评估指标——适用于 Pipeline / Joint / LLM 三种方法。

公开接口一览:
  TripleMetrics         — 严格精确匹配，微平均（训练循环 / 快速验证）
  ComprehensiveMetrics  — 严格+宽松 × 微平均+宏平均（最终评估报告）
  InferenceTimer        — 推理速度计时器（samples/s、ms/sample）
  per_relation_metrics  — 按关系类型分解 P/R/F1
  count_parameters      — 统计模型总参数量 / 可训练参数量 / 冻结量
  model_size_mb         — 估算模型参数占用内存（MB）
  gpu_memory_stats      — CUDA 显存使用快照（MB）
  reset_gpu_peak_memory — 重置显存峰值统计
  analyze_errors        — 四层九类错误分类（支持返回逐条详情）
  export_error_cases    — 错误案例导出到文本文件
  save_error_report     — 错误类型汇总报告
  format_comparison_table — 多方法 Markdown 对比表格
  print_per_relation_table — 按关系类型打印 P/R/F1 表格
"""

import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════
# 一、三元组规范化与匹配原语
# ═══════════════════════════════════════════════════════════════════════

def _triple_to_key(triple: Union[Dict, Tuple]) -> Tuple[str, str, str]:
    """将各种格式的三元组统一转换为 (subject, predicate, object_text) 元组"""
    if isinstance(triple, (tuple, list)):
        return str(triple[0]), str(triple[1]), str(triple[2])
    s = str(triple.get("subject", ""))
    p = str(triple.get("predicate", ""))
    obj = triple.get("object", "")
    o = str(obj.get("@value", "")) if isinstance(obj, dict) else str(obj)
    return s, p, o


def _triples_to_set(triples: List) -> Set[Tuple[str, str, str]]:
    return {_triple_to_key(t) for t in triples}


def _is_partial_match(pred_text: str, gold_text: str) -> bool:
    """
    宽松实体匹配：任一文本为另一方的子串（含完全相等）。
    适合中文信息抽取的字符级边界容忍评估。
    """
    if not pred_text or not gold_text:
        return pred_text == gold_text
    return pred_text in gold_text or gold_text in pred_text


def _triple_partial_match(pred: Tuple[str, str, str], gold: Tuple[str, str, str]) -> bool:
    """
    三元组宽松匹配：谓词精确匹配，主宾语允许子串包含。
    与 DuIE2.0 Partial Match 评估标准一致。
    """
    ps, pp, po = pred
    gs, gp, go = gold
    return pp == gp and _is_partial_match(ps, gs) and _is_partial_match(po, go)


def _max_bipartite_match(
    pred_list: List[Tuple],
    gold_list: List[Tuple],
    match_fn,
) -> int:
    """
    最大二部图匹配（增广路 DFS），用于宽松 TP 计数，避免同一金标被多个预测重复计为正确。
    O(|pred|²×|gold|)；每句三元组数通常 <10，开销可忽略。
    """
    m = len(gold_list)
    match_gold = [-1] * m

    def _dfs(i: int, visited: List[bool]) -> bool:
        for j in range(m):
            if not visited[j] and match_fn(pred_list[i], gold_list[j]):
                visited[j] = True
                if match_gold[j] == -1 or _dfs(match_gold[j], visited):
                    match_gold[j] = i
                    return True
        return False

    tp = 0
    for i in range(len(pred_list)):
        visited = [False] * m
        if _dfs(i, visited):
            tp += 1
    return tp


def _compute_prf(correct: int, predicted: int, gold: int) -> Tuple[float, float, float]:
    """由 TP/PP/GP 计算 Precision / Recall / F1"""
    p = correct / max(predicted, 1)
    r = correct / max(gold, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return p, r, f1


# ═══════════════════════════════════════════════════════════════════════
# 二、严格精确匹配·微平均计算器（训练循环首选）
# ═══════════════════════════════════════════════════════════════════════

class TripleMetrics:
    """
    三元组严格精确匹配，跨样本累积计数（Micro-averaged P/R/F1）。
    训练验证循环、快速评估的首选；与旧接口完全兼容。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct = self._predicted = self._gold = 0

    def update(self, pred_triples: List, gold_triples: List) -> None:
        pred_set = _triples_to_set(pred_triples)
        gold_set = _triples_to_set(gold_triples)
        self._correct   += len(pred_set & gold_set)
        self._predicted += len(pred_set)
        self._gold      += len(gold_set)

    def compute(self) -> Dict[str, Any]:
        p, r, f1 = _compute_prf(self._correct, self._predicted, self._gold)
        return {
            "precision": round(p, 6),
            "recall":    round(r, 6),
            "f1":        round(f1, 6),
            "correct":   self._correct,
            "predicted": self._predicted,
            "gold":      self._gold,
        }

    def save_report(self, path: Union[str, Path]) -> None:
        from utils.io_utils import save_json
        save_json(self.compute(), path)
        print(f"指标已保存到: {path}")

    def print_report(self, title: str = "评估结果") -> None:
        r = self.compute()
        print("\n" + "=" * 50)
        print(f"  {title}")
        print("=" * 50)
        print(f"  Precision : {r['precision']:.4f}")
        print(f"  Recall    : {r['recall']:.4f}")
        print(f"  F1        : {r['f1']:.4f}")
        print(f"  Correct   : {r['correct']}")
        print(f"  Predicted : {r['predicted']}")
        print(f"  Gold      : {r['gold']}")
        print("=" * 50)


# ═══════════════════════════════════════════════════════════════════════
# 三、综合评估计算器（严格+宽松 × 微平均+宏平均）
# ═══════════════════════════════════════════════════════════════════════

class ComprehensiveMetrics:
    """
    四维三元组评估，适合最终测试报告。

    strict_micro  — 严格精确匹配，跨样本微平均（等价于 TripleMetrics）
    relaxed_micro — 宽松匹配（谓词精确 + 主宾语子串容忍），跨样本微平均
    strict_macro  — 严格匹配，逐样本 F1 后宏平均（mean/median/std）
    relaxed_macro — 宽松匹配，逐样本 F1 后宏平均

    宏平均规则：
      - 预测和金标均为空 → F1=1（完美空集）
      - 仅一方为空 → F1=0
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._s_correct = self._s_predicted = self._s_gold = 0
        self._r_correct = self._r_predicted = self._r_gold = 0
        self._per_strict_f1:  List[float] = []
        self._per_relaxed_f1: List[float] = []
        self._num_samples = 0

    def update(self, pred_triples: List, gold_triples: List) -> None:
        pred_set  = _triples_to_set(pred_triples)
        gold_set  = _triples_to_set(gold_triples)
        pred_list = list(pred_set)
        gold_list = list(gold_set)

        # 严格匹配
        s_tp = len(pred_set & gold_set)
        self._s_correct   += s_tp
        self._s_predicted += len(pred_set)
        self._s_gold      += len(gold_set)
        _, _, sf1 = _compute_prf(s_tp, len(pred_set), len(gold_set))
        self._per_strict_f1.append(sf1)

        # 宽松匹配（最大二部图匹配，避免重复计 TP）
        r_tp = _max_bipartite_match(pred_list, gold_list, _triple_partial_match)
        self._r_correct   += r_tp
        self._r_predicted += len(pred_set)
        self._r_gold      += len(gold_set)
        _, _, rf1 = _compute_prf(r_tp, len(pred_set), len(gold_set))
        self._per_relaxed_f1.append(rf1)

        self._num_samples += 1

    def compute(self) -> Dict[str, Any]:
        sp, sr, sf1 = _compute_prf(self._s_correct, self._s_predicted, self._s_gold)
        rp, rr, rf1 = _compute_prf(self._r_correct, self._r_predicted, self._r_gold)

        def _stat(lst: List[float]) -> Dict[str, float]:
            if not lst:
                return {"mean": 0.0, "median": 0.0, "std": 0.0}
            return {
                "mean":   round(sum(lst) / len(lst), 6),
                "median": round(statistics.median(lst), 6),
                "std":    round(statistics.pstdev(lst), 6),
            }

        return {
            "strict_micro": {
                "precision": round(sp, 6), "recall": round(sr, 6), "f1": round(sf1, 6),
                "correct": self._s_correct, "predicted": self._s_predicted, "gold": self._s_gold,
            },
            "relaxed_micro": {
                "precision": round(rp, 6), "recall": round(rr, 6), "f1": round(rf1, 6),
                "correct": self._r_correct, "predicted": self._r_predicted, "gold": self._r_gold,
            },
            "strict_macro":  _stat(self._per_strict_f1),
            "relaxed_macro": _stat(self._per_relaxed_f1),
            "num_samples": self._num_samples,
        }

    def print_report(self, title: str = "综合评估结果") -> None:
        d = self.compute()
        sm, rm = d["strict_micro"], d["relaxed_micro"]
        sf, rf = d["strict_macro"], d["relaxed_macro"]
        W = 34
        print("\n" + "=" * 70)
        print(f"  {title}  （样本数={d['num_samples']}）")
        print("=" * 70)
        print(f"  {'指标':{W}} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*64}")
        print(f"  {'严格匹配·微平均 (Strict Micro)':{W}} "
              f"{sm['precision']:>10.4f} {sm['recall']:>10.4f} {sm['f1']:>10.4f}")
        print(f"  {'宽松匹配·微平均 (Relaxed Micro)':{W}} "
              f"{rm['precision']:>10.4f} {rm['recall']:>10.4f} {rm['f1']:>10.4f}")
        print(f"  {'-'*64}")
        print(f"  {'指标':{W}} {'Mean F1':>10} {'Median F1':>10} {'Std':>10}")
        print(f"  {'-'*64}")
        print(f"  {'严格匹配·宏平均 (Strict Macro)':{W}} "
              f"{sf['mean']:>10.4f} {sf['median']:>10.4f} {sf['std']:>10.4f}")
        print(f"  {'宽松匹配·宏平均 (Relaxed Macro)':{W}} "
              f"{rf['mean']:>10.4f} {rf['median']:>10.4f} {rf['std']:>10.4f}")
        print(f"  {'-'*64}")
        print(f"  TP(严格)={sm['correct']}  TP(宽松)={rm['correct']}  "
              f"PP={sm['predicted']}  GP={sm['gold']}")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# 四、按关系类型分解 P/R/F1
# ═══════════════════════════════════════════════════════════════════════

def per_relation_metrics(
    pred_triples_list: List[List],
    gold_triples_list: List[List],
) -> Dict[str, Dict[str, Any]]:
    """
    按关系类型分别计算严格匹配微平均 P/R/F1。

    Args:
        pred_triples_list: 外层按样本，内层为该样本的预测三元组列表。
        gold_triples_list: 外层按样本，内层为该样本的金标三元组列表。

    Returns:
        dict — 键为关系名（含 "__overall__" 汇总行），值为
               {"precision", "recall", "f1", "support", "predicted", "correct"}
    """
    rel_correct:   Dict[str, int] = defaultdict(int)
    rel_predicted: Dict[str, int] = defaultdict(int)
    rel_gold:      Dict[str, int] = defaultdict(int)

    for pred_triples, gold_triples in zip(pred_triples_list, gold_triples_list):
        pred_set = _triples_to_set(pred_triples)
        gold_set = _triples_to_set(gold_triples)
        for t in pred_set:
            rel_predicted[t[1]] += 1
        for t in gold_set:
            rel_gold[t[1]] += 1
        for t in pred_set & gold_set:
            rel_correct[t[1]] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for rel in sorted(set(rel_predicted) | set(rel_gold)):
        p, r, f1 = _compute_prf(rel_correct[rel], rel_predicted[rel], rel_gold[rel])
        result[rel] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "support": rel_gold[rel], "predicted": rel_predicted[rel], "correct": rel_correct[rel],
        }

    tot_c = sum(rel_correct.values())
    tot_p = sum(rel_predicted.values())
    tot_g = sum(rel_gold.values())
    p, r, f1 = _compute_prf(tot_c, tot_p, tot_g)
    result["__overall__"] = {
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "support": tot_g, "predicted": tot_p, "correct": tot_c,
    }
    return result


def print_per_relation_table(rel_metrics: Dict[str, Dict], top_n: int = 20) -> None:
    """
    打印按关系类型排列的 P/R/F1 表格（按 gold support 降序，最多 top_n 行）。
    "__overall__" 汇总行始终显示在末尾。
    """
    rows = [(k, v) for k, v in rel_metrics.items() if k != "__overall__"]
    rows.sort(key=lambda x: -x[1]["support"])
    rows = rows[:top_n]

    header = f"  {'关系类型':<22} {'P':>8} {'R':>8} {'F1':>8} {'Support':>9} {'Pred':>7} {'Correct':>8}"
    sep    = f"  {'-'*72}"
    print(header)
    print(sep)
    for rel, m in rows:
        print(f"  {rel:<22} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['support']:>9} {m['predicted']:>7} {m['correct']:>8}")
    if "__overall__" in rel_metrics:
        ov = rel_metrics["__overall__"]
        print(sep)
        print(f"  {'Overall':<22} {ov['precision']:>8.4f} {ov['recall']:>8.4f} {ov['f1']:>8.4f} "
              f"{ov['support']:>9} {ov['predicted']:>7} {ov['correct']:>8}")


# ═══════════════════════════════════════════════════════════════════════
# 五、模型参数量 & 内存 & 显存
# ═══════════════════════════════════════════════════════════════════════

def count_parameters(model) -> Dict[str, int]:
    """
    统计模型参数量。

    Returns:
        {"total": int, "trainable": int, "frozen": int}
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def model_size_mb(model) -> float:
    """
    估算模型参数 + 缓冲区占用内存（MB）。
    实际显存还包含激活值和优化器状态，本函数仅统计权重本身，供参考。
    """
    nbytes = (
        sum(p.numel() * p.element_size() for p in model.parameters())
        + sum(b.numel() * b.element_size() for b in model.buffers())
    )
    return round(nbytes / 1024 / 1024, 2)


def gpu_memory_stats(device=None) -> Dict[str, Any]:
    """
    返回当前 CUDA 设备的显存使用情况（MB）。
    CUDA 不可用时返回 {"available": False}。

    Returns:
        {available, allocated_mb, reserved_mb, peak_allocated_mb, peak_reserved_mb}
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        dev = device if device is not None else torch.cuda.current_device()
        return {
            "available":         True,
            "allocated_mb":      round(torch.cuda.memory_allocated(dev)     / 1024 ** 2, 2),
            "reserved_mb":       round(torch.cuda.memory_reserved(dev)      / 1024 ** 2, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(dev) / 1024 ** 2, 2),
            "peak_reserved_mb":  round(torch.cuda.max_memory_reserved(dev)  / 1024 ** 2, 2),
        }
    except Exception:
        return {"available": False}


def reset_gpu_peak_memory(device=None) -> None:
    """重置 CUDA 显存峰值统计，令下一次 gpu_memory_stats 的 peak_* 从当前值重新计量。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# 六、推理速度计时器
# ═══════════════════════════════════════════════════════════════════════

class InferenceTimer:
    """
    推理速度计时器，支持 start/stop 和直接 update 两种使用方式。

    示例::

        timer = InferenceTimer()
        for sample in test_set:
            timer.start()
            result = model.predict(sample)
            timer.stop(num_samples=1, num_triples=len(result))

        speed = timer.compute()
        print(speed["samples_per_sec"], speed["avg_latency_ms"])
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total_time  = 0.0
        self._num_samples = 0
        self._num_triples = 0
        self._t_start: Optional[float] = None

    def start(self) -> None:
        """开始计时（CUDA 环境请在 torch.no_grad() 外调用，使计时更准确）"""
        self._t_start = time.perf_counter()

    def stop(self, num_samples: int = 1, num_triples: int = 0) -> float:
        """结束计时并累积，返回本次耗时（秒）"""
        assert self._t_start is not None, "必须先调用 start()"
        elapsed = time.perf_counter() - self._t_start
        self._t_start = None
        self._total_time  += elapsed
        self._num_samples += num_samples
        self._num_triples += num_triples
        return elapsed

    def update(self, elapsed: float, num_samples: int = 1, num_triples: int = 0) -> None:
        """直接传入外部测量的耗时（秒），适合已有独立计时逻辑的场景"""
        self._total_time  += elapsed
        self._num_samples += num_samples
        self._num_triples += num_triples

    def compute(self) -> Dict[str, Any]:
        t = self._total_time
        n = max(self._num_samples, 1)
        return {
            "total_time_s":    round(t, 3),
            "num_samples":     self._num_samples,
            "num_triples":     self._num_triples,
            "samples_per_sec": round(n / max(t, 1e-9), 2),
            "triples_per_sec": round(self._num_triples / max(t, 1e-9), 2),
            "avg_latency_ms":  round(t / n * 1000, 2),
        }


# ═══════════════════════════════════════════════════════════════════════
# 七、错误类型分析（供三种方法共用）
# ═══════════════════════════════════════════════════════════════════════

ERROR_TYPES = {
    # 实体层面
    "entity_boundary_error":      "实体边界切分错误（定位到但边界不准）",
    "entity_fn_error":            "实体漏抽（实体完全未被捕获）",
    "entity_fp_error":            "实体无中生有（幻觉实体）",
    # 关系层面
    "relation_misclassification": "关系分类错误（实体正确，关系类型错）",
    "relation_direction_error":   "关系方向反转（主宾角色颠倒）",
    # 配对层面
    "wrong_pairing_error":        "张冠李戴（无关实体被错误配对）",
    "triplet_fn_error":           "三元组完全漏报（整体缺失）",
    # 复杂场景（Joint 模型特有场景）
    "seo_failure":                "一主多客遗漏 SEO（共主语多宾语场景部分遗漏）",
    "epo_failure":                "实体对多关系遗漏 EPO（同实体对多关系场景部分遗漏）",
    # 兜底
    "complex_sentence_error":     "复杂句子混淆（其他多余预测）",
}


def _has_boundary_overlap(entity_text: str, entity_set: Set[str]) -> bool:
    """判断 entity_text 是否与 entity_set 中某实体有子串重叠（用于边界错误检测）"""
    if not entity_text:
        return False
    return any(e and e != entity_text and (entity_text in e or e in entity_text)
               for e in entity_set)


def analyze_errors(
    pred_triples: List,
    gold_triples: List,
    pred_entity_texts: Optional[Set[str]] = None,
    verbose: bool = False,
    return_detail: bool = False,
) -> Union[Dict[str, int], Tuple[Dict[str, int], List[Dict]]]:
    """
    分析预测错误类型，返回各错误类型的计数。

    错误分类遵循以下四层体系：
      1. 实体层面：边界错误 / 漏抽 / 无中生有
      2. 关系层面：分类错误 / 方向反转
      3. 配对层面：张冠李戴 / 三元组完全漏报
      4. 复杂场景：SEO 遗漏 / EPO 遗漏

    Args:
        pred_triples:      预测三元组列表
        gold_triples:      标注三元组列表
        pred_entity_texts: 预测实体文本集合；Pipeline 传 NER 输出实体，
                           Joint 不传（自动从预测三元组中提取）
        verbose:           True 时在终端打印每条错误的分类细节
        return_detail:     True 时同时返回逐条详情列表；
                           详情格式: {"direction": "miss"/"extra",
                                      "triple": (s, p, o), "error_type": str}

    Returns:
        return_detail=False（默认）: 错误类型计数字典
        return_detail=True: (计数字典, 详情列表)
    """
    pred_set = _triples_to_set(pred_triples)
    gold_set = _triples_to_set(gold_triples)
    counts: Dict[str, int] = defaultdict(int)
    detail_list: List[Dict] = []

    gold_entities: Set[str] = {e for gs, _, go in gold_set for e in (gs, go)}
    if pred_entity_texts is None:
        pred_entity_texts = {e for ps, _, po in pred_set for e in (ps, po)}

    if verbose:
        print(f"\n[analyze_errors] pred={len(pred_set)} gold={len(gold_set)} "
              f"pred_entities={len(pred_entity_texts)} gold_entities={len(gold_entities)}")

    # ── 缺失三元组（gold - pred）────────────────────────────────────
    for gs, gp, go in gold_set - pred_set:
        if (go, gp, gs) in pred_set:
            label = "relation_direction_error"

        elif any(ps == gs and po == go and pp != gp for ps, pp, po in pred_set):
            sibling_predicted = any(
                (gs, gp2, go) in pred_set
                for gs2, gp2, go2 in gold_set
                if gs2 == gs and go2 == go and gp2 != gp
            )
            label = "epo_failure" if sibling_predicted else "relation_misclassification"

        elif any(ps == gs and pp == gp and po != go for ps, pp, po in pred_set):
            sibling_predicted = any(
                (gs, gp, go2) in pred_set
                for gs2, gp2, go2 in gold_set
                if gs2 == gs and gp2 == gp and go2 != go
            )
            label = "seo_failure" if sibling_predicted else "triplet_fn_error"

        elif _has_boundary_overlap(gs, pred_entity_texts) or \
                _has_boundary_overlap(go, pred_entity_texts):
            label = "entity_boundary_error"

        elif gs not in pred_entity_texts or go not in pred_entity_texts:
            label = "entity_fn_error"

        else:
            label = "triplet_fn_error"

        counts[label] += 1
        if verbose:
            print(f"  [MISS]  ({gs}, {gp}, {go})  → {label}")
        if return_detail:
            detail_list.append({"direction": "miss", "triple": (gs, gp, go), "error_type": label})

    # ── 多余三元组（pred - gold）────────────────────────────────────
    for ps, pp, po in pred_set - gold_set:
        if (po, pp, ps) in gold_set:
            label = "relation_direction_error"

        elif any(gs == ps and go == po for gs, _, go in gold_set):
            label = "relation_misclassification"

        elif ps not in gold_entities and po not in gold_entities:
            label = "entity_fp_error"

        elif ps in gold_entities and po in gold_entities:
            label = "wrong_pairing_error"

        else:
            label = "complex_sentence_error"

        counts[label] += 1
        if verbose:
            print(f"  [EXTRA] ({ps}, {pp}, {po})  → {label}")
        if return_detail:
            detail_list.append({"direction": "extra", "triple": (ps, pp, po), "error_type": label})

    if return_detail:
        return dict(counts), detail_list
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════
# 八、错误案例导出
# ═══════════════════════════════════════════════════════════════════════

def export_error_cases(
    sample_outputs: List[Dict],
    output_path: Union[str, Path],
    error_type: Optional[str] = None,
    max_cases: int = 200,
    pred_key: str = "pred_triples",
    gold_key: str = "gold_triples",
    text_key: str = "text",
) -> int:
    """
    将预测有误的样本导出为可读文本文件，便于人工分析。

    Args:
        sample_outputs: 每条样本字典，需含 text / pred_triples / gold_triples 字段。
        output_path:    输出文件路径（建议 .txt）。
        error_type:     若指定，只导出含该错误类型的样本
                        （合法值见 ERROR_TYPES，例如 "entity_fn_error"）。
        max_cases:      最多导出样本数。
        pred_key / gold_key / text_key: 字典中对应字段名（兼容不同输出格式）。

    Returns:
        实际导出的案例数。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with open(path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(sample_outputs):
            if exported >= max_cases:
                break

            text         = sample.get(text_key, "")
            pred_triples = sample.get(pred_key, [])
            gold_triples = sample.get(gold_key, [])

            pred_set = _triples_to_set(pred_triples)
            gold_set = _triples_to_set(gold_triples)

            if pred_set == gold_set:
                continue

            # 若指定错误类型，用 analyze_errors 过滤
            if error_type is not None:
                cnt, details = analyze_errors(pred_triples, gold_triples, return_detail=True)
                if cnt.get(error_type, 0) == 0:
                    continue
            else:
                _, details = analyze_errors(pred_triples, gold_triples, return_detail=True)

            # 建立 triple → error_type 查找表
            miss_labels:  Dict[Tuple, str] = {}
            extra_labels: Dict[Tuple, str] = {}
            for d in details:
                if d["direction"] == "miss":
                    miss_labels[d["triple"]] = d["error_type"]
                else:
                    extra_labels[d["triple"]] = d["error_type"]

            f.write(f"{'─' * 60}\n")
            f.write(f"样本 #{idx}\n")
            f.write(f"文本: {text}\n")

            correct = pred_set & gold_set
            if correct:
                f.write(f"✓ 正确 ({len(correct)}):\n")
                for t in sorted(correct):
                    f.write(f"    {t[0]} | {t[1]} | {t[2]}\n")

            extra = pred_set - gold_set
            if extra:
                f.write(f"✗ 多余预测 ({len(extra)}):\n")
                for t in sorted(extra):
                    lbl = extra_labels.get(t, "?")
                    f.write(f"    {t[0]} | {t[1]} | {t[2]}   [{lbl}]\n")

            missed = gold_set - pred_set
            if missed:
                f.write(f"✗ 漏报金标 ({len(missed)}):\n")
                for t in sorted(missed):
                    lbl = miss_labels.get(t, "?")
                    f.write(f"    {t[0]} | {t[1]} | {t[2]}   [{lbl}]\n")

            f.write("\n")
            exported += 1

    print(f"错误案例已导出: {output_path}  （共 {exported} 条）")
    return exported


# ═══════════════════════════════════════════════════════════════════════
# 九、跨方法对比表格
# ═══════════════════════════════════════════════════════════════════════

def format_comparison_table(method_results: Dict[str, Dict]) -> str:
    """
    生成多方法对比表格（Markdown 格式），方便写入论文或报告。

    Args:
        method_results: {method_name: ComprehensiveMetrics.compute() 返回的字典}

    Returns:
        Markdown 格式的对比表格字符串
    """
    header = ("| 方法 | S-Micro-P | S-Micro-R | S-Micro-F1 | "
              "R-Micro-F1 | S-Macro-F1 | R-Macro-F1 |")
    sep    = "|------|---:|---:|---:|---:|---:|---:|"
    rows   = [header, sep]
    for method, result in method_results.items():
        sm  = result.get("strict_micro",  {})
        rm  = result.get("relaxed_micro", {})
        smf = result.get("strict_macro",  {})
        rmf = result.get("relaxed_macro", {})
        rows.append(
            f"| {method} "
            f"| {sm.get('precision', 0):.4f} "
            f"| {sm.get('recall', 0):.4f} "
            f"| {sm.get('f1', 0):.4f} "
            f"| {rm.get('f1', 0):.4f} "
            f"| {smf.get('mean', 0):.4f} "
            f"| {rmf.get('mean', 0):.4f} |"
        )
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════
# 十、错误报告保存
# ═══════════════════════════════════════════════════════════════════════

def save_error_report(
    error_counts: Dict[str, int],
    output_path: Union[str, Path],
    model_name: str = "model",
) -> None:
    """保存错误类型汇总报告为文本文件"""
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
