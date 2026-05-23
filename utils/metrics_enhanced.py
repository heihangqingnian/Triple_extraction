# -*- coding: utf-8 -*-
"""
增强指标计算工具 - 用于消融实验
依赖 utils/metrics.py 的通用功能，专注于消融实验特定指标
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from utils.metrics import InferenceTimer, PerformanceMonitor as BasePerformanceMonitor


class EnhancedNERMetrics:
    """增强的 NER 指标计算类（Pipeline 消融实验）"""

    def __init__(self):
        self.reset()
        self._perf_monitor = BasePerformanceMonitor()

    def reset(self):
        """重置所有计数器"""
        # 基础计数
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        # 按实体类型统计
        self.by_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0}
        )

        # 错误分析
        self.boundary_errors = 0  # 类型正确但边界错误
        self.type_errors = 0      # 边界正确但类型错误
        self.partial_matches = 0  # 部分匹配（边界有交集）
        self.entity_lengths = []  # 实体长度分布

    def update(
        self,
        pred_entities: List[Dict],
        gold_entities: List[Dict],
    ):
        """
        更新指标

        Args:
            pred_entities: 预测实体列表
            gold_entities: 真实实体列表
        """
        # 转换为集合便于查找
        pred_set = self._entities_to_set(pred_entities)
        gold_set = self._entities_to_set(gold_entities)

        # 基础计数
        self.true_positives += len(pred_set & gold_set)
        self.false_positives += len(pred_set - gold_set)
        self.false_negatives += len(gold_set - pred_set)

        # 按类型统计
        for e in pred_entities:
            e_type = e["type"]
            if (e["start"], e["end"], e["text"]) in gold_set:
                self.by_type[e_type]["tp"] += 1
            else:
                self.by_type[e_type]["fp"] += 1

        for e in gold_entities:
            e_type = e["type"]
            if (e["start"], e["end"], e["text"]) not in pred_set:
                self.by_type[e_type]["fn"] += 1
            # 记录实体长度
            self.entity_lengths.append(e["end"] - e["start"] + 1)

        # 错误分析
        self._analyze_errors(pred_entities, gold_entities)

    def _entities_to_set(self, entities: List[Dict]) -> set:
        """将实体列表转换为集合"""
        return {(e["start"], e["end"], e["text"]) for e in entities}

    def _analyze_errors(
        self, pred_entities: List[Dict], gold_entities: List[Dict]
    ):
        """分析错误类型"""
        pred_by_text = {e["text"]: e for e in pred_entities}
        gold_by_text = {e["text"]: e for e in gold_entities}

        # 查找类型错误（文本相同但边界或类型不同）
        for text, e_pred in pred_by_text.items():
            if text in gold_by_text:
                e_gold = gold_by_text[text]
                if e_pred["type"] == e_gold["type"]:
                    # 类型相同但边界不同 → 边界错误
                    if e_pred["start"] != e_gold["start"] or e_pred["end"] != e_gold["end"]:
                        self.boundary_errors += 1
                        # 检查是否有交集
                        if self._has_overlap(e_pred, e_gold):
                            self.partial_matches += 1
                else:
                    # 类型错误
                    self.type_errors += 1

    def _has_overlap(self, e1: Dict, e2: Dict) -> bool:
        """检查两个实体是否有边界交集"""
        return not (e1["end"] < e2["start"] or e2["end"] < e1["start"])

    def get_basic_metrics(self) -> Dict[str, float]:
        """获取基础指标"""
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    def get_by_type_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取按实体类型的指标"""
        results = {}
        for e_type, counts in self.by_type.items():
            precision = (
                counts["tp"] / (counts["tp"] + counts["fp"])
                if (counts["tp"] + counts["fp"]) > 0
                else 0.0
            )
            recall = (
                counts["tp"] / (counts["tp"] + counts["fn"])
                if (counts["tp"] + counts["fn"]) > 0
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            results[e_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": counts["tp"] + counts["fn"],
            }
        return results

    def get_error_metrics(self) -> Dict[str, float]:
        """获取错误分析指标"""
        total_errors = self.false_positives + self.false_negatives
        return {
            "boundary_error_rate": (
                self.boundary_errors / total_errors if total_errors > 0 else 0.0
            ),
            "type_error_rate": (
                self.type_errors / total_errors if total_errors > 0 else 0.0
            ),
            "partial_match_rate": (
                self.partial_matches / total_errors if total_errors > 0 else 0.0
            ),
            "boundary_errors": self.boundary_errors,
            "type_errors": self.type_errors,
            "partial_matches": self.partial_matches,
        }


class EnhancedTripleMetrics:
    """增强的三元组指标计算类（CasRel 消融实验）

    专注于消融实验特定指标，基础指标使用 utils/metrics.py 的 TripleMetrics
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有计数器"""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_samples = 0

        # 按关系类型统计
        self.by_relation: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0}
        )

        # 按部分统计
        self.subject_correct = 0  # 主语正确
        self.object_correct = 0   # 宾语正确
        self.relation_correct = 0  # 关系正确

        # 重叠关系指标（消融实验重点关注）
        self.overlapping_samples = 0
        self.overlapping_tp = 0
        self.overlapping_fp = 0
        self.overlapping_fn = 0
        self.subject_sharing_tp = 0
        self.object_sharing_tp = 0
        self.nested_entity_tp = 0

        # 错误分析
        self.subject_errors = 0
        self.object_errors = 0
        self.relation_errors = 0

    def update(
        self,
        pred_triples: List[Tuple],
        gold_triples: List[Tuple],
    ):
        """
        更新指标

        Args:
            pred_triples: 预测三元组列表
            gold_triples: 真实三元组列表
        """
        self.total_samples += 1
        pred_set = set(pred_triples)
        gold_set = set(gold_triples)

        # 基础计数
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn

        # 检测重叠关系（消融实验重点关注）
        has_overlap = self._detect_overlap(gold_triples)
        if has_overlap:
            self.overlapping_samples += 1
            self.overlapping_tp += tp
            self.overlapping_fp += fp
            self.overlapping_fn += fn

            # 细分重叠类型
            self._classify_overlap(pred_triples, gold_triples)

        # 按关系类型统计
        self._update_by_relation(pred_triples, gold_triples)

        # 按部分统计
        self._update_by_parts(pred_triples, gold_triples)

        # 错误分析
        self._analyze_errors(pred_triples, gold_triples)

    def _detect_overlap(self, triples: List[Tuple]) -> bool:
        """检测是否存在重叠关系"""
        subjects = [t[0] for t in triples]
        objects = [t[2] for t in triples]
        return len(subjects) != len(set(subjects)) or len(objects) != len(set(objects))

    def _classify_overlap(
        self, pred_triples: List[Tuple], gold_triples: List[Tuple]
    ):
        """分类重叠关系类型"""
        gold_subjects = [t[0] for t in gold_triples]
        gold_objects = [t[2] for t in gold_triples]

        for t in pred_triples:
            if t in gold_triples:
                # 正确预测
                if gold_subjects.count(t[0]) > 1:
                    self.subject_sharing_tp += 1
                if gold_objects.count(t[2]) > 1:
                    self.object_sharing_tp += 1
                # 检测嵌套实体（主语包含宾语或反之）
                if t[0] in t[2] or t[2] in t[0]:
                    self.nested_entity_tp += 1

    def _update_by_relation(
        self, pred_triples: List[Tuple], gold_triples: List[Tuple]
    ):
        """按关系类型统计"""
        pred_by_rel = defaultdict(list)
        gold_by_rel = defaultdict(list)

        for t in pred_triples:
            pred_by_rel[t[1]].append(t)

        for t in gold_triples:
            gold_by_rel[t[1]].append(t)

        for rel in set(pred_by_rel.keys()) | set(gold_by_rel.keys()):
            pred_set = set(pred_by_rel[rel])
            gold_set = set(gold_by_rel[rel])
            self.by_relation[rel]["tp"] += len(pred_set & gold_set)
            self.by_relation[rel]["fp"] += len(pred_set - gold_set)
            self.by_relation[rel]["fn"] += len(gold_set - pred_set)

    def _update_by_parts(
        self, pred_triples: List[Tuple], gold_triples: List[Tuple]
    ):
        """按部分统计（主语/宾语/关系）"""
        gold_by_subject = {t[0]: t for t in gold_triples}
        gold_by_object = {t[2]: t for t in gold_triples}

        for pred_t in pred_triples:
            if pred_t in gold_triples:
                continue

            # 检查哪部分匹配
            if pred_t[0] in gold_by_subject:
                self.subject_correct += 1
            if pred_t[2] in gold_by_object:
                self.object_correct += 1

            # 检查关系类型是否正确
            for gold_t in gold_triples:
                if pred_t[1] == gold_t[1]:
                    self.relation_correct += 1
                    break

        # 错误统计
        for pred_t in pred_triples:
            if pred_t not in gold_triples:
                self.subject_errors += 1  # 主语部分错误（至少）
                self.object_errors += 1   # 宾语部分错误（至少）
                self.relation_errors += 1  # 关系部分错误（至少）

    def _analyze_errors(
        self, pred_triples: List[Tuple], gold_triples: List[Tuple]
    ):
        """分析错误类型"""
        # 错误分布已在 _update_by_parts 中更新
        pass

    def get_basic_metrics(self) -> Dict[str, float]:
        """获取基础指标"""
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    def get_by_relation_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取按关系类型的指标"""
        results = {}
        for rel, counts in self.by_relation.items():
            precision = (
                counts["tp"] / (counts["tp"] + counts["fp"])
                if (counts["tp"] + counts["fp"]) > 0
                else 0.0
            )
            recall = (
                counts["tp"] / (counts["tp"] + counts["fn"])
                if (counts["tp"] + counts["fn"]) > 0
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            results[rel] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": counts["tp"] + counts["fn"],
            }
        return results

    def get_overlap_metrics(self) -> Dict[str, float]:
        """获取重叠关系指标（消融实验重点关注）"""
        if self.overlapping_samples == 0:
            return {
                "overlapping_f1": 0.0,
                "overlapping_precision": 0.0,
                "overlapping_recall": 0.0,
                "non_overlapping_f1": 0.0,
                "subject_sharing_f1": 0.0,
                "object_sharing_f1": 0.0,
                "nested_entity_f1": 0.0,
                "overlapping_sample_count": 0,
            }

        # 非重叠样本指标
        non_overlap_tp = self.true_positives - self.overlapping_tp
        non_overlap_fp = self.false_positives - self.overlapping_fp
        non_overlap_fn = self.false_negatives - self.overlapping_fn

        non_overlap_precision = (
            non_overlap_tp / (non_overlap_tp + non_overlap_fp)
            if (non_overlap_tp + non_overlap_fp) > 0
            else 0.0
        )
        non_overlap_recall = (
            non_overlap_tp / (non_overlap_tp + non_overlap_fn)
            if (non_overlap_tp + non_overlap_fn) > 0
            else 0.0
        )
        non_overlap_f1 = (
            2 * non_overlap_precision * non_overlap_recall
            / (non_overlap_precision + non_overlap_recall)
            if (non_overlap_precision + non_overlap_recall) > 0
            else 0.0
        )

        overlapping_precision = (
            self.overlapping_tp / (self.overlapping_tp + self.overlapping_fp)
            if (self.overlapping_tp + self.overlapping_fp) > 0
            else 0.0
        )
        overlapping_recall = (
            self.overlapping_tp / (self.overlapping_tp + self.overlapping_fn)
            if (self.overlapping_tp + self.overlapping_fn) > 0
            else 0.0
        )
        overlapping_f1 = (
            2 * overlapping_precision * overlapping_recall
            / (overlapping_precision + overlapping_recall)
            if (overlapping_precision + overlapping_recall) > 0
            else 0.0
        )

        return {
            "overlapping_f1": overlapping_f1,
            "overlapping_precision": overlapping_precision,
            "overlapping_recall": overlapping_recall,
            "non_overlapping_f1": non_overlap_f1,
            "subject_sharing_f1": (
                self.subject_sharing_tp / self.overlapping_tp
                if self.overlapping_tp > 0
                else 0.0
            ),
            "object_sharing_f1": (
                self.object_sharing_tp / self.overlapping_tp
                if self.overlapping_tp > 0
                else 0.0
            ),
            "nested_entity_f1": (
                self.nested_entity_tp / self.overlapping_tp
                if self.overlapping_tp > 0
                else 0.0
            ),
            "overlapping_sample_count": self.overlapping_samples,
        }

    def get_error_metrics(self) -> Dict[str, float]:
        """获取错误分析指标"""
        total_errors = self.false_positives + self.false_negatives
        return {
            "subject_error_rate": (
                self.subject_errors / total_errors if total_errors > 0 else 0.0
            ),
            "object_error_rate": (
                self.object_errors / total_errors if total_errors > 0 else 0.0
            ),
            "relation_error_rate": (
                self.relation_errors / total_errors if total_errors > 0 else 0.0
            ),
            "subject_errors": self.subject_errors,
            "object_errors": self.object_errors,
            "relation_errors": self.relation_errors,
        }