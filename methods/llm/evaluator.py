# -*- coding: utf-8 -*-
"""
LLM 方法预测结果评估
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from utils.common import get_logger
from utils.io_utils import load_jsonl, save_json
from utils.metrics import TripleMetrics, analyze_errors, save_error_report


def parse_triple_string(triple_str: str) -> Set[Tuple[str, str, str]]:
    """
    解析模型输出的三元组字符串为集合

    支持格式：[("主体", "关系", "客体"), ...]
    """
    if not triple_str:
        return set()
    triple_str = triple_str.strip().strip("[]")
    if not triple_str:
        return set()

    triples = set()
    start = 0
    in_quote = False
    for i, ch in enumerate(triple_str):
        if ch == '"':
            in_quote = not in_quote
        elif ch == "(" and not in_quote:
            start = i + 1
        elif ch == ")" and not in_quote:
            segment = triple_str[start:i].strip()
            parts = segment.split(",")
            if len(parts) == 3:
                s = parts[0].strip().strip("\"'")
                p = parts[1].strip().strip("\"'")
                o = parts[2].strip().strip("\"'")
                triples.add((s, p, o))
    return triples


def _parse_gold_triples(label_str: str) -> Set[Tuple[str, str, str]]:
    """解析标注三元组字符串"""
    return parse_triple_string(label_str)


def _extract_text(prompt: str) -> str:
    """从 prompt 中提取原始文本"""
    if "### 文本\n" in prompt:
        return prompt.split("### 文本\n")[-1].split("\n\n答")[0]
    return prompt


def evaluate(
    cfg: dict,
    predictions_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    LLM 预测结果评估

    Args:
        cfg: configs/llm.yaml 配置 dict
        predictions_file: 预测结果 JSONL 文件（每行含 predict / label / prompt 字段）
        output_dir: 结果输出目录

    Returns:
        评估指标字典
    """
    logger = get_logger("llm_eval")
    pred_file = predictions_file or cfg["data"].get("predictions_file") or cfg["output"]["predictions"]
    out_dir = output_dir or cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"预测文件不存在: {pred_file}\n"
            "请先运行推理：python main.py --method llm --mode predict"
        )

    logger.info(f"加载预测文件: {pred_file}")
    records = load_jsonl(pred_file)

    metrics = TripleMetrics()
    total_errors: Dict[str, int] = defaultdict(int)
    parse_errors = 0

    for rec in records:
        pred_str = rec.get("predict", "[]")
        label_str = rec.get("label", "[]")
        prompt = rec.get("prompt", "")
        text = _extract_text(prompt)

        if rec.get("parse_error", False):
            parse_errors += 1
            continue

        pred_set = parse_triple_string(pred_str)
        gold_set = _parse_gold_triples(label_str)

        # 转换为 list of dict 供 TripleMetrics 使用
        pred_triples = [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in pred_set]
        gold_triples = [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in gold_set]

        metrics.update(pred_triples, gold_triples)
        errs = analyze_errors(pred_triples, gold_triples)
        for k, v in errs.items():
            total_errors[k] += v

    result = metrics.compute()
    result.update({
        "model": "llm_lora",
        "total_predictions": len(records),
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / max(len(records), 1), 6),
        "error_counts": dict(total_errors),
    })

    metrics.print_report("LLM 评估结果")
    logger.info(f"解析失败数: {parse_errors} / {len(records)}")

    save_json(result, os.path.join(out_dir, "metrics.json"))
    save_error_report(dict(total_errors), os.path.join(out_dir, "error_report.txt"), model_name="llm_lora")
    logger.info(f"评估完成，结果已保存到: {out_dir}")
    return result


def run(cfg: dict, mode: str, input_path: Optional[str] = None, output_path: Optional[str] = None) -> None:
    """LLM 方法统一入口"""
    from utils.common import set_seed
    set_seed(cfg["seed"])

    if mode == "predict":
        from methods.llm.infer import predict_file
        predict_file(cfg, input_file=input_path, output_file=output_path)

    elif mode == "evaluate":
        evaluate(cfg, predictions_file=input_path, output_dir=output_path)

    elif mode == "train":
        print(
            "LLM 训练通过 LlamaFactory 框架完成，不在本仓库中。\n"
            "请参考 README.md 中的 LLM 训练说明。"
        )
    else:
        raise ValueError(f"未知 mode: {mode}")
