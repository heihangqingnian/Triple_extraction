# -*- coding: utf-8 -*-
"""
LLM 方法最终测试评估（单 Prompt / 单 LoRA）。

读取 infer.predict_file 产出的 predictions.jsonl，用 ComprehensiveMetrics 计算指标，
输出结构与 Pipeline / Joint 基线完全一致的 metrics.json（严格/宽松 × 微平均/宏平均
+ 逐关系 + 错误报告 + 错误案例），保证三种方法在同一套测试集、同一套指标下可比。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from utils.common import get_logger
from utils.io_utils import load_jsonl, save_json
from utils.metrics import (
    ComprehensiveMetrics,
    analyze_errors,
    export_error_cases,
    parse_triple_string,
    per_relation_metrics,
    save_error_report,
)


def _to_triple_dicts(triple_set) -> List[Dict]:
    return [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in triple_set]


def evaluate(cfg: dict, predictions_file: Optional[str] = None, output_dir: Optional[str] = None) -> Dict:
    """
    评估 LLM 测试预测结果。

    Args:
        cfg:              configs/llm.yaml 配置 dict
        predictions_file: 指定预测文件（默认 cfg["output"]["dir"]/predictions.jsonl）
        output_dir:       结果输出目录（默认 cfg["output"]["dir"]）

    Returns:
        指标字典（与 pipeline/joint 的 metrics.json 同结构）。
    """
    logger = get_logger("llm_eval")
    out_dir = output_dir or cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    pred_file = predictions_file or os.path.join(out_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"预测文件不存在: {pred_file}\n"
            "请先运行推理：python main.py --method llm --mode predict"
        )

    logger.info(f"加载预测文件: {pred_file}")
    records = load_jsonl(pred_file)

    metrics = ComprehensiveMetrics()
    error_counts: Dict[str, int] = {}
    pred_triples_all: List[List] = []
    gold_triples_all: List[List] = []
    sample_outputs: List[Dict] = []
    parse_errors = 0

    for rec in records:
        if rec.get("parse_error", False):
            parse_errors += 1

        pred_triples = _to_triple_dicts(parse_triple_string(rec.get("predict", "[]")))
        gold_triples = _to_triple_dicts(parse_triple_string(rec.get("label", "[]")))

        metrics.update(pred_triples, gold_triples)
        pred_triples_all.append(pred_triples)
        gold_triples_all.append(gold_triples)

        for k, v in analyze_errors(pred_triples, gold_triples).items():
            error_counts[k] = error_counts.get(k, 0) + v

        sample_outputs.append({
            "text": rec.get("text", ""),
            "pred_triples": pred_triples,
            "gold_triples": gold_triples,
        })

    # ── 汇总结果（与 pipeline/joint 同结构）──────────────────────────────
    result = metrics.compute()
    result["model"] = "llm_lora"
    result["total_predictions"] = len(records)
    result["parse_errors"] = parse_errors
    result["parse_error_rate"] = round(parse_errors / max(len(records), 1), 6)
    result["error_counts"] = error_counts

    rel_metrics = per_relation_metrics(pred_triples_all, gold_triples_all)
    result["per_relation"] = rel_metrics

    metrics.print_report("LLM 最终测试评估")
    logger.info(f"解析失败数: {parse_errors} / {len(records)}")

    # ── 持久化输出（文件名与 pipeline/joint 一致）────────────────────────
    save_json(result, cfg["output"].get("metrics", os.path.join(out_dir, "metrics.json")))
    save_error_report(
        error_counts,
        cfg["output"].get("error_report", os.path.join(out_dir, "error_report.txt")),
        model_name="llm_lora",
    )
    export_error_cases(sample_outputs, Path(out_dir) / "error_cases.txt")

    rel_path = Path(out_dir) / "per_relation.txt"
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(f"{'关系类型':<22} {'P':>8} {'R':>8} {'F1':>8} {'Support':>9} {'Pred':>7} {'Correct':>8}\n")
        f.write(f"{'-' * 72}\n")
        rows = sorted(
            [(k, v) for k, v in rel_metrics.items() if k != "__overall__"],
            key=lambda x: -x[1]["support"],
        )
        for rel, m in rows:
            f.write(f"{rel:<22} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                    f"{m['support']:>9} {m['predicted']:>7} {m['correct']:>8}\n")
        if "__overall__" in rel_metrics:
            ov = rel_metrics["__overall__"]
            f.write(f"{'-' * 72}\n")
            f.write(f"{'Overall':<22} {ov['precision']:>8.4f} {ov['recall']:>8.4f} {ov['f1']:>8.4f} "
                    f"{ov['support']:>9} {ov['predicted']:>7} {ov['correct']:>8}\n")
    print(f"关系级指标已保存到: {rel_path}")

    logger.info(f"评估完成，所有结果已保存到: {out_dir}")
    return result


def run(
    cfg: dict,
    mode: str,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    LLM 方法统一入口（最终测试阶段）。

    Args:
        cfg:         configs/llm.yaml 配置 dict
        mode:        predict / evaluate / train
        input_path:  predict 时为测试输入文件；evaluate 时为指定预测文件
        output_path: predict 时为输出文件；evaluate 时为输出目录
    """
    from utils.common import set_seed
    set_seed(cfg["seed"])

    if mode == "predict":
        from methods.llm.infer import predict_file
        predict_file(cfg, input_file=input_path, output_file=output_path)

    elif mode == "evaluate":
        evaluate(cfg, predictions_file=input_path, output_dir=output_path)

    elif mode == "train":
        print(
            "LLM 微调通过 LLaMA-Factory 完成，不在本仓库中。\n"
            "请先用 scripts/prompt_search.py 选最优 Prompt，再用 scripts/build_llm_dataset.py "
            "构造 train.json，然后在 LLaMA-Factory 中微调。详见 WORKFLOW.md。"
        )
    else:
        raise ValueError(f"未知 mode: {mode}")
