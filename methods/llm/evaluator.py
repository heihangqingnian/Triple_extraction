# -*- coding: utf-8 -*-
"""
LLM 方法预测结果评估

支持对三种 Prompt 类型（base / schema / cot）分别评估，
也可一次性评估全部并汇总写入 metrics.json。
"""

import os
from collections import defaultdict
from typing import Dict, Optional

from utils.common import get_logger
from utils.io_utils import load_jsonl, save_json
from utils.metrics import TripleMetrics, analyze_errors, save_error_report, parse_triple_string

PROMPT_TYPES = ["base", "schema", "cot"]



def evaluate(
    cfg: dict,
    prompt_type: Optional[str] = None,
    predictions_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    LLM 预测结果评估。

    Args:
        cfg:              configs/llm.yaml 配置 dict
        prompt_type:      指定评估某一类型（base/schema/cot）；
                          为 None 时自动遍历全部三种并汇总。
        predictions_file: 指定预测文件路径（仅在 prompt_type 非 None 时生效）。
        output_dir:       结果输出目录。

    Returns:
        评估结果字典。prompt_type=None 时顶层 key 为 prompt_type，
        否则直接返回单类型结果。
    """
    logger = get_logger("llm_eval")
    out_dir = output_dir or cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    if prompt_type is None:
        # 评估全部三种 prompt 类型
        all_results: Dict = {}
        for pt in PROMPT_TYPES:
            pred_file = os.path.join(cfg["output"]["dir"], f"predictions_{pt}.jsonl")
            if not os.path.exists(pred_file):
                logger.warning(f"预测文件不存在，跳过: {pred_file}")
                continue
            all_results[pt] = _evaluate_single(pred_file, out_dir, pt, logger)

        if not all_results:
            raise FileNotFoundError(
                "未找到任何预测文件，请先运行：python main.py --method llm --mode predict"
            )

        save_json(all_results, os.path.join(out_dir, "metrics.json"))
        logger.info(f"全部 prompt 类型评估完成，结果已保存到: {out_dir}/metrics.json")
        _print_summary(all_results)
        return all_results

    # 评估单一 prompt 类型
    pred_file = predictions_file or os.path.join(out_dir, f"predictions_{prompt_type}.jsonl")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"预测文件不存在: {pred_file}\n"
            "请先运行推理：python main.py --method llm --mode predict"
        )
    result = _evaluate_single(pred_file, out_dir, prompt_type, logger)
    save_json(result, os.path.join(out_dir, f"metrics_{prompt_type}.json"))
    logger.info(f"评估完成，结果已保存到: {out_dir}/metrics_{prompt_type}.json")
    return result


def _evaluate_single(pred_file: str, out_dir: str, prompt_type: str, logger) -> Dict:
    """对单个预测文件执行评估，返回指标字典并写入错误报告。"""
    logger.info(f"[{prompt_type}] 加载预测文件: {pred_file}")
    records = load_jsonl(pred_file)

    metrics = TripleMetrics()
    total_errors: Dict[str, int] = defaultdict(int)
    parse_errors = 0

    for rec in records:
        if rec.get("parse_error", False):
            parse_errors += 1
            continue

        pred_set = parse_triple_string(rec.get("predict", "[]"))
        gold_set = parse_triple_string(rec.get("label", "[]"))

        pred_triples = [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in pred_set]
        gold_triples = [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in gold_set]

        metrics.update(pred_triples, gold_triples)
        for k, v in analyze_errors(pred_triples, gold_triples).items():
            total_errors[k] += v

    result = metrics.compute()
    result.update({
        "model": f"llm_lora_{prompt_type}",
        "prompt_type": prompt_type,
        "total_predictions": len(records),
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / max(len(records), 1), 6),
        "error_counts": dict(total_errors),
    })

    metrics.print_report(f"LLM [{prompt_type}] 评估结果")
    logger.info(f"[{prompt_type}] 解析失败数: {parse_errors} / {len(records)}")

    save_error_report(
        dict(total_errors),
        os.path.join(out_dir, f"error_report_{prompt_type}.txt"),
        model_name=f"llm_lora_{prompt_type}",
    )
    return result


def _print_summary(all_results: Dict) -> None:
    """打印三种 prompt 类型的 F1 对比"""
    print("\n" + "=" * 58)
    print(f"  {'Prompt':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("=" * 58)
    for pt, r in all_results.items():
        print(f"  {pt:<10} {r.get('precision', 0):>10.4f} "
              f"{r.get('recall', 0):>10.4f} {r.get('f1', 0):>10.4f}")
    print("=" * 58)


def run(
    cfg: dict,
    mode: str,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    variant: str = "all",
) -> None:
    """
    LLM 方法统一入口

    Args:
        cfg:         configs/llm.yaml 配置 dict
        mode:        train / predict / evaluate
        input_path:  predict 时为输入文件；evaluate 时为指定预测文件（仅 variant 非 all 时生效）
        output_path: predict 时为输出文件（单变体）或忽略（all）；evaluate 时为输出目录
        variant:     LoRA 变体 base/schema/cot/all（默认 all）
    """
    from utils.common import set_seed
    set_seed(cfg["seed"])

    if mode == "predict":
        from methods.llm.infer import predict_file
        predict_file(cfg, input_file=input_path, output_file=output_path, variant=variant)

    elif mode == "evaluate":
        # 优先级：命令行 --variant > config 中 model.prompt_type > 全部运行
        cfg_prompt_type = cfg.get("model", {}).get("prompt_type")
        if variant != "all":
            prompt_type = variant
        elif cfg_prompt_type:
            prompt_type = cfg_prompt_type
        else:
            prompt_type = None  # evaluate all
        pred_file = input_path if prompt_type is not None else None
        evaluate(cfg, prompt_type=prompt_type, predictions_file=pred_file, output_dir=output_path)

    elif mode == "train":
        print(
            "LLM 训练通过 LlamaFactory 框架完成，不在本仓库中。\n"
            "请参考 WORKFLOW.md 中的 LLM 训练说明。"
        )
    else:
        raise ValueError(f"未知 mode: {mode}")
