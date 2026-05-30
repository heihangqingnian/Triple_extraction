# -*- coding: utf-8 -*-
"""
免训练 Prompt 寻优（Zero-shot / Few-shot）。

在**不加载任何 LoRA**的前提下，用基座 ChatGLM2 对验证集（或训练集）的小样本分别运行
各 Prompt 变体，按三元组 F1 选出「最优 Prompt」。该最优 Prompt 随后用于 LoRA 微调数据构造。

数据隔离硬约束
--------------
* 评估样本只能取自 ``dev`` 或 ``train``；代码断言禁止 ``test``，杜绝数据泄露。
* few-shot 示例从 ``train_raw`` 采样，且排除评估样本，绝不取 test。

输出
----
* results/llm/prompt_search/predictions_<pt>.jsonl  各变体逐条预测
* results/llm/prompt_search/ranking.json            各变体 P/R/F1 排名 + 最优 prompt
* 终端打印对比表与「最优 Prompt」。
"""

import os
import random
from typing import Dict, List, Tuple

from utils.common import get_logger, set_seed
from utils.io_utils import load_jsonl, save_json, save_jsonl
from utils.metrics import TripleMetrics, parse_triple_string

from methods.llm import infer
from methods.llm.prompt_templates import build_instruction, build_query

_ALLOWED_SPLITS = {"dev", "train"}


def _load_raw(raw_file: str) -> List[Dict]:
    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"raw 文件不存在: {raw_file}\n"
            "请先运行：python scripts/preprocess.py --input data/raw/DuIE2.0 --method llm"
        )
    return load_jsonl(raw_file)


def _triples_to_str(triples: List) -> str:
    parts = [f'("{t[0]}", "{t[1]}", "{t[2]}")' for t in triples if t[0] and t[1] and t[2]]
    return f"[{', '.join(parts)}]"


def _sample_fewshot(
    fs_cfg: dict,
    eval_texts: set,
    seed: int,
) -> List[Tuple[str, List]]:
    """从 train_raw 采样 K 个 few-shot 示例，排除评估集中出现的文本。"""
    if not fs_cfg.get("enabled", False):
        return []
    k = int(fs_cfg.get("k", 2))
    source = fs_cfg.get("source", "data/processed/llm/train_raw.jsonl")
    pool = [r for r in _load_raw(source) if r.get("text") and r["text"] not in eval_texts]
    rng = random.Random(seed + 99991)
    rng.shuffle(pool)
    examples = [(r["text"], r.get("triples", [])) for r in pool[:k]]
    return examples


def run_prompt_search(cfg: dict) -> Dict:
    """执行免训练 Prompt 寻优，返回 ranking 字典。"""
    set_seed(cfg["seed"])
    logger = get_logger("prompt_search", log_file=os.path.join(cfg["output"]["dir"], "search.log"))
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    search_cfg = cfg["search"]
    source_split = search_cfg.get("source_split", "dev")
    if source_split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"source_split 只能是 {_ALLOWED_SPLITS}，收到 {source_split!r}。"
            "免训练寻优绝不允许使用测试集（test），以防数据泄露。"
        )

    raw_file = search_cfg.get("raw_file", f"data/processed/llm/{source_split}_raw.jsonl")
    if "test" in os.path.basename(raw_file):
        raise ValueError(f"raw_file 指向了疑似测试集文件: {raw_file}，已拒绝以防数据泄露。")

    records = _load_raw(raw_file)
    sample_size = int(search_cfg.get("sample_size", 150))
    rng = random.Random(cfg["seed"])
    if sample_size < len(records):
        records = rng.sample(records, sample_size)
    logger.info(f"寻优数据：{raw_file}（split={source_split}），采样 {len(records)} 条")

    eval_texts = {r["text"] for r in records}
    few_shot_examples = _sample_fewshot(search_cfg.get("few_shot", {}), eval_texts, cfg["seed"])
    if few_shot_examples:
        logger.info(f"Few-shot 已启用：K={len(few_shot_examples)}（取自 train_raw，已排除评估样本）")
    else:
        logger.info("Few-shot 未启用，纯 Zero-shot 寻优")

    prompt_types = search_cfg.get("prompt_types", ["base", "schema", "cot"])

    # 基座模型只加载一次（无 LoRA），各 prompt 变体复用
    logger.info("加载基座模型（无 LoRA，零样本）...")
    model, tokenizer = infer.load_model(cfg, lora_weights=None)

    ranking: Dict[str, Dict] = {}
    try:
        for pt in prompt_types:
            logger.info(f"=== 评估 Prompt 变体: {pt} ===")
            system = build_instruction(pt)
            items = []
            for r in records:
                text = r["text"]
                # few-shot 块拼进 query（system 不变，保证变体间唯一差异是指令）
                query = build_query(text, few_shot_examples)
                items.append({
                    "system": system,
                    "query": query,
                    "text": text,
                    "label": _triples_to_str(r.get("triples", [])),
                })

            outputs = infer.run_inference(model, tokenizer, cfg, items, desc=f"search_{pt}")
            save_jsonl(outputs, os.path.join(cfg["output"]["dir"], f"predictions_{pt}.jsonl"))

            metrics = TripleMetrics()
            parse_errors = 0
            for rec in outputs:
                if rec.get("parse_error"):
                    parse_errors += 1
                pred = [list(t) for t in parse_triple_string(rec["predict"])]
                gold = [list(t) for t in parse_triple_string(rec["label"])]
                metrics.update(
                    [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in pred],
                    [{"subject": s, "predicate": p, "object": {"@value": o}} for s, p, o in gold],
                )
            res = metrics.compute()
            res["parse_errors"] = parse_errors
            ranking[pt] = res
            logger.info(f"[{pt}] P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f}")
    finally:
        import torch
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_prompt = max(ranking, key=lambda k: ranking[k]["f1"]) if ranking else None
    result = {
        "source_split": source_split,
        "raw_file": raw_file,
        "sample_size": len(records),
        "few_shot": bool(few_shot_examples),
        "ranking": ranking,
        "best_prompt": best_prompt,
    }
    save_json(result, os.path.join(cfg["output"]["dir"], "ranking.json"))
    _print_ranking(result)
    return result


def _print_ranking(result: Dict) -> None:
    print("\n" + "=" * 64)
    print(f"  免训练 Prompt 寻优结果  （split={result['source_split']}，"
          f"n={result['sample_size']}，few_shot={result['few_shot']}）")
    print("=" * 64)
    print(f"  {'Prompt':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'解析失败':>8}")
    print("-" * 64)
    for pt, r in sorted(result["ranking"].items(), key=lambda x: -x[1]["f1"]):
        print(f"  {pt:<10} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r.get('parse_errors', 0):>8}")
    print("=" * 64)
    print(f"  ★ 最优 Prompt: {result['best_prompt']}")
    print(f"  下一步：python scripts/build_llm_dataset.py --prompt {result['best_prompt']} --split all")
    print("=" * 64)
