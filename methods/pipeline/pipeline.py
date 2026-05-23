# -*- coding: utf-8 -*-
"""
Pipeline 方法端到端流程：NER → RE → 三元组评估
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from methods.pipeline.ner.trainer import _predict_single, _build_label_maps
from methods.pipeline.ner.model import BertNer
from methods.pipeline.re.model import BertForRelationExtraction
from utils.common import get_device, get_logger, set_seed
from utils.io_utils import load_checkpoint, load_jsonl, save_json, save_jsonl
from utils.metrics import (
    ComprehensiveMetrics, InferenceTimer,
    count_parameters, model_size_mb, gpu_memory_stats, reset_gpu_peak_memory,
    analyze_errors, save_error_report, export_error_cases, per_relation_metrics,
)


# ──────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────

def _load_ner_model(cfg: dict, device: torch.device):
    ner_cfg = cfg["ner"]
    data_cfg = cfg["data"]
    _, txt2label, label2txt = _build_label_maps(data_cfg["entity2id"])
    tokenizer = BertTokenizer.from_pretrained(ner_cfg["bert_model"])
    model = BertNer(bert_path=ner_cfg["bert_model"], num_tags=len(txt2label)).to(device)
    load_checkpoint(model, ner_cfg["checkpoint"], device=device)
    model.eval()
    return model, tokenizer, txt2label, label2txt


def _load_re_model(cfg: dict, device: torch.device):
    re_cfg = cfg["re"]
    data_cfg = cfg["data"]
    with open(data_cfg["rel2id"], "r", encoding="utf-8") as f:
        rel_data = json.load(f)
    rel2id = rel_data.get("relation2id", rel_data)
    id2rel = {int(v): k for k, v in rel2id.items()}

    tokenizer = BertTokenizer.from_pretrained(re_cfg["bert_model"])
    model = BertForRelationExtraction(
        bert_path=re_cfg["bert_model"],
        num_tags=len(rel2id),
        dropout=re_cfg["dropout"],
    ).to(device)
    load_checkpoint(model, re_cfg["checkpoint"], device=device)
    model.eval()
    return model, tokenizer, rel2id, id2rel


# ──────────────────────────────────────────
# 推理核心
# ──────────────────────────────────────────

def _mark_entities(text: str, sub: Dict, obj: Dict) -> str:
    """用 # 标记主语，$ 标记宾语"""
    entities = [
        (sub["start"], sub["end"], sub["text"], "#"),
        (obj["start"], obj["end"], obj["text"], "$"),
    ]
    entities.sort(key=lambda x: x[0], reverse=True)
    marked = text
    for start, end, ent_text, marker in entities:
        marked = marked[:start] + f"{marker}{ent_text}{marker}" + marked[end + 1:]
    return marked


def _get_marked_entity_positions(marked: str, sub_ent: Dict, obj_ent: Dict):
    """
    在标记文本中查找实体的正确字符位置（BUG-1 fix 推理端）。

    原来: 用原始文本坐标 +1（[CLS] offset）作为 BERT 提取位置，
          实际指向的是 '#'/'$' 标记符而非实体 token，存在系统性错位。
    现在: 在标记文本中搜索 '#sub_text' 和 '$obj_text'，精确定位实体首字符。
    """
    sub_pos = marked.find("#" + sub_ent["text"])
    obj_pos = marked.find("$" + obj_ent["text"])
    if sub_pos == -1 or obj_pos == -1:
        return None
    sub_start_m = sub_pos + 1
    sub_end_m   = sub_start_m + len(sub_ent["text"]) - 1
    obj_start_m = obj_pos + 1
    obj_end_m   = obj_start_m + len(obj_ent["text"]) - 1
    return [sub_start_m, sub_end_m, obj_start_m, obj_end_m]


def _predict_re_for_sample(
    text: str,
    entities: List[Dict],
    re_model,
    re_tokenizer,
    rel2id: Dict,
    id2rel: Dict,
    re_cfg: dict,
    device: torch.device,
    debug_first_n: int = 0,
) -> Tuple[List[Dict], int]:
    """
    对文本中所有实体对运行 RE 模型，返回三元组列表和截断对数。

    Args:
        debug_first_n: 大于 0 时打印前 N 个实体对的位置信息，用于验证 BUG-1 修复效果。
    """
    no_rel_id = rel2id.get("无关系") or rel2id.get("NA") or max(id2rel.keys())
    triples: List[Dict] = []
    seen = set()
    trunc_count = 0
    pair_idx = 0

    for sub_ent in entities:
        for obj_ent in entities:
            if sub_ent is obj_ent:
                continue
            marked = _mark_entities(text, sub_ent, obj_ent)
            enc = re_tokenizer(
                list(marked),
                is_split_into_words=True,
                add_special_tokens=True,
                max_length=re_cfg["max_seq_len"],
                truncation=True,
                padding="max_length",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # BUG-1 fix: 计算实体在标记文本中的位置，而非原始文本位置
            positions = _get_marked_entity_positions(marked, sub_ent, obj_ent)
            if positions is None:
                continue  # 标记文本中找不到实体，跳过
            sub_s, sub_e, obj_s, obj_e = positions

            if any(p > re_cfg["max_seq_len"] - 2 for p in positions):
                trunc_count += 1

            # +1 for [CLS] token at position 0
            ids_tensor = torch.tensor([[p + 1 for p in positions]], dtype=torch.long, device=device)

            # 调试输出：验证位置指向的 token 是否是实体本身
            if pair_idx < debug_first_n:
                marked_chars = list(marked)
                sub_token = "".join(marked_chars[sub_s:sub_e + 1])
                obj_token = "".join(marked_chars[obj_s:obj_e + 1])
                print(
                    f"  [RE 位置调试 pair#{pair_idx}] "
                    f"sub='{sub_ent['text']}' 标记后位置[{sub_s},{sub_e}] 对应字符='{sub_token}' "
                    f"obj='{obj_ent['text']}' 标记后位置[{obj_s},{obj_e}] 对应字符='{obj_token}'"
                )

            with torch.no_grad():
                logits = re_model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], ids_tensor)
                pred_id = int(torch.argmax(logits, dim=1).item())

            if pred_id == no_rel_id:
                pair_idx += 1
                continue

            predicate = id2rel[pred_id]
            key = (sub_ent["text"], predicate, obj_ent["text"])
            if key in seen:
                pair_idx += 1
                continue
            seen.add(key)
            triples.append({
                "subject": sub_ent["text"],
                "predicate": predicate,
                "object": {"@value": obj_ent["text"]},
                "subject_type": sub_ent.get("type", ""),
                "object_type": {"@value": obj_ent.get("type", "")},
            })
            pair_idx += 1

    return triples, trunc_count


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────

def run(cfg: dict, mode: str, input_path: Optional[str] = None, output_path: Optional[str] = None) -> None:
    """
    Pipeline 方法统一入口

    Args:
        cfg: configs/pipeline.yaml 解析后的配置 dict
        mode: "train" / "evaluate" / "predict"
        input_path: predict 模式下的输入文件（JSONL 格式，每行含 text 字段）
        output_path: 预测结果输出路径（默认取 cfg 中的 predictions）
    """
    set_seed(cfg["seed"])

    if mode == "train":
        from methods.pipeline.ner.trainer import train as ner_train
        from methods.pipeline.re.trainer import train as re_train
        print("=== 第一阶段：训练 NER 模型 ===")
        ner_train(cfg)
        print("=== 第二阶段：训练 RE 模型 ===")
        re_train(cfg)

    elif mode == "evaluate":
        _evaluate(cfg)

    elif mode == "predict":
        test_file = input_path or cfg["data"].get("test_file", "data/processed/pipeline/test/pipeline_test.jsonl")
        out_file = output_path or cfg["output"]["predictions"]
        _predict(cfg, test_file, out_file)

    else:
        raise ValueError(f"未知 mode: {mode}，可选 train / evaluate / predict")


def _evaluate(cfg: dict) -> Dict:
    """
    在测试集上运行 Pipeline 端到端评估。

    输出文件（全部持久化到 cfg["output"]["dir"]）：
      metrics.json        — 严格/宽松 × 微平均/宏平均 + 推理速度 + 模型参数
      predictions.jsonl   — 每条样本的预测实体、预测三元组与金标三元组
      error_report.txt    — 错误类型汇总（带占比柱状图）
      error_cases.txt     — 错误案例详情（每条样本的具体错误三元组及类型）
      per_relation.txt    — 按关系类型分解的 P/R/F1 表格
    """
    device = get_device()
    out_dir = cfg["output"]["dir"]
    logger = get_logger("pipeline_eval", log_file=cfg["output"]["log"])
    os.makedirs(out_dir, exist_ok=True)

    logger.info("加载 NER 模型...")
    ner_model, ner_tokenizer, _, ner_label2txt = _load_ner_model(cfg, device)
    logger.info("加载 RE 模型...")
    re_model, re_tokenizer, rel2id, id2rel = _load_re_model(cfg, device)

    # 模型参数统计（NER + RE 分别记录）
    ner_params = count_parameters(ner_model)
    re_params  = count_parameters(re_model)
    ner_mb     = model_size_mb(ner_model)
    re_mb      = model_size_mb(re_model)
    total_params = ner_params["total"] + re_params["total"]
    logger.info(
        f"NER 参数: {ner_params['total']:,} ({ner_mb} MB)  "
        f"RE 参数: {re_params['total']:,} ({re_mb} MB)  "
        f"Pipeline 总参数: {total_params:,}"
    )

    reset_gpu_peak_memory()

    test_file = os.path.join(cfg["data"]["processed_dir"], "test", "pipeline_test.jsonl")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")

    samples = load_jsonl(test_file)
    metrics       = ComprehensiveMetrics()
    timer         = InferenceTimer()
    sample_outputs:   List[Dict] = []
    error_counts:     Dict       = {}
    pred_triples_all: List[List] = []
    gold_triples_all: List[List] = []

    DEBUG_SAMPLES = 3  # 前 N 条打印 RE 实体位置调试信息（验证 BUG-1 修复效果）

    for sample_idx, sample in enumerate(tqdm(samples, desc="pipeline_eval")):
        text         = sample["text"]
        gold_triples = sample.get("gold_triples", [])

        timer.start()
        entities, _ = _predict_single(
            text, ner_model, ner_tokenizer, ner_label2txt,
            cfg["ner"]["max_length"], device
        )

        dbg = 2 if sample_idx < DEBUG_SAMPLES else 0
        if dbg:
            print(f"\n[位置调试] 样本 #{sample_idx}: text='{text[:40]}...' "
                  f"实体={[e['text'] for e in entities]}")

        pred_triples, _ = _predict_re_for_sample(
            text, entities, re_model, re_tokenizer,
            rel2id, id2rel, cfg["re"], device,
            debug_first_n=dbg,
        )
        timer.stop(num_samples=1, num_triples=len(pred_triples))

        metrics.update(pred_triples, gold_triples)
        pred_triples_all.append(pred_triples)
        gold_triples_all.append(gold_triples)

        pred_entity_texts = {e["text"] for e in entities}
        errs = analyze_errors(pred_triples, gold_triples, pred_entity_texts)
        for k, v in errs.items():
            error_counts[k] = error_counts.get(k, 0) + v

        sample_outputs.append({
            "text":          text,
            "pred_entities": entities,
            "pred_triples":  pred_triples,
            "gold_triples":  gold_triples,
        })

    # ── 汇总结果 ──────────────────────────────────────────────────────
    result  = metrics.compute()
    speed   = timer.compute()
    gpu_mem = gpu_memory_stats()

    result["inference"]    = speed
    result["model"] = {
        "ner":   {"params": ner_params, "size_mb": ner_mb},
        "re":    {"params": re_params,  "size_mb": re_mb},
        "total_params": total_params,
    }
    result["gpu_memory"]   = gpu_mem
    result["error_counts"] = error_counts

    rel_metrics = per_relation_metrics(pred_triples_all, gold_triples_all)
    result["per_relation"] = rel_metrics

    # ── 打印报告 ─────────────────────────────────────────────────────
    metrics.print_report("Pipeline 端到端评估")
    logger.info(
        f"推理速度: {speed['samples_per_sec']} samples/s  "
        f"| 平均延迟 {speed['avg_latency_ms']} ms/sample  "
        f"| 总耗时 {speed['total_time_s']} s"
    )
    if gpu_mem.get("available"):
        logger.info(
            f"显存峰值: allocated={gpu_mem['peak_allocated_mb']} MB  "
            f"reserved={gpu_mem['peak_reserved_mb']} MB"
        )

    # ── 持久化输出文件 ────────────────────────────────────────────────
    save_json(result, cfg["output"]["metrics"])
    save_jsonl(sample_outputs, cfg["output"]["predictions"])
    save_error_report(error_counts, cfg["output"]["error_report"], model_name="pipeline")

    cases_path = Path(out_dir) / "error_cases.txt"
    export_error_cases(sample_outputs, cases_path)

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


def _predict(cfg: dict, input_file: str, output_file: str) -> None:
    """对任意 JSONL 输入文件批量预测"""
    device = get_device()
    ner_model, ner_tokenizer, _, ner_label2txt = _load_ner_model(cfg, device)
    re_model, re_tokenizer, rel2id, id2rel = _load_re_model(cfg, device)

    samples = load_jsonl(input_file)
    outputs = []
    for sample in tqdm(samples, desc="predicting"):
        text = sample.get("text", "")
        entities, _ = _predict_single(
            text, ner_model, ner_tokenizer, ner_label2txt,
            cfg["ner"]["max_length"], device
        )
        pred_triples, _ = _predict_re_for_sample(
            text, entities, re_model, re_tokenizer,
            rel2id, id2rel, cfg["re"], device
        )
        outputs.append({"text": text, "pred_triples": pred_triples})

    save_jsonl(outputs, output_file)
    print(f"预测结果已保存到: {output_file}")
