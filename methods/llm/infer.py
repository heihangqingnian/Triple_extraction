# -*- coding: utf-8 -*-
"""
ChatGLM2-6B + LoRA 推理接口
提供 LoRA 权重加载和批量推理功能
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from utils.common import get_logger, set_seed
from utils.io_utils import load_json, save_jsonl


def load_model(cfg: dict):
    """
    加载 ChatGLM2-6B 基础模型并挂载 LoRA 权重

    Args:
        cfg: configs/llm.yaml 解析后的完整配置 dict

    Returns:
        (model, tokenizer)
    """
    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "请安装 peft 和 transformers：pip install peft transformers"
        )

    model_cfg = cfg["model"]
    base_model = model_cfg["base_model"]
    lora_weights = model_cfg["lora_weights"]
    device_arg = model_cfg.get("device", "auto")

    logger = get_logger("llm_infer")
    logger.info(f"加载基础模型: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_arg if device_arg != "cpu" else None,
    )

    if os.path.exists(lora_weights):
        logger.info(f"加载 LoRA 权重: {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
    else:
        logger.warning(f"LoRA 权重目录不存在: {lora_weights}，使用基础模型推理")

    model = model.eval()
    return model, tokenizer


def _build_prompt(text: str) -> str:
    """构建与训练时相同的推理 prompt（与 Alpaca 格式训练数据 instruction+input 一致）"""
    instruction = (
        "你是一个信息抽取专家。请根据给定的Schema和文本，抽取出所有满足关系的事实三元组。"
        "输出格式为：[(\"主体\", \"关系\", \"客体\"), ...]，如果没有符合的三元组则输出 []。"
    )
    prompt = f"{instruction}\n\n### 文本\n{text}\n\n答："
    return prompt


def predict_sample(text: str, model, tokenizer, cfg: dict) -> List[Tuple[str, str, str]]:
    """
    对单条文本推理，返回三元组列表

    Args:
        text: 输入文本
        model: 已加载的模型
        tokenizer: tokenizer
        cfg: llm 配置 dict

    Returns:
        [(subject, predicate, object), ...]
    """
    model_cfg = cfg["model"]
    prompt = _build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")

    import torch
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=model_cfg.get("max_new_tokens", 512),
            temperature=model_cfg.get("temperature", 0.1),
            do_sample=False,
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_triples(generated)


def _parse_triples(output_str: str) -> List[Tuple[str, str, str]]:
    """解析模型生成的三元组字符串，兼容各种格式偏差"""
    output_str = output_str.strip().strip("[]")
    if not output_str:
        return []

    triples = []
    start = 0
    in_quote = False
    for i, ch in enumerate(output_str):
        if ch == '"':
            in_quote = not in_quote
        elif ch == "(" and not in_quote:
            start = i + 1
        elif ch == ")" and not in_quote:
            triple_str = output_str[start:i].strip()
            parts = triple_str.split(",")
            if len(parts) == 3:
                s = parts[0].strip().strip('"\'')
                p = parts[1].strip().strip('"\'')
                o = parts[2].strip().strip('"\'')
                triples.append((s, p, o))
    return triples


def predict_file(cfg: dict, input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """
    批量推理：读取 Alpaca 格式 JSON 文件，逐条推理并保存预测结果

    Args:
        cfg: configs/llm.yaml 配置 dict
        input_file: 输入文件（Alpaca JSON 格式），默认使用 cfg 中的 test_file
        output_file: 输出 JSONL 文件，默认使用 cfg 中的 predictions
    """
    set_seed(cfg["seed"])
    logger = get_logger("llm_infer", log_file=cfg["output"]["log"])
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    test_file = input_file or cfg["data"]["test_file"]
    out_file = output_file or cfg["output"]["predictions"]

    logger.info(f"加载模型...")
    model, tokenizer = load_model(cfg)

    logger.info(f"开始推理: {test_file}")
    samples = load_json(test_file)
    outputs = []

    for sample in tqdm(samples, desc="llm_infer"):
        text = _extract_text_from_sample(sample)
        label_str = sample.get("output", "[]")

        try:
            pred_triples = predict_sample(text, model, tokenizer, cfg)
            parse_error = False
        except Exception as e:
            logger.warning(f"推理失败: {e}")
            pred_triples = []
            parse_error = True

        outputs.append({
            "prompt": _build_prompt(text),
            "predict": str(pred_triples),
            "label": label_str,
            "parse_error": parse_error,
        })

    save_jsonl(outputs, out_file)
    logger.info(f"推理完成，结果已保存到: {out_file}")


def _extract_text_from_sample(sample: dict) -> str:
    """从 Alpaca 格式样本中提取原始文本"""
    inp = sample.get("input", "")
    if "### 文本\n" in inp:
        return inp.split("### 文本\n")[-1].strip()
    return inp.strip()
