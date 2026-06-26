# -*- coding: utf-8 -*-
"""
ChatGLM2-6B 推理接口（基座零样本 / 基座 + 单 LoRA）。

两种调用方式：
  1. 免训练 Prompt 寻优（prompt_search.py）：``load_model(cfg, lora_weights=None)`` 纯基座
  2. 最终测试推理（main.py --method llm --mode predict）：``load_model(cfg, lora_weights="...")``

Prompt 拼接走 prompt_templates.py：``system`` 字段 = 固定指令，``instruction`` 字段 = query。
喂给 ChatGLM2 前套上 ``[Round 1]\\n\\n问：{query}\\n\\n答：`` 模板（由 ``tokenizer.build_prompt`` 生成），
与 LLaMA-Factory 的 ``chatglm2`` 模板 + system 对齐。

PEFT LoRA 与 ChatGLM2 的 KV-cache 不兼容，推理统一传 ``use_cache=False``。
"""

import os
from typing import Dict, List, Optional

from tqdm import tqdm

from methods.llm.prompt_templates import extract_text_from_input
from utils.common import get_logger, set_seed
from utils.io_utils import load_json, save_jsonl
from utils.metrics import parse_triple_string


def load_model(cfg: dict, lora_weights: Optional[str] = None):
    """加载 ChatGLM2-6B 基座 + 可选 LoRA。``lora_weights=None`` 时纯基座（用于寻优）。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    logger = get_logger("llm_infer")
    model_cfg = cfg["model"]
    base_model_path = model_cfg["base_model"]
    device_arg = model_cfg.get("device", "auto")

    use_gpu = torch.cuda.is_available() and device_arg != "cpu"
    dtype = torch.float16 if use_gpu else torch.float32
    device_map = device_arg if use_gpu else None
    logger.info(f"加载基座模型: {base_model_path}（dtype={dtype}, device_map={device_map}）")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModel.from_pretrained(base_model_path, **load_kwargs)

    # ChatGLM2Config 用 num_layers，PEFT 注册 LoRA 层时会访问 num_hidden_layers
    if not hasattr(model.config, "num_hidden_layers"):
        for attr in ("num_layers", "n_layer", "n_layers"):
            if hasattr(model.config, attr):
                model.config.num_hidden_layers = getattr(model.config, attr)
                break

    if lora_weights:
        if not os.path.exists(lora_weights):
            raise FileNotFoundError(f"LoRA 权重目录不存在: {lora_weights}")
        logger.info(f"挂载 LoRA: {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
    else:
        logger.info("纯基座推理（无 LoRA）")

    return model.eval(), tokenizer


def _build_prompt(tokenizer, system: Optional[str], query: str) -> str:
    """system + ChatGLM2 模板 ``[Round 1]\\n\\n问：{query}\\n\\n答：``。"""
    if hasattr(tokenizer, "build_prompt"):
        turn = tokenizer.build_prompt(query, history=[])
    else:
        turn = query
    return f"{system}\n\n{turn}" if system else turn


def run_inference(
    model,
    tokenizer,
    cfg: dict,
    items: List[Dict],
    desc: str = "llm_infer",
) -> List[Dict]:
    """批量推理循环（prompt_search 与最终测试共用）。

    Args:
        items: 每条 ``{"system": str|None, "query": str, "text": str, "label": str}``
        desc:  tqdm 进度条标签

    Returns:
        每条 ``{"prompt", "text", "predict", "label", "parse_error"}``
    """
    import torch

    logger = get_logger("llm_infer")
    model_cfg = cfg["model"]
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.1)
    per_sample_timeout = cfg.get("inference", {}).get("per_sample_timeout", 0)
    max_time = float(per_sample_timeout) if per_sample_timeout > 0 else None

    device = next(model.parameters()).device
    outputs: List[Dict] = []

    for item in tqdm(items, desc=desc):
        system = item.get("system")
        query = item.get("query", "")
        text = item.get("text", "")
        label = item.get("label", "[]")

        prompt = _build_prompt(tokenizer, system, query)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(device)

        gen_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].to(device),
            "max_new_tokens": max_new_tokens,
            "use_cache": False,  # PEFT LoRA 与 ChatGLM2 KV-cache 不兼容
        }
        if max_time:
            gen_kwargs["max_time"] = max_time
        if temperature > 1e-4:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False

        try:
            with torch.no_grad():
                seqs = model.generate(**gen_kwargs)
            generated = seqs[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            for m in (model, getattr(model, "base_model", None)):
                if m is not None and hasattr(m, "process_response"):
                    response = m.process_response(response)
                    break
            pred_triples = list(parse_triple_string(response or ""))
            parse_error = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"推理失败: {exc}")
            pred_triples, parse_error = [], True

        outputs.append({
            "prompt": prompt,
            "text": text,
            "predict": str(pred_triples),
            "label": label,
            "parse_error": parse_error,
        })

    return outputs


def predict_file(
    cfg: dict,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """读 Alpaca 测试文件（system/instruction/output）→ 挂 LoRA 推理 → 写 predictions.jsonl。"""
    import torch

    set_seed(cfg["seed"])
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger("llm_infer", log_file=cfg["output"].get("log"))

    test_file = input_file or cfg["data"]["test_file"]
    lora_weights = cfg["model"].get("lora_weights")
    if not isinstance(lora_weights, (str, type(None))):
        raise ValueError(
            f"cfg.model.lora_weights 必须是字符串或 null，收到 {type(lora_weights).__name__}: "
            f"{lora_weights!r}。旧 dict 多变体配置已弃用，请改为单一 LoRA 路径。"
        )
    logger.info(f"测试推理：测试集={test_file}，LoRA={lora_weights!r}")

    samples = load_json(test_file)
    items = [
        {
            "system": s.get("system"),
            "query": s.get("instruction", ""),
            "text": extract_text_from_input(s.get("instruction", "")),
            "label": s.get("output", "[]"),
        }
        for s in samples
    ]

    model, tokenizer = load_model(cfg, lora_weights=lora_weights)
    outputs = run_inference(model, tokenizer, cfg, items, desc="llm_test_infer")

    out_path = output_file or os.path.join(out_dir, "predictions.jsonl")
    save_jsonl(outputs, out_path)
    logger.info(f"完成 → {out_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path
