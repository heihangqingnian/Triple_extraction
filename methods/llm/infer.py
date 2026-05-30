# -*- coding: utf-8 -*-
"""
ChatGLM2-6B 推理接口（基座零样本 / 基座+单 LoRA）。

两种使用场景共用本模块：
  1. 免训练 Prompt 寻优（prompt_search.py）：``load_model(cfg, lora_weights=None)`` 纯基座。
  2. 最终测试推理（evaluator.py / main.py）：``load_model(cfg, lora_weights="models/...")`` 挂单 LoRA。

Prompt 拼接统一走 methods/llm/prompt_templates.py，并在喂给 ChatGLM2 前套上
``system + [Round 1]\\n\\n问：{query}\\n\\n答：`` 模板，与 LLaMA-Factory 的 ``chatglm2``
模板 + system 对齐（务必用 build_llm_dataset.py 打印的同一份 system 训练，否则掉点）。

PEFT LoRA 与 ChatGLM2 的 KV-cache 不兼容，推理统一传 ``use_cache=False``。
"""

import os
import threading
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.common import get_logger, set_seed
from utils.io_utils import load_json, save_jsonl
from utils.metrics import parse_triple_string


def load_model(cfg: dict, lora_weights: Optional[str] = None):
    """
    加载 ChatGLM2-6B 基础模型，可选挂载单一 LoRA 权重。

    Args:
        cfg:          配置 dict（需含 model.base_model、可选 model.device 等）
        lora_weights: LoRA 权重目录；None 或空字符串表示「纯基座零样本推理」。

    Returns:
        (model, tokenizer)
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("请安装 peft 和 transformers：pip install peft transformers")

    model_cfg = cfg["model"]
    base_model_path = model_cfg["base_model"]
    device_arg = model_cfg.get("device", "auto")

    logger = get_logger("llm_infer")

    use_gpu = torch.cuda.is_available() and device_arg != "cpu"
    if use_gpu:
        dtype = torch.float16
        device_map = device_arg  # "auto" 或指定 GPU
        logger.info(f"使用 GPU（{torch.cuda.get_device_name(0)}），dtype=float16")
    else:
        dtype = torch.float32
        device_map = None
        logger.warning("CUDA 不可用，回退到 CPU（float32），推理速度会非常慢")

    logger.info(f"加载基础模型: {base_model_path}，dtype={dtype}, device_map={device_map}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = AutoModel.from_pretrained(base_model_path, **load_kwargs)

    # ChatGLM2Config 用 num_layers 而非 num_hidden_layers，PEFT 注册 LoRA 层时会访问后者。
    if not hasattr(model.config, "num_hidden_layers"):
        for _attr in ("num_layers", "n_layer", "n_layers"):
            if hasattr(model.config, _attr):
                model.config.num_hidden_layers = getattr(model.config, _attr)
                logger.info(f"已将 config.{_attr} → config.num_hidden_layers = {model.config.num_hidden_layers}")
                break

    if lora_weights and os.path.exists(lora_weights):
        logger.info(f"加载 LoRA 权重: {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
    elif lora_weights:
        logger.warning(f"LoRA 权重目录不存在: {lora_weights!r}，回退到基座模型推理")
    else:
        logger.info("未配置 LoRA（零样本模式），使用纯基座模型推理")

    model = model.eval()
    return model, tokenizer


# ──────────────────────────────────────────
# Prompt → ChatGLM2 模板
# ──────────────────────────────────────────

def build_model_input(tokenizer, query: str, system: Optional[str] = None) -> str:
    """
    把 (system, query) 拼成喂给 ChatGLM2 的完整输入串。

    对齐 LLaMA-Factory ``chatglm2`` 模板：system 文本置于轮次块之前，轮次块由
    ``tokenizer.build_prompt(query)`` 生成（``[Round 1]\\n\\n问：{query}\\n\\n答：``）。

    若你的 LLaMA-Factory 版本对 system 与轮次块之间的分隔有差异，只需在此处调整
    ``sep`` 一个旋钮即可（务必让训练/推理两侧一致）。
    """
    if hasattr(tokenizer, "build_prompt"):
        turn = tokenizer.build_prompt(query, history=[])
    else:
        turn = query
    if system:
        sep = "\n\n"
        return f"{system}{sep}{turn}"
    return turn


def predict_sample(
    model,
    tokenizer,
    cfg: dict,
    query: str,
    system: Optional[str] = None,
    timeout: float = 0,
) -> List[Tuple[str, str, str]]:
    """
    对单条样本推理，返回三元组列表。

    Args:
        query:   query/input 文本（如 ``### 文本\\n...``，可含 few-shot 块）。
        system:  system 指令文本；None 表示无 system。
        timeout: >0 时把 max_time 传给 generate，超时即停。
    """
    import torch

    logger = get_logger("llm_infer")
    model_cfg = cfg["model"]
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.1)
    max_time = float(timeout) if timeout > 0 else None

    formatted = build_model_input(tokenizer, query, system)

    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None:
        raise ValueError("tokenizer 返回了 None 作为 input_ids")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    input_ids = input_ids.to(device)

    gen_kwargs: dict = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "use_cache": False,  # PEFT LoRA 与 ChatGLM2 KV-cache 不兼容
    }
    if max_time:
        gen_kwargs["max_time"] = max_time
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask.to(device)
    if temperature > 1e-4:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        raw_out = model.generate(**gen_kwargs)

    if raw_out is None:
        raise ValueError("model.generate() 返回了 None")
    if hasattr(raw_out, "sequences"):
        seqs = raw_out.sequences
    elif isinstance(raw_out, torch.Tensor):
        seqs = raw_out
    else:
        raise ValueError(f"无法识别的 generate() 输出类型: {type(raw_out)}")

    input_len = input_ids.shape[1]
    generated_ids = seqs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 用 ChatGLM2 的 process_response 清理输出（如果可用）
    for m_obj in (model, getattr(model, "base_model", None)):
        if m_obj is not None and hasattr(m_obj, "process_response"):
            try:
                response = m_obj.process_response(response)
            except Exception:
                pass
            break

    return list(parse_triple_string(response or ""))


def run_inference(
    model,
    tokenizer,
    cfg: dict,
    items: List[Dict],
    desc: str = "llm_infer",
) -> List[Dict]:
    """
    共享批量推理循环（prompt_search 与最终测试复用）。

    Args:
        items: 每条为 dict，需含 query / 可选 system / text / label 字段：
               {"system": str|None, "query": str, "text": str, "label": str}
        desc:  进度条与命名标签。

    Returns:
        每条 dict：{"prompt", "text", "predict", "label", "parse_error"}
    """
    import torch

    per_sample_timeout: float = cfg.get("inference", {}).get("per_sample_timeout", 60)
    logger = get_logger("llm_infer")
    outputs: List[Dict] = []

    for item in tqdm(items, desc=desc):
        system = item.get("system")
        query = item.get("query", "")
        text = item.get("text", "")
        label_str = item.get("label", "[]")

        result_box: list = [None]
        err_box: list = [None]
        done = threading.Event()

        def _infer(_q=query, _s=system, _rb=result_box, _eb=err_box, _done=done):
            try:
                _rb[0] = predict_sample(
                    model, tokenizer, cfg, _q, system=_s, timeout=per_sample_timeout
                )
            except Exception as exc:  # noqa: BLE001
                _eb[0] = exc
            _done.set()

        threading.Thread(target=_infer, daemon=True).start()
        timed_out = not done.wait(timeout=per_sample_timeout if per_sample_timeout > 0 else None)

        if timed_out:
            logger.warning(f"推理超时（>{per_sample_timeout:.0f}s），已跳过")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred_triples, parse_error = [], True
        elif err_box[0] is not None:
            logger.warning(f"推理失败: {err_box[0]}")
            pred_triples, parse_error = [], True
        else:
            pred_triples, parse_error = (result_box[0] or []), False

        prompt_record = build_model_input(tokenizer, query, system)
        outputs.append({
            "prompt": prompt_record,
            "text": text,
            "predict": str(pred_triples),
            "label": label_str,
            "parse_error": parse_error,
        })

    return outputs


# ──────────────────────────────────────────
# 最终测试推理：单 prompt / 单 LoRA
# ──────────────────────────────────────────

def predict_file(
    cfg: dict,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """
    最终测试推理：读 Alpaca 测试文件（含 system/instruction/output），单 LoRA 推理并保存。

    测试文件由 scripts/build_llm_dataset.py 用「最优 Prompt」生成，system 字段承载固定指令，
    与 LoRA 微调时使用的训练数据格式完全一致。

    Returns:
        预测结果 JSONL 路径。
    """
    import torch

    set_seed(cfg["seed"])
    out_dir = cfg["output"]["dir"]
    logger = get_logger("llm_infer", log_file=cfg["output"].get("log"))
    os.makedirs(out_dir, exist_ok=True)

    test_file = input_file or cfg["data"]["test_file"]
    lora_weights = cfg.get("model", {}).get("lora_weights")
    logger.info(f"最终测试推理：测试文件={test_file}，LoRA={lora_weights!r}")

    samples = load_json(test_file)
    items = [
        {
            "system": s.get("system"),
            "query": s.get("instruction", ""),
            "text": _extract_text(s),
            "label": s.get("output", "[]"),
        }
        for s in samples
    ]

    model, tokenizer = load_model(cfg, lora_weights=lora_weights)
    outputs = run_inference(model, tokenizer, cfg, items, desc="llm_test_infer")

    out_path = output_file or os.path.join(out_dir, "predictions.jsonl")
    save_jsonl(outputs, out_path)
    logger.info(f"测试推理完成 → {out_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def _extract_text(sample: dict) -> str:
    """从 Alpaca 测试样本的 instruction 字段反解原始文本（用于结果展示）。"""
    from methods.llm.prompt_templates import extract_text_from_input
    return extract_text_from_input(sample.get("instruction", ""))
