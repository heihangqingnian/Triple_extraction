# -*- coding: utf-8 -*-
"""
ChatGLM2-6B + LoRA 推理接口
提供 LoRA 权重加载和批量推理功能

LoRA 权重配置支持两种格式（向后兼容）：
  旧格式: lora_weights: "models/chatglm2-lora"
  新格式: lora_weights:
            base:   "models/chatglm2-lora-base"
            schema: "models/chatglm2-lora-schema"
            cot:    "models/chatglm2-lora-cot"
"""

import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.common import get_logger, set_seed
from utils.io_utils import load_json, save_jsonl
from utils.metrics import parse_triple_string


def load_model(cfg: dict, prompt_type: str = "base"):
    """
    加载 ChatGLM2-6B 基础模型并挂载对应 prompt_type 的 LoRA 权重

    Args:
        cfg:         configs/llm.yaml 解析后的完整配置 dict
        prompt_type: 对应的 LoRA 变体，可选 base / schema / cot

    Returns:
        (model, tokenizer)
    """
    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("请安装 peft 和 transformers：pip install peft transformers")

    model_cfg = cfg["model"]
    base_model = model_cfg["base_model"]
    device_arg = model_cfg.get("device", "auto")

    # 支持新格式（dict）和旧格式（str）
    lora_cfg = model_cfg["lora_weights"]
    if isinstance(lora_cfg, dict):
        lora_weights = lora_cfg.get(prompt_type, "")
    else:
        lora_weights = lora_cfg

    logger = get_logger("llm_infer")
    logger.info(f"加载基础模型: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_arg if device_arg != "cpu" else None,
    )

    if lora_weights and os.path.exists(lora_weights):
        logger.info(f"[{prompt_type}] 加载 LoRA 权重: {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
    else:
        logger.warning(f"[{prompt_type}] LoRA 权重目录不存在或未配置: {lora_weights!r}，使用基础模型推理")

    model = model.eval()
    return model, tokenizer


def _build_prompt_base(text: str) -> str:
    """Prompt A (Base): 直接给出句子，要求输出三元组"""
    instruction = (
        "请从以下文本中抽取事实三元组。"
        "输出格式为：[(\"主体\", \"关系\", \"客体\"), ...]，如果没有符合的三元组则输出 []。"
    )
    return f"{instruction}\n\n### 文本\n{text}\n\n答："


def _build_prompt_with_schema(text: str) -> str:
    """Prompt B (添加 Schema 约束): 在 Prompt 中明确告诉模型 DUIE 包含哪些关系类型和实体类型"""
    subject_types = "人物、电视综艺、娱乐人物、影视作品、企业/品牌、歌曲、图书作品、学科专业、机构、行政区、企业、文学作品、学校、国家、历史人物、景点、地点"
    object_types = "学校、人物、歌曲、音乐专辑、Date、Text、Number、气候、城市、地点、奖项、作品、语言、影视作品、企业、国家"
    schema_relations = "导演、出生地、毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言"

    instruction = (
        f"你是一个信息抽取专家。请从以下文本中抽取事实三元组。\n"
        f"主体实体类型只能从以下列表中选择：{subject_types}\n"
        f"对象实体类型只能从以下列表中选择：{object_types}\n"
        f"关系类型只能从以下列表中选择：{schema_relations}\n"
        "输出格式为：[(\"主体\", \"关系\", \"客体\"), ...]，如果没有符合的三元组则输出 []。"
    )
    return f"{instruction}\n\n### 文本\n{text}\n\n答："


def _build_prompt_with_cot(text: str) -> str:
    """Prompt C (添加 CoT / 思想链): 提示模型先找出实体，再判断关系，最后输出，同时指定 Schema 约束"""
    subject_types = "人物、电视综艺、娱乐人物、影视作品、企业/品牌、歌曲、图书作品、学科专业、机构、行政区、企业、文学作品、学校、国家、历史人物、景点、地点"
    object_types = "学校、人物、歌曲、音乐专辑、Date、Text、Number、气候、城市、地点、奖项、作品、语言、影视作品、企业、国家"
    schema_relations = "导演、出生地、毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言"

    instruction = (
        "你是一个信息抽取专家。请按照以下步骤从文本中抽取事实三元组：\n"
        f"主体实体类型只能从以下列表中选择：{subject_types}\n"
        f"对象实体类型只能从以下列表中选择：{object_types}\n"
        f"关系类型只能从以下列表中选择：{schema_relations}\n"
        "1. 首先识别文本中出现的所有符合上述类型的实体；\n"
        "2. 然后根据给定的关系类型列表，判断这些实体之间可能存在的关系；\n"
        "3. 最后将符合条件的三元组以列表形式输出。\n"
        "输出格式为：[(\"主体\", \"关系\", \"客体\"), ...]，如果没有符合的三元组则输出 []。"
    )
    return f"{instruction}\n\n### 文本\n{text}\n\n答："


def _build_prompt(text: str, prompt_type: str = "base") -> str:
    """构建推理 prompt，支持三种类型：base, schema, cot"""
    if prompt_type == "schema":
        return _build_prompt_with_schema(text)
    elif prompt_type == "cot":
        return _build_prompt_with_cot(text)
    else:
        return _build_prompt_base(text)


def predict_sample(
    text: str,
    model,
    tokenizer,
    cfg: dict,
    prompt_type: str = "base",
) -> List[Tuple[str, str, str]]:
    """
    对单条文本推理，返回三元组列表

    Args:
        text:        输入文本
        model:       已加载的模型
        tokenizer:   tokenizer
        cfg:         llm 配置 dict
        prompt_type: prompt 类型，可选值: base, schema, cot

    Returns:
        [(subject, predicate, object), ...]
    """
    import torch

    model_cfg = cfg["model"]
    prompt = _build_prompt(text, prompt_type)
    inputs = tokenizer(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=model_cfg.get("max_new_tokens", 512),
            temperature=model_cfg.get("temperature", 0.1),
            do_sample=False,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return list(parse_triple_string(generated))


def _extract_text_from_sample(sample: dict) -> str:
    """从 Alpaca 格式样本中提取原始文本"""
    inp = sample.get("input", "")
    if "### 文本\n" in inp:
        return inp.split("### 文本\n")[-1].strip()
    return inp.strip()


def predict_file(
    cfg: dict,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """
    批量推理：读取 Alpaca 格式 JSON 文件，逐条推理并保存预测结果。
    对每种 Prompt 类型分别加载对应 LoRA 权重后推理，推理完毕后释放显存再加载下一个。

    Args:
        cfg:         configs/llm.yaml 配置 dict
        input_file:  输入文件（Alpaca JSON 格式），默认使用 cfg 中的 test_file
        output_file: 输出 JSONL 文件前缀，默认使用 cfg 中的 output.dir
    """
    import torch

    set_seed(cfg["seed"])
    logger = get_logger("llm_infer", log_file=cfg["output"]["log"])
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    test_file = input_file or cfg["data"]["test_file"]
    logger.info(f"推理输入文件: {test_file}")
    samples = load_json(test_file)

    for prompt_type in ["base", "schema", "cot"]:
        logger.info(f"=== 开始 Prompt 类型: {prompt_type} ===")

        model, tokenizer = load_model(cfg, prompt_type=prompt_type)
        outputs = []

        for sample in tqdm(samples, desc=f"llm_infer_{prompt_type}"):
            text = _extract_text_from_sample(sample)
            label_str = sample.get("output", "[]")

            try:
                pred_triples = predict_sample(text, model, tokenizer, cfg, prompt_type)
                parse_error = False
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                pred_triples = []
                parse_error = True

            outputs.append({
                "prompt_type": prompt_type,
                "prompt": _build_prompt(text, prompt_type),
                "text": text,
                "predict": str(pred_triples),
                "label": label_str,
                "parse_error": parse_error,
            })

        out_path = output_file or f"{cfg['output']['dir']}/predictions_{prompt_type}.jsonl"
        save_jsonl(outputs, out_path)
        logger.info(f"Prompt {prompt_type} 推理完成 → {out_path}")

        # 推理完毕后释放显存，再加载下一个 LoRA
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
