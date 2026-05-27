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

推理 prompt 直接从测试文件（Alpaca 格式）读取，而非在代码中动态拼接，
确保推理与训练所用 prompt 格式完全一致。
"""

import os
from typing import List, Optional, Tuple

from tqdm import tqdm

from utils.common import get_logger, set_seed
from utils.io_utils import load_json, save_jsonl
from utils.metrics import parse_triple_string

# ──────────────────────────────────────────
# 实体与关系类型常量（与 preprocess.py 共享同一份）
# ──────────────────────────────────────────
_SUBJECT_TYPES = "人物、电视综艺、娱乐人物、影视作品、企业/品牌、歌曲、图书作品、学科专业、机构、行政区、企业、文学作品、学校、国家、历史人物、景点、地点"
_OBJECT_TYPES = "学校、人物、歌曲、音乐专辑、Date、Text、Number、气候、城市、地点、奖项、作品、语言、影视作品、企业、国家"
_SCHEMA_RELATIONS = "导演、出生地、毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言"

_FORMAT_REQ_BASE = (
    "【重要格式要求】\n"
    "你只能输出一个合法的 Python 列表结构，形如：[(\"主体\", \"关系\", \"客体\"), ...]\n"
    "如果没有符合的三元组则输出 []。\n"
    "绝对不要输出任何前言、后语、解释或 Markdown 代码块标记（如 ```python）。"
    "你的回复必须直接以 '[' 开头，以 ']' 结尾。"
)
_FORMAT_REQ_STRICT = (
    "【重要格式要求】\n"
    "你只能输出一个合法的 Python 列表结构，形如：[(\"主体\", \"关系\", \"客体\"), ...]\n"
    "如果没有符合的三元组则输出 []。\n"
    "不要捏造列表以外的关系。绝对不要输出任何解释性文本、废话或 Markdown 代码块标记（如 ```python）。"
    "你的回复必须直接以 '[' 开头，以 ']' 结尾。"
)


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


# ──────────────────────────────────────────
# Prompt 构建（用于单条调用 / 兜底）
# ──────────────────────────────────────────

def _build_prompt_base(text: str) -> str:
    """Prompt A (Base)：与 test_base.json 中的 instruction+input 拼接结果一致"""
    instruction = "请从以下文本中抽取事实三元组。\n" + _FORMAT_REQ_BASE
    return f"{instruction}\n\n### 文本\n{text}\n\n答："


def _build_prompt_with_schema(text: str) -> str:
    """Prompt B (Schema)：与 test_schema.json 中的 instruction+input 拼接结果一致"""
    instruction = (
        "你是一个信息抽取专家。请从以下文本中抽取事实三元组。\n"
        "【Schema约束】\n"
        f"1. 主体只能是：{_SUBJECT_TYPES}\n"
        f"2. 客体只能是：{_OBJECT_TYPES}\n"
        f"3. 关系只能是：{_SCHEMA_RELATIONS}\n"
        + _FORMAT_REQ_STRICT
    )
    return f"{instruction}\n\n### 文本\n{text}\n\n答："


def _build_prompt_with_cot(text: str) -> str:
    """Prompt C (CoT)：与 test_cot.json 中的 instruction+input 拼接结果一致"""
    instruction = (
        "你是一个信息抽取专家。请按照以下步骤从文本中抽取事实三元组：\n"
        f"主体可选类型：{_SUBJECT_TYPES}\n"
        f"客体可选类型：{_OBJECT_TYPES}\n"
        f"关系可选类型：{_SCHEMA_RELATIONS}\n\n"
        "步骤 1: 识别文本中出现的符合类型的主体和客体实体。\n"
        "步骤 2: 根据限定的关系列表，判断实体之间存在的关系。\n"
        "步骤 3: 严格按格式输出最终结果。\n\n"
        + _FORMAT_REQ_STRICT
    )
    return f"{instruction}\n\n### 文本\n{text}\n\n答："


def _build_prompt(text: str, prompt_type: str = "base") -> str:
    """从原始文本动态构建 prompt（兜底用；批量推理时应优先使用 _build_prompt_from_sample）"""
    if prompt_type == "schema":
        return _build_prompt_with_schema(text)
    elif prompt_type == "cot":
        return _build_prompt_with_cot(text)
    else:
        return _build_prompt_base(text)


def _build_prompt_from_sample(sample: dict) -> str:
    """
    从 Alpaca 格式样本直接重建推理 prompt。

    格式：instruction + "\\n\\n" + input + "\\n\\n答："

    这样推理时使用的 prompt 与该样本在训练集中的格式完全一致，
    避免动态拼接带来的细微格式差异。
    """
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    parts = [p for p in (instruction, inp) if p]
    return "\n\n".join(parts) + "\n\n答："


def _extract_text_from_sample(sample: dict) -> str:
    """从 Alpaca 格式样本中提取原始文本（用于结果记录展示）"""
    inp = sample.get("input", "")
    if "### 文本\n" in inp:
        return inp.split("### 文本\n")[-1].strip()
    return inp.strip()


def predict_sample(
    text: str,
    model,
    tokenizer,
    cfg: dict,
    prompt_type: str = "base",
    prompt: str = None,
) -> List[Tuple[str, str, str]]:
    """
    对单条样本推理，返回三元组列表。

    Args:
        text:        原始文本（当 prompt=None 时用于动态构建 prompt）
        model:       已加载的模型
        tokenizer:   tokenizer
        cfg:         llm 配置 dict
        prompt_type: prompt 类型（prompt=None 时生效）
        prompt:      预构建好的完整 prompt（优先使用；来自测试文件时应传入此参数）

    Returns:
        [(subject, predicate, object), ...]
    """
    import torch

    model_cfg = cfg["model"]
    final_prompt = prompt if prompt is not None else _build_prompt(text, prompt_type)
    inputs = tokenizer(final_prompt, return_tensors="pt")

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


def predict_file(
    cfg: dict,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    variant: str = "all",
) -> None:
    """
    批量推理：读取 Alpaca 格式 JSON 文件，逐条推理并保存预测结果。
    对每种 Prompt 类型分别加载对应 LoRA 权重后推理，推理完毕后释放显存再加载下一个。

    推理 prompt 直接从测试文件的 instruction+input 字段重建，不在代码中动态拼接，
    保证与 LlamaFactory 微调时的训练格式完全一致。

    Args:
        cfg:         configs/llm.yaml 配置 dict
        input_file:  覆盖配置文件中的测试集路径（可选）
        output_file: 输出 JSONL 文件路径（仅在 variant 为单一类型时生效）；
                     variant="all" 时忽略此参数，按 predictions_{type}.jsonl 自动命名
        variant:     要运行的 LoRA 变体，可选 base/schema/cot/all（默认 all）
    """
    import torch

    set_seed(cfg["seed"])
    logger = get_logger("llm_infer", log_file=cfg["output"]["log"])
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    # 优先级：命令行 --variant > config 中 model.prompt_type > 全部运行
    cfg_prompt_type = cfg.get("model", {}).get("prompt_type")
    if variant != "all":
        types_to_run = [variant]
    elif cfg_prompt_type:
        types_to_run = [cfg_prompt_type]
    else:
        types_to_run = ["base", "schema", "cot"]
    logger.info(f"将运行变体: {types_to_run}")

    # 各变体的默认测试文件映射（支持 llm.yaml 中的 test_files 字典）
    test_files_map = cfg.get("data", {}).get("test_files", {})

    for prompt_type in types_to_run:
        # 选择测试文件：命令行指定 > test_files 字典 > test_file 默认值
        if input_file:
            test_file = input_file
        elif test_files_map.get(prompt_type):
            test_file = test_files_map[prompt_type]
        else:
            test_file = cfg["data"]["test_file"]

        logger.info(f"=== 开始 Prompt 类型: {prompt_type}，测试文件: {test_file} ===")
        samples = load_json(test_file)

        model, tokenizer = load_model(cfg, prompt_type=prompt_type)
        outputs = []

        for sample in tqdm(samples, desc=f"llm_infer_{prompt_type}"):
            # 从文件中直接读取 prompt，确保与训练格式一致
            prompt_str = _build_prompt_from_sample(sample)
            text = _extract_text_from_sample(sample)
            label_str = sample.get("output", "[]")

            try:
                pred_triples = predict_sample("", model, tokenizer, cfg, prompt_type, prompt=prompt_str)
                parse_error = False
            except Exception as e:
                logger.warning(f"推理失败: {e}")
                pred_triples = []
                parse_error = True

            outputs.append({
                "prompt_type": prompt_type,
                "prompt": prompt_str,
                "text": text,
                "predict": str(pred_triples),
                "label": label_str,
                "parse_error": parse_error,
            })

        # variant="all" 时固定按类型命名，避免多文件相互覆盖
        if variant != "all" and output_file:
            out_path = output_file
        else:
            out_path = f"{cfg['output']['dir']}/predictions_{prompt_type}.jsonl"
        save_jsonl(outputs, out_path)
        logger.info(f"Prompt {prompt_type} 推理完成 → {out_path}")

        # 推理完毕后释放显存，再加载下一个 LoRA
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
