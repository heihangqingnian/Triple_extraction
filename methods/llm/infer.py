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
import threading
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
    加载 ChatGLM2-6B 基础模型并挂载对应 prompt_type 的 LoRA 权重。

    自动检测 CUDA 可用性：
      - 有 GPU：使用 float16 + device_map，推理速度快
      - 无 GPU：使用 float32 + CPU，推理速度慢但可运行

    Args:
        cfg:         configs/llm.yaml 解析后的完整配置 dict
        prompt_type: 对应的 LoRA 变体，可选 base / schema / cot

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

    # 支持新格式（dict）和旧格式（str）
    lora_cfg = model_cfg["lora_weights"]
    if isinstance(lora_cfg, dict):
        lora_weights = lora_cfg.get(prompt_type, "")
    else:
        lora_weights = lora_cfg

    logger = get_logger("llm_infer")

    # 根据是否有 GPU 选择 dtype 和 device_map
    use_gpu = torch.cuda.is_available() and device_arg != "cpu"
    if use_gpu:
        dtype = torch.float16
        device_map = device_arg  # "auto" 或指定 GPU
        logger.info(f"使用 GPU（{torch.cuda.get_device_name(0)}），dtype=float16")
    else:
        dtype = torch.float32
        device_map = None  # CPU 不使用 device_map
        logger.warning("CUDA 不可用，回退到 CPU（float32），推理速度会非常慢")

    logger.info(f"加载基础模型: {base_model_path}，dtype={dtype}, device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModel.from_pretrained(base_model_path, **load_kwargs)

    # ChatGLM2Config 使用 num_layers 而非标准的 num_hidden_layers，
    # PEFT 在注册 LoRA 层时会访问 num_hidden_layers，缺失则报 AttributeError。
    if not hasattr(model.config, "num_hidden_layers"):
        for _attr in ("num_layers", "n_layer", "n_layers"):
            if hasattr(model.config, _attr):
                model.config.num_hidden_layers = getattr(model.config, _attr)
                logger.info(f"已将 config.{_attr} → config.num_hidden_layers = {model.config.num_hidden_layers}")
                break

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
    timeout: float = 0,
) -> List[Tuple[str, str, str]]:
    """
    对单条样本推理，返回三元组列表。
    优先使用 model.chat()；若失败则回退到手动 tokenize + model.generate()。
    两条路径均传入 use_cache=False，规避 PEFT LoRA 与 ChatGLM2 KV-cache 的兼容问题。
    timeout > 0 时将 max_time 传给 generate，超时后在当前 token 生成完毕后即停止。
    """
    import torch

    logger = get_logger("llm_infer")
    model_cfg = cfg["model"]
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.1)

    full_prompt = prompt if prompt is not None else _build_prompt(text, prompt_type)

    # 去掉末尾的 "答：" 后缀，避免传入 chat() 时重复
    query = full_prompt
    if query.endswith("\n\n答："):
        query = query[: -len("\n\n答：")]

    max_time = float(timeout) if timeout > 0 else None

    # ── 方法一：model.chat() ──────────────────────────────────
    if hasattr(model, "chat"):
        try:
            chat_kwargs: dict = {"use_cache": False}
            if max_time:
                chat_kwargs["max_time"] = max_time
            response, _ = model.chat(
                tokenizer,
                query,
                history=[],
                max_length=max_new_tokens + 2048,
                temperature=max(temperature, 1e-4),
                do_sample=temperature > 1e-4,
                **chat_kwargs,
            )
            return list(parse_triple_string(response or ""))
        except Exception as e:
            logger.warning(f"[DBG] chat() failed ({type(e).__name__}): {e}")

    # ── 方法二：手动 tokenize + model.generate() ─────────────
    if hasattr(tokenizer, "build_prompt"):
        formatted = tokenizer.build_prompt(query, history=[])
    else:
        formatted = query

    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
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
        "use_cache": False,  # PEFT LoRA 与 ChatGLM2 KV-cache 不兼容，禁用缓存
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

    # 兼容 Tensor 和 GenerateOutput（HuggingFace 新版返回 ModelOutput）
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
            response = m_obj.process_response(response)
            break

    return list(parse_triple_string(response))


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
    per_sample_timeout: float = cfg.get("inference", {}).get("per_sample_timeout", 60)

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

            result_box: list = [None]
            err_box: list = [None]
            done = threading.Event()

            def _infer(ps=prompt_str, _rb=result_box, _eb=err_box, _done=done):
                try:
                    _rb[0] = predict_sample(
                        "", model, tokenizer, cfg, prompt_type,
                        prompt=ps, timeout=per_sample_timeout,
                    )
                except Exception as exc:
                    _eb[0] = exc
                _done.set()

            threading.Thread(target=_infer, daemon=True).start()
            timed_out = not done.wait(
                timeout=per_sample_timeout if per_sample_timeout > 0 else None
            )

            if timed_out:
                logger.warning(f"推理超时（>{per_sample_timeout:.0f}s），已跳过")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pred_triples = []
                parse_error = True
            elif err_box[0] is not None:
                logger.warning(f"推理失败: {err_box[0]}")
                pred_triples = []
                parse_error = True
            else:
                pred_triples = result_box[0] or []
                parse_error = False

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