# -*- coding: utf-8 -*-
"""
LLM 抽取 Prompt 的单一事实来源（Single Source of Truth）。

本模块集中定义所有 Prompt 变体的指令文本与拼接逻辑，供三处共用：
  1. 免训练寻优       methods/llm/prompt_search.py
  2. 最终测试推理     methods/llm/infer.py
  3. 微调数据构造     scripts/build_llm_dataset.py（并据此生成 LLaMA-Factory dataset_info.json）

设计要点
--------
* **固定指令 = system 文本**：每个 Prompt 变体的指令对所有样本完全相同，
  因此作为 system 提示，不在每条数据里重复（用户确认采用该方案）。
* **query/input = 句子文本**：所有变体共用 ``build_input_field``，差异只在 system。
* **训练/推理必须逐字一致**：微调（LLaMA-Factory chatglm2 模板 + system）与推理
  （infer.py 复现同一模板 + system）的提示串必须一致，否则 LoRA「肌肉记忆」失配会掉点。
  ``assemble_preview`` 用于打印两侧拼接串以便人工 diff 校验。
"""

from typing import List, Optional, Tuple

# ──────────────────────────────────────────
# 实体与关系类型常量（DuIE2.0，三处共享同一份）
# ──────────────────────────────────────────
SUBJECT_TYPES = "人物、电视综艺、娱乐人物、影视作品、企业/品牌、歌曲、图书作品、学科专业、机构、行政区、企业、文学作品、学校、国家、历史人物、景点、地点"
OBJECT_TYPES = "学校、人物、歌曲、音乐专辑、Date、Text、Number、气候、城市、地点、奖项、作品、语言、影视作品、企业、国家"
SCHEMA_RELATIONS = "导演、出生地、毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言"

# 候选 Prompt 变体（免训练寻优在这些里选最优）
PROMPT_TYPES: List[str] = ["base", "schema", "cot"]

# 输入文本字段前缀（query 部分的统一标记，所有变体一致）
_TEXT_MARKER = "### 文本"

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


# ──────────────────────────────────────────
# 指令（system 文本）
# ──────────────────────────────────────────

def build_instruction(prompt_type: str = "base") -> str:
    """返回指定 Prompt 变体的固定指令（即 system 文本）。

    Args:
        prompt_type: base / schema / cot

    Raises:
        ValueError: 未知的 prompt_type
    """
    if prompt_type == "base":
        return "请从以下文本中抽取事实三元组。\n" + _FORMAT_REQ_BASE
    if prompt_type == "schema":
        return (
            "你是一个信息抽取专家。请从以下文本中抽取事实三元组。\n"
            "【Schema约束】\n"
            f"1. 主体只能是：{SUBJECT_TYPES}\n"
            f"2. 客体只能是：{OBJECT_TYPES}\n"
            f"3. 关系只能是：{SCHEMA_RELATIONS}\n"
            + _FORMAT_REQ_STRICT
        )
    if prompt_type == "cot":
        return (
            "你是一个信息抽取专家。请按照以下步骤从文本中抽取事实三元组：\n"
            f"主体可选类型：{SUBJECT_TYPES}\n"
            f"客体可选类型：{OBJECT_TYPES}\n"
            f"关系可选类型：{SCHEMA_RELATIONS}\n\n"
            "步骤 1: 识别文本中出现的符合类型的主体和客体实体。\n"
            "步骤 2: 根据限定的关系列表，判断实体之间存在的关系。\n"
            "步骤 3: 严格按格式输出最终结果。\n\n"
            + _FORMAT_REQ_STRICT
        )
    raise ValueError(f"未知的 prompt_type: {prompt_type!r}，可选 {PROMPT_TYPES}")


# ──────────────────────────────────────────
# 输入（query）字段
# ──────────────────────────────────────────

def build_input_field(text: str) -> str:
    """构造 query/input 字段（所有变体一致）：``### 文本\\n{text}``"""
    return f"{_TEXT_MARKER}\n{text}"


def extract_text_from_input(input_str: str) -> str:
    """从 input 字段反解出原始句子文本（去掉 ``### 文本`` 标记）。"""
    if f"{_TEXT_MARKER}\n" in input_str:
        return input_str.split(f"{_TEXT_MARKER}\n")[-1].strip()
    return input_str.strip()


def _triples_to_output(triples: List) -> str:
    """把 [(s,p,o), ...] / [[s,p,o], ...] 序列化为模型输出字符串。"""
    parts = []
    for t in triples:
        s, p, o = t[0], t[1], t[2]
        if s and p and o:
            parts.append(f'("{s}", "{p}", "{o}")')
    return f"[{', '.join(parts)}]"


def build_fewshot_block(examples: List[Tuple[str, List]]) -> str:
    """构造 few-shot 示例块（仅免训练寻优使用）。

    Args:
        examples: [(text, triples), ...]，triples 为 [(s,p,o), ...]

    Returns:
        形如 "【示例】\\n文本：...\\n输出：[...]\\n\\n文本：...\\n输出：[...]\\n" 的字符串；
        examples 为空时返回空串。
    """
    if not examples:
        return ""
    lines = ["【示例】"]
    for text, triples in examples:
        lines.append(f"文本：{text}")
        lines.append(f"输出：{_triples_to_output(triples)}")
        lines.append("")
    return "\n".join(lines)


def build_query(text: str, few_shot_examples: Optional[List[Tuple[str, List]]] = None) -> str:
    """构造完整 query：可选 few-shot 示例块 + 目标文本字段。

    few-shot 示例只在免训练寻优时拼入 query；微调/最终推理始终为 zero-shot（few_shot_examples=None）。
    """
    block = build_fewshot_block(few_shot_examples or [])
    field = build_input_field(text)
    return f"{block}\n{field}" if block else field


# ──────────────────────────────────────────
# 预览 / 校验
# ──────────────────────────────────────────

def assemble_preview(
    prompt_type: str,
    text: str,
    few_shot_examples: Optional[List[Tuple[str, List]]] = None,
) -> str:
    """拼出「逻辑提示串」（system + query），用于人工核对训练侧与推理侧是否一致。

    注意：这是与模板无关的逻辑拼接（system 段 + query 段），用于直观 diff；
    实际喂给 ChatGLM2 时还会套上 ``[Round 1]\\n\\n问：…\\n\\n答：`` 模板（见 infer.py），
    该模板由 LLaMA-Factory 的 ``chatglm2`` 模板与 infer.py 共同保证一致。
    """
    system = build_instruction(prompt_type)
    query = build_query(text, few_shot_examples)
    return f"[system]\n{system}\n\n[query]\n{query}"
