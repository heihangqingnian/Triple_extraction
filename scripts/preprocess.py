#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本：将原始 DuIE2.0 数据一键转换为三种方法所需格式

整合来源：
  - tradition/experiments/data_processing/duie_preprocessor.py
  - tradition/experiments/data_processing/pipeline_formatter.py
  - tradition/experiments/data_processing/joint_formatter.py
  - convert_to_alpaca.py

Usage::

    # 转换所有方法所需格式
    python scripts/preprocess.py --input data/raw/DuIE2.0 --method all

    # 只转换 pipeline 格式
    python scripts/preprocess.py --input data/raw/DuIE2.0 --method pipeline

    # 只转换 joint 格式
    python scripts/preprocess.py --input data/raw/DuIE2.0 --method joint

    # 只转换 llm 格式
    python scripts/preprocess.py --input data/raw/DuIE2.0 --method llm

数据集目录结构（DuIE2.0 原始格式）::

    data/raw/DuIE2.0/
    ├── duie_train.json     # 训练集（每行一个 JSON 对象）
    ├── duie_dev.json       # 验证集
    ├── duie_test2.json     # 测试集（无标注，可选）
    └── duie_schema/
        └── duie_schema.json  # 关系 Schema 定义

每行数据格式::

    {"text": "...", "spo_list": [{"subject": "...", "predicate": "...", "object": {"@value": "..."}}]}
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.common import get_logger, set_seed
from utils.io_utils import save_json

logger = get_logger("preprocess")


# ──────────────────────────────────────────
# DuIE2.0 数据加载
# ──────────────────────────────────────────

def load_duie_file(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载 DuIE2.0 JSONL 文件"""
    samples = []
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在，跳过: {file_path}")
        return samples
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"第 {i+1} 行 JSON 解析失败，跳过")
            if max_samples and len(samples) >= max_samples:
                break
    logger.info(f"从 {file_path} 加载 {len(samples)} 条样本")
    return samples


def load_schema(schema_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    从 Schema 文件提取关系和实体类型映射

    Returns:
        (relation2id, entity_type2id)
    """
    relation2id: Dict[str, int] = {}
    entity_types: Set[str] = set()

    if not os.path.exists(schema_path):
        logger.warning(f"Schema 文件不存在: {schema_path}")
        return relation2id, {}

    with open(schema_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            schema = json.loads(line)
            predicate = schema["predicate"]
            if predicate not in relation2id:
                relation2id[predicate] = len(relation2id)
            entity_types.add(schema["subject_type"])
            obj_type = schema["object_type"]
            if isinstance(obj_type, dict):
                for v in obj_type.values():
                    entity_types.add(str(v))
            else:
                entity_types.add(str(obj_type))

    # 在末尾追加"无关系"标签（Pipeline RE 方法需要）
    if "无关系" not in relation2id:
        relation2id["无关系"] = len(relation2id)

    entity_type2id = {et: i for i, et in enumerate(sorted(entity_types))}
    logger.info(f"Schema 加载完成: {len(relation2id)} 个关系, {len(entity_type2id)} 个实体类型")
    return relation2id, entity_type2id


# ──────────────────────────────────────────
# 共享辅助函数
# ──────────────────────────────────────────

def _extract_entities(sample: Dict) -> List[Dict]:
    """从样本中提取所有实体，带位置信息"""
    text = sample.get("text", "")
    spo_list = sample.get("spo_list", [])
    seen: Set[Tuple] = set()
    entities = []

    for spo in spo_list:
        for entity_text, etype in [
            (spo.get("subject", ""), spo.get("subject_type", "")),
        ]:
            if entity_text and entity_text in text:
                pos = text.find(entity_text)
                key = (pos, entity_text)
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "text": entity_text,
                        "type": etype,
                        "start": pos,
                        "end": pos + len(entity_text) - 1,
                    })

        obj = spo.get("object", {})
        obj_val = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
        obj_type = spo.get("object_type", {})
        obj_type_str = obj_type.get("@value", "") if isinstance(obj_type, dict) else str(obj_type)
        if obj_val and obj_val in text:
            pos = text.find(obj_val)
            key = (pos, obj_val)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": obj_val,
                    "type": obj_type_str,
                    "start": pos,
                    "end": pos + len(obj_val) - 1,
                })

    return entities


# ──────────────────────────────────────────
# Pipeline 格式：NER BIO + RE 实体对分类
# ──────────────────────────────────────────

def _sample_to_bio(sample: Dict) -> Optional[str]:
    """将样本转换为 BIO 格式字符串（字符 + 制表符 + 标签，句子末尾空行）"""
    text = sample.get("text", "")
    if not text:
        return None
    entities = _extract_entities(sample)
    tags = ["O"] * len(text)
    for ent in entities:
        s, e, et = ent["start"], ent["end"], ent["type"]
        if 0 <= s <= e < len(text):
            for i in range(s, e + 1):
                tags[i] = f"B-{et}" if i == s else f"I-{et}"
    lines = []
    for ch, tag in zip(text, tags):
        ch_out = {"\\n": "\\n", "\t": "\\t"}.get(ch, ch)
        lines.append(f"{ch_out}\t{tag}")
    return "\n".join(lines) + "\n"


def _sample_to_re(sample: Dict, relation2id: Dict[str, int], neg_ratio: float = 2.0) -> List[str]:
    """将样本转换为 RE 格式字符串列表"""
    text = sample.get("text", "")
    spo_list = sample.get("spo_list", [])
    entities = _extract_entities(sample)
    no_rel_id = relation2id.get("无关系", max(relation2id.values()))

    # 正样本
    pos_keys: Set[Tuple] = set()
    re_lines: List[str] = []
    for spo in spo_list:
        pred = spo.get("predicate", "")
        sub_text = spo.get("subject", "")
        obj = spo.get("object", {})
        obj_val = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
        if not (pred and sub_text and obj_val):
            continue
        rel_id = relation2id.get(pred)
        if rel_id is None:
            continue
        sub_ent = next((e for e in entities if e["text"] == sub_text), None)
        obj_ent = next((e for e in entities if e["text"] == obj_val), None)
        if not (sub_ent and obj_ent):
            continue
        key = (sub_ent["start"], sub_ent["end"], obj_ent["start"], obj_ent["end"])
        if key in pos_keys:
            continue
        pos_keys.add(key)
        marked = _mark_entities(text, sub_ent, obj_ent)
        re_lines.append(
            f"{rel_id}\t{marked}\t{sub_ent['start']}\t{sub_ent['end']}\t{obj_ent['start']}\t{obj_ent['end']}"
        )

    # 负样本采样
    if pos_keys:
        neg_pairs = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j:
                    continue
                key = (e1["start"], e1["end"], e2["start"], e2["end"])
                if key not in pos_keys:
                    neg_pairs.append((e1, e2))
        n_neg = min(int(len(pos_keys) * neg_ratio), len(neg_pairs))
        for e1, e2 in random.sample(neg_pairs, n_neg):
            marked = _mark_entities(text, e1, e2)
            re_lines.append(
                f"{no_rel_id}\t{marked}\t{e1['start']}\t{e1['end']}\t{e2['start']}\t{e2['end']}"
            )

    return re_lines


def _mark_entities(text: str, sub: Dict, obj: Dict) -> str:
    """用 # 标记主语、$ 标记宾语"""
    pairs = sorted(
        [(sub["start"], sub["end"], sub["text"], "#"), (obj["start"], obj["end"], obj["text"], "$")],
        key=lambda x: x[0], reverse=True,
    )
    marked = text
    for start, end, ent_text, marker in pairs:
        marked = marked[:start] + f"{marker}{ent_text}{marker}" + marked[end + 1:]
    return marked


def format_pipeline(
    samples: List[Dict],
    output_dir: str,
    split: str,
    relation2id: Dict[str, int],
) -> None:
    """生成 Pipeline 方法数据"""
    os.makedirs(output_dir, exist_ok=True)
    ner_file = os.path.join(output_dir, f"{split}_ner.txt")
    re_file = os.path.join(output_dir, f"{split}_re.txt")

    with open(ner_file, "w", encoding="utf-8") as fn:
        for sample in samples:
            bio = _sample_to_bio(sample)
            if bio:
                fn.write(bio + "\n")
    logger.info(f"NER 数据已保存: {ner_file}")

    with open(re_file, "w", encoding="utf-8") as fr:
        for sample in samples:
            for line in _sample_to_re(sample, relation2id):
                fr.write(line + "\n")
    logger.info(f"RE 数据已保存: {re_file}")


# ──────────────────────────────────────────
# Pipeline 测试集：带 gold_triples 的 JSONL 格式
# ──────────────────────────────────────────

def format_pipeline_test(samples: List[Dict], output_file: str) -> None:
    """生成 Pipeline 端到端评估用的测试文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            text = sample.get("text", "")
            spo_list = sample.get("spo_list", [])
            gold_triples = []
            for spo in spo_list:
                obj = spo.get("object", {})
                obj_val = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
                gold_triples.append({
                    "subject": spo.get("subject", ""),
                    "predicate": spo.get("predicate", ""),
                    "object": {"@value": obj_val},
                })
            f.write(json.dumps({"id": i, "text": text, "gold_triples": gold_triples}, ensure_ascii=False) + "\n")
    logger.info(f"Pipeline 测试数据已保存: {output_file}")


# ──────────────────────────────────────────
# Joint 格式：CasRel JSONL
# ──────────────────────────────────────────

def _sample_to_casrel(sample: Dict, relation2id: Dict[str, int], tokenizer) -> Optional[Dict]:
    """将样本转换为 CasRel 格式"""
    text = sample.get("text", "")
    spo_list = sample.get("spo_list", [])
    if not text or not spo_list:
        return None

    tokens = tokenizer.tokenize(text)
    if len(tokens) > 512:
        tokens = tokens[:512]
    text_len = len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    masks = [1] * text_len

    # 构建 s2ro_map
    s2ro_map: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}

    for spo in spo_list:
        pred = spo.get("predicate", "")
        sub_text = spo.get("subject", "")
        obj = spo.get("object", {})
        obj_val = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
        if not (pred and sub_text and obj_val):
            continue
        rel_id = relation2id.get(pred)
        if rel_id is None:
            continue
        sub_toks = tokenizer.tokenize(sub_text)
        obj_toks = tokenizer.tokenize(obj_val)
        sh = _find_head(tokens, sub_toks)
        oh = _find_head(tokens, obj_toks)
        if sh == -1 or oh == -1:
            continue
        sub_span = (sh, sh + len(sub_toks) - 1)
        obj_span = (oh, oh + len(obj_toks) - 1, rel_id)
        s2ro_map.setdefault(sub_span, []).append(obj_span)

    if not s2ro_map:
        return None

    sub_heads_arr = [0] * text_len
    sub_tails_arr = [0] * text_len
    for sh, st in s2ro_map:
        sub_heads_arr[sh] = 1
        sub_tails_arr[st] = 1

    sh_idx, st_idx = random.choice(list(s2ro_map.keys()))
    sub_head_arr = [0] * text_len
    sub_tail_arr = [0] * text_len
    sub_head_arr[sh_idx] = 1
    sub_tail_arr[st_idx] = 1

    rel_num = len(relation2id)
    obj_heads = [[0] * rel_num for _ in range(text_len)]
    obj_tails = [[0] * rel_num for _ in range(text_len)]
    for oh, ot, rid in s2ro_map.get((sh_idx, st_idx), []):
        if oh < text_len and ot < text_len:
            obj_heads[oh][rid] = 1
            obj_tails[ot][rid] = 1

    return {
        "token_ids": token_ids,
        "masks": masks,
        "text_len": text_len,
        "sub_heads": sub_heads_arr,
        "sub_tails": sub_tails_arr,
        "sub_head": sub_head_arr,
        "sub_tail": sub_tail_arr,
        "obj_heads": obj_heads,
        "obj_tails": obj_tails,
        "tokens": tokens,
        "original_spo_list": spo_list,
    }


def _find_head(source: List[str], target: List[str]) -> int:
    tl = len(target)
    for i in range(len(source) - tl + 1):
        if source[i: i + tl] == target:
            return i
    return -1


def format_joint(
    samples: List[Dict],
    output_dir: str,
    split: str,
    relation2id: Dict[str, int],
    bert_model: str,
) -> None:
    """生成 Joint 方法（CasRel）数据"""
    from transformers import BertTokenizer
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"加载 BERT tokenizer: {bert_model}")
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    out_file = os.path.join(output_dir, f"{split}.json")

    converted = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in samples:
            casrel = _sample_to_casrel(sample, relation2id, tokenizer)
            if casrel is None:
                continue
            # s2ro_map 中元组键无法直接序列化，忽略该字段
            f.write(json.dumps(casrel, ensure_ascii=False) + "\n")
            converted += 1

    logger.info(f"Joint 数据已保存: {out_file} ({converted}/{len(samples)} 条)")


# ──────────────────────────────────────────
# LLM 格式：Alpaca JSON
# ──────────────────────────────────────────

ALPACA_INSTRUCTION = (
    "你是一个信息抽取专家。请根据给定的Schema和文本，抽取出所有满足关系的事实三元组，"
    "并以列表形式输出。"
)

ALPACA_SCHEMA = (
    "### Schema\n"
    "主体实体类型:人物,电视综艺,娱乐人物,影视作品,企业/品牌,歌曲,图书作品,学科专业,"
    "机构,行政区,企业,文学作品,学校,国家,历史人物,景点,地点\n"
    "对象实体类型:学校,人物,歌曲,音乐专辑,Date,Text,Number,气候,城市,地点,奖项,作品,"
    "语言,影视作品,企业,国家\n"
    "关系类型:毕业院校,嘉宾,配音,主题曲,代言人,所属专辑,父亲,作者,上映时间,母亲,"
    "专业代码,占地面积,邮政编码,票房,注册资本,主角,妻子,编剧,气候,歌手,获奖,校长,"
    "创始人,首都,丈夫,朝代,饰演,面积,总部地点,祖籍,人口数量,制片人,修业年限,"
    "所在城市,董事长,作词,改编自,出品公司,导演,作曲,主演,主持人,成立日期,简称,"
    "海拔,号,国籍,官方语言"
)


def _sample_to_alpaca(sample: Dict) -> Dict:
    text = sample.get("text", "")
    spo_list = sample.get("spo_list", [])
    triples = []
    for spo in spo_list:
        s = spo.get("subject", "")
        p = spo.get("predicate", "")
        obj = spo.get("object", {})
        o = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
        if s and p and o:
            triples.append(f'("{s}", "{p}", "{o}")')
    return {
        "instruction": ALPACA_INSTRUCTION,
        "input": f"{ALPACA_SCHEMA}\n\n### 文本\n{text}",
        "output": f"[{', '.join(triples)}]",
    }


def format_llm(samples: List[Dict], output_file: str) -> None:
    """生成 LLM（Alpaca 格式）数据"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = [_sample_to_alpaca(s) for s in samples]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"LLM Alpaca 数据已保存: {output_file} ({len(data)} 条)")


# ──────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DuIE2.0 数据预处理：一键转换为三种方法所需格式",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="原始 DuIE2.0 数据目录（包含 duie_train.json / duie_dev.json / duie_schema/）"
    )
    parser.add_argument(
        "--method", default="all",
        choices=["all", "pipeline", "joint", "llm"],
        help="要生成哪种方法的数据（默认 all）"
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="输出根目录（默认 data/processed）"
    )
    parser.add_argument(
        "--bert_model", default="hfl/chinese-bert-wwm",
        help="Joint 方法使用的 BERT tokenizer（默认 hfl/chinese-bert-wwm）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="调试模式：只处理前 200 条"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    input_dir = args.input
    output_dir = args.output
    max_samples = 200 if args.debug else None

    # 加载 Schema
    schema_path = os.path.join(input_dir, "duie_schema", "duie_schema.json")
    relation2id, entity_type2id = load_schema(schema_path)

    # 保存共享映射文件
    shared_dir = os.path.join(output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    save_json(
        {"relation2id": relation2id, "id2relation": {str(v): k for k, v in relation2id.items()}},
        os.path.join(shared_dir, "rel2id.json"),
    )
    save_json(
        {"entity_type2id": entity_type2id, "id2entity_type": {str(v): k for k, v in entity_type2id.items()}},
        os.path.join(shared_dir, "entity2id.json"),
    )
    logger.info(f"映射文件已保存到: {shared_dir}")

    # ── 统一数据划分：三种方法共用同一套 8:1:1 分割 ──────────────────────
    # 将 duie_train.json + duie_dev.json 合并，随机打乱，按 8:1:1 重新划分。
    # set_seed 已在上方调用，shuffle 结果完全可复现。
    train_file = os.path.join(input_dir, "duie_train.json")
    dev_file   = os.path.join(input_dir, "duie_dev.json")
    raw_train  = load_duie_file(train_file, max_samples)
    raw_dev    = load_duie_file(dev_file,   max_samples)
    all_samples = raw_train + raw_dev
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * 0.8)
    n_dev   = int(n * 0.1)
    splits = {
        "train": all_samples[:n_train],
        "dev":   all_samples[n_train: n_train + n_dev],
        "test":  all_samples[n_train + n_dev:],
    }
    logger.info(
        f"统一数据划分（8:1:1）："
        f"train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}"
    )
    # ─────────────────────────────────────────────────────────────────────

    # Pipeline
    # NER 和 RE 阶段只需 train / dev 训练；test 分割仅生成带 gold_triples 的 JSONL
    # 供端到端 Pipeline 评估使用，不参与 NER / RE 训练。
    if args.method in ("all", "pipeline"):
        logger.info("=== 生成 Pipeline 数据 ===")
        pipeline_dir = os.path.join(output_dir, "pipeline")
        for split_name in ("train", "dev"):
            format_pipeline(splits[split_name], os.path.join(pipeline_dir, split_name), split_name, relation2id)
        format_pipeline_test(splits["test"], os.path.join(pipeline_dir, "test", "pipeline_test.jsonl"))
        logger.info("Pipeline 数据生成完成")

    # Joint
    if args.method in ("all", "joint"):
        logger.info("=== 生成 Joint 数据 ===")
        joint_dir = os.path.join(output_dir, "joint")
        for split_name, samples in splits.items():
            format_joint(samples, os.path.join(joint_dir, split_name), split_name, relation2id, args.bert_model)
        save_json(relation2id, os.path.join(joint_dir, "rel2id.json"))
        logger.info("Joint 数据生成完成")

    # LLM
    if args.method in ("all", "llm"):
        logger.info("=== 生成 LLM 数据 ===")
        llm_dir = os.path.join(output_dir, "llm")
        os.makedirs(llm_dir, exist_ok=True)
        for split_name, samples in splits.items():
            format_llm(samples, os.path.join(llm_dir, f"{split_name}.json"))
        logger.info("LLM 数据生成完成")

    logger.info(f"所有数据已保存到: {output_dir}")


if __name__ == "__main__":
    main()
