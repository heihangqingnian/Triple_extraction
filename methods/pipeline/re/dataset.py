# -*- coding: utf-8 -*-
"""
关系抽取数据集加载与预处理
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# 特征数据类
# ──────────────────────────────────────────

@dataclass
class ReFeature:
    token_ids: List[int]
    attention_masks: List[float]
    token_type_ids: List[int]
    labels: int           # 关系 ID
    ids: List[int]        # [sub_start, sub_end, obj_start, obj_end]（已 +1 for [CLS]）


# ──────────────────────────────────────────
# 预处理：.txt → ReFeature 列表
# ──────────────────────────────────────────

def load_and_featurize(
    txt_path: str,
    bert_model: str,
    max_seq_len: int,
    cache_path: Optional[str] = None,
) -> Tuple[List[ReFeature], List[Tuple]]:
    """
    读取关系抽取 .txt 文件并转换为 BERT 特征

    数据格式（每行 tab 分隔）::

        关系ID \\t 文本 \\t sub_start \\t sub_end \\t obj_start \\t obj_end

    Args:
        txt_path: 数据文件路径
        bert_model: BERT 模型路径
        max_seq_len: 最大序列长度
        cache_path: 特征缓存路径（.pt 文件），存在时直接加载

    Returns:
        (features, callback_info)
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"加载特征缓存: {cache_path}")
        out = torch.load(cache_path, weights_only=False)
        return out[0], out[1]

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    raw = open(txt_path, encoding="utf-8").read().strip()
    features: List[ReFeature] = []
    callback_info: List[Tuple] = []

    for i, line in enumerate(raw.split("\n")):
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        label = int(parts[0])
        text = parts[1]
        ids_raw = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
        # +1 因为 [CLS] 占位置 0
        ids = [x + 1 for x in ids_raw]

        chars = list(text)
        enc = tokenizer(
            chars,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        if i < 3:
            logger.debug(f"sample-{i}: text={text[:30]}... label={label} ids={ids}")

        features.append(ReFeature(
            token_ids=enc["input_ids"],
            attention_masks=enc["attention_mask"],
            token_type_ids=enc["token_type_ids"],
            labels=label,
            ids=ids,
        ))
        callback_info.append((text, label))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save((features, callback_info), cache_path)
        logger.info(f"特征缓存已保存: {cache_path}")

    logger.info(f"共提取 {len(features)} 个特征（from {txt_path}）")
    return features, callback_info


# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────

class ReDataset(Dataset):
    """关系抽取 Dataset"""

    def __init__(self, features: List[ReFeature]):
        self.token_ids = [torch.tensor(f.token_ids, dtype=torch.long) for f in features]
        self.attention_masks = [torch.tensor(f.attention_masks, dtype=torch.float) for f in features]
        self.token_type_ids = [torch.tensor(f.token_type_ids, dtype=torch.long) for f in features]
        self.labels = [torch.tensor(f.labels, dtype=torch.long) for f in features]
        self.ids = [torch.tensor(f.ids, dtype=torch.long) for f in features]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "token_ids": self.token_ids[idx],
            "attention_masks": self.attention_masks[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
            "ids": self.ids[idx],
        }
