# -*- coding: utf-8 -*-
"""
CasRel 数据加载器
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class CasRelDataset(Dataset):
    """CasRel 数据集，每条数据为预处理后的 JSON 对象"""

    def __init__(self, data_file: str, rel2id: dict, rel_num: int, tokenizer: BertTokenizer, debug: bool = False):
        """
        Args:
            data_file: 数据文件路径（JSONL 格式，每行一个预处理样本）
            rel2id: 关系名称 -> ID 映射
            rel_num: 关系数量
            tokenizer: BERT tokenizer（用于 padding token id 参考）
            debug: True 时只加载前 500 条
        """
        self.rel2id = rel2id
        self.rel_num = rel_num
        self.tokenizer = tokenizer
        self.samples = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
                if debug and len(self.samples) >= 500:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        token_ids = np.array(sample.get("token_ids", []), dtype=np.int64)
        masks = np.array(sample.get("masks", []), dtype=np.int64)
        text_len = sample.get("text_len", len(token_ids))

        sub_heads = np.array(sample.get("sub_heads", [0] * text_len)[:text_len])
        sub_tails = np.array(sample.get("sub_tails", [0] * text_len)[:text_len])
        sub_head = np.array(sample.get("sub_head", [0] * text_len)[:text_len])
        sub_tail = np.array(sample.get("sub_tail", [0] * text_len)[:text_len])

        # obj_heads / obj_tails: [text_len, rel_num]
        obj_heads = np.zeros((text_len, self.rel_num), dtype=np.float32)
        obj_tails = np.zeros((text_len, self.rel_num), dtype=np.float32)

        for i, row in enumerate(sample.get("obj_heads", [])[:text_len]):
            if isinstance(row, list):
                for j in range(min(self.rel_num, len(row))):
                    if row[j]:
                        obj_heads[i][j] = 1.0

        for i, row in enumerate(sample.get("obj_tails", [])[:text_len]):
            if isinstance(row, list):
                for j in range(min(self.rel_num, len(row))):
                    if row[j]:
                        obj_tails[i][j] = 1.0

        bert_tokens = sample.get("tokens", [])
        triple_list = sample.get("original_spo_list", [])

        return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triple_list, bert_tokens


def casrel_collate_fn(batch):
    """将一个 batch 的样本 padding 至相同长度"""
    batch = [x for x in batch if x is not None]
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(*batch)

    B = len(batch)
    max_len = max(text_len)
    rel_num = obj_heads[0].shape[1] if len(obj_heads) > 0 else 49

    batch_token_ids = torch.LongTensor(B, max_len).zero_()
    batch_masks = torch.LongTensor(B, max_len).zero_()
    batch_sub_heads = torch.zeros(B, max_len)
    batch_sub_tails = torch.zeros(B, max_len)
    batch_sub_head = torch.zeros(B, max_len)
    batch_sub_tail = torch.zeros(B, max_len)
    batch_obj_heads = torch.zeros(B, max_len, rel_num)
    batch_obj_tails = torch.zeros(B, max_len, rel_num)

    for i in range(B):
        tl = text_len[i]
        batch_token_ids[i, :tl].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :tl].copy_(torch.from_numpy(masks[i]))
        batch_sub_heads[i, :tl].copy_(torch.from_numpy(sub_heads[i]))
        batch_sub_tails[i, :tl].copy_(torch.from_numpy(sub_tails[i]))
        batch_sub_head[i, :tl].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :tl].copy_(torch.from_numpy(sub_tail[i]))
        batch_obj_heads[i, :tl, :].copy_(torch.from_numpy(obj_heads[i]))
        batch_obj_tails[i, :tl, :].copy_(torch.from_numpy(obj_tails[i]))

    return {
        "token_ids": batch_token_ids,
        "mask": batch_masks,
        "sub_heads": batch_sub_heads,
        "sub_tails": batch_sub_tails,
        "sub_head": batch_sub_head,
        "sub_tail": batch_sub_tail,
        "obj_heads": batch_obj_heads,
        "obj_tails": batch_obj_tails,
        "triples": triples,
        "tokens": tokens,
    }


def get_loader(
    data_dir: str,
    prefix: str,
    rel2id: dict,
    rel_num: int,
    tokenizer: BertTokenizer,
    batch_size: int,
    is_test: bool = False,
    debug: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    构建 DataLoader

    Args:
        data_dir: 数据根目录（data/processed/joint）
        prefix: 子目录名称（"train" / "dev" / "test"）
        rel2id: 关系映射
        rel_num: 关系数量
        tokenizer: BERT tokenizer
        batch_size: 批大小
        is_test: True 时 batch_size=1，不 shuffle
        debug: 调试模式，只加载少量数据
    """
    data_file = os.path.join(data_dir, prefix, f"{prefix}.json")
    dataset = CasRelDataset(data_file, rel2id, rel_num, tokenizer, debug=debug)

    return DataLoader(
        dataset,
        batch_size=1 if is_test else batch_size,
        shuffle=not is_test,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=casrel_collate_fn,
    )


class DataPreFetcher:
    """CUDA stream 预取器，提升数据加载效率（需要 CUDA）"""

    def __init__(self, loader: DataLoader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = v.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
