# -*- coding: utf-8 -*-
"""
NER 数据集加载
"""

import copy
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class NerDataset(Dataset):
    """NER 数据集，读取 BIO 格式的 .txt 文件"""

    def __init__(
        self,
        data_dir: str,
        mode: str,
        tokenizer: BertTokenizer,
        txt2label: Dict[str, int],
        device: torch.device,
    ):
        """
        Args:
            data_dir: 预处理数据目录（data/processed/pipeline/train 等）
            mode: "train" / "dev" / "test"
            tokenizer: BERT tokenizer
            txt2label: 标签文本 -> 标签 ID 映射
            device: 数据加载到的设备
        """
        self.tokenizer = tokenizer
        self.txt2label = txt2label
        self.device = device
        self.data = self._load(data_dir, mode)

    def _load(self, data_dir: str, mode: str) -> List[Tuple[List[int], List[int]]]:
        """加载 BIO 格式文件，返回 (token_id_list, label_id_list) 对"""
        # 按优先级查找数据文件
        candidates = [
            os.path.join(data_dir, mode, f"{mode}_ner.txt"),
            os.path.join(data_dir, f"{mode}_ner.txt"),
            os.path.join(data_dir, mode, f"{mode}.txt"),
        ]
        data_file = None
        for c in candidates:
            if os.path.exists(c):
                data_file = c
                break
        if data_file is None:
            raise FileNotFoundError(
                f"找不到 {mode} NER 数据文件，已尝试路径：{candidates}"
            )

        lines_buf: List[str] = []
        flags_buf: List[str] = []
        all_lines: List[List[str]] = []
        all_flags: List[List[str]] = []

        with open(data_file, encoding="utf-8") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                if not raw.strip():
                    if lines_buf:
                        all_lines.append(copy.deepcopy(lines_buf))
                        all_flags.append(copy.deepcopy(flags_buf))
                        lines_buf.clear()
                        flags_buf.clear()
                    continue
                parts = raw.split("\t", 1) if "\t" in raw else raw.split(" ", 1)
                if len(parts) != 2:
                    continue
                word, flag = parts
                # 还原转义字符
                word = {"\\n": "\n", "\\t": "\t"}.get(word, word)
                lines_buf.append(word)
                flags_buf.append(flag.strip())

        # 处理最后一个未被空行结束的句子
        if lines_buf:
            all_lines.append(lines_buf)
            all_flags.append(flags_buf)

        data = []
        for words, tags in zip(all_lines, all_flags):
            token_ids = [self.tokenizer.convert_tokens_to_ids(w) for w in words]
            label_ids = [self.txt2label.get(t, 0) for t in tags]
            assert len(token_ids) == len(label_ids)
            data.append((token_ids, label_ids))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def ner_collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    tokenizer: BertTokenizer,
    txt2label: Dict[str, int],
    device: torch.device,
    max_len: int = 510,
):
    """
    将一个 batch 的样本填充为相同长度并构建 BERT 输入

    Args:
        batch: [(token_ids, label_ids), ...]
        tokenizer: BERT tokenizer
        txt2label: 标签文本 -> ID 映射
        device: 目标设备
        max_len: 最大截断长度（不含 [CLS][SEP]，默认 510）
    """
    sentences = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    # 截断
    sentences = [s[:max_len] for s in sentences]
    labels = [l[:max_len] for l in labels]

    batch_max = max(len(s) for s in sentences) + 2  # +2 for [CLS][SEP]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    cls_label = txt2label.get(tokenizer.cls_token, 0)
    sep_label = txt2label.get(tokenizer.sep_token, 0)
    pad_label = txt2label.get(tokenizer.pad_token, 0)

    input_ids_batch, token_type_ids_batch, attention_mask_batch, label_batch = [], [], [], []
    for sent, lbl in zip(sentences, labels):
        seq = [cls_id] + sent + [sep_id]
        pad_len = batch_max - len(seq)
        attn = [1] * len(seq) + [0] * pad_len
        seq = seq + [pad_id] * pad_len
        lbl_seq = [cls_label] + lbl + [sep_label] + [pad_label] * pad_len

        input_ids_batch.append(seq)
        token_type_ids_batch.append([0] * batch_max)
        attention_mask_batch.append(attn)
        label_batch.append(lbl_seq)

    bert_input = {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long, device=device),
        "token_type_ids": torch.tensor(token_type_ids_batch, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long, device=device),
    }
    label_tensor = torch.tensor(label_batch, dtype=torch.long, device=device)
    return bert_input, label_tensor
