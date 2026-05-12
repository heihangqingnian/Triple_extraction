# -*- coding: utf-8 -*-
"""
CasRel 联合三元组抽取模型
"""

import torch
import torch.nn as nn
from transformers import BertModel


class Casrel(nn.Module):
    """
    CasRel：基于 BERT 的级联二元标注关系抽取模型

    流程：
    1. BERT 编码文本 → encoded_text
    2. 线性层预测主语头/尾（2 个二值序列）
    3. 输入选定主语的位置向量，预测宾语头/尾（2 个 rel_num 维序列）
    """

    def __init__(self, bert_model: str, rel_num: int):
        """
        Args:
            bert_model: BERT 模型路径或 HuggingFace 名称
            rel_num: 关系类别数量
        """
        super().__init__()
        self.bert_dim = 768
        self.bert_encoder = BertModel.from_pretrained(bert_model)
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.bert_dim, rel_num)
        self.obj_tails_linear = nn.Linear(self.bert_dim, rel_num)

    def get_encoded_text(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """BERT 编码，返回 [B, L, 768]"""
        return self.bert_encoder(token_ids, attention_mask=mask)[0]

    def get_subs(self, encoded_text: torch.Tensor):
        """预测主语头尾概率，各 [B, L, 1]"""
        heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return heads, tails

    def get_objs_for_specific_sub(
        self,
        sub_head_mapping: torch.Tensor,
        sub_tail_mapping: torch.Tensor,
        encoded_text: torch.Tensor,
    ):
        """
        给定主语位置，预测宾语头尾概率

        Args:
            sub_head_mapping: [B, 1, L]
            sub_tail_mapping: [B, 1, L]
            encoded_text:     [B, L, 768]

        Returns:
            (pred_obj_heads, pred_obj_tails)，各 [B, L, rel_num]
        """
        sub_head = torch.matmul(sub_head_mapping, encoded_text)  # [B, 1, 768]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)  # [B, 1, 768]
        sub = (sub_head + sub_tail) / 2                          # [B, 1, 768]
        encoded_text = encoded_text + sub                        # [B, L, 768]
        obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))  # [B, L, rel_num]
        obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))  # [B, L, rel_num]
        return obj_heads, obj_tails

    def forward(self, data: dict):
        token_ids = data["token_ids"]
        mask = data["mask"]
        encoded_text = self.get_encoded_text(token_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head_mapping = data["sub_head"].unsqueeze(1)   # [B, 1, L]
        sub_tail_mapping = data["sub_tail"].unsqueeze(1)   # [B, 1, L]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(
            sub_head_mapping, sub_tail_mapping, encoded_text
        )
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
