# -*- coding: utf-8 -*-
"""
关系抽取模型：Bert + 线性分类头
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertForRelationExtraction(nn.Module):
    """
    基于 BERT 的关系分类模型

    输入标注了实体边界的文本，拼接主语/宾语的 BERT 表示，
    通过线性层预测关系类别
    """

    def __init__(self, bert_path: str, num_tags: int, dropout: float = 0.1):
        """
        Args:
            bert_path: BERT 预训练模型路径或 HuggingFace 名称
            num_tags: 关系类别数（含"无关系"）
            dropout: Dropout 概率
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(p=dropout)
        # 拼接主语头、主语尾、宾语头、宾语尾共 4 个 token 的表示
        self.linear = nn.Linear(hidden * 4, num_tags)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        token_type_ids: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_ids:        [B, L] 输入 token ID
            attention_masks:  [B, L] 注意力掩码
            token_type_ids:   [B, L] token 类型 ID
            ids:              [B, 4] 四个实体边界位置（+1 是因为加了 [CLS]）

        Returns:
            logits: [B, num_tags]
        """
        bert_out = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )[0]  # [B, L, 768]

        B, L, _ = bert_out.shape
        # 将越界索引裁剪到合法范围
        safe_ids = torch.clamp(ids, min=0, max=L - 1)

        # 提取四个位置的 token 表示并拼接
        ent_repr = torch.cat(
            [torch.index_select(bert_out[i], 0, safe_ids[i].long()).unsqueeze(0) for i in range(B)],
            dim=0,
        )  # [B, 4, 768]
        ent_repr = self.dropout(ent_repr).view(B, -1)  # [B, 4*768]
        return self.linear(ent_repr)  # [B, num_tags]
