# -*- coding: utf-8 -*-
"""
NER 模型定义：Bert + BiLSTM + CRF
"""

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel


class BertNer(nn.Module):
    """Bert-BiLSTM-CRF 命名实体识别模型"""

    def __init__(self, bert_path: str, num_tags: int, hidden_dim: int = 768, dropout: float = 0.1):
        """
        Args:
            bert_path: BERT 预训练模型路径或 HuggingFace 模型名
            num_tags: 标签总数（BIO 标签数 + 特殊 token 标签数）
            hidden_dim: BERT 隐藏层维度（默认 768）
            dropout: Dropout 概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bert = BertModel.from_pretrained(
            bert_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        # BiLSTM：输入 768 维，输出 768 维（双向各 384）
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_tags)

    def forward(self, bert_input: dict, batch_labels=None):
        """
        Args:
            bert_input: tokenizer 返回的字典（input_ids / attention_mask / token_type_ids）
            batch_labels: 训练时传入的标签 tensor；推理时为 None

        Returns:
            训练时返回 (loss, logits)，推理时返回 (logits,)
        """
        attention_mask = bert_input["attention_mask"]
        bert_out = self.bert(**bert_input).last_hidden_state   # [B, L, 768]
        lstm_out, _ = self.bilstm(bert_out)                    # [B, L, 768]
        logits = self.classifier(lstm_out)                     # [B, L, num_tags]

        if batch_labels is not None:
            loss = self.crf(logits, batch_labels, mask=attention_mask.gt(0)) * -1
            return loss, logits
        return (logits,)
