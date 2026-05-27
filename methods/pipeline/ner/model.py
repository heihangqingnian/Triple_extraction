# -*- coding: utf-8 -*-
"""
NER 模型定义：Bert + BiLSTM + CRF（支持消融实验）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertModel


class BertNer(nn.Module):
    """Bert-BiLSTM-CRF 命名实体识别模型（支持消融实验）"""

    def __init__(
        self,
        bert_path: str,
        num_tags: int,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        use_bilstm: bool = True,
        use_crf: bool = True,
    ):
        """
        Args:
            bert_path: BERT 预训练模型路径或 HuggingFace 模型名
            num_tags: 标签总数（BIO 标签数 + 特殊 token 标签数）
            hidden_dim: BERT 隐藏层维度（默认 768）
            dropout: Dropout 概率
            use_bilstm: 是否使用 BiLSTM 层（消融实验：False = BERT-CRF）
            use_crf: 是否使用 CRF 层（消融实验：False = BERT-Linear/Softmax）
        """
        super().__init__()
        self.use_bilstm = use_bilstm
        self.use_crf = use_crf
        self.dropout = nn.Dropout(p=dropout)
        self.bert = BertModel.from_pretrained(
            bert_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        # BiLSTM：输入 768 维，输出 768 维（双向各 384）
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        if self.use_crf:
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

        # 消融：根据参数选择是否使用 BiLSTM
        if self.use_bilstm:
            lstm_out, _ = self.bilstm(bert_out)                    # [B, L, 768]
            hidden = lstm_out
        else:
            hidden = bert_out                                      # [B, L, 768]

        logits = self.classifier(hidden)                           # [B, L, num_tags]

        if batch_labels is not None:
            if self.use_crf:
                # 使用 CRF 损失
                loss = self.crf(logits, batch_labels, mask=attention_mask.gt(0)) * -1
                return loss, logits
            else:
                # 消融：使用交叉熵损失（Linear + Softmax）
                log_probs = F.log_softmax(logits, dim=-1)           # [B, L, num_tags]
                valid_mask = attention_mask.gt(0)
                # 计算有效位置的平均损失
                loss = F.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)),
                    batch_labels.view(-1),
                    ignore_index=-1,
                    reduction='none'
                )
                loss = (loss * valid_mask.view(-1).float()).sum() / valid_mask.sum()
                return loss, logits
        return (logits,)

    def decode(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        解码 logits 为标签序列

        Args:
            logits: [B, L, num_tags]
            mask: [B, L] 有效位置掩码

        Returns:
            decoded: [B, L] 标签序列
        """
        if self.use_crf:
            return self.crf.decode(logits, mask=mask)
        else:
            # 贪心解码：取 argmax，转为 list of lists 与 CRF decode 保持一致
            return logits.argmax(dim=-1).cpu().tolist()
