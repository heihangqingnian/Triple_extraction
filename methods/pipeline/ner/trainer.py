# -*- coding: utf-8 -*-
"""
NER 训练、评估、预测
"""

import json
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from methods.pipeline.ner.dataset import NerDataset, ner_collate_fn
from methods.pipeline.ner.model import BertNer
from utils.common import get_device, get_logger, set_seed
from utils.io_utils import load_checkpoint, save_checkpoint, save_json


def _build_label_maps(entity2id_path: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """根据实体类型映射文件构建 BIO 标签列表"""
    with open(entity2id_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_types = list(data.get("entity_type2id", {}).keys())
    if not entity_types:
        # 默认 DuIE2.0 实体类型
        entity_types = [
            "人物", "机构", "企业", "影视作品", "歌曲", "图书作品",
            "国家", "行政区", "地点", "学校", "电视综艺", "学科专业",
            "历史人物", "娱乐人物", "音乐专辑", "作品", "气候", "语言",
        ]

    tags = ["O"]
    for etype in entity_types:
        tags.extend([f"B-{etype}", f"I-{etype}"])
    tags.extend(["[CLS]", "[SEP]", "[PAD]"])

    txt2label = {t: i for i, t in enumerate(tags)}
    label2txt = {i: t for i, t in enumerate(tags)}
    return tags, txt2label, label2txt


@torch.no_grad()
def _evaluate(model: BertNer, loader: DataLoader, label2txt: Dict[int, str]) -> Dict:
    """在 loader 上评估，返回 precision / recall / f1 及详细报告"""
    model.eval()
    true_all, pred_all = [], []
    special = {"[CLS]", "[SEP]", "[PAD]"}

    for bert_input, batch_labels in loader:
        logits = model(bert_input, batch_labels)[1]
        decoded = model.crf.decode(logits, mask=bert_input["attention_mask"].gt(0))
        masks = bert_input["attention_mask"].detach().cpu().tolist()
        true_ids = batch_labels.detach().cpu().tolist()

        for pred, true, mask in zip(decoded, true_ids, masks):
            valid = int(sum(mask))
            # 去掉 [CLS] 和 [SEP]
            true_seq = [label2txt.get(i, "O") for i in true[1: valid - 1]]
            pred_seq = [label2txt.get(i, "O") for i in pred[1: valid - 1]]
            # 将特殊标签替换为 O
            true_seq = ["O" if t in special else t for t in true_seq]
            pred_seq = ["O" if t in special else t for t in pred_seq]
            true_all.append(true_seq)
            pred_all.append(pred_seq)

    p = precision_score(true_all, pred_all)
    r = recall_score(true_all, pred_all)
    f1 = f1_score(true_all, pred_all)
    report = classification_report(true_all, pred_all)
    return {"precision": p, "recall": r, "f1": f1, "report": report}


def train(cfg: dict) -> None:
    """
    NER 训练入口

    Args:
        cfg: configs/pipeline.yaml 中的完整配置 dict
    """
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    logger = get_logger("ner_train", log_file=cfg["output"]["log"])

    ner_cfg = cfg["ner"]
    data_cfg = cfg["data"]

    # 构建标签映射
    _, txt2label, label2txt = _build_label_maps(data_cfg["entity2id"])
    num_tags = len(txt2label)
    logger.info(f"标签总数: {num_tags}")

    tokenizer = BertTokenizer.from_pretrained(ner_cfg["bert_model"])

    collate = partial(ner_collate_fn, tokenizer=tokenizer, txt2label=txt2label, device=device)

    train_set = NerDataset(data_cfg["processed_dir"], "train", tokenizer, txt2label, device)
    dev_set = NerDataset(data_cfg["processed_dir"], "dev", tokenizer, txt2label, device)
    train_loader = DataLoader(train_set, batch_size=ner_cfg["batch_size"], shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=ner_cfg["batch_size"], shuffle=False, collate_fn=collate)

    model = BertNer(
        bert_path=ner_cfg["bert_model"],
        num_tags=num_tags,
        hidden_dim=768,
        dropout=ner_cfg["dropout"],
    ).to(device)

    # 差分学习率：CRF 层使用更高学习率
    bert_params = [p for n, p in model.named_parameters() if "bert" in n]
    other_params = [p for n, p in model.named_parameters() if "bert" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": ner_cfg["learning_rate"]},
            {"params": other_params, "lr": ner_cfg["crf_lr"]},
        ],
        weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1 = 0.0
    patience_cnt = 0
    total_loss = 0.0
    global_step = 0

    for epoch in range(ner_cfg["epochs"]):
        model.train()
        for bert_input, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            loss = model(bert_input, batch_labels)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1

            if global_step % 200 == 0:
                logger.info(f"Step {global_step} | avg_loss={total_loss/200:.4f}")
                total_loss = 0.0

        # 每 epoch 评估
        metrics = _evaluate(model, dev_loader, label2txt)
        scheduler.step(metrics["f1"])
        logger.info(
            f"Epoch {epoch+1} | P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_cnt = 0
            save_checkpoint(model, ner_cfg["checkpoint"], optimizer=optimizer, epoch=epoch, f1=best_f1)
            logger.info(f"  -> 最佳模型已保存 (F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= ner_cfg["patience"]:
                logger.info(f"早停触发，最佳 F1={best_f1:.4f}")
                break

    logger.info("NER 训练完成")


def evaluate(cfg: dict) -> Dict:
    """
    NER 评估入口（加载最佳模型在测试集上评估）

    Args:
        cfg: configs/pipeline.yaml 中的完整配置 dict

    Returns:
        评估指标字典
    """
    set_seed(cfg["seed"])
    device = get_device()
    ner_cfg = cfg["ner"]
    data_cfg = cfg["data"]

    _, txt2label, label2txt = _build_label_maps(data_cfg["entity2id"])
    tokenizer = BertTokenizer.from_pretrained(ner_cfg["bert_model"])
    collate = partial(ner_collate_fn, tokenizer=tokenizer, txt2label=txt2label, device=device)

    test_set = NerDataset(data_cfg["processed_dir"], "test", tokenizer, txt2label, device)
    test_loader = DataLoader(test_set, batch_size=ner_cfg["batch_size"], shuffle=False, collate_fn=collate)

    model = BertNer(bert_path=ner_cfg["bert_model"], num_tags=len(txt2label)).to(device)
    load_checkpoint(model, ner_cfg["checkpoint"], device=device)

    metrics = _evaluate(model, test_loader, label2txt)
    print(metrics["report"])
    save_json(metrics, os.path.join(cfg["output"]["dir"], "ner_metrics.json"))
    return metrics


def predict(texts: List[str], cfg: dict) -> List[List[Dict]]:
    """
    NER 批量预测

    Args:
        texts: 文本列表
        cfg: configs/pipeline.yaml 中的完整配置 dict

    Returns:
        每条文本对应的实体列表（每个实体为 dict：text/type/start/end）
    """
    device = get_device()
    ner_cfg = cfg["ner"]
    data_cfg = cfg["data"]

    _, txt2label, label2txt = _build_label_maps(data_cfg["entity2id"])
    tokenizer = BertTokenizer.from_pretrained(ner_cfg["bert_model"])

    model = BertNer(bert_path=ner_cfg["bert_model"], num_tags=len(txt2label)).to(device)
    load_checkpoint(model, ner_cfg["checkpoint"], device=device)
    model.eval()

    results = []
    for text in texts:
        entities, _ = _predict_single(text, model, tokenizer, label2txt, ner_cfg["max_length"], device)
        results.append(entities)
    return results


def _predict_single(
    text: str,
    model: BertNer,
    tokenizer: BertTokenizer,
    label2txt: Dict[int, str],
    max_length: int,
    device: torch.device,
) -> Tuple[List[Dict], bool]:
    """对单条文本预测实体，返回 (entities, was_truncated)"""
    chars = list(text)
    enc = tokenizer(
        chars,
        is_split_into_words=True,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    was_truncated = len(chars) > 510

    with torch.no_grad():
        logits = model(bert_input=enc)[0]
        decoded = model.crf.decode(logits, mask=enc["attention_mask"].gt(0))[0]

    valid_len = int(enc["attention_mask"][0].sum().item())
    char_label_ids = decoded[1: valid_len - 1]

    match_len = min(len(text), len(char_label_ids))
    entities = _decode_bio(text[:match_len], char_label_ids[:match_len], label2txt)
    return entities, was_truncated


def _decode_bio(text: str, label_ids: List[int], label2txt: Dict[int, str]) -> List[Dict]:
    """将 BIO 标签序列解码为实体列表"""
    entities = []
    cur_chars: List[str] = []
    cur_type: Optional[str] = None
    cur_start: Optional[int] = None

    def flush(end: int):
        nonlocal cur_chars, cur_type, cur_start
        if cur_chars and cur_type is not None:
            entities.append({
                "text": "".join(cur_chars),
                "type": cur_type,
                "start": cur_start,
                "end": end,
            })
        cur_chars, cur_type, cur_start = [], None, None

    for idx, lid in enumerate(label_ids):
        label = label2txt.get(lid, "O")
        char = text[idx]
        if label.startswith("B-"):
            flush(idx - 1)
            cur_chars = [char]
            cur_type = label[2:]
            cur_start = idx
        elif label.startswith("I-") and cur_type == label[2:]:
            cur_chars.append(char)
        else:
            flush(idx - 1)

    flush(len(label_ids) - 1)
    return entities
