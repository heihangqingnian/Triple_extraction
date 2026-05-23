# -*- coding: utf-8 -*-
"""
关系抽取训练、评估、预测
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from methods.pipeline.re.dataset import ReDataset, load_and_featurize
from methods.pipeline.re.model import BertForRelationExtraction
from utils.common import get_device, get_logger, set_seed
from utils.io_utils import load_checkpoint, save_checkpoint, save_json


def _get_cache_path(cfg: dict, split: str) -> str:
    """特征缓存文件路径"""
    cache_dir = os.path.join(cfg["output"]["dir"], "re_cache")
    return os.path.join(cache_dir, f"{split}_features.pt")


def _load_rel2id(rel2id_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """加载关系映射文件，兼容多种格式"""
    with open(rel2id_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "relation2id" in data:
        rel2id = data["relation2id"]
    else:
        rel2id = data
    id2rel = {int(v): k for k, v in rel2id.items()}
    return rel2id, id2rel


def train(cfg: dict) -> None:
    """
    RE 训练入口

    Args:
        cfg: configs/pipeline.yaml 中的完整配置 dict
    """
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    logger = get_logger("re_train", log_file=cfg["output"]["log"])

    re_cfg = cfg["re"]
    data_cfg = cfg["data"]

    rel2id, id2rel = _load_rel2id(data_cfg["rel2id"])
    num_tags = len(rel2id)
    logger.info(f"关系类别数: {num_tags}")

    # 构建数据
    processed_dir = data_cfg["processed_dir"]
    train_features, _ = load_and_featurize(
        os.path.join(processed_dir, "train", "train_re.txt"),
        bert_model=re_cfg["bert_model"],
        max_seq_len=re_cfg["max_seq_len"],
        cache_path=_get_cache_path(cfg, "train"),
    )
    dev_features, _ = load_and_featurize(
        os.path.join(processed_dir, "dev", "dev_re.txt"),
        bert_model=re_cfg["bert_model"],
        max_seq_len=re_cfg["max_seq_len"],
        cache_path=_get_cache_path(cfg, "dev"),
    )

    # BUG-2 fix: 只创建一个 ReDataset 对象给 DataLoader，避免重复构建
    # 原来: sampler=RandomSampler(ReDataset(train_features)) 创建了第二份冗余对象
    train_dataset = ReDataset(train_features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=re_cfg["train_batch_size"],
        sampler=RandomSampler(train_dataset),
    )
    dev_loader = DataLoader(
        ReDataset(dev_features),
        batch_size=re_cfg["eval_batch_size"],
    )

    model = BertForRelationExtraction(
        bert_path=re_cfg["bert_model"],
        num_tags=num_tags,
        dropout=re_cfg["dropout"],
    ).to(device)
    # BUG-18 fix: 使用 AdamW（BERT fine-tuning 推荐）并加梯度裁剪
    optimizer = torch.optim.AdamW(model.parameters(), lr=re_cfg["bert_lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience_cnt = 0
    patience = re_cfg.get("patience", 5)  # BUG-11 fix: 从配置读取早停耐心值
    total_steps = len(train_loader) * re_cfg["epochs"]
    global_step = 0

    logger.info(f"[RE训练] 共 {len(train_dataset)} 条训练样本，"
                f"关系类别数={num_tags}，早停 patience={patience}")

    for epoch in range(re_cfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{re_cfg['epochs']}")
        for batch in pbar:
            token_ids = batch["token_ids"].to(device)
            attn = batch["attention_masks"].to(device)
            ttype = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            ids = batch["ids"].to(device)

            out = model(token_ids, attn, ttype, ids)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            # BUG-18 fix: 梯度裁剪，防止 BERT fine-tuning 梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % 100 == 0:
                logger.info(f"step {global_step}/{total_steps} loss={loss.item():.4f}")

        # 验证集评估
        dev_loss, dev_preds, dev_targets = _dev_eval(model, dev_loader, criterion, device)
        p, r, f1 = _metrics(dev_preds, dev_targets, num_tags, no_rel_id=rel2id.get("无关系"))
        logger.info(f"[dev] epoch={epoch+1} loss={dev_loss:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f}")

        # BUG-11 fix: 添加早停机制（与 NER trainer 保持一致）
        if f1 > best_f1:
            best_f1 = f1
            patience_cnt = 0
            save_checkpoint(model, re_cfg["checkpoint"], optimizer=optimizer, epoch=epoch, f1=best_f1)
            logger.info(f"  -> 最佳模型已保存 (F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"早停触发，最佳 F1={best_f1:.4f}")
                break

    logger.info("RE 训练完成")


def evaluate(cfg: dict) -> Dict:
    """RE 评估入口"""
    set_seed(cfg["seed"])
    device = get_device()
    re_cfg = cfg["re"]
    data_cfg = cfg["data"]

    rel2id, id2rel = _load_rel2id(data_cfg["rel2id"])
    num_tags = len(rel2id)

    test_features, _ = load_and_featurize(
        os.path.join(data_cfg["processed_dir"], "test", "test_re.txt"),
        bert_model=re_cfg["bert_model"],
        max_seq_len=re_cfg["max_seq_len"],
    )
    test_loader = DataLoader(ReDataset(test_features), batch_size=re_cfg["eval_batch_size"])

    model = BertForRelationExtraction(re_cfg["bert_model"], num_tags, re_cfg["dropout"]).to(device)
    load_checkpoint(model, re_cfg["checkpoint"], device=device)

    criterion = nn.CrossEntropyLoss()
    _, preds, targets = _dev_eval(model, test_loader, criterion, device)
    no_rel_id = rel2id.get("无关系")
    p, r, f1 = _metrics(preds, targets, num_tags=num_tags, no_rel_id=no_rel_id)
    result = {"precision": p, "recall": r, "f1": f1}
    save_json(result, os.path.join(cfg["output"]["dir"], "re_metrics.json"))
    print(f"RE 评估结果（排除'无关系'类的宏平均）: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return result


def _dev_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attn = batch["attention_masks"].to(device)
            ttype = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            ids = batch["ids"].to(device)
            out = model(token_ids, attn, ttype, ids)
            loss = criterion(out, labels)
            total_loss += loss.item()
            preds = np.argmax(out.cpu().detach().numpy(), axis=1).tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())
    return total_loss / len(loader), all_preds, all_targets


def _metrics(preds, targets, num_tags: int = None, no_rel_id: int = None):
    """
    计算 RE 关系分类的 P/R/F1。

    BUG-9 fix: 排除"无关系"类别再做宏平均。
    原来: average="macro" 把"无关系"也纳入平均，导致指标虚高（"无关系"占主导）。
    现在: 只在真实关系类别上评估，"无关系"类别用于过滤但不参与宏平均计算。
    """
    # 构建需要评估的标签集（排除"无关系"）
    if num_tags is not None and no_rel_id is not None:
        labels = [i for i in range(num_tags) if i != no_rel_id]
    else:
        labels = None  # 回退：评估全部类别（不推荐）

    p = precision_score(targets, preds, labels=labels, average="macro", zero_division=0)
    r = recall_score(targets, preds, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, labels=labels, average="macro", zero_division=0)
    return p, r, f1
