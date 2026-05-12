# -*- coding: utf-8 -*-
"""
CasRel 训练、评估、预测全流程
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer

from methods.joint.dataset import DataPreFetcher, get_loader
from methods.joint.model import Casrel
from utils.common import get_device, get_logger, set_seed
from utils.io_utils import load_checkpoint, save_checkpoint, save_json, save_jsonl
from utils.metrics import TripleMetrics, analyze_errors, save_error_report


def _load_rel2id(rel2id_path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """加载关系映射，兼容多种格式"""
    with open(rel2id_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "relation2id" in data:
        rel2id = data["relation2id"]
    elif isinstance(data, list):
        rel2id = data[1]
    else:
        rel2id = data
    id2rel = {str(v): k for k, v in rel2id.items()}
    return rel2id, id2rel


def _bce_loss(gold: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """带掩码的二元交叉熵损失"""
    pred = pred.squeeze(-1)
    loss = F.binary_cross_entropy(pred, gold, reduction="none")
    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    return torch.sum(loss * mask) / torch.sum(mask)


def _decode_span(tokens: List[str], start: int, end: int) -> str:
    """将 BERT token 序列解码为文本（去掉 ## 前缀）"""
    span = tokens[start: end + 1]
    return "".join(t.lstrip("##") for t in span)


def _triple_to_tuple(triple) -> Tuple[str, str, str]:
    """统一三元组格式为 (subject, predicate, object_text)"""
    if isinstance(triple, dict):
        s = triple.get("subject", "")
        p = triple.get("predicate", "")
        obj = triple.get("object", {})
        o = obj.get("@value", "") if isinstance(obj, dict) else str(obj)
        return str(s), str(p), str(o)
    return tuple(triple)


# ──────────────────────────────────────────
# 训练
# ──────────────────────────────────────────

def train(cfg: dict) -> None:
    """
    CasRel 训练入口

    Args:
        cfg: configs/joint.yaml 解析后的完整配置 dict
    """
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    logger = get_logger("joint_train", log_file=cfg["output"]["log"])

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    rel2id, id2rel = _load_rel2id(data_cfg["rel2id"])
    rel_num = len(rel2id)
    logger.info(f"关系数量: {rel_num}")

    tokenizer = BertTokenizer.from_pretrained(model_cfg["bert_model"])

    model = Casrel(bert_model=model_cfg["bert_model"], rel_num=rel_num).cuda()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_cfg["learning_rate"],
    )

    train_loader = get_loader(
        data_cfg["processed_dir"], data_cfg["train_prefix"],
        rel2id, rel_num, tokenizer,
        batch_size=model_cfg["batch_size"],
    )
    dev_loader = get_loader(
        data_cfg["processed_dir"], data_cfg["dev_prefix"],
        rel2id, rel_num, tokenizer,
        batch_size=1, is_test=True,
    )

    best_f1 = 0.0
    global_step = 0
    loss_sum = 0.0

    for epoch in range(model_cfg["epochs"]):
        model.train()
        prefetcher = DataPreFetcher(train_loader)
        data = prefetcher.next()
        epoch_loss = 0.0

        while data is not None:
            pred_sh, pred_st, pred_oh, pred_ot = model(data)
            sub_h_loss = _bce_loss(data["sub_heads"], pred_sh, data["mask"])
            sub_t_loss = _bce_loss(data["sub_tails"], pred_st, data["mask"])
            obj_h_loss = _bce_loss(data["obj_heads"], pred_oh, data["mask"])
            obj_t_loss = _bce_loss(data["obj_tails"], pred_ot, data["mask"])
            total_loss = sub_h_loss + sub_t_loss + obj_h_loss + obj_t_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            loss_sum += total_loss.item()
            epoch_loss += total_loss.item()

            if global_step % model_cfg["log_every"] == 0:
                avg = loss_sum / model_cfg["log_every"]
                logger.info(f"epoch={epoch} step={global_step} loss={avg:.4f}")
                loss_sum = 0.0

            data = prefetcher.next()

        # 定期评估
        if (epoch + 1) % model_cfg["eval_every"] == 0:
            model.eval()
            p, r, f1 = _eval_loop(dev_loader, model, id2rel, model_cfg["threshold"])
            logger.info(f"[dev] epoch={epoch} F1={f1:.4f} P={p:.4f} R={r:.4f}")
            model.train()

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(model, model_cfg["checkpoint"], optimizer=optimizer, epoch=epoch, f1=best_f1)
                logger.info(f"  -> 最佳模型已保存 (F1={best_f1:.4f})")

        torch.cuda.empty_cache()

    logger.info(f"训练完成，最佳 F1={best_f1:.4f}")


# ──────────────────────────────────────────
# 评估循环
# ──────────────────────────────────────────

def _eval_loop(
    data_loader,
    model: Casrel,
    id2rel: Dict[str, str],
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """在 data_loader 上运行 CasRel 推理，返回 (precision, recall, f1)"""
    correct = predict_n = gold_n = 0

    prefetcher = DataPreFetcher(data_loader)
    data = prefetcher.next()

    while data is not None:
        with torch.no_grad():
            token_ids = data["token_ids"]
            tokens = data["tokens"][0]
            mask = data["mask"]

            encoded = model.get_encoded_text(token_ids, mask)
            pred_sh, pred_st = model.get_subs(encoded)
            sub_heads = np.where(pred_sh.cpu()[0] > threshold)[0]
            sub_tails = np.where(pred_st.cpu()[0] > threshold)[0]

            subjects = []
            for sh in sub_heads:
                st = sub_tails[sub_tails >= sh]
                if len(st) > 0:
                    st = st[0]
                    subjects.append((sh, st))

            pred_triples = set()
            if subjects:
                rep_enc = encoded.repeat(len(subjects), 1, 1)
                sh_map = torch.zeros(len(subjects), 1, encoded.size(1))
                st_map = torch.zeros(len(subjects), 1, encoded.size(1))
                for i, (sh, st) in enumerate(subjects):
                    sh_map[i][0][sh] = 1
                    st_map[i][0][st] = 1
                sh_map = sh_map.to(rep_enc)
                st_map = st_map.to(rep_enc)

                pred_oh, pred_ot = model.get_objs_for_specific_sub(sh_map, st_map, rep_enc)
                for i, (sh, st) in enumerate(subjects):
                    sub_text = _decode_span(tokens, sh, st)
                    obj_hs = np.where(pred_oh.cpu()[i] > threshold)
                    obj_ts = np.where(pred_ot.cpu()[i] > threshold)
                    for oh, rel_h in zip(*obj_hs):
                        for ot, rel_t in zip(*obj_ts):
                            if oh <= ot and rel_h == rel_t:
                                rel = id2rel[str(int(rel_h))]
                                obj_text = _decode_span(tokens, oh, ot)
                                pred_triples.add((sub_text, rel, obj_text))
                                break

            gold_triples = set(_triple_to_tuple(t) for t in data["triples"][0])
            correct += len(pred_triples & gold_triples)
            predict_n += len(pred_triples)
            gold_n += len(gold_triples)

        data = prefetcher.next()

    p = correct / (predict_n + 1e-10)
    r = correct / (gold_n + 1e-10)
    f1 = 2 * p * r / (p + r + 1e-10)
    return p, r, f1


# ──────────────────────────────────────────
# 评估 & 预测入口
# ──────────────────────────────────────────

def evaluate(cfg: dict) -> Dict:
    """加载最佳模型，在测试集上评估"""
    set_seed(cfg["seed"])
    device = get_device()
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    logger = get_logger("joint_eval", log_file=cfg["output"]["log"])

    rel2id, id2rel = _load_rel2id(data_cfg["rel2id"])
    rel_num = len(rel2id)
    tokenizer = BertTokenizer.from_pretrained(model_cfg["bert_model"])

    test_loader = get_loader(
        data_cfg["processed_dir"], data_cfg["test_prefix"],
        rel2id, rel_num, tokenizer,
        batch_size=1, is_test=True,
    )

    model = Casrel(bert_model=model_cfg["bert_model"], rel_num=rel_num).cuda()
    load_checkpoint(model, model_cfg["checkpoint"], device=device)
    model.eval()

    p, r, f1 = _eval_loop(test_loader, model, id2rel, model_cfg["threshold"])
    result = {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}
    save_json(result, cfg["output"]["metrics"])
    logger.info(f"Joint 评估结果: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return result


def predict(cfg: dict, input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """对任意输入文件批量预测"""
    set_seed(cfg["seed"])
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    rel2id, id2rel = _load_rel2id(data_cfg["rel2id"])
    rel_num = len(rel2id)
    tokenizer = BertTokenizer.from_pretrained(model_cfg["bert_model"])

    # 使用测试集 DataLoader 预测
    prefix = data_cfg["test_prefix"]
    test_loader = get_loader(
        data_cfg["processed_dir"], prefix,
        rel2id, rel_num, tokenizer,
        batch_size=1, is_test=True,
    )

    model = Casrel(bert_model=model_cfg["bert_model"], rel_num=rel_num).cuda()
    load_checkpoint(model, model_cfg["checkpoint"])
    model.eval()

    outputs = []
    prefetcher = DataPreFetcher(test_loader)
    data = prefetcher.next()
    threshold = model_cfg["threshold"]

    while data is not None:
        with torch.no_grad():
            token_ids = data["token_ids"]
            tokens = data["tokens"][0]
            mask = data["mask"]
            encoded = model.get_encoded_text(token_ids, mask)
            pred_sh, pred_st = model.get_subs(encoded)
            sub_heads = np.where(pred_sh.cpu()[0] > threshold)[0]
            sub_tails = np.where(pred_st.cpu()[0] > threshold)[0]

            subjects = []
            for sh in sub_heads:
                st = sub_tails[sub_tails >= sh]
                if len(st) > 0:
                    subjects.append((sh, st[0]))

            pred_triples = []
            if subjects:
                rep_enc = encoded.repeat(len(subjects), 1, 1)
                sh_map = torch.zeros(len(subjects), 1, encoded.size(1)).to(rep_enc)
                st_map = torch.zeros(len(subjects), 1, encoded.size(1)).to(rep_enc)
                for i, (sh, st) in enumerate(subjects):
                    sh_map[i][0][sh] = 1
                    st_map[i][0][st] = 1

                pred_oh, pred_ot = model.get_objs_for_specific_sub(sh_map, st_map, rep_enc)
                seen = set()
                for i, (sh, st) in enumerate(subjects):
                    sub_text = _decode_span(tokens, sh, st)
                    obj_hs = np.where(pred_oh.cpu()[i] > threshold)
                    obj_ts = np.where(pred_ot.cpu()[i] > threshold)
                    for oh, rh in zip(*obj_hs):
                        for ot, rt in zip(*obj_ts):
                            if oh <= ot and rh == rt:
                                rel = id2rel[str(int(rh))]
                                obj_text = _decode_span(tokens, oh, ot)
                                key = (sub_text, rel, obj_text)
                                if key not in seen:
                                    seen.add(key)
                                    pred_triples.append({
                                        "subject": sub_text,
                                        "predicate": rel,
                                        "object": {"@value": obj_text},
                                    })
                                break

            gold_triples = list(data["triples"][0])
            outputs.append({"text": "".join(tokens), "pred_triples": pred_triples, "gold_triples": gold_triples})
        data = prefetcher.next()

    out_path = output_file or cfg["output"]["predictions"]
    save_jsonl(outputs, out_path)
    print(f"预测结果已保存到: {out_path}")


def run(cfg: dict, mode: str, input_path: Optional[str] = None, output_path: Optional[str] = None) -> None:
    """Joint 方法统一入口"""
    set_seed(cfg["seed"])
    if mode == "train":
        train(cfg)
    elif mode == "evaluate":
        evaluate(cfg)
    elif mode == "predict":
        predict(cfg, input_file=input_path, output_file=output_path)
    else:
        raise ValueError(f"未知 mode: {mode}")
