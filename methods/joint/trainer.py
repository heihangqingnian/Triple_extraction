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
    """
    带掩码的二元交叉熵损失。

    BUG-6 fix: 修正 obj_heads/obj_tails 的归一化错误。
    原来: squeeze(-1) 对 [B,L,rel_num] 无效（最后维度!=1），导致 obj loss 的分母
          仍是 sum(mask)（有效位置数），而非 sum(mask)*rel_num，
          使 obj loss 比 sub loss 大约 rel_num（~48）倍，破坏四项 loss 的平衡。
    修复: 区分 2D（sub_heads/tails）和 3D（obj_heads/tails）两种情况，分别归一化。
    """
    if gold.dim() == 2:
        # sub_heads / sub_tails: pred=[B,L,1] → squeeze → [B,L]，gold=[B,L]
        pred_sq = pred.squeeze(-1)
        loss = F.binary_cross_entropy(pred_sq, gold, reduction="none")  # [B,L]
        return torch.sum(loss * mask) / torch.sum(mask)
    else:
        # obj_heads / obj_tails: pred=[B,L,rel_num]，gold=[B,L,rel_num]
        loss = F.binary_cross_entropy(pred, gold, reduction="none")     # [B,L,rel_num]
        mask_exp = mask.unsqueeze(-1).expand_as(loss)                   # [B,L,rel_num]
        # 归一化分母 = 有效位置数 × 关系数，保持与 sub loss 同量级
        denom = torch.sum(mask) * gold.size(-1)
        return torch.sum(loss * mask_exp) / denom


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

    # BUG-7 fix: 使用 get_device() 而非硬编码 .cuda()，支持 CPU 环境
    # 原来: model = Casrel(...).cuda()  ← CUDA 不可用时直接崩溃
    model = Casrel(bert_model=model_cfg["bert_model"], rel_num=rel_num).to(device)
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
    # 用于调试 BUG-6 修复：记录各分项 loss 历史，验证 sub 和 obj loss 是否处于同一量级
    sub_loss_sum = obj_loss_sum = 0.0

    for epoch in range(model_cfg["epochs"]):
        model.train()
        # BUG-7 fix: DataPreFetcher 内部使用 CUDA stream，仅 CUDA 可用时启用
        if device.type == "cuda":
            prefetcher = DataPreFetcher(train_loader)
            data = prefetcher.next()
        else:
            data_iter = iter(train_loader)
            data = next(data_iter, None)
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
            # BUG-18 fix: 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            loss_sum    += total_loss.item()
            sub_loss_sum += (sub_h_loss + sub_t_loss).item()
            obj_loss_sum += (obj_h_loss + obj_t_loss).item()
            epoch_loss  += total_loss.item()

            if global_step % model_cfg["log_every"] == 0:
                avg_total = loss_sum / model_cfg["log_every"]
                avg_sub   = sub_loss_sum / model_cfg["log_every"]
                avg_obj   = obj_loss_sum / model_cfg["log_every"]
                # BUG-6 调试：打印 sub/obj loss 分量，修复后两者应处于同一量级（比例约 1:1）
                # 修复前 obj_loss 约是 sub_loss 的 rel_num 倍（约 48 倍）
                logger.info(
                    f"epoch={epoch} step={global_step} "
                    f"total={avg_total:.4f} sub={avg_sub:.4f} obj={avg_obj:.4f} "
                    f"[sub:obj比={avg_sub/(avg_obj+1e-9):.2f}，理想接近1.0]"
                )
                loss_sum = sub_loss_sum = obj_loss_sum = 0.0

            if device.type == "cuda":
                data = prefetcher.next()
            else:
                data = next(data_iter, None)

        # 定期评估
        if (epoch + 1) % model_cfg["eval_every"] == 0:
            model.eval()
            p, r, f1 = _eval_loop(dev_loader, model, id2rel, model_cfg["threshold"], device)
            logger.info(f"[dev] epoch={epoch} F1={f1:.4f} P={p:.4f} R={r:.4f}")
            model.train()

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(model, model_cfg["checkpoint"], optimizer=optimizer, epoch=epoch, f1=best_f1)
                logger.info(f"  -> 最佳模型已保存 (F1={best_f1:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"训练完成，最佳 F1={best_f1:.4f}")


# ──────────────────────────────────────────
# 评估循环
# ──────────────────────────────────────────

def _extract_triples_from_batch(
    data: dict,
    model: Casrel,
    id2rel: Dict[str, str],
    threshold: float,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    对单条样本（batch_size=1）运行 CasRel 推理，返回 (pred_triples, gold_triples)。
    抽取为独立函数，供 _eval_loop 和 evaluate 复用。

    BUG-8 fix: 将内层循环改为按关系分组查找，逻辑更清晰，
    语义等价于原 break 策略（贪心最近尾），但避免混淆。
    """
    token_ids = data["token_ids"]
    tokens    = data["tokens"][0]   # List[str]，含 [CLS] 和 [SEP]（BUG-5 修复后）
    mask      = data["mask"]
    seq_len   = len(tokens)

    encoded = model.get_encoded_text(token_ids, mask)
    pred_sh, pred_st = model.get_subs(encoded)

    # pred_sh: [1, L, 1]  → [L, 1]  → np.where 返回 (row_idxs, col_idxs)
    sub_heads_arr = np.where(pred_sh.cpu()[0] > threshold)[0]  # token positions
    sub_tails_arr = np.where(pred_st.cpu()[0] > threshold)[0]

    # 过滤：[CLS](index 0) 和 [SEP](最后一个) 不应是实体
    sub_heads_arr = sub_heads_arr[(sub_heads_arr > 0) & (sub_heads_arr < seq_len - 1)]
    sub_tails_arr = sub_tails_arr[(sub_tails_arr > 0) & (sub_tails_arr < seq_len - 1)]

    subjects = []
    for sh in sub_heads_arr:
        valid_tails = sub_tails_arr[sub_tails_arr >= sh]
        if len(valid_tails) > 0:
            subjects.append((sh, valid_tails[0]))  # 贪心最近尾

    pred_triples: List[Tuple] = []
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
        seen = set()
        for i, (sh, st) in enumerate(subjects):
            sub_text = _decode_span(tokens, sh, st)
            obj_hs = np.where(pred_oh.cpu()[i] > threshold)
            obj_ts = np.where(pred_ot.cpu()[i] > threshold)

            # BUG-8 fix: 按关系分组预先建索引，避免 O(|hs|×|ts|) 双层遍历混淆
            # obj_ts = (position_array, relation_array)
            ts_by_rel: Dict[int, List[int]] = {}
            if len(obj_ts[0]) > 0:
                for ot, rel_t in zip(obj_ts[0], obj_ts[1]):
                    ts_by_rel.setdefault(int(rel_t), []).append(int(ot))

            if len(obj_hs[0]) > 0:
                for oh, rel_h in zip(obj_hs[0], obj_hs[1]):
                    oh, rel_h = int(oh), int(rel_h)
                    if oh == 0 or oh >= seq_len - 1:
                        continue  # 跳过 [CLS]/[SEP] 位置
                    if rel_h not in ts_by_rel:
                        continue
                    for ot in ts_by_rel[rel_h]:
                        if ot >= oh and ot < seq_len - 1:
                            rel = id2rel[str(rel_h)]
                            obj_text = _decode_span(tokens, oh, ot)
                            key = (sub_text, rel, obj_text)
                            if key not in seen:
                                seen.add(key)
                                pred_triples.append(key)
                            break  # 贪心最近尾

    gold_triples = [_triple_to_tuple(t) for t in data["triples"][0]]
    return pred_triples, gold_triples


def _eval_loop(
    data_loader,
    model: Casrel,
    id2rel: Dict[str, str],
    threshold: float = 0.5,
    device: torch.device = None,
) -> Tuple[float, float, float]:
    """在 data_loader 上运行 CasRel 推理，返回 (precision, recall, f1)"""
    correct = predict_n = gold_n = 0
    use_cuda = (device is not None and device.type == "cuda")

    if use_cuda:
        prefetcher = DataPreFetcher(data_loader)
        data = prefetcher.next()
    else:
        data_iter = iter(data_loader)
        data = next(data_iter, None)

    while data is not None:
        with torch.no_grad():
            pred_triples, gold_triples = _extract_triples_from_batch(
                data, model, id2rel, threshold
            )
        pred_set = set(pred_triples)
        gold_set = set(gold_triples)
        correct   += len(pred_set & gold_set)
        predict_n += len(pred_set)
        gold_n    += len(gold_set)

        if use_cuda:
            data = prefetcher.next()
        else:
            data = next(data_iter, None)

    p = correct / (predict_n + 1e-10)
    r = correct / (gold_n + 1e-10)
    f1 = 2 * p * r / (p + r + 1e-10)
    return p, r, f1


# ──────────────────────────────────────────
# 评估 & 预测入口
# ──────────────────────────────────────────

def evaluate(cfg: dict) -> Dict:
    """
    加载最佳模型，在测试集上评估。

    BUG-7 fix: .cuda() → .to(device)，支持 CPU 环境。
    BUG-10 fix: 使用 TripleMetrics（与 Pipeline 保持一致），替代手写计数。
    BUG-13 fix: 保存样本级预测结果到 predictions.jsonl，支持事后分析。
    同时调用 analyze_errors 生成错误类型报告（与 Pipeline evaluate 对齐）。
    """
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

    # BUG-7 fix: .to(device) 替代 .cuda()
    model = Casrel(bert_model=model_cfg["bert_model"], rel_num=rel_num).to(device)
    load_checkpoint(model, model_cfg["checkpoint"], device=device)
    model.eval()

    # BUG-10 fix: 使用 TripleMetrics 与 Pipeline 保持完全一致的评估逻辑
    metrics = TripleMetrics()
    sample_outputs = []
    error_counts: Dict = {}
    use_cuda = (device.type == "cuda")

    t_start = time.perf_counter()
    if use_cuda:
        prefetcher = DataPreFetcher(test_loader)
        data = prefetcher.next()
    else:
        data_iter = iter(test_loader)
        data = next(data_iter, None)

    sample_idx = 0
    while data is not None:
        with torch.no_grad():
            pred_triples_tuples, gold_triples_tuples = _extract_triples_from_batch(
                data, model, id2rel, model_cfg["threshold"]
            )

        # 转为 dict 格式供 TripleMetrics 使用（兼容 _triple_to_key）
        pred_dicts = [{"subject": s, "predicate": p, "object": {"@value": o}}
                      for s, p, o in pred_triples_tuples]
        gold_dicts = [{"subject": s, "predicate": p, "object": {"@value": o}}
                      for s, p, o in gold_triples_tuples]

        metrics.update(pred_dicts, gold_dicts)

        # 错误分析（BUG-15 fix 已在 metrics.py 中：Joint 也提取预测实体集）
        errs = analyze_errors(pred_dicts, gold_dicts)
        for k, v in errs.items():
            error_counts[k] = error_counts.get(k, 0) + v

        # BUG-13 fix: 保存样本级预测结果
        tokens = data["tokens"][0]
        sample_outputs.append({
            "text": "".join(t.lstrip("##") for t in tokens if t not in ("[CLS]", "[SEP]")),
            "pred_triples": pred_dicts,
            "gold_triples": gold_dicts,
        })

        if use_cuda:
            data = prefetcher.next()
        else:
            data = next(data_iter, None)
        sample_idx += 1

    elapsed = time.perf_counter() - t_start
    result = metrics.compute()
    result["inference_time_seconds"] = round(elapsed, 2)
    result["samples_per_second"] = round(sample_idx / elapsed, 2)
    result["error_counts"] = error_counts

    metrics.print_report("Joint 端到端评估")
    save_json(result, cfg["output"]["metrics"])
    save_jsonl(sample_outputs, cfg["output"]["predictions"])
    save_error_report(error_counts, cfg["output"]["error_report"], model_name="joint")
    logger.info(f"Joint 评估完成，结果已保存到: {cfg['output']['dir']}")
    return result


def predict(cfg: dict, input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """
    对测试集批量预测。

    注意: Joint（CasRel）方法依赖预处理好的 CasRel 格式数据（含 token_ids、masks、
    sub_heads 等字段），无法直接接收原始文本文件作为输入。
    若传入 input_file 参数，当前实现会记录警告并忽略，仍使用配置中的测试集。

    BUG-7 fix: .cuda() → .to(device)。
    BUG-8 fix: 复用 _extract_triples_from_batch，消除重复的推理逻辑。
    """
    if input_file is not None:
        print(
            f"[警告] Joint predict 暂不支持自定义 input_file（CasRel 需要预处理格式数据）。"
            f"忽略 input_file='{input_file}'，使用配置中的测试集。"
        )

    set_seed(cfg["seed"])
    device = get_device()
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    rel2id, id2rel = _load_rel2id(data_cfg["rel2id"])
    rel_num = len(rel2id)
    tokenizer = BertTokenizer.from_pretrained(model_cfg["bert_model"])

    test_loader = get_loader(
        data_cfg["processed_dir"], data_cfg["test_prefix"],
        rel2id, rel_num, tokenizer,
        batch_size=1, is_test=True,
    )

    # BUG-7 fix: .to(device) 替代 .cuda()
    model = Casrel(bert_model=model_cfg["bert_model"], rel_num=rel_num).to(device)
    load_checkpoint(model, model_cfg["checkpoint"], device=device)
    model.eval()

    outputs = []
    use_cuda = (device.type == "cuda")
    if use_cuda:
        prefetcher = DataPreFetcher(test_loader)
        data = prefetcher.next()
    else:
        data_iter = iter(test_loader)
        data = next(data_iter, None)

    while data is not None:
        with torch.no_grad():
            # BUG-8 fix: 复用统一推理函数，不再有重复代码
            pred_triples_tuples, gold_triples_tuples = _extract_triples_from_batch(
                data, model, id2rel, model_cfg["threshold"]
            )

        pred_dicts = [{"subject": s, "predicate": p, "object": {"@value": o}}
                      for s, p, o in pred_triples_tuples]
        gold_dicts = [{"subject": s, "predicate": p, "object": {"@value": o}}
                      for s, p, o in gold_triples_tuples]

        tokens = data["tokens"][0]
        text = "".join(t.lstrip("##") for t in tokens if t not in ("[CLS]", "[SEP]"))
        outputs.append({"text": text, "pred_triples": pred_dicts, "gold_triples": gold_dicts})

        if use_cuda:
            data = prefetcher.next()
        else:
            data = next(data_iter, None)

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
