# -*- coding: utf-8 -*-
"""
统一数据读写工具：JSON/JSONL 文件读写、模型检查点保存加载
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


# ──────────────────────────────────────────
# JSON / JSONL 文件操作
# ──────────────────────────────────────────

def load_json(path: Union[str, Path]) -> Any:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """保存为 JSON 文件，自动创建父目录"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    """加载 JSONL 文件（每行一个 JSON 对象）"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(data: List[Any], path: Union[str, Path]) -> None:
    """保存为 JSONL 文件，自动创建父目录"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_txt_lines(path: Union[str, Path]) -> List[str]:
    """读取文本文件，返回各行列表（已去除换行符）"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# ──────────────────────────────────────────
# 模型检查点 保存 / 加载
# ──────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    **meta: Any,
) -> None:
    """
    保存模型检查点（包含 state_dict、optimizer 状态和任意元数据）

    Args:
        model: PyTorch 模型
        path: 保存路径（.pt 文件）
        optimizer: 优化器（可选，用于恢复训练）
        **meta: 任意额外信息（epoch、f1、loss 等）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        **meta,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict:
    """
    加载模型检查点

    Args:
        model: 待加载参数的模型实例
        path: 检查点路径
        optimizer: 优化器（非 None 时同步恢复优化器状态）
        device: 映射设备

    Returns:
        完整检查点字典（可读取 epoch、f1 等元信息）
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # 兼容旧格式：直接保存 state_dict 而非完整 checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # 兼容 Bert_based_extraction 旧格式
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer and checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        # 直接是 state_dict
        model.load_state_dict(checkpoint)

    return checkpoint
