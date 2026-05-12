# -*- coding: utf-8 -*-
"""
通用工具函数：随机种子、日志、路径管理
三种方法（Pipeline / Joint / LLM）共用，确保实验可复现
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """统一设置所有随机种子，保证实验结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_project_root() -> Path:
    """获取项目根目录（包含 main.py 的目录）"""
    return Path(__file__).parent.parent


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    创建统一格式的日志器

    Args:
        name: 日志器名称
        log_file: 日志文件路径（None 则只输出到控制台）
        level: 日志级别

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件输出（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_yaml(path: str) -> dict:
    """加载 YAML 配置文件"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """自动选择可用设备（CUDA 优先）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
