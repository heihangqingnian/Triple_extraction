# -*- coding: utf-8 -*-
"""
下载 ChatGLM2-6B 模型到本地目录

Usage::

    python scripts/download_model.py --model THUDM/chatglm2-6b --save_dir models/chatglm2-6b
"""

import argparse
import os


def download_from_huggingface(model_id: str, save_dir: str) -> None:
    """从 HuggingFace Hub 下载模型"""
    from huggingface_hub import snapshot_download
    print(f"从 HuggingFace 下载: {model_id} -> {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=save_dir)
    print(f"下载完成: {save_dir}")


def download_from_modelscope(model_id: str, save_dir: str) -> None:
    """从 ModelScope 下载模型（国内推荐）"""
    from modelscope import snapshot_download
    print(f"从 ModelScope 下载: {model_id} -> {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    snapshot_download(model_id=model_id, cache_dir=save_dir)
    print(f"下载完成: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载预训练模型")
    parser.add_argument("--model", default="THUDM/chatglm2-6b", help="模型 ID")
    parser.add_argument("--save_dir", default="models/chatglm2-6b", help="本地保存目录")
    parser.add_argument(
        "--source", default="huggingface",
        choices=["huggingface", "modelscope"],
        help="下载源（国内推荐 modelscope）"
    )
    args = parser.parse_args()

    if args.source == "huggingface":
        download_from_huggingface(args.model, args.save_dir)
    else:
        download_from_modelscope(args.model, args.save_dir)


if __name__ == "__main__":
    main()
