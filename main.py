# -*- coding: utf-8 -*-
"""
三元组抽取对比研究

Usage::

    # Pipeline 方法
    python main.py --method pipeline --mode train
    python main.py --method pipeline --mode evaluate
    python main.py --method pipeline --mode predict --input data/processed/pipeline/test/pipeline_test.jsonl

    # Joint 方法
    python main.py --method joint --mode train
    python main.py --method joint --mode evaluate
    python main.py --method joint --mode predict

    # LLM 方法（需要已训练的 LoRA 权重）
    python main.py --method llm --mode predict
    python main.py --method llm --mode evaluate
    python main.py --method llm --mode evaluate --input results/llm/predictions.jsonl

    # 对比所有方法（evaluate 模式）
    python main.py --method all --mode evaluate

    # 使用自定义配置文件
    python main.py --method pipeline --mode train --config my_pipeline.yaml
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="三元组抽取对比研究 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["pipeline", "joint", "llm", "all"],
        help="选择方法：pipeline / joint / llm / all（对比评估）",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "evaluate", "predict"],
        help="运行模式：train / evaluate / predict",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="自定义配置文件路径（可选，默认使用 configs/<method>.yaml）",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="predict 或 evaluate 模式下的输入文件路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="预测结果或评估报告的输出路径（可选，覆盖配置文件中的默认路径）",
    )
    return parser.parse_args()


def load_config(method: str, config_path: str = None) -> dict:
    """加载 YAML 配置文件"""
    from utils.common import load_yaml
    if config_path:
        path = config_path
    else:
        path = Path("configs") / f"{method}.yaml"
    if not Path(path).exists():
        print(f"错误：配置文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    return load_yaml(str(path))


def run_pipeline(cfg: dict, mode: str, input_path: str = None, output_path: str = None):
    from methods.pipeline.pipeline import run
    run(cfg, mode, input_path=input_path, output_path=output_path)


def run_joint(cfg: dict, mode: str, input_path: str = None, output_path: str = None):
    from methods.joint.trainer import run
    run(cfg, mode, input_path=input_path, output_path=output_path)


def run_llm(cfg: dict, mode: str, input_path: str = None, output_path: str = None):
    from methods.llm.evaluator import run
    run(cfg, mode, input_path=input_path, output_path=output_path)


def run_all_evaluate():
    """并排对比三种方法的评估结果"""
    from utils.common import load_yaml

    results = {}
    for method in ("pipeline", "joint", "llm"):
        cfg_path = Path("configs") / f"{method}.yaml"
        metrics_path = Path(load_yaml(str(cfg_path))["output"]["metrics"])
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as f:
                results[method] = json.load(f)
        else:
            results[method] = {"note": f"文件不存在: {metrics_path}"}

    print("\n" + "=" * 65)
    print(f"  {'方法':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("=" * 65)
    for method, r in results.items():
        p = r.get("precision", "N/A")
        rec = r.get("recall", "N/A")
        f1 = r.get("f1", "N/A")
        if isinstance(p, float):
            print(f"  {method:<12} {p:>10.4f} {rec:>10.4f} {f1:>10.4f}")
        else:
            print(f"  {method:<12} {'N/A':>10} {'N/A':>10} {r.get('note', 'N/A'):>10}")
    print("=" * 65)


def main():
    args = parse_args()

    # 初始化：统一设置随机种子（由配置文件决定）
    if args.method != "all":
        cfg = load_config(args.method, args.config)
        from utils.common import set_seed
        set_seed(cfg.get("seed", 42))

    # 路由到对应方法
    if args.method == "pipeline":
        run_pipeline(cfg, args.mode, input_path=args.input, output_path=args.output)

    elif args.method == "joint":
        run_joint(cfg, args.mode, input_path=args.input, output_path=args.output)

    elif args.method == "llm":
        run_llm(cfg, args.mode, input_path=args.input, output_path=args.output)

    elif args.method == "all":
        if args.mode != "evaluate":
            print("--method all 仅支持 --mode evaluate", file=sys.stderr)
            sys.exit(1)
        run_all_evaluate()


if __name__ == "__main__":
    main()
