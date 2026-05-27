# -*- coding: utf-8 -*-
"""
三元组抽取对比研究

Usage::

    # Pipeline 方法
    python main.py --method pipeline --mode train
    python main.py --method pipeline --mode train --component re   # 单独训练 RE 模型
    python main.py --method pipeline --mode train --component ner  # 单独训练 NER 模型
    python main.py --method pipeline --mode evaluate
    python main.py --method pipeline --mode predict --input data/processed/pipeline/test/pipeline_test.jsonl

    # Joint 方法
    python main.py --method joint --mode train
    python main.py --method joint --mode evaluate
    python main.py --method joint --mode predict

    # LLM 方法（需要已训练的 LoRA 权重）
    # 方式一：使用各变体独立配置文件（推荐）
    python main.py --method llm --mode predict  --config configs/llm_base.yaml
    python main.py --method llm --mode evaluate --config configs/llm_base.yaml
    python main.py --method llm --mode predict  --config configs/llm_schema.yaml
    python main.py --method llm --mode evaluate --config configs/llm_schema.yaml
    python main.py --method llm --mode predict  --config configs/llm_cot.yaml
    python main.py --method llm --mode evaluate --config configs/llm_cot.yaml

    # 方式二：使用主配置 + --variant 参数
    python main.py --method llm --mode predict                          # 推理全部 3 个 LoRA 变体
    python main.py --method llm --mode predict --variant base           # 只推理 base LoRA
    python main.py --method llm --mode predict --variant schema         # 只推理 schema LoRA
    python main.py --method llm --mode predict --variant cot            # 只推理 cot LoRA
    python main.py --method llm --mode evaluate                         # 评估全部 3 个变体
    python main.py --method llm --mode evaluate --variant base          # 只评估 base 变体
    python main.py --method llm --mode evaluate --variant base --input results/llm/predictions_base.jsonl

    # 对比所有方法（evaluate 模式）
    python main.py --method all --mode evaluate

    # 使用自定义配置文件
    python main.py --method pipeline --mode train --config my_pipeline.yaml

    # 消融实验示例
    # Pipeline 消融 1：去除 BiLSTM（BERT-CRF）
    python main.py --method pipeline --mode train --config configs/ablation_pipeline_1_no_bilstm.yaml
    python main.py --method pipeline --mode evaluate --config configs/ablation_pipeline_1_no_bilstm.yaml

    # Pipeline 消融 2：去除 CRF（BERT-Linear）
    python main.py --method pipeline --mode train --config configs/ablation_pipeline_2_no_crf.yaml
    python main.py --method pipeline --mode evaluate --config configs/ablation_pipeline_2_no_crf.yaml

    # CasRel 消融 1：移除主实体反馈
    python main.py --method joint --mode train --config configs/ablation_joint_1_no_feedback.yaml
    python main.py --method joint --mode evaluate --config configs/ablation_joint_1_no_feedback.yaml

    # CasRel 消融 2：双编码器
    python main.py --method joint --mode train --config configs/ablation_joint_2_dual_encoder.yaml
    python main.py --method joint --mode evaluate --config configs/ablation_joint_2_dual_encoder.yaml

    # CasRel 消融 3a：损失权重 - 侧重实体
    python main.py --method joint --mode train --config configs/ablation_joint_3_weights_entity.yaml
    python main.py --method joint --mode evaluate --config configs/ablation_joint_3_weights_entity.yaml

    # CasRel 消融 3b：损失权重 - 侧重关系
    python main.py --method joint --mode train --config configs/ablation_joint_3_weights_relation.yaml
    python main.py --method joint --mode evaluate --config configs/ablation_joint_3_weights_relation.yaml
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="三元组抽取对比研究 - 统一入口（含消融实验支持）",
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
        "--component",
        default=None,
        choices=["ner", "re"],
        help="pipeline train 模式下只训练指定子模型（ner / re），省略则顺序训练 NER→RE",
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
    parser.add_argument(
        "--variant",
        default="all",
        choices=["base", "schema", "cot", "all"],
        help="LLM 方法：指定要推理/评估的 LoRA 变体（base/schema/cot/all，默认 all）",
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


def run_pipeline(cfg: dict, mode: str, input_path: str = None, output_path: str = None, component: str = None):
    if mode == "train" and component == "re":
        from methods.pipeline.re.trainer import train as re_train
        print("=== 单独训练 RE 模型 ===")
        re_train(cfg)
    elif mode == "train" and component == "ner":
        from methods.pipeline.ner.trainer import train as ner_train
        print("=== 单独训练 NER 模型 ===")
        ner_train(cfg)
    else:
        from methods.pipeline.pipeline import run
        run(cfg, mode, input_path=input_path, output_path=output_path)


def run_joint(cfg: dict, mode: str, input_path: str = None, output_path: str = None):
    from methods.joint.trainer import run
    run(cfg, mode, input_path=input_path, output_path=output_path)


def run_llm(cfg: dict, mode: str, input_path: str = None, output_path: str = None, variant: str = "all"):
    from methods.llm.evaluator import run
    run(cfg, mode, input_path=input_path, output_path=output_path, variant=variant)


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
        run_pipeline(cfg, args.mode, input_path=args.input, output_path=args.output, component=args.component)

    elif args.method == "joint":
        run_joint(cfg, args.mode, input_path=args.input, output_path=args.output)

    elif args.method == "llm":
        run_llm(cfg, args.mode, input_path=args.input, output_path=args.output, variant=args.variant)

    elif args.method == "all":
        if args.mode != "evaluate":
            print("--method all 仅支持 --mode evaluate", file=sys.stderr)
            sys.exit(1)
        run_all_evaluate()


if __name__ == "__main__":
    main()
