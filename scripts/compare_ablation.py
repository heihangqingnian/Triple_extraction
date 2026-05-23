# -*- coding: utf-8 -*-
"""
消融实验结果对比工具
用于快速对比多个消融实验的指标
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_metrics(result_dir: str) -> Dict[str, Any]:
    """加载实验指标"""
    result_dir = Path(result_dir)

    # 尝试加载基础指标
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def load_all_metrics(result_dir: str) -> Dict[str, Any]:
    """加载所有指标文件"""
    result_dir = Path(result_dir)
    all_metrics = {}

    for metrics_file in ["metrics.json", "metrics_detailed.json", "metrics_error.json", "metrics_performance.json", "metrics_overlap.json"]:
        path = result_dir / metrics_file
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                key = metrics_file.replace(".json", "")
                all_metrics[key] = json.load(f)

    return all_metrics


def compare_pipeline_ablations(result_dirs: List[str]) -> Dict[str, Any]:
    """对比 Pipeline 消融实验"""
    comparison = {
        "method": "pipeline",
        "experiments": {},
        "baseline": None,
    }

    for result_dir in result_dirs:
        dir_name = Path(result_dir).name
        all_metrics = load_all_metrics(result_dir)

        if not all_metrics:
            print(f"警告：未找到指标文件: {result_dir}")
            continue

        # 确定实验类型
        if "baseline" in dir_name.lower() or dir_name == "pipeline":
            exp_type = "baseline"
            comparison["baseline"] = all_metrics
        elif "ablation1" in dir_name or "no_bilstm" in dir_name:
            exp_type = "ablation1_no_bilstm"
        elif "ablation2" in dir_name or "no_crf" in dir_name:
            exp_type = "ablation2_no_crf"
        else:
            exp_type = dir_name

        comparison["experiments"][exp_type] = {
            "dir": result_dir,
            "metrics": all_metrics,
        }

    return comparison


def compare_casrel_ablations(result_dirs: List[str]) -> Dict[str, Any]:
    """对比 CasRel 消融实验"""
    comparison = {
        "method": "casrel",
        "experiments": {},
        "baseline": None,
    }

    for result_dir in result_dirs:
        dir_name = Path(result_dir).name
        all_metrics = load_all_metrics(result_dir)

        if not all_metrics:
            print(f"警告：未找到指标文件: {result_dir}")
            continue

        # 确定实验类型
        if "baseline" in dir_name.lower() or dir_name == "joint":
            exp_type = "baseline"
            comparison["baseline"] = all_metrics
        elif "ablation1" in dir_name or "no_feedback" in dir_name:
            exp_type = "ablation1_no_feedback"
        elif "ablation2" in dir_name or "dual_encoder" in dir_name:
            exp_type = "ablation2_dual_encoder"
        elif "ablation3" in dir_name or "weights_entity" in dir_name:
            exp_type = "ablation3_weights_entity"
        elif "ablation4" in dir_name or "weights_relation" in dir_name:
            exp_type = "ablation4_weights_relation"
        else:
            exp_type = dir_name

        comparison["experiments"][exp_type] = {
            "dir": result_dir,
            "metrics": all_metrics,
        }

    return comparison


def generate_markdown_report(comparison: Dict[str, Any]) -> str:
    """生成 Markdown 格式的对比报告"""
    md_lines = []

    method = comparison["method"]
    method_name = "Pipeline (NER)" if method == "pipeline" else "CasRel (Joint Extraction)"

    md_lines.append(f"# {method_name} 消融实验对比报告\n")

    # 基础指标对比表格
    md_lines.append("## 基础指标对比\n")
    md_lines.append("| 实验名称 | Precision | Recall | F1 |")
    md_lines.append("|---------|----------|--------|-----|")

    baseline_f1 = 0.0
    for exp_name, exp_data in comparison["experiments"].items():
        metrics = exp_data["metrics"]
        basic = metrics.get("metrics", {})

        # 尝试获取 F1
        if "f1" in basic:
            f1 = basic["f1"]
            precision = basic.get("precision", "N/A")
            recall = basic.get("recall", "N/A")
        elif "strict" in basic:
            f1 = basic["strict"]["micro"]["f1"]
            precision = basic["strict"]["micro"]["precision"]
            recall = basic["strict"]["micro"]["recall"]
        else:
            f1 = "N/A"
            precision = "N/A"
            recall = "N/A"

        # 记录基线 F1
        if exp_name == "baseline":
            baseline_f1 = f1 if isinstance(f1, float) else 0.0

        # 格式化数值
        if isinstance(f1, float):
            delta = f1 - baseline_f1
            delta_str = f" ({delta:+.4f})" if baseline_f1 > 0 else ""
            f1_str = f"{f1:.4f}{delta_str}"
        else:
            f1_str = str(f1)

        if isinstance(precision, float):
            precision_str = f"{precision:.4f}"
        else:
            precision_str = str(precision)

        if isinstance(recall, float):
            recall_str = f"{recall:.4f}"
        else:
            recall_str = str(recall)

        md_lines.append(f"| {exp_name} | {precision_str} | {recall_str} | {f1_str} |")

    md_lines.append("")

    # 性能指标对比
    md_lines.append("## 性能指标对比\n")
    md_lines.append("| 实验名称 | 训练时间 (秒) | 峰值显存 (MB) | 平均推理时间 (ms) |")
    md_lines.append("|---------|-------------|-------------|-----------------|")

    for exp_name, exp_data in comparison["experiments"].items():
        metrics = exp_data["metrics"]
        perf = metrics.get("metrics_performance", {})

        training_time = perf.get("training_time", "N/A")
        peak_memory = perf.get("peak_gpu_memory_mb", "N/A")
        if peak_memory == 0 or peak_memory == "N/A":
            peak_memory = perf.get("peak_gpu_memory", "N/A")

        inference_time = perf.get("avg_inference_time", "N/A")
        if inference_time != "N/A":
            inference_time = f"{inference_time * 1000:.2f}"  # 转换为毫秒

        md_lines.append(f"| {exp_name} | {training_time} | {peak_memory} | {inference_time} |")

    md_lines.append("")

    # 错误分析指标
    md_lines.append("## 错误分析指标\n")
    md_lines.append("| 实验名称 | 边界错误率 | 类型错误率 | 遗漏率 | 误判率 |")
    md_lines.append("|---------|-----------|-----------|--------|--------|")

    for exp_name, exp_data in comparison["experiments"].items():
        metrics = exp_data["metrics"]
        error = metrics.get("metrics_error", {})
        basic = metrics.get("metrics", {})

        # 计算遗漏率和误判率
        if "precision" in basic and "recall" in basic:
            omission_rate = 1.0 - basic["recall"]
            false_positive_rate = 1.0 - basic["precision"]
        else:
            omission_rate = "N/A"
            false_positive_rate = "N/A"

        boundary_error_rate = error.get("boundary_error_rate", "N/A")
        type_error_rate = error.get("type_error_rate", "N/A")

        # 格式化
        def format_rate(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        md_lines.append(
            f"| {exp_name} | {format_rate(boundary_error_rate)} | {format_rate(type_error_rate)} "
            f"| {format_rate(omission_rate)} | {format_rate(false_positive_rate)} |"
        )

    md_lines.append("")

    # 重叠关系指标（仅 CasRel）
    if method == "casrel":
        md_lines.append("## 重叠关系特化指标\n")
        md_lines.append("| 实验名称 | 重叠三元组 F1 | 非重叠三元组 F1 | 主语共享 F1 | 宾语共享 F1 | 嵌套实体 F1 |")
        md_lines.append("|---------|-------------|----------------|-----------|-----------|-----------|")

        for exp_name, exp_data in comparison["experiments"].items():
            metrics = exp_data["metrics"]
            overlap = metrics.get("metrics_overlap", {})

            def format_float(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return "N/A"

            md_lines.append(
                f"| {exp_name} | {format_float(overlap.get('overlapping_f1', 'N/A'))} "
                f"| {format_float(overlap.get('non_overlapping_f1', 'N/A'))} "
                f"| {format_float(overlap.get('subject_sharing_f1', 'N/A'))} "
                f"| {format_float(overlap.get('object_sharing_f1', 'N/A'))} "
                f"| {format_float(overlap.get('nested_entity_f1', 'N/A'))} |"
            )

        md_lines.append("")

    # 消融配置
    md_lines.append("## 消融配置摘要\n")
    for exp_name, exp_data in comparison["experiments"].items():
        perf = exp_data["metrics"].get("metrics_performance", {})

        config_items = []
        if "use_bilstm" in perf:
            config_items.append(f"BiLSTM: {perf['use_bilstm']}")
        if "use_crf" in perf:
            config_items.append(f"CRF: {perf['use_crf']}")
        if "use_subject_feedback" in perf:
            config_items.append(f"Subject Feedback: {perf['use_subject_feedback']}")
        if "dual_encoder" in perf:
            config_items.append(f"Dual Encoder: {perf['dual_encoder']}")
        if "loss_alpha" in perf:
            config_items.append(f"Loss α: {perf['loss_alpha']}")
        if "loss_beta" in perf:
            config_items.append(f"Loss β: {perf['loss_beta']}")

        if config_items:
            md_lines.append(f"**{exp_name}**: {', '.join(config_items)}")

    md_lines.append("")

    return "\n".join(md_lines)


def generate_csv_report(comparison: Dict[str, Any]) -> str:
    """生成 CSV 格式的对比数据"""
    method = comparison["method"]

    csv_lines = []
    csv_lines.append("experiment,precision,recall,f1,training_time,peak_memory_mb")

    for exp_name, exp_data in comparison["experiments"].items():
        metrics = exp_data["metrics"]
        basic = metrics.get("metrics", {})
        perf = metrics.get("metrics_performance", {})

        # 获取 F1
        if "f1" in basic:
            f1 = basic["f1"]
            precision = basic.get("precision", "")
            recall = basic.get("recall", "")
        elif "strict" in basic:
            f1 = basic["strict"]["micro"]["f1"]
            precision = basic["strict"]["micro"]["precision"]
            recall = basic["strict"]["micro"]["recall"]
        else:
            f1 = ""
            precision = ""
            recall = ""

        training_time = perf.get("training_time", "")
        peak_memory = perf.get("peak_gpu_memory_mb", "")

        csv_lines.append(f"{exp_name},{precision},{recall},{f1},{training_time},{peak_memory}")

    return "\n".join(csv_lines)


def main():
    parser = argparse.ArgumentParser(description="消融实验结果对比工具")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="实验结果目录列表，如: results/pipeline results/pipeline_ablation1",
    )
    parser.add_argument(
        "--output",
        default="ablation_reports",
        help="输出目录（默认：ablation_reports）",
    )
    parser.add_argument(
        "--method",
        choices=["pipeline", "casrel", "auto"],
        default="auto",
        help="实验类型（默认：auto 自动检测）",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 自动检测方法类型
    if args.method == "auto":
        # 检查第一个结果目录
        first_dir = Path(args.results[0])
        if "joint" in first_dir.name.lower() or "casrel" in first_dir.name.lower():
            args.method = "casrel"
        else:
            args.method = "pipeline"

    print(f"检测到方法类型: {args.method}")

    # 加载和对比
    if args.method == "pipeline":
        comparison = compare_pipeline_ablations(args.results)
        prefix = "pipeline"
    else:
        comparison = compare_casrel_ablations(args.results)
        prefix = "casrel"

    # 生成 Markdown 报告
    md_report = generate_markdown_report(comparison)
    md_path = os.path.join(args.output, f"{prefix}_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Markdown 报告已保存: {md_path}")

    # 生成 CSV 报告
    csv_report = generate_csv_report(comparison)
    csv_path = os.path.join(args.output, f"{prefix}_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_report)
    print(f"CSV 报告已保存: {csv_path}")

    # 保存原始 JSON
    json_path = os.path.join(args.output, f"{prefix}_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告已保存: {json_path}")

    print("\n对比完成！")


if __name__ == "__main__":
    main()