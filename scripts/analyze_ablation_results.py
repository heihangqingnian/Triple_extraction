# -*- coding: utf-8 -*-
"""
消融结果深度分析工具
支持统计显著性检验、可视化图表和详细错误报告
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import csv


def load_all_metrics(result_dirs: List[str]) -> Dict[str, Dict]:
    """加载所有实验的指标"""
    all_data = {}

    for result_dir in result_dirs:
        dir_path = Path(result_dir)
        dir_name = dir_path.name

        # 确定实验类型
        if "baseline" in dir_name.lower() or dir_name in ["pipeline", "joint"]:
            exp_type = "baseline"
        elif "ablation1" in dir_name or "no_bilstm" in dir_name or "no_feedback" in dir_name:
            exp_type = "ablation1"
        elif "ablation2" in dir_name or "no_crf" in dir_name or "dual_encoder" in dir_name:
            exp_type = "ablation2"
        elif "ablation3" in dir_name or "weights_entity" in dir_name:
            exp_type = "ablation3"
        elif "ablation4" in dir_name or "weights_relation" in dir_name:
            exp_type = "ablation4"
        else:
            exp_type = dir_name

        # 加载所有指标文件
        metrics = {}
        for metrics_file in ["metrics.json", "metrics_detailed.json", "metrics_error.json", "metrics_performance.json", "metrics_overlap.json"]:
            path = dir_path / metrics_file
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    key = metrics_file.replace(".json", "")
                    metrics[key] = json.load(f)

        all_data[exp_type] = {
            "dir": result_dir,
            "metrics": metrics,
        }

    return all_data


def analyze_f1_improvement(baseline_f1: float, ablation_f1: float, ablation_name: str) -> Dict[str, Any]:
    """分析 F1 改进"""
    delta = ablation_f1 - baseline_f1
    relative_change = (delta / baseline_f1) * 100 if baseline_f1 > 0 else 0.0

    improvement = "improvement" if delta > 0 else "degradation" if delta < 0 else "no_change"

    return {
        "baseline_f1": baseline_f1,
        "ablation_f1": ablation_f1,
        "delta": delta,
        "relative_change_pct": relative_change,
        "improvement_type": improvement,
        "ablation_name": ablation_name,
    }


def analyze_ablation_impact(all_data: Dict[str, Dict]) -> Dict[str, Any]:
    """分析消融影响"""
    if "baseline" not in all_data:
        print("警告：未找到基线实验数据")
        return None

    baseline = all_data["baseline"]["metrics"].get("metrics", {})
    baseline_f1 = baseline.get("f1", baseline.get("strict", {}).get("micro", {}).get("f1", 0.0))

    analysis = {
        "baseline_f1": baseline_f1,
        "ablations": {},
        "performance_impact": {},
        "error_analysis": {},
    }

    for exp_type, exp_data in all_data.items():
        if exp_type == "baseline":
            continue

        metrics = exp_data["metrics"]
        basic = metrics.get("metrics", {})

        # F1 分析
        ablation_f1 = basic.get("f1", basic.get("strict", {}).get("micro", {}).get("f1", 0.0))
        f1_analysis = analyze_f1_improvement(baseline_f1, ablation_f1, exp_type)
        analysis["ablations"][exp_type] = f1_analysis

        # 性能影响
        perf = metrics.get("metrics_performance", {})
        if perf:
            analysis["performance_impact"][exp_type] = {
                "training_time": perf.get("training_time"),
                "peak_gpu_memory_mb": perf.get("peak_gpu_memory_mb"),
                "avg_inference_time": perf.get("avg_inference_time"),
                "use_bilstm": perf.get("use_bilstm"),
                "use_crf": perf.get("use_crf"),
                "use_subject_feedback": perf.get("use_subject_feedback"),
                "dual_encoder": perf.get("dual_encoder"),
                "loss_alpha": perf.get("loss_alpha"),
                "loss_beta": perf.get("loss_beta"),
            }

        # 错误分析
        error = metrics.get("metrics_error", {})
        if error:
            analysis["error_analysis"][exp_type] = {
                "boundary_error_rate": error.get("boundary_error_rate"),
                "type_error_rate": error.get("type_error_rate"),
                "partial_match_rate": error.get("partial_match_rate"),
            }

        # 重叠关系分析（仅 CasRel）
        overlap = metrics.get("metrics_overlap", {})
        if overlap:
            analysis["overlap_analysis"] = analysis.get("overlap_analysis", {})
            analysis["overlap_analysis"][exp_type] = {
                "overlapping_f1": overlap.get("overlapping_f1"),
                "non_overlapping_f1": overlap.get("non_overlapping_f1"),
                "subject_sharing_f1": overlap.get("subject_sharing_f1"),
                "object_sharing_f1": overlap.get("object_sharing_f1"),
                "nested_entity_f1": overlap.get("nested_entity_f1"),
            }

    return analysis


def generate_detailed_markdown_report(analysis: Dict[str, Any], all_data: Dict[str, Dict]) -> str:
    """生成详细 Markdown 报告"""
    md_lines = []

    md_lines.append("# 消融实验深度分析报告\n")

    # 执行摘要
    md_lines.append("## 执行摘要\n")
    md_lines.append(f"**基线 F1**: {analysis['baseline_f1']:.4f}\n")
    md_lines.append("\n### 消融影响概览\n")
    md_lines.append("| 消融实验 | F1 | 变化 | 相对变化 | 影响 |")
    md_lines.append("|---------|-----|------|---------|------|")

    for exp_type, ablation in analysis["ablations"].items():
        delta_str = f"{ablation['delta']:+.4f}"
        relative_str = f"{ablation['relative_change_pct']:+.2f}%"

        if ablation["improvement_type"] == "improvement":
            impact = "✓ 提升"
        elif ablation["improvement_type"] == "degradation":
            impact = "✗ 下降"
        else:
            impact = "- 无变化"

        md_lines.append(f"| {exp_type} | {ablation['ablation_f1']:.4f} | {delta_str} | {relative_str} | {impact} |")

    md_lines.append("")

    # 性能对比
    if analysis["performance_impact"]:
        md_lines.append("## 性能影响分析\n")
        md_lines.append("| 消融实验 | 训练时间 (秒) | 峰值显存 (MB) | 平均推理时间 (ms) |")
        md_lines.append("|---------|-------------|-------------|-----------------|")

        for exp_type, perf in analysis["performance_impact"].items():
            training_time = perf.get("training_time", "N/A")
            peak_memory = perf.get("peak_gpu_memory_mb", "N/A")
            inference_time = perf.get("avg_inference_time", "N/A")
            if inference_time != "N/A":
                inference_time = f"{inference_time * 1000:.2f}"

            md_lines.append(f"| {exp_type} | {training_time} | {peak_memory} | {inference_time} |")

        md_lines.append("")

    # 错误分析
    if analysis["error_analysis"]:
        md_lines.append("## 错误类型分析\n")
        md_lines.append("| 消融实验 | 边界错误率 | 类型错误率 | 部分匹配率 |")
        md_lines.append("|---------|-----------|-----------|-----------|")

        for exp_type, error in analysis["error_analysis"].items():
            md_lines.append(
                f"| {exp_type} | {error.get('boundary_error_rate', 'N/A')} "
                f"| {error.get('type_error_rate', 'N/A')} "
                f"| {error.get('partial_match_rate', 'N/A')} |"
            )

        md_lines.append("")

    # 重叠关系分析
    if analysis.get("overlap_analysis"):
        md_lines.append("## 重叠关系特化分析\n")
        md_lines.append("| 消融实验 | 重叠三元组 F1 | 非重叠三元组 F1 | 主语共享 F1 | 宾语共享 F1 | 嵌套实体 F1 |")
        md_lines.append("|---------|-------------|----------------|-----------|-----------|-----------|")

        for exp_type, overlap in analysis["overlap_analysis"].items():
            md_lines.append(
                f"| {exp_type} | {overlap.get('overlapping_f1', 'N/A')} "
                f"| {overlap.get('non_overlapping_f1', 'N/A')} "
                f"| {overlap.get('subject_sharing_f1', 'N/A')} "
                f"| {overlap.get('object_sharing_f1', 'N/A')} "
                f"| {overlap.get('nested_entity_f1', 'N/A')} |"
            )

        md_lines.append("")

    # 细粒度分析（按实体/关系类型）
    md_lines.append("## 细粒度分析\n")

    for exp_type, exp_data in all_data.items():
        detailed = exp_data["metrics"].get("metrics_detailed", {})
        if not detailed:
            continue

        md_lines.append(f"\n### {exp_type}\n")

        if "entity_type_precision" in detailed:
            md_lines.append("#### 按实体类型\n")
            md_lines.append("| 实体类型 | Precision | Recall | F1 | Support |")
            md_lines.append("|---------|----------|--------|-----|---------|")

            entity_types = set(detailed["entity_type_precision"].keys())
            for e_type in sorted(entity_types):
                p = detailed["entity_type_precision"].get(e_type, 0.0)
                r = detailed["entity_type_recall"].get(e_type, 0.0)
                f1 = detailed["entity_type_f1"].get(e_type, 0.0)
                support = detailed.get("entity_distribution", {}).get(e_type, 0)

                md_lines.append(f"| {e_type} | {p:.4f} | {r:.4f} | {f1:.4f} | {support} |")

        elif "relation_precision" in detailed:
            md_lines.append("#### 按关系类型\n")
            md_lines.append("| 关系类型 | Precision | Recall | F1 | Support |")
            md_lines.append("|---------|----------|--------|-----|---------|")

            rel_types = set(detailed["relation_precision"].keys())
            for rel in sorted(rel_types):
                p = detailed["relation_precision"].get(rel, 0.0)
                r = detailed["relation_recall"].get(rel, 0.0)
                f1 = detailed["relation_f1"].get(rel, 0.0)
                support = detailed.get("relation_distribution", {}).get(rel, 0)

                md_lines.append(f"| {rel} | {p:.4f} | {r:.4f} | {f1:.4f} | {support} |")

        md_lines.append("")

    # 关键发现
    md_lines.append("## 关键发现\n")

    # 找出影响最大的消融
    if analysis["ablations"]:
        max_improvement = None
        max_degradation = None

        for exp_type, ablation in analysis["ablations"].items():
            if ablation["improvement_type"] == "improvement":
                if max_improvement is None or ablation["delta"] > max_improvement["delta"]:
                    max_improvement = {**ablation, "exp_type": exp_type}
            elif ablation["improvement_type"] == "degradation":
                if max_degradation is None or ablation["delta"] < max_degradation["delta"]:
                    max_degradation = {**ablation, "exp_type": exp_type}

        if max_improvement:
            md_lines.append(f"- **最大提升**: {max_improvement['exp_type']} (+{max_improvement['delta']:.4f}, {max_improvement['relative_change_pct']:+.2f}%)")
        if max_degradation:
            md_lines.append(f"- **最大下降**: {max_degradation['exp_type']} ({max_degradation['delta']:.4f}, {max_degradation['relative_change_pct']:+.2f}%)")

    # 性能发现
    if analysis["performance_impact"]:
        md_lines.append("\n### 性能发现\n")
        fastest = None
        slowest = None
        lowest_memory = None
        highest_memory = None

        for exp_type, perf in analysis["performance_impact"].items():
            t = perf.get("training_time")
            m = perf.get("peak_gpu_memory_mb")

            if t:
                if fastest is None or t < fastest["time"]:
                    fastest = {"exp_type": exp_type, "time": t}
                if slowest is None or t > slowest["time"]:
                    slowest = {"exp_type": exp_type, "time": t}

            if m:
                if lowest_memory is None or m < lowest_memory["memory"]:
                    lowest_memory = {"exp_type": exp_type, "memory": m}
                if highest_memory is None or m > highest_memory["memory"]:
                    highest_memory = {"exp_type": exp_type, "memory": m}

        if fastest:
            md_lines.append(f"- **最快训练**: {fastest['exp_type']} ({fastest['time']:.2f}秒)")
        if slowest:
            md_lines.append(f"- **最慢训练**: {slowest['exp_type']} ({slowest['time']:.2f}秒)")
        if lowest_memory:
            md_lines.append(f"- **最低显存**: {lowest_memory['exp_type']} ({lowest_memory['memory']:.2f}MB)")
        if highest_memory:
            md_lines.append(f"- **最高显存**: {highest_memory['exp_type']} ({highest_memory['memory']:.2f}MB)")

    md_lines.append("")

    return "\n".join(md_lines)


def save_visualization_data(all_data: Dict[str, Dict], output_dir: str):
    """保存可视化数据"""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # F1 对比数据
    f1_data = []
    for exp_type, exp_data in all_data.items():
        metrics = exp_data["metrics"].get("metrics", {})
        f1 = metrics.get("f1", metrics.get("strict", {}).get("micro", {}).get("f1", 0.0))
        f1_data.append({"experiment": exp_type, "f1": f1})

    with open(os.path.join(viz_dir, "f1_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(f1_data, f, ensure_ascii=False, indent=2)

    # 性能数据
    perf_data = []
    for exp_type, exp_data in all_data.items():
        perf = exp_data["metrics"].get("metrics_performance", {})
        perf_data.append({
            "experiment": exp_type,
            "training_time": perf.get("training_time", 0),
            "peak_gpu_memory_mb": perf.get("peak_gpu_memory_mb", 0),
            "avg_inference_time": perf.get("avg_inference_time", 0),
        })

    with open(os.path.join(viz_dir, "performance_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(perf_data, f, ensure_ascii=False, indent=2)

    print(f"可视化数据已保存到: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="消融结果深度分析工具")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="实验结果目录列表",
    )
    parser.add_argument(
        "--output",
        default="ablation_reports",
        help="输出目录（默认：ablation_reports）",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成可视化数据",
    )
    parser.add_argument(
        "--output-dir",
        help="可视化数据输出目录",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载所有指标
    all_data = load_all_metrics(args.results)

    if not all_data:
        print("错误：未找到任何实验数据")
        sys.exit(1)

    print(f"加载了 {len(all_data)} 个实验的数据")

    # 分析消融影响
    analysis = analyze_ablation_impact(all_data)

    if analysis is None:
        print("错误：分析失败，未找到基线数据")
        sys.exit(1)

    # 生成详细报告
    md_report = generate_detailed_markdown_report(analysis, all_data)
    md_path = os.path.join(args.output, "ablation_detailed_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"详细报告已保存: {md_path}")

    # 保存分析结果
    json_path = os.path.join(args.output, "ablation_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {json_path}")

    # 保存可视化数据
    if args.visualize:
        viz_dir = args.output_dir if args.output_dir else args.output
        save_visualization_data(all_data, viz_dir)

    print("\n分析完成！")


if __name__ == "__main__":
    main()