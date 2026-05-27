# 三元组抽取对比研究 — 全流程操作指南

本文档覆盖从原始数据集到模型推理的完整操作步骤，适用于三种方法：
**Pipeline（NER→RE）**、**Joint（CasRel）**、**LLM（ChatGLM2-6B + LoRA）**。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据下载与预处理](#2-数据下载与预处理)
3. [Pipeline 方法](#3-pipeline-方法)
4. [Joint 方法（CasRel）](#4-joint-方法casrel)
5. [LLM 方法（ChatGLM2-6B）](#5-llm-方法chatglm2-6b)
6. [消融实验](#6-消融实验)
7. [细粒度分析](#7-细粒度分析)
8. [低资源对比实验](#8-低资源对比实验)
9. [三方法结果对比](#9-三方法结果对比)
10. [目录结构说明](#10-目录结构说明)

---

## 1. 环境准备

```bash
# 建议使用 Python 3.9+，CUDA 11.8+
pip install -r requirements.txt
```

主要依赖：`torch`, `transformers`, `peft`, `pytorch-crf`, `seqeval`, `PyYAML`, `tqdm`

---

## 2. 数据下载与预处理

### 2.1 下载原始数据集

从 Gitee 下载 DuIE2.0 数据集，放置到 `data/raw/DuIE2.0/`：

```
data/raw/DuIE2.0/
├── duie_train.json     # 训练集（JSONL 格式，每行一条样本）
├── duie_dev.json       # 验证/测试集
└── duie_schema/        # 关系 Schema 定义
```

### 2.2 下载预训练模型

```bash
# 下载 chinese-bert-wwm（Pipeline NER + Joint 共用）
python scripts/download_model.py --model hfl/chinese-bert-wwm --output methods/pipeline/ner/model

# ChatGLM2-6B（LLM 方法）— 约 12GB，请确保磁盘空间充足
# 方式一：HuggingFace
python scripts/download_model.py --model THUDM/chatglm2-6b --output models/chatglm2-6b

# 方式二：ModelScope（国内推荐）
# pip install modelscope
# python -c "from modelscope import snapshot_download; snapshot_download('ZhipuAI/chatglm2-6b', local_dir='models/chatglm2-6b')"
```

### 2.3 一键预处理（三种方法格式）

```bash
# 生成全部三种方法的训练/验证/测试数据
python scripts/preprocess.py --method all

# 单独生成某种方法的数据
python scripts/preprocess.py --method pipeline
python scripts/preprocess.py --method joint
python scripts/preprocess.py --method llm
```

预处理后的文件目录：

```
data/processed/
├── shared/
│   ├── rel2id.json         # 49 个关系 → ID 映射
│   └── entity2id.json      # 实体类型映射
├── pipeline/
│   ├── train/train_ner.txt, train_re.txt
│   ├── dev/dev_ner.txt, dev_re.txt
│   └── test/test_ner.txt, test_re.txt, pipeline_test.jsonl
├── joint/
│   ├── train/train.json    # 预分词 + head/tail mask（约 3.2GB）
│   ├── dev/dev.json
│   └── test/test.json
└── llm/
    ├── train_base.json     # 三种变体的 Alpaca 格式训练集（用于 LlamaFactory 微调）
    ├── train_schema.json
    ├── train_cot.json
    ├── dev_{base,schema,cot}.json   # 验证集（三种格式）
    ├── test_base.json      # 测试集（三种格式），推理时直接从文件读取 prompt
    ├── test_schema.json    # instruction 字段与训练数据完全一致，无需代码动态拼接
    └── test_cot.json
```

---

## 3. Pipeline 方法

### 3.1 训练

```bash
# 完整训练（NER 阶段 → RE 阶段，含早停）
python main.py --method pipeline --mode train

# 单独训练 RE 模型（NER 权重已存在时使用，等价于 train_re_only.py）
python main.py --method pipeline --mode train --component re

# 单独训练 NER 模型
python main.py --method pipeline --mode train --component ner

# 自定义配置（--component 与 --config 可组合使用）
python main.py --method pipeline --mode train --config configs/pipeline.yaml
python main.py --method pipeline --mode train --component re --config configs/ablation_pipeline_1_no_bilstm.yaml
```

训练产出：

- `results/pipeline/ner_best.pt` — NER 最优模型权重
- `results/pipeline/re_best.pt` — RE 最优模型权重
- `results/pipeline/train.log` — 训练日志

### 3.2 评估

```bash
# 在测试集上端到端评估
python main.py --method pipeline --mode evaluate
```

评估产出（`results/pipeline/`）：

- `metrics.json` — 整体 P/R/F1
- `predictions.jsonl` — 每条样本的预测结果（含 `text/pred_triples/gold_triples`）
- `error_report.txt` — 九类错误统计
- `error_cases.txt` — 错误样本详情（最多 200 条）
- `per_relation.txt` — 49 类关系的逐类 P/R/F1

### 3.3 推理（自定义输入）

```bash
# 对任意 JSONL 文件（每行含 "text" 字段）进行批量预测
python main.py --method pipeline --mode predict \
    --input  data/processed/pipeline/test/pipeline_test.jsonl \
    --output results/pipeline/my_predictions.jsonl
```

---

## 4. Joint 方法（CasRel）

### 4.1 训练

```bash
# 完整训练（CasRel 端到端，含早停，约 300 epoch）
python main.py --method joint --mode train
```

训练产出：

- `results/joint/best.pt` — 最优模型权重
- `results/joint/train.log` — 训练日志

### 4.2 评估

```bash
python main.py --method joint --mode evaluate
```

评估产出（`results/joint/`）：

- `metrics.json`, `predictions.jsonl`, `error_report.txt`, `error_cases.txt`, `per_relation.txt`

### 4.3 推理

```bash
python main.py --method joint --mode predict \
    --output results/joint/my_predictions.jsonl
```

---

## 5. LLM 方法（ChatGLM2-6B）

LLM 方法分为两个阶段：**LoRA 微调**（通过 LlamaFactory）和**推理评估**（本仓库）。

三个 LoRA 变体对应三种 Prompt 策略，分别有独立配置文件：

| 变体   | 配置文件                  | LoRA 权重路径                  | Prompt 策略                   |
| ------ | ------------------------- | ------------------------------ | ----------------------------- |
| base   | `configs/llm_base.yaml`   | `models/chatglm2-lora-base/`   | 基础指令，无约束              |
| schema | `configs/llm_schema.yaml` | `models/chatglm2-lora-schema/` | 加入 Schema 实体/关系类型约束 |
| cot    | `configs/llm_cot.yaml`    | `models/chatglm2-lora-cot/`    | Chain-of-Thought 分步推理     |

### 5.1 LoRA 微调（外部 LlamaFactory）

```bash
# 安装 LlamaFactory
pip install llamafactory

# 训练 base LoRA（以 base 为例，schema/cot 同理，替换 output_dir 和 prompt 格式）
llamafactory-cli train \
    --model_name_or_path ZhipuAI/chatglm2-6b \
    --finetuning_type lora \
    --dataset data/processed/llm/train.json \
    --dataset_dir . \
    --output_dir models/chatglm2-lora-base \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --lora_rank 8
```

若暂无 LoRA 权重，推理将自动回退到基础模型（日志中会提示 Warning）。

### 5.2 推理

**方式一（推荐）：使用各变体独立配置文件**

每个配置文件指定对应的 LoRA 路径、Prompt 类型和结果输出目录，互不干扰：

```bash
# base 变体：结果保存到 results/llm/base/
python main.py --method llm --mode predict --config configs/llm_base.yaml

# schema 变体：结果保存到 results/llm/schema/
python main.py --method llm --mode predict --config configs/llm_schema.yaml

# cot 变体：结果保存到 results/llm/cot/
python main.py --method llm --mode predict --config configs/llm_cot.yaml
```

**方式二：使用主配置 + `--variant` 参数**

```bash
# 一次性对全部三个变体推理（按顺序加载 LoRA，完成后释放显存）
python main.py --method llm --mode predict

# 只推理指定变体
python main.py --method llm --mode predict --variant base
python main.py --method llm --mode predict --variant schema
python main.py --method llm --mode predict --variant cot
```

推理产出（方式一，各变体独立目录）：

```
results/llm/
├── base/
│   └── predictions_base.jsonl
├── schema/
│   └── predictions_schema.jsonl
└── cot/
    └── predictions_cot.jsonl
```

每条记录格式：

```json
{
  "prompt_type": "base",
  "prompt": "...",
  "text": "原始文本",
  "predict": "[('主体', '关系', '客体'), ...]",
  "label": "[('主体', '关系', '客体'), ...]",
  "parse_error": false
}
```

### 5.3 评估

**方式一（推荐）：使用各变体独立配置文件**

```bash
# base 变体：指标保存到 results/llm/base/metrics_base.json
python main.py --method llm --mode evaluate --config configs/llm_base.yaml

# schema 变体
python main.py --method llm --mode evaluate --config configs/llm_schema.yaml

# cot 变体
python main.py --method llm --mode evaluate --config configs/llm_cot.yaml

# 指定已有预测文件（--input 仅在单变体时生效）
python main.py --method llm --mode evaluate --config configs/llm_base.yaml \
    --input results/llm/base/predictions_base.jsonl
```

**方式二：使用主配置**

```bash
# 同时评估全部三种变体，汇总写入 results/llm/metrics.json
python main.py --method llm --mode evaluate

# 只评估指定变体
python main.py --method llm --mode evaluate --variant base
python main.py --method llm --mode evaluate --variant schema
python main.py --method llm --mode evaluate --variant cot
```

评估产出（方式一，以 base 为例）：

```
results/llm/base/
├── predictions_base.jsonl     — 推理结果
├── metrics_base.json          — P/R/F1 指标
└── error_report_base.txt      — 错误类型统计
```

---

## 6. 消融实验

### Pipeline 消融

```bash
# 消融 1：去除 BiLSTM（BERT-CRF）
python main.py --method pipeline --mode train    --config configs/ablation_pipeline_1_no_bilstm.yaml
python main.py --method pipeline --mode evaluate --config configs/ablation_pipeline_1_no_bilstm.yaml

# 消融 2：去除 CRF（BERT-Linear）
python main.py --method pipeline --mode train    --config configs/ablation_pipeline_2_no_crf.yaml
python main.py --method pipeline --mode evaluate --config configs/ablation_pipeline_2_no_crf.yaml
```

`--component` 可与消融配置组合使用，适合 NER/RE 任一阶段已完成、只需重跑另一阶段的场景：

```bash
# 仅重训消融 1 的 RE 部分（NER 权重已存在）
python main.py --method pipeline --mode train --component ner --config configs/ablation_pipeline_1_no_bilstm.yaml

# 仅重训消融 2 的 NER 部分
python main.py --method pipeline --mode train --component ner --config configs/ablation_pipeline_2_no_crf.yaml
```

### Joint 消融

```bash
# 消融 1：移除主实体条件反馈
python main.py --method joint --mode train    --config configs/ablation_joint_1_no_feedback.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_1_no_feedback.yaml

# 消融 2：双独立编码器
python main.py --method joint --mode train    --config configs/ablation_joint_2_dual_encoder.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_2_dual_encoder.yaml

# 消融 3a：损失权重偏实体
python main.py --method joint --mode train    --config configs/ablation_joint_3_weights_entity.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_3_weights_entity.yaml

# 消融 3b：损失权重偏关系
python main.py --method joint --mode train    --config configs/ablation_joint_3_weights_relation.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_3_weights_relation.yaml
```

### 消融结果对比

```bash
python scripts/compare_ablation.py
python scripts/analyze_ablation_results.py
```

---

## 7. 细粒度分析

以下两个脚本均**读取已有预测文件**，在内存中分类，不生成子集文件。

### 7.1 重叠类型细粒度评测（Normal / EPO / SEO）

```bash
python scripts/fine_grained_eval.py

# 指定文件路径（可选）
python scripts/fine_grained_eval.py \
    --pipeline results/pipeline/predictions.jsonl \
    --joint    results/joint/predictions.jsonl \
    --llm_dir  results/llm
```

输出示例：

```
子集       样本数   Pipeline     Joint  LLM-base  LLM-schema   LLM-cot
Normal       2850    0.7821    0.8134    0.6210      0.6890    0.7102
SEO           680    0.6910    0.7230    0.5110      0.5830    0.6210
EPO           210    0.5820    0.6410    0.4210      0.5010    0.5630
```

**分类定义**：

- **Normal**：句中所有三元组之间无任何实体共享
- **SEO**（SingleEntityOverlap）：至少一个实体（主体或客体）出现在多个三元组中，且无 EPO
- **EPO**（EntityPairOverlap）：同一实体对（主体+客体）以不同谓词出现在多个三元组中

### 7.2 三元组密度分组评测 + LLM 截断分析

```bash
python scripts/density_eval.py

# 指定路径
python scripts/density_eval.py \
    --pipeline results/pipeline/predictions.jsonl \
    --joint    results/joint/predictions.jsonl \
    --llm_dir  results/llm
```

输出包含两部分：

1. 按三元组密度（1 / 2-3 / 4+）分桶的各方法 F1 对比表
2. LLM 在 ≥5 个三元组的长句上的专项分析（recall、截断比例、遗忘信号）

---

## 8. 低资源对比实验

随机抽取 1%、5%、10% 训练集，分别训练并评估 Pipeline 和 Joint。

```bash
# 完整运行（采样 + 训练 + 评估）
python scripts/low_resource_eval.py

# 仅指定比例和方法
python scripts/low_resource_eval.py --ratios 0.01 0.05 0.10 --methods pipeline joint

# 仅生成采样数据，不训练（适用于 LLM 手动调用 LlamaFactory）
python scripts/low_resource_eval.py --data_only

# 仅生成 LLM 采样数据
python scripts/low_resource_eval.py --methods llm --data_only
```

采样数据目录：

```
data/processed/low_resource/
├── 1pct/
│   ├── pipeline/train/, dev/, test/
│   ├── joint/train/, dev/, test/
│   └── llm/train_base.json, train_schema.json, train_cot.json
├── 5pct/
└── 10pct/
```

结果目录：

```
results/low_resource/
├── 1pct/pipeline/metrics.json
├── 1pct/joint/metrics.json
├── 5pct/...
└── 10pct/...
```

---

## 9. 三方法结果对比

```bash
# 前提：三种方法均已完成评估（生成了各自的 metrics.json）
python main.py --method all --mode evaluate
```

输出示例：

```
方法             Precision     Recall         F1
Pipeline          0.7234       0.7012     0.7121
Joint             0.7891       0.7634     0.7760
llm               0.6123       0.5834     0.5975
```

---

## 10. 目录结构说明

```
Research1/
├── main.py                     # 统一入口（train/evaluate/predict）
├── WORKFLOW.md                 # 本文档
├── requirements.txt
│
├── configs/                    # YAML 超参数配置
│   ├── pipeline.yaml
│   ├── joint.yaml
│   ├── llm.yaml                # LLM 主配置（三路 LoRA 权重，--variant 控制）
│   ├── llm_base.yaml           # base LoRA 独立配置（推荐单独使用）
│   ├── llm_schema.yaml         # schema LoRA 独立配置
│   ├── llm_cot.yaml            # cot LoRA 独立配置
│   └── ablation_*.yaml
│
├── methods/
│   ├── pipeline/               # NER（BERT-BiLSTM-CRF）→ RE（BERT 分类器）
│   ├── joint/                  # CasRel 端到端联合抽取
│   └── llm/
│       ├── infer.py            # 推理：按 prompt_type 加载独立 LoRA
│       └── evaluator.py        # 评估：三种 Prompt 类型统一评估
│
├── utils/
│   ├── common.py               # 随机种子、日志、YAML 加载
│   ├── io_utils.py             # JSON/JSONL 读写、模型保存
│   └── metrics.py              # P/R/F1、错误分析、parse_triple_string
│
├── scripts/
│   ├── preprocess.py           # 数据预处理（DuIE2.0 → 三种格式）
│   ├── fine_grained_eval.py    # 细粒度评测（Normal/EPO/SEO）
│   ├── density_eval.py         # 密度分组评测 + LLM 截断分析
│   ├── low_resource_eval.py    # 低资源训练对比（1%/5%/10%）
│   ├── compare_ablation.py     # 消融结果对比
│   └── smoke_test.py           # 快速冒烟测试
│
├── data/
│   ├── raw/DuIE2.0/            # 原始数据集（手动下载）
│   └── processed/              # 预处理后数据（脚本生成）
│
├── models/
│   ├── chatglm2-6b/            # ChatGLM2-6B 基础模型（手动下载）
│   ├── chatglm2-lora-base/     # LoRA 权重（LlamaFactory 训练产出）
│   ├── chatglm2-lora-schema/
│   └── chatglm2-lora-cot/
│
└── results/                    # 实验结果（自动生成）
    ├── pipeline/
    ├── joint/
    ├── llm/
    │   ├── base/               # base LoRA 推理/评估结果
    │   ├── schema/             # schema LoRA 推理/评估结果
    │   └── cot/                # cot LoRA 推理/评估结果
    └── low_resource/
```

---

## 常见问题

**Q: CUDA 内存不足（OOM）**

- 减小 `batch_size`（configs/joint.yaml 或 pipeline.yaml）
- LLM 推理时设置 `device: cpu`（速度慢但无 OOM）
- 确保每次只加载一套 LoRA 权重（infer.py 已自动释放）

**Q: LoRA 权重不存在**

- 推理会自动回退到基础模型，观察日志中 `Warning: LoRA 权重目录不存在`
- 基础模型推理效果通常低于微调后的模型

**Q: fine_grained_eval.py 或 density_eval.py 报"未找到预测文件"**

- 请先运行各方法的 evaluate 步骤生成 `predictions.jsonl`
- LLM 需要先运行 `--mode predict` 再运行 `--mode evaluate`
- LLM 三个变体需分别运行（使用 `configs/llm_base.yaml` / `llm_schema.yaml` / `llm_cot.yaml`）

**Q: LLM 推理/评估如何选择配置文件？**

- 推荐为每个变体使用独立配置文件（`configs/llm_*.yaml`），结果自动保存到独立目录
- 需要一次性跑全部三个变体时，使用主配置 `configs/llm.yaml`（不加 `--variant`）
- `--variant` 参数可以在任意配置文件基础上强制覆盖 prompt 类型

**Q: low_resource_eval.py 提示训练文件不存在**

- 请先运行 `python scripts/preprocess.py --method all` 完成预处理
