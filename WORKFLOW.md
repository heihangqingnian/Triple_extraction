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
    ├── train_raw.jsonl     # Prompt 无关的规范化样本（每行 {text, triples}）
    ├── dev_raw.jsonl       # 验证集（供免训练 Prompt 寻优，绝不用 test）
    └── test_raw.jsonl      # 测试集（仅最终评估时由 build_llm_dataset.py 派生 test.json）
```

> LLM 的 Prompt 不在预处理阶段固化。预处理只产出与 Prompt 无关的 `*_raw.jsonl`，
> 之后由 `scripts/prompt_search.py`（免训练寻优）与 `scripts/build_llm_dataset.py`
> （选定最优 Prompt 后构造 `train.json` / `test.json`）分别消费。

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

> **设计变更**：不再为每个 Prompt 各训一套 LoRA（LoRA 的「肌肉记忆」会掩盖 Prompt
> 指令设计本身的优劣）。改为**免训练寻优**先选出最优 Prompt，再**只用这一个 Prompt**
> 微调出**单一 LoRA**。全程严格隔离 Train/Val/Test，免训练寻优**绝不触碰测试集**。

四个阶段：

| 阶段             | 脚本 / 命令                                   | 是否需要模型 | 数据集     |
| ---------------- | --------------------------------------------- | ------------ | ---------- |
| ① 免训练 Prompt 寻优 | `scripts/prompt_search.py`                | 仅基座，无 LoRA | dev（或 train） |
| ② 构造微调数据   | `scripts/build_llm_dataset.py --prompt <best>` | 否           | train + test |
| ③ LoRA 微调      | 外部 LLaMA-Factory                            | 是           | train.json |
| ④ 最终测试评估   | `main.py --method llm`                        | 基座 + 单 LoRA | **test** |

候选 Prompt 变体（指令文本集中在 `methods/llm/prompt_templates.py`，三处共用）：

| 变体   | Prompt 策略                   |
| ------ | ----------------------------- |
| base   | 基础指令，无约束              |
| schema | 加入 Schema 实体/关系类型约束 |
| cot    | Chain-of-Thought 分步推理     |

### 5.1 免训练 Prompt 寻优（不加载任何 LoRA）

```bash
# 用基座模型在验证集小样本（默认 150 条）上对比各 Prompt 变体的三元组 F1
python scripts/prompt_search.py --config configs/llm_prompt_search.yaml
```

- 配置 `configs/llm_prompt_search.yaml`：`model.lora_weights: null`（纯基座），
  `search.source_split` 仅允许 `dev` / `train`（指向 `test` 会被代码直接拒绝，杜绝泄露）。
- 可在配置里开启 Few-shot（`search.few_shot.enabled: true`），示例从 `train_raw` 采样且排除评估样本。
- 产出 `results/llm/prompt_search/ranking.json` 与终端排名表，并打印 **★ 最优 Prompt**。

### 5.2 构造微调 / 测试数据（选定最优 Prompt 后）

```bash
# 假设寻优选出 schema 最优
python scripts/build_llm_dataset.py --prompt schema --split all
```

产出（`data/processed/llm/`）：

- `train.json` — LLaMA-Factory 微调数据；**固定指令放入 `system` 字段**，每条不重复
- `test.json`  — 最终测试推理用（格式与训练完全一致）
- `dataset_info.json` — LLaMA-Factory 数据集注册片段（映射 system/instruction/output，模板 `chatglm2`）

> **关键一致性**：训练（LLaMA-Factory `chatglm2` 模板 + system）与推理
> （`methods/llm/infer.py` 复现同一模板 + system）的提示串必须逐字一致，否则 LoRA 失配掉点。
> 二者共用 `prompt_templates.py` 的同一份定义；如需核对，可用
> `python -c "from methods.llm.prompt_templates import assemble_preview; print(assemble_preview('schema','示例文本'))"`
> 打印逻辑提示串，对照你的 LLaMA-Factory 版本确认。

### 5.3 LoRA 微调（外部 LLaMA-Factory）

`build_llm_dataset.py` 运行结束会打印对齐好的训练命令，核心如下（务必 `--template chatglm2`）：

```bash
llamafactory-cli train \
    --stage sft --do_train \
    --model_name_or_path ZhipuAI/chatglm2-6b \
    --finetuning_type lora --template chatglm2 \
    --dataset duie_lora_train --dataset_dir data/processed/llm \
    --output_dir models/chatglm2-lora \
    --num_train_epochs 3 --per_device_train_batch_size 4 \
    --learning_rate 2e-4 --lora_rank 8 --cutoff_len 1024
```

若 `models/chatglm2-lora` 不存在，推理会自动回退到基座模型（日志中提示 Warning）。

### 5.4 最终测试推理与评估

```bash
# 基座 + 单 LoRA，在测试集上推理
python main.py --method llm --mode predict

# 评估：ComprehensiveMetrics（与 Pipeline/Joint 完全一致）
python main.py --method llm --mode evaluate

# 评估指定预测文件
python main.py --method llm --mode evaluate --input results/llm/predictions.jsonl
```

产出（`results/llm/`，文件名与 Pipeline/Joint 一致）：

```
results/llm/
├── predictions.jsonl   — 推理结果（text/predict/label/parse_error）
├── metrics.json        — 严格/宽松 × 微平均/宏平均 + 逐关系 + 错误统计
├── error_report.txt    — 错误类型汇总
├── error_cases.txt     — 错误案例详情
└── per_relation.txt    — 逐关系 P/R/F1
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
    --llm      results/llm/predictions.jsonl
```

输出示例：

```
子集       样本数   Pipeline     Joint       LLM
Normal       2850    0.7821    0.8134    0.6890
SEO           680    0.6910    0.7230    0.5830
EPO           210    0.5820    0.6410    0.5010
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
    --llm      results/llm/predictions.jsonl
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

# 仅生成 LLM 采样数据（用最优 Prompt，例如 schema）
python scripts/low_resource_eval.py --methods llm --data_only --prompt schema
```

采样数据目录：

```
data/processed/low_resource/
├── 1pct/
│   ├── pipeline/train/, dev/, test/
│   ├── joint/train/, dev/, test/
│   └── llm/train.json          # 单 Prompt 的 Alpaca 微调数据（system 字段承载指令）
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
│   ├── llm.yaml                # LLM 最终测试配置（单 LoRA）
│   ├── llm_prompt_search.yaml  # 免训练 Prompt 寻优配置（无 LoRA）
│   └── ablation_*.yaml
│
├── methods/
│   ├── pipeline/               # NER（BERT-BiLSTM-CRF）→ RE（BERT 分类器）
│   ├── joint/                  # CasRel 端到端联合抽取
│   └── llm/
│       ├── prompt_templates.py # 所有 Prompt 指令的单一事实来源（三处共用）
│       ├── prompt_search.py    # 免训练 Prompt 寻优核心逻辑（仅基座，禁用 test）
│       ├── infer.py            # 推理：基座零样本 / 基座+单 LoRA
│       └── evaluator.py        # 最终测试评估（ComprehensiveMetrics，与基线一致）
│
├── utils/
│   ├── common.py               # 随机种子、日志、YAML 加载
│   ├── io_utils.py             # JSON/JSONL 读写、模型保存
│   └── metrics.py              # P/R/F1、错误分析、parse_triple_string
│
├── scripts/
│   ├── preprocess.py           # 数据预处理（DuIE2.0 → pipeline/joint/llm-raw）
│   ├── prompt_search.py        # 免训练 Prompt 寻优入口
│   ├── build_llm_dataset.py    # 选定最优 Prompt 后构造 train.json/test.json
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
│   └── chatglm2-lora/          # 单一 LoRA 权重（LLaMA-Factory 用最优 Prompt 训练产出）
│
└── results/                    # 实验结果（自动生成）
    ├── pipeline/
    ├── joint/
    ├── llm/
    │   ├── prompt_search/      # 免训练寻优结果（ranking.json + 各变体预测）
    │   ├── predictions.jsonl   # 最终测试预测
    │   └── metrics.json        # 最终测试指标
    └── low_resource/
```

---

## 常见问题

**Q: CUDA 内存不足（OOM）**

- 减小 `batch_size`（configs/joint.yaml 或 pipeline.yaml）
- LLM 推理时设置 `device: cpu`（速度慢但无 OOM）

**Q: LoRA 权重不存在**

- 推理会自动回退到基础模型，观察日志中 `Warning: LoRA 权重目录不存在`
- 基础模型推理效果通常低于微调后的模型

**Q: 免训练寻优会不会用到测试集？**

- 不会。`configs/llm_prompt_search.yaml` 的 `search.source_split` 仅允许 `dev` / `train`，
  指向 `test` 或文件名疑似测试集时，`methods/llm/prompt_search.py` 会直接报错退出。

**Q: 训练和推理的 Prompt 怎么保证一致？**

- 指令文本集中在 `methods/llm/prompt_templates.py`，免训练寻优、最终推理、
  `dataset_info.json` 三处共用同一份。微调务必用 `--template chatglm2`，
  并用 `assemble_preview` 对照核对（见 5.2）。

**Q: fine_grained_eval.py 或 density_eval.py 报"未找到预测文件"**

- 请先运行各方法的 evaluate 步骤生成 `predictions.jsonl`
- LLM 需要先运行 `--mode predict` 再运行 `--mode evaluate`，预测文件为 `results/llm/predictions.jsonl`

**Q: low_resource_eval.py 提示训练文件不存在**

- 请先运行 `python scripts/preprocess.py --input data/raw/DuIE2.0 --method all` 完成预处理
