# 三元组抽取范式对比研究

本仓库系统对比三种关系三元组抽取范式，并配套**消融实验**与 **LLM Prompt 探索实验**，基于 **DuIE2.0** 中文数据集（48 类关系）。

| 方法     | 模型                          | 核心思路             |
| -------- | ----------------------------- | -------------------- |
| Pipeline | BERT-BiLSTM-CRF + BERT 分类器 | 先 NER 后 RE，两阶段 |
| Joint    | CasRel（级联二元标注）        | 端到端联合抽取       |
| LLM      | ChatGLM2-6B + LoRA            | 指令微调生成式抽取   |

---

## 实验总览

仓库共包含 **3 个主实验 + 5 个消融实验 + 2 个 LLM 探索实验**，全部输出到 `results/`：

### 主实验

| 结果目录           | 配置                    | 说明                                                     |
| ------------------ | ----------------------- | -------------------------------------------------------- |
| `results/pipeline` | `configs/pipeline.yaml` | Pipeline 主模型（BERT-BiLSTM-CRF → BERT RE）             |
| `results/joint`    | `configs/joint.yaml`    | Joint 主模型（CasRel，主语反馈 + 共享编码器 + 等权损失） |
| `results/llm`      | `configs/llm.yaml`      | LLM 主模型（最优 Prompt + 单 LoRA 微调，最终测试）       |

### 消融实验

| 结果目录                     | 配置                                             | 改动 / 对照                                        |
| ---------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| `results/joint_ablation1`    | `configs/ablation_joint_1_no_feedback.yaml`      | **移除主实体反馈**（`use_subject_feedback=false`） |
| `results/joint_ablation3`    | `configs/ablation_joint_3_weights_entity.yaml`   | **损失权重 2:1 侧重实体**（α=1.333, β=0.667）      |
| `results/joint_ablation4`    | `configs/ablation_joint_3_weights_relation.yaml` | **损失权重 1:2 侧重关系**（α=0.667, β=1.333）      |
| `results/pipeline_ablation1` | `configs/ablation_pipeline_1_no_bilstm.yaml`     | **去 BiLSTM**（NER 退化为 BERT-CRF）               |
| `results/pipeline_ablation2` | `configs/ablation_pipeline_2_no_crf.yaml`        | **去 CRF**（NER 退化为 BERT-Linear+Softmax）       |

### LLM 探索实验（免训练 Prompt 寻优）

| 结果目录                            | 配置                                     | 说明                                           |
| ----------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| `results/llm/prompt_search`         | `configs/llm_prompt_search.yaml`         | **零样本**对比 base / schema / cot 三种 Prompt |
| `results/llm/prompt_search_fewshot` | `configs/llm_prompt_search_fewshot.yaml` | **2-shot**（k=2）对比 base / schema / cot      |

> 寻优只在 **dev 集 150 条采样**上进行，仅用基座模型（不挂 LoRA），用于挑选最终微调所用的最优 Prompt。

---

## 目录结构

```
Research1/
├── configs/                          # 超参数配置
│   ├── pipeline.yaml / joint.yaml / llm.yaml          # 三个主实验
│   ├── ablation_joint_1_no_feedback.yaml              # Joint 消融
│   ├── ablation_joint_2_dual_encoder.yaml
│   ├── ablation_joint_3_weights_entity.yaml
│   ├── ablation_joint_3_weights_relation.yaml
│   ├── ablation_pipeline_1_no_bilstm.yaml             # Pipeline 消融
│   ├── ablation_pipeline_2_no_crf.yaml
│   ├── llm_prompt_search.yaml                         # LLM 探索
│   └── llm_prompt_search_fewshot.yaml
├── data/
│   ├── raw/                          # 原始 DuIE2.0 数据
│   └── processed/                    # 预处理后数据
├── methods/
│   ├── pipeline/                     # Pipeline 方法实现（NER + RE）
│   ├── joint/                        # Joint 方法实现（CasRel）
│   └── llm/                          # LLM 方法（推理 + 评估）
├── utils/                            # 三种方法共用工具库
│   ├── common.py                     # 随机种子、日志、路径
│   ├── metrics.py                    # 统一 F1 计算（ComprehensiveMetrics）
│   └── io_utils.py                   # 数据读写、模型保存加载
├── scripts/
│   ├── preprocess.py                 # 数据预处理一键脚本
│   ├── download_model.py             # 模型下载脚本
│   ├── prompt_search.py              # LLM 免训练 Prompt 寻优
│   └── build_llm_dataset.py          # 按最优 Prompt 构造 LLM 数据集
├── draw/                             # 实验结果可视化（图表 + LaTeX 表）
├── results/                          # 统一输出目录（不纳入版本控制）
├── main.py                           # 唯一入口
└── requirements.txt
```

---

## 环境安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 模型下载

### Pipeline / Joint 方法所需 BERT 模型

```bash
# 从 HuggingFace 下载（hfl/chinese-bert-wwm）
python scripts/download_model.py --model hfl/chinese-bert-wwm --save_dir methods/pipeline/ner/model
```

> BERT 模型在 Pipeline NER、Pipeline RE、Joint CasRel 中共用，三处均指向同一本地路径。
> 若不下载到本地，脚本会自动从 HuggingFace 联网拉取 `hfl/chinese-bert-wwm`。

### LLM 方法所需 ChatGLM2-6B 模型

```bash
# 下载基础模型
python scripts/download_model.py --model THUDM/chatglm2-6b --save_dir models/chatglm2-6b --source modelscope

# LoRA 权重由 LLaMA-Factory 训练生成（--template chatglm2）
```

---

## 数据集准备

### 下载 DuIE2.0

从 [gitee](https://gitee.com/open-datasets/DuIE2.0) 下载 DuIE2.0 数据集，解压后放置到：

```
data/raw/DuIE2.0/
├── duie_train.json       # 训练集（每行一个 JSON 对象）
├── duie_dev.json         # 验证集
└── duie_schema/
    └── duie_schema.json  # 关系 Schema 定义
```

每行数据格式：

```json
{
  "text": "原文",
  "spo_list": [
    { "subject": "主体", "predicate": "关系", "object": { "@value": "客体" } }
  ]
}
```

### 数据预处理（一键转换）

```bash
# 转换所有方法所需格式
python scripts/preprocess.py --input data/raw/DuIE2.0 --method all

# 只转换某种方法的格式
python scripts/preprocess.py --input data/raw/DuIE2.0 --method pipeline
python scripts/preprocess.py --input data/raw/DuIE2.0 --method joint
python scripts/preprocess.py --input data/raw/DuIE2.0 --method llm
```

生成文件说明：

- `data/processed/shared/rel2id.json` — 关系映射（三种方法共用）
- `data/processed/pipeline/train/train_ner.txt` — BIO 格式 NER 数据
- `data/processed/pipeline/train/train_re.txt` — 实体对关系分类数据
- `data/processed/joint/train/train.json` — CasRel 格式数据
- `data/processed/llm/{train,dev,test}_raw.jsonl` — Prompt 无关的 LLM 规范化样本（{text, triples}）

---

## 快速复现

### 主实验

#### Pipeline 方法

```bash
# 1. 训练（NER → RE 顺序训练）
python main.py --method pipeline --mode train
python main.py --method pipeline --mode train --component ner   # 单独训练 NER
python main.py --method pipeline --mode train --component re    # 单独训练 RE

# 2. 端到端评估
python main.py --method pipeline --mode evaluate

# 3. 对自定义输入预测
python main.py --method pipeline --mode predict --input data/processed/pipeline/test/pipeline_test.jsonl
```

#### Joint 方法（CasRel）

```bash
python main.py --method joint --mode train       # 训练
python main.py --method joint --mode evaluate     # 评估
python main.py --method joint --mode predict      # 预测
```

#### LLM 方法（ChatGLM2-6B + 单 LoRA）

四步流程（详见 [WORKFLOW.md](WORKFLOW.md) 第 5 节）：

```bash
# 1. 免训练 Prompt 寻优（仅基座模型，绝不用测试集）→ 选出最优 Prompt
python scripts/prompt_search.py --config configs/llm_prompt_search.yaml

# 2. 用最优 Prompt 构造微调/测试数据（固定指令放入 system 字段）
python scripts/build_llm_dataset.py --prompt <best> --split all

# 3. 在 LLaMA-Factory 中用 data/processed/llm/train.json 微调出单一 LoRA（--template chatglm2）

# 4. 最终测试推理与评估（与 Pipeline/Joint 同一测试集、同一 ComprehensiveMetrics 指标）
python main.py --method llm --mode predict
python main.py --method llm --mode evaluate
```

### 消融实验

消融实验通过 `--config` 指定对应 YAML 复用主流程，结果按各配置的 `output.dir` 自动落盘。

#### Pipeline 消融（NER 结构）

```bash
# 去 BiLSTM（BERT-CRF）
python main.py --method pipeline --mode train    --config configs/ablation_pipeline_1_no_bilstm.yaml
python main.py --method pipeline --mode evaluate --config configs/ablation_pipeline_1_no_bilstm.yaml

# 去 CRF（BERT-Linear + Softmax）
python main.py --method pipeline --mode train    --config configs/ablation_pipeline_2_no_crf.yaml
python main.py --method pipeline --mode evaluate --config configs/ablation_pipeline_2_no_crf.yaml
```

#### Joint 消融（机制 / 损失权重）

```bash
# 移除主实体反馈
python main.py --method joint --mode train    --config configs/ablation_joint_1_no_feedback.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_1_no_feedback.yaml

# 损失权重 2:1 侧重实体（α=1.333, β=0.667）
python main.py --method joint --mode train    --config configs/ablation_joint_3_weights_entity.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_3_weights_entity.yaml

# 损失权重 1:2 侧重关系（α=0.667, β=1.333）
python main.py --method joint --mode train    --config configs/ablation_joint_3_weights_relation.yaml
python main.py --method joint --mode evaluate --config configs/ablation_joint_3_weights_relation.yaml
```

### LLM 探索实验（Prompt 寻优）

```bash
# 零样本：对比 base / schema / cot
python scripts/prompt_search.py --config configs/llm_prompt_search.yaml

# 2-shot（k=2）：对比 base / schema / cot
python scripts/prompt_search.py --config configs/llm_prompt_search_fewshot.yaml
```

每次寻优输出 `predictions_{base,cot,schema}.jsonl` 与 `ranking.json`（含各 Prompt 的 P/R/F1 与 `best_prompt`）。

### 对比三种方法

```bash
# 所有方法评估完成后，输出对比表格
python main.py --method all --mode evaluate
```

---

## 实验结果

> 测试集 strict-micro / relaxed-micro 指标，由统一的 `utils.metrics.ComprehensiveMetrics` 计算。完整数据见 `results/*/metrics.json`。

### 三范式主对比

| 方法     | Strict P | Strict R | **Strict F1** | Relaxed F1 |
| -------- | -------- | -------- | ------------- | ---------- |
| Pipeline | 0.592    | 0.605    | 0.598         | 0.633      |
| Joint    | 0.701    | 0.677    | 0.689         | 0.725      |
| **LLM**  | 0.749    | 0.673    | **0.709**     | **0.749**  |

### Joint 消融

| 模型                 | 设置                     | Strict F1 | ΔF1    |
| -------------------- | ------------------------ | --------- | ------ |
| Joint (CasRel)       | baseline（α=1.0, β=1.0） | 0.689     | --     |
| w/o Subject-Feedback | 移除主实体反馈           | 0.644     | −0.045 |
| Loss 2:1 (Entity)    | α=1.333, β=0.667         | 0.695     | +0.006 |
| Loss 1:2 (Relation)  | α=0.667, β=1.333         | **0.706** | +0.018 |

> 主实体反馈是 CasRel 级联机制的核心，移除后 F1 下降最明显；适度提升关系（object）损失权重带来小幅增益。

### Pipeline 消融

| 模型       | 设置                        | Strict F1 | ΔF1    |
| ---------- | --------------------------- | --------- | ------ |
| Pipeline   | baseline（BERT-BiLSTM-CRF） | 0.598     | --     |
| w/o BiLSTM | BERT-CRF                    | 0.590     | −0.009 |
| w/o CRF    | BERT-Linear + Softmax       | 0.584     | −0.014 |

> BiLSTM 与 CRF 对 NER 均有正向贡献，去 CRF 影响略大于去 BiLSTM。

### LLM Prompt 探索（dev 150 条，仅基座模型）

| Prompt | 零样本 F1 | 2-shot F1 |
| ------ | --------- | --------- |
| base   | 0.027     | 0.107     |
| schema | **0.063** | 0.144     |
| cot    | 0.047     | **0.146** |

> 零样本最优为 **schema**，2-shot 最优为 **cot**；few-shot 显著优于 zero-shot。最优 Prompt 用于最终单 LoRA 微调，测试集 strict-F1 达 **0.709**。

---

## 超参数修改

所有超参数集中在 `configs/` 目录下，修改对应 YAML 文件即可：

```yaml
# configs/pipeline.yaml（示例）
seed: 42
ner:
  learning_rate: 3.0e-5
  batch_size: 16
  epochs: 15
```

消融实验只需在对应 `ablation_*.yaml` 中调整开关字段（如 `use_subject_feedback`、`use_bilstm`、`use_crf`、`loss_alpha`/`loss_beta`）。

---

## 输出文件说明

所有结果统一输出到 `results/` 目录，每个 run 目录结构一致：

```
results/<run>/
├── metrics.json          # strict/relaxed micro+macro P/R/F1、错误统计、各关系指标、推理速度、模型规模
├── predictions.jsonl     # 预测结果（每行：text + pred_triples + gold_triples）
├── error_report.txt      # 错误类型分布汇总
├── error_cases.txt       # 错误样例
├── per_relation.txt      # 各关系 P/R/F1
└── train.log             # 训练日志（含每轮 dev F1，用于学习曲线）

results/llm/prompt_search[_fewshot]/
├── predictions_{base,cot,schema}.jsonl
├── ranking.json          # 各 Prompt 的 P/R/F1 与 best_prompt
└── search.log
```

---
