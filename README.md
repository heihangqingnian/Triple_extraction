# 三元组抽取范式对比研究

本仓库对比三种关系三元组抽取范式，基于 **DuIE2.0** 数据集：

| 方法     | 模型                          | 核心思路             |
| -------- | ----------------------------- | -------------------- |
| Pipeline | Bert-BiLSTM-CRF + Bert 分类器 | 先 NER 后 RE，两阶段 |
| Joint    | CasRel                        | 端到端联合抽取       |
| LLM      | ChatGLM2-6B + LoRA            | 指令微调生成式抽取   |

---

## 目录结构

```
Research2/
├── configs/               # 超参数配置
│   ├── pipeline.yaml
│   ├── joint.yaml
│   └── llm.yaml
├── data/
│   ├── raw/               # 原始 DuIE2.0 数据
│   └── processed/         # 预处理后数据
├── methods/
│   ├── pipeline/          # Pipeline 方法实现（NER + RE）
│   ├── joint/             # Joint 方法实现（CasRel）
│   └── llm/               # LLM 方法（推理 + 评估）
├── utils/                 # 三种方法共用工具库
│   ├── common.py          # 随机种子、日志、路径
│   ├── metrics.py         # 统一 F1 计算
│   └── io_utils.py        # 数据读写、模型保存加载
├── scripts/
│   ├── preprocess.py      # 数据预处理一键脚本
│   └── download_model.py  # 模型下载脚本
├── results/               # 统一输出目录（不纳入版本控制）
├── main.py                # 唯一入口
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

# LoRA 权重由 LlamaFactory 训练生成，此仓库里未提供方法说明
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
- `data/processed/llm/train.json` — Alpaca 格式 LLM 训练数据

---

## 快速复现

### Pipeline 方法

```bash
# 1. 训练（NER → RE 顺序训练）
python main.py --method pipeline --mode train

# 2. 端到端评估
python main.py --method pipeline --mode evaluate

# 3. 对自定义输入预测
python main.py --method pipeline --mode predict --input data/processed/pipeline/test/pipeline_test.jsonl
```

### Joint 方法（CasRel）

```bash
# 1. 训练
python main.py --method joint --mode train

# 2. 评估
python main.py --method joint --mode evaluate

# 3. 预测
python main.py --method joint --mode predict
```

### LLM 方法（ChatGLM2-6B + LoRA）

**训练说明：** LLM 微调通过 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 完成，本仓库不含训练代码。  
训练数据已生成于 `data/processed/llm/train.json`（Alpaca 格式），可直接用于 LlamaFactory。

```bash
# 推理（需要模型权重）
python main.py --method llm --mode predict

# 评估（使用已有预测结果，无需重新推理）
python main.py --method llm --mode evaluate --input results/llm/predictions.jsonl
```

### 对比三种方法

```bash
# 所有方法评估完成后，输出对比表格
python main.py --method all --mode evaluate
```

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

---

## 输出文件说明

所有结果统一输出到 `results/` 目录：

```
results/
├── pipeline/
│   ├── metrics.json          # F1 指标
│   ├── predictions.jsonl     # 预测结果（每行：text + pred_triples + gold_triples）
│   └── error_report.txt      # 错误类型分析
├── joint/
│   └── ...（同上）
└── llm/
    └── ...（同上）
```

---

## 可复现性保证

- **随机种子**：所有实验统一使用 `seed=42`（在 YAML 配置文件中设置），由 `utils.common.set_seed()` 负责设置 Python / NumPy / PyTorch / CUDA 的随机状态。
- **超参数外置**：所有超参数在 `configs/*.yaml` 中声明，代码中无硬编码。
- **数据处理复现**：`scripts/preprocess.py --input data/raw/DuIE2.0 --method all` 可完整重现预处理结果。

---
