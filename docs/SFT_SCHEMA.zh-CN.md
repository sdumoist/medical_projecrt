# SFT 样本格式与数据规范

## 概述

本文档定义 SFT 训练数据的样本格式、4 种任务类型、质量过滤策略和字段说明。

---

## 1. 通用样本格式

每条 JSONL 样本的完整结构：

```json
{
  "exam_id": "MR202212250010",
  "task_type": "label_binary",
  "quality_bucket": "A",
  "instruction": "[system] 你是一位专业的...\n[user] 请根据...",
  "output": "{\"labels\": {...}}",
  "train_policy": {
    "use_for_label": true,
    "use_for_chain": true,
    "use_for_findings": true,
    "use_for_impression": true
  },
  "metadata": {
    "quality_flag": "high",
    "laterality": "right",
    "sex": "女",
    "age": "52岁",
    "num_valid_labels": 7,
    "has_evidence": true,
    "has_key_slice": true,
    "has_roi": false,
    "source_summary": {"SST": "both", "IST": "findings", ...}
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| exam_id | string | 检查唯一标识 |
| task_type | string | 任务类型，见下文 |
| quality_bucket | string | 质量分桶标记 A/B/C |
| instruction | string | 指令文本（训练时由 tokenizer 渲染为 ChatML） |
| output | string | 目标输出（JSON 字符串） |
| train_policy | object | 按任务的可用性标记 |
| metadata | object | 样本元信息 |

---

## 2. 四种任务类型

### 2.1 `label_binary` — 七病种标签

**output 格式**：
```json
{
  "labels": {
    "SST": {"label": 1, "status": "explicit_positive"},
    "IST": {"label": 0, "status": "explicit_negative"},
    "SSC": {"label": 0, "status": "implicit_negative"},
    "LHBT": {"label": 0, "status": "explicit_negative"},
    "IGHL": {"label": 0, "status": "explicit_negative"},
    "RIPI": {"label": 0, "status": "implicit_negative"},
    "GHOA": {"label": 0, "status": "explicit_negative"}
  }
}
```

**label 值**：1=阳性, 0=阴性, -1=无法判断

**status 值**：explicit_positive, explicit_negative, implicit_negative, uncertain, postop_unmappable

### 2.2 `diagnosis_chain` — 结构化诊断链

**output 格式**：
```json
{
  "labels": {"SST": 1, "IST": 0, ...},
  "evidence": {
    "SST": {
      "positive": ["冈上肌腱远侧段增粗，信号增高"],
      "negative": []
    },
    "IST": {
      "positive": [],
      "negative": ["冈下肌腱连续"]
    }
  },
  "anchor_sequence": {
    "SST": "coronal_PD",
    "IST": "axial_PD",
    "SSC": "axial_PD",
    "LHBT": "coronal_PD",
    "IGHL": "coronal_PD",
    "RIPI": "sagittal_PD",
    "GHOA": "coronal_PD"
  },
  "key_slice": {"SST": 12, "IST": null, ...},
  "roi_box": {"SST": [3, 45, 120, 8, 89, 210], "IST": null, ...}
}
```

**key_slice**：整数索引（来自 cache_loc），无数据时为 null

**roi_box**：`[z_min, h_min, w_min, z_max, h_max, w_max]`，无数据时为 null

### 2.3 `structured_findings` — 结构化所见

**output 格式**：
```json
{
  "structured_findings": [
    "冈上肌腱远侧段增粗，信号增高",
    "冈上肌腱远侧段实质内撕裂伴肌腱病"
  ]
}
```

### 2.4 `structured_impression` — 结构化印象

**output 格式**：
```json
{
  "structured_impression": [
    "右侧肩峰下缘有骨刺形成；喙肩韧带增厚",
    "冈上肌腱远侧段实质内撕裂伴肌腱病"
  ]
}
```

---

## 3. 质量过滤策略

### 3.1 按任务过滤（主逻辑）

每个样本的可用性由 `train_policy` 控制，不做统一的桶级排除。

| 任务 | 准入条件 |
|------|----------|
| label_binary | ≥5 个病种 label ∈ {0,1,2} + exclude==0 + postop==0 |
| diagnosis_chain | label 条件同上 + evidence_text 至少 1 个病种非空 |
| structured_findings | structured_findings 非空 + quality_flag ≠ "low" |
| structured_impression | structured_impression 非空 + quality_flag ≠ "low" |

### 3.2 质量分桶标记

分桶仅用于分析和可选采样权重，不硬性排除：

| 桶 | 条件 | 建议采样权重 |
|----|------|-------------|
| A | quality_flag=="high" + exclude==0 | 1.0 |
| B | quality_flag=="medium" + exclude==0 | 0.7 |
| C | 其余非 low | 0.3 |

### 3.3 全局排除条件

以下样本**不生成任何任务的 JSONL**：
- `quality_flag == "low"`
- `postoperative == 1`（术后样本，标签不可靠）

---

## 4. 数据来源

| 来源 | 路径 | 提供字段 |
|------|------|----------|
| case_json | `{JSON_ROOT}/{exam_id}.json` | labels, label_status, evidence_text, negative_evidence, structured_findings, structured_impression, quality_flag, laterality, sex, age, source_summary |
| cache_loc_index | `outputs/cache_loc/cache_loc_index.csv` | exam_id → cache 路径映射 |
| cache_loc .pt | `outputs/cache_loc/{exam_id}.pt` | key_slices [7], roi_boxes [7,6] |
| DISEASE_ANCHOR_SEQ | `utils/io.py` | 病种 → 锚定序列映射（固定规则） |

---

## 5. 数据划分

- 复用 `create_train_val_split(seed=42, val_ratio=0.2)` 对齐现有 CV 训练集
- CV val_ids 进一步 50/50 → SFT val + SFT test
- 按 exam_id 级别划分，同一 exam 不跨集
- 输出 `split_meta.json` 记录各集 exam_id 列表

---

## 6. 输出目录结构

```
outputs/sft_data/
├── sft_label_binary.jsonl
├── sft_diagnosis_chain.jsonl
├── sft_structured_findings.jsonl
├── sft_structured_impression.jsonl
├── sft_summary.json
└── split/
    ├── sft_label_binary_train.jsonl
    ├── sft_label_binary_val.jsonl
    ├── sft_label_binary_test.jsonl
    ├── sft_diagnosis_chain_train.jsonl
    ├── ...
    └── split_meta.json
```

---

## 7. sft_summary.json 内容

```json
{
  "total_cases_scanned": 7870,
  "excluded": {"low_quality": 6, "postoperative": 120},
  "per_task": {
    "label_binary": {"count": 7500, "bucket_A": 6800, "bucket_B": 600, "bucket_C": 100},
    "diagnosis_chain": {"count": 7200, ...},
    "structured_findings": {"count": 7400, ...},
    "structured_impression": {"count": 7300, ...}
  },
  "label_distribution": {
    "SST": {"positive": 5800, "negative": 1500, "uncertain": 200, "masked": 0},
    ...
  },
  "field_completeness": {
    "evidence_text_nonempty_rate": 0.85,
    "key_slice_available_rate": 0.60,
    "roi_box_available_rate": 0.55,
    ...
  }
}
```
