# 肩关节 MRI 的 SFT 数据格式与构建方案

## 1. 目标

本文件用于定义肩关节 MRI 项目的 **结构化 SFT 数据格式**，并说明如何从当前 JSON 标签自动构建多种 SFT 训练样本。

当前建议的整体路线不是直接做自由报告生成，而是分层进行：

1. 标签与中间诊断步骤 SFT
2. structured findings / structured impression SFT
3. 最终自由报告 SFT

这样更容易稳定训练，也更适合后续做 reward 设计与 RL。

---

## 2. 当前可直接利用的原始字段

仓库中的 `data/json_parser.py` 已支持读取以下字段：

- `labels`
- `label_status`
- `evidence_text`
- `negative_evidence`
- `structured_findings`
- `structured_impression`
- `quality_flag`
- `exclude_from_main_training`
- `laterality`
- `raw_findings`
- `raw_impression`
- `source_summary`

这些字段已经足够支持第一版结构化 SFT 数据构建。

---

## 3. 推荐同时构建的四类 jsonl

### 3.1 `sft_label_binary.jsonl`

目标：只学习七病种标签。

### 3.2 `sft_diagnosis_chain.jsonl`

目标：学习病种、证据、主序列、key slice、ROI 等中间诊断链。

### 3.3 `sft_structured_findings.jsonl`

目标：学习结构化 findings。

### 3.4 `sft_structured_impression.jsonl`

目标：学习结构化 impression。

> 第一阶段不建议直接把自由报告作为主训练目标。

---

## 4. 统一主样本结构

建议每条样本至少保留以下字段：

```json
{
  "exam_id": "MR202511100183",
  "task_type": "diagnosis_chain",
  "instruction": "请根据肩关节MRI输出结构化诊断链。",
  "input_meta": {
    "laterality": "right",
    "quality_flag": "high",
    "source_summary": {
      "findings": "present",
      "impression": "present"
    }
  },
  "vision_inputs": {
    "sequence_order": [
      "axial_PD",
      "coronal_PD",
      "coronal_T2WI",
      "sagittal_PD",
      "sagittal_T1WI"
    ],
    "cls_cache_path": "outputs_clean/cache_cls/MR202511100183.pt",
    "loc_cache_path": "outputs_clean/cache_loc/MR202511100183.pt"
  },
  "_cache_note": "cache_loc .pt 中存储的是 image/mask/key_slices/roi_boxes/spatial_meta，**不存 labels**。labels 在运行时由 ShoulderCacheDataset 根据 metadata_master.csv + LabelMapper 动态生成。",
  "targets": {},
  "output_text": "...",
  "train_policy": {
    "use_for_label": true,
    "use_for_chain": true,
    "use_for_findings": true,
    "use_for_impression": true,
    "use_for_report": false
  },
  "sample_weight": {
    "label": 1.0,
    "chain": 1.0,
    "findings": 1.0,
    "impression": 1.0,
    "report": 0.0
  }
}
```

---

## 5. 四类任务的输出设计

## 5.1 标签任务

```json
{
  "task_type": "label_binary",
  "instruction": "请判断肩关节MRI中七个病种的状态，并以JSON输出。",
  "targets": {
    "SST": 1,
    "IST": 0,
    "SSC": 0,
    "LHBT": 0,
    "IGHL": 1,
    "RIPI": 1,
    "GHOA": 0
  },
  "output_text": "{\n  \"SST\": 1,\n  \"IST\": 0,\n  \"SSC\": 0,\n  \"LHBT\": 0,\n  \"IGHL\": 1,\n  \"RIPI\": 1,\n  \"GHOA\": 0\n}"
}
```

### 说明

- `1 -> positive`
- `0 -> negative`
- `2 -> uncertain` 在二分类任务里默认映射为 `0`
- `-1 -> mask`，不建议放入 label 主监督

---

## 5.2 诊断链任务

```json
{
  "task_type": "diagnosis_chain",
  "instruction": "请根据肩关节MRI输出结构化诊断链，包括病种、证据、主序列、关键切片和ROI。",
  "targets": {
    "SST": {
      "raw_label": 1,
      "train_label_binary": 1,
      "label_status": "explicit_positive",
      "positive_evidence": [
        "冈上肌腱远端PD压脂信号增高",
        "冈上肌腱远端腱病"
      ],
      "negative_evidence": [],
      "uncertain": false,
      "anchor_sequence": "coronal_PD",
      "key_slice": 11,
      "roi": [120, 80, 210, 170]
    }
  },
  "output_text": "{\n  \"SST\": {\n    \"label\": 1,\n    \"evidence\": [\"冈上肌腱远端PD压脂信号增高\", \"冈上肌腱远端腱病\"],\n    \"anchor_sequence\": \"coronal_PD\",\n    \"key_slice\": 11,\n    \"roi\": [120, 80, 210, 170]\n  }\n}"
}
```

### 说明

这是最推荐的 SFT 主任务，因为它最接近后续 RL 所需要的可验证中间步骤。

---

## 5.3 structured findings 任务

```json
{
  "task_type": "structured_findings",
  "instruction": "请根据肩关节MRI输出结构化 findings。",
  "targets": [
    "冈上肌腱远端腱病，PD压脂信号增高。",
    "盂肱下韧带增粗。",
    "肩袖间隙滑膜增生。"
  ],
  "output_text": "[\n  \"冈上肌腱远端腱病，PD压脂信号增高。\",\n  \"盂肱下韧带增粗。\",\n  \"肩袖间隙滑膜增生。\"\n]"
}
```

---

## 5.4 structured impression 任务

```json
{
  "task_type": "structured_impression",
  "instruction": "请根据肩关节MRI输出结构化 impression。",
  "targets": [
    "冈上肌腱损伤。",
    "IGHL异常。",
    "肩袖间隙异常。"
  ],
  "output_text": "[\n  \"冈上肌腱损伤。\",\n  \"IGHL异常。\",\n  \"肩袖间隙异常。\"\n]"
}
```

---

## 6. 训练策略字段建议

建议为每个样本增加：

### 6.1 `train_policy`

```json
{
  "train_policy": {
    "use_for_label": true,
    "use_for_chain": true,
    "use_for_findings": true,
    "use_for_impression": true,
    "use_for_report": false
  }
}
```

### 6.2 `sample_weight`

```json
{
  "sample_weight": {
    "label": 1.0,
    "chain": 1.0,
    "findings": 1.0,
    "impression": 1.0,
    "report": 0.0
  }
}
```

这些字段可由 `quality_flag`、laterality 冲突、source_summary 完整性等规则自动生成。

---

## 7. 数据分桶建议

### 桶 1，高置信主训练样本

满足：

- `quality_flag = high`
- `exclude_from_main_training = 0`
- laterality 无明显冲突
- structured 字段完整

用途：

- label
- diagnosis_chain
- structured_findings
- structured_impression
- 可选 report

### 桶 2，中置信结构化样本

满足：

- `quality_flag = medium`
- 标签和证据基本可靠
- 但最终自由报告不够稳定

用途：

- label
- diagnosis_chain
- structured_findings
- structured_impression

### 桶 3，不确定样本

满足：

- uncertain 较多
- laterality 可能有冲突
- 或只在 findings / impression 中部分可用

用途：

- diagnosis_chain
- 不建议直接作为 report 主监督

---

## 8. 自动构建脚本

仓库提供：

- `scripts/build_sft_jsonl.py`

该脚本默认会生成四类 jsonl：

- `sft_label_binary.jsonl`
- `sft_diagnosis_chain.jsonl`
- `sft_structured_findings.jsonl`
- `sft_structured_impression.jsonl`

### 典型命令

> **重要**：`outputs_clean/` 下的 cache/metadata 已在构建时完成 clean 过滤
> （quality_flag != low, postoperative != 1, exclude_from_main_training != 1, raw_labels 仅含 0/1）。
> 因此 `build_sft_jsonl.py` 的 `--metadata_csv` 应指向 `outputs_clean/metadata/`，
> 而不是原始 `outputs/metadata/`，这样过滤已在上游完成，此处无需重复处理。

```bash
python scripts/build_sft_jsonl.py \
  --output_dir outputs_clean/sft_data \
  --metadata_csv outputs_clean/metadata/metadata_master.csv \
  --min_quality medium
```

如果已构建过 `cache_loc_index.csv`，还可以加：

```bash
python scripts/build_sft_jsonl.py \
  --output_dir outputs_clean/sft_data \
  --metadata_csv outputs_clean/metadata/metadata_master.csv \
  --loc_index outputs_clean/cache_loc/cache_loc_index.csv \
  --min_quality medium
```

---

## 9. 推荐的 SFT 训练顺序

### Stage 1

先用：

- `sft_label_binary.jsonl`
- `sft_diagnosis_chain.jsonl`

### Stage 2

再加入：

- `sft_structured_findings.jsonl`
- `sft_structured_impression.jsonl`

### Stage 3

最后再考虑自由报告 SFT。

---

## 10. 一句话结论

第一版不要直接追求自由报告，而是优先把 **标签 + 诊断链 + structured findings / impression** 做成稳定可训练的 SFT 数据，这样最适合当前肩关节 MRI 项目的多阶段路线。