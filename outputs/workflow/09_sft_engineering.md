# SFT 第一阶段工程实现 + 冒烟验证 (2026-04-23)

## 概述

完成结构化 SFT 全管线工程实现：数据生成 → 划分 → 训练 → 评估。通过代码审查修复 3 个关键问题，并在集群上完成 3 种 encoder 的冒烟测试。

---

## 一、新增文件（15 个）

### sft/ 模块

| 文件 | 说明 |
|---|---|
| `sft/__init__.py` | 包初始化 |
| `sft/prompts.py` | 4 类任务 instruction 模板，`build_prompt()` 用 `tokenizer.apply_chat_template()` |
| `sft/dataset.py` | JSONL + MRI cache 多任务数据集，`sft_collate_fn` 含 visual token 占位 |
| `sft/losses.py` | SFT loss（当前 LM-only，aux heads 占位） |
| `sft/eval_utils.py` | JSON 解析 + fuzzy matching + 4 任务评估指标 + 字段级成功率 |
| `sft/train_sft.py` | 训练主脚本：VisualProjector + ShoulderSFTModel + freeze strategy |

### 配置文件 (configs/)

| 文件 | 说明 |
|---|---|
| `sft_stage1_frozen.yaml` | Stage 1 冻结版（swin3d_tiny, lr=2e-4, 5 epochs） |
| `sft_stage2_partial_unfreeze.yaml` | Stage 2 部分解冻（lr=5e-5, LM-only） |
| `sft_stage3_full_ft.yaml` | Stage 3 占位（lr=1e-5） |
| `sft_stage1_smoke.yaml` | 冒烟测试配置（swin3d_tiny, bs=1, 2 epochs） |
| `sft_smoke_resnet50.yaml` | 冒烟测试配置（resnet50） |
| `sft_smoke_densenet201.yaml` | 冒烟测试配置（densenet201） |

### 脚本 (scripts/)

| 文件 | 说明 |
|---|---|
| `build_sft_jsonl.py` | 从 case_json + cache_loc 生成 4 类 JSONL |
| `inspect_sft_jsonl.py` | 统计样本数、字段完整率、标签分布 |
| `split_sft_data.py` | exam_id 级 train/val/test 划分 |
| `extract_branch_features.py` | 预提取 branch features 为 .npz（可选加速） |

### 文档 (docs/)

| 文件 | 说明 |
|---|---|
| `SFT_PLAN.zh-CN.md` | 三阶段训练路线文档 |
| `SFT_SCHEMA.zh-CN.md` | SFT 样本格式、过滤策略文档 |

---

## 二、核心架构

```
MRI [B,5,1,Z,H,W]
  → ShoulderCoPASModel (frozen)
  → sag_feat, cor_feat, axi_feat [B,C]
  → torch.stack → [B, 3, C]
  → VisualProjector (Linear→GELU→Linear)
  → [B, 3, H]  (H=3584, Qwen2.5-7B hidden_size)
  → inputs_embeds 替换前 3 个位置
  → Qwen2.5-7B + LoRA (r=16, target=q/k/v/o_proj)
  → 结构化 JSON 输出
```

---

## 三、数据管线验证结果

| JSONL | 样本数 | Parse 成功率 |
|-------|--------|-------------|
| label_binary | 7,847 | 100% |
| diagnosis_chain | 7,313 | 100% |
| structured_findings | 7,839 | 100% |
| structured_impression | 7,842 | 100% |

- 训练/验证/测试：6,236 / 778 / 779 exams
- 排除：23 例 postop，0 例 low quality
- 质量分桶：A 桶 ~96%，B 桶 ~4%

---

## 四、代码审查修复（3 个关键问题）

### 1. label_binary 语义不一致
- **问题**：`raw_label=2`（uncertain）直接进入 label_binary 输出，但任务名是 "binary"
- **修复**：新增 `map_label_to_binary()`，`2→0`（映射为阴性），status 保留 "uncertain"
- **影响文件**：`scripts/build_sft_jsonl.py`

### 2. gradient_accumulation_steps 未生效
- **问题**：config 写了 `gradient_accumulation_steps: 4`，但训练循环每 batch 直接 step
- **修复**：`train_epoch()` 实现完整累积逻辑：`loss /= accum_steps`，按步数 clip/step/schedule
- **影响文件**：`sft/train_sft.py`

### 3. 文档写了 DeepSpeed 但代码没接
- **问题**：SFT_PLAN 声称用 DeepSpeed ZeRO-2，实际是标准 torchrun
- **修复**：文档改为"当前 MVP: torchrun (DDP)；后续: DeepSpeed ZeRO-2"，补充实现状态说明
- **影响文件**：`docs/SFT_PLAN.zh-CN.md`

### 额外运行时修复
- **BatchNorm bs=1 报错**：冻结 MRI-CV 强制 `eval()` 模式
- **PeftModel embed_tokens 路径**：新增 `_get_embed_fn()` 方法适配 LoRA 包装后的模型结构

### 4. validate() generate() 输出切分错误（二次审查发现）
- **问题**：`model.generate()` 使用 `inputs_embeds` 路线，但未传 `input_ids`。此时 HuggingFace `generate()` 返回的序列**仅包含新生成的 token**（不含 prompt），但 `validate()` 仍按 `gen_ids[0][plen:]` 切分，导致生成结果被错误截断
- **根因**：`transformers.GenerationMixin` 在仅接收 `inputs_embeds` 时，内部 `input_ids` 初始化为空张量 `(B, 0)`，最终返回的序列长度 = 新生成 token 数
- **修复**：`generate()` 方法同时传递 `input_ids` 和 `inputs_embeds`（前者用于构建返回序列，后者用于首次 forward pass）；`validate()` 增加长度检查的 fallback 逻辑
- **影响文件**：`sft/train_sft.py`

---

## 五、冒烟测试结果（10 samples × 2 epochs）

| Encoder | CV F1 | feat_dim | Epoch 2 Train Loss | Epoch 2 Val Loss | 状态 |
|---------|-------|----------|--------------------|--------------------|------|
| **resnet50** | 0.516 | 2048 | 0.0642 | 0.0689 | Pass |
| swin3d_tiny | 0.283 | 768 | 0.0648 | 0.0646 | Pass |
| densenet201 | 0.510 | 1920 | — | — | FAIL (avg_pool3d) |

- resnet50 和 swin3d_tiny 均 loss 正常下降
- densenet201 因 avg_pool3d kernel 大于输入时间维度而失败（模型架构限制）
- experiments_copas/ 的 CoPAS 独立模型使用不同架构，不兼容当前 SFT 管线

**推荐**：resnet50 (g2_resnet_binary) 作为正式 Stage 1 的 MRI-CV backbone。

---

## 六、现有 CV 管线零修改

以下文件未做任何改动：
- `train.py`
- `models/*.py`
- `data/*.py`
- `utils/*.py`
- 原有 `configs/g*.yaml`

---

## 七、下一步

1. 用 resnet50 backbone 启动正式 Stage 1 训练（8 GPU, 全量数据, 5 epochs）
2. 重新生成 JSONL（label_binary 语义修复后）
3. 等 swin3d_tiny 训练完成后再做对比
4. 后续考虑 differentiable auxiliary heads + Stage 2
