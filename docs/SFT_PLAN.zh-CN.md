# 肩关节 MRI 结构化 SFT 三阶段训练路线（Grounded v1）

## 总目标

在现有 MRI 视觉分类前端（ShoulderCoPASModel）基础上，通过 **grounded SFT** 将视觉特征接入 Qwen2.5-7B-Instruct，实现以下目标：

1. **推理时只输入原始多序列 MRI**，不依赖外部 nnUNet 预处理
2. 模型内部通过 disease-specific grounding heads 自动完成关键层面定位
3. LLM 统一输出完整诊断链：标签 + 证据 + 锚定序列 + 关键层面 + findings + impression

当前阶段**不以自由报告生成为主目标**。

### 核心原则

**mask / key_slice / ROI 在训练时作为监督信号，不作为推理时必需的外部前置条件。**

---

## 架构总览

```
MRI [B, 5, 1, Z, H, W]
  → ShoulderCoPASModel (Swin3D-Tiny × 5 编码器 + 3 分支 CoPlane/CrossModal 注意力)
  → sag_feat [B,C], cor_feat [B,C], axi_feat [B,C]       ← 3 global tokens
  → DiseaseSpecificSliceHeads (cor/axi/sag 三路)
    → slice_logits [B, 7, D']
    → SoftAttentionLocalTokenPooler
    → local_tokens [B, 7, C]                               ← 7 disease-aware tokens
  → stack → [B, 10, C]
  → VisualProjector (2 层 MLP, 共享)
  → visual_tokens [B, 10, H]        (H = Qwen2.5-7B hidden_size = 3584)
  → inputs_embeds 拼接
  → Qwen2.5-7B-Instruct + LoRA
  → 结构化 JSON 文本输出
```

关键设计：
- **10 个 visual tokens**：3 global (sag/cor/axi) + 7 local (SST/IST/SSC/LHBT/IGHL/RIPI/GHOA)
- **Disease-specific branch routing**：
  - cor_slice_head → SST, LHBT, IGHL, GHOA (coronal_PD)
  - axi_slice_head → IST, SSC (axial_PD)
  - sag_slice_head → RIPI (sagittal_PD)
- **Soft attention pooling**：完全可微，训练稳定，推理时自然
- **VisualProjector**：所有 token 过同一个 `Linear(C,H) → GELU → Linear(H,H)`
- **Token 注入**：使用 `inputs_embeds` 替换前 10 个位置

### Backbone 选择

**Swin3D-Tiny 是当前最适合作为可解释定位与 Qwen SFT 主线的前端**：
- D' = 10（保留 slice 维度，可做 local token 提取）
- 已训练 SliceHead localizer（±1 命中率 97.7%）
- ResNet18/34/50 的 D' ≈ 1-2（conv1/maxpool/layer2/3/4 均压缩 Z 维），定位粒度过粗

**MedicalNet ResNet18/34 的定位**（作为对照，不作 grounding 主线）：
- 优势：医学预训练，分类 AUC 0.744/0.747，明显优于从头训练的 Swin
- 局限：D' ≈ 1-2，local_tokens 退化为"粗局部/分支 token"，无法支撑 slice-level grounding
- 用途：**分类性能上限参考 / 迁移学习有效性对照 / 非定位 SFT baseline**
- **KS ±1 指标对 D'=1-2 不可信**（总共只有 1-2 个 feature slice，±1 几乎覆盖全范围）；对这类 backbone 更可信的是 KS top1 + 可视化检查

**三条并行线最终定位**：
```
MedicalNet ResNet18/34 G2L  = 分类性能与迁移学习对照
Swin3D-Tiny G3 grounded     = 可解释定位与 Qwen SFT 主线
CoPAS                       = 多序列融合对照，后续可吸收其 attention 机制到 grounded 主线
```

---

## 训练流程

### Step 0：Grounded visual backbone 预训练

**目标**：训练 disease-specific SliceHeads + 分类头

**Loss**：`L = L_cls + 0.3 * L_branch + 0.5 * L_keyslice`

**Config**：`configs/g3_grounded_swin_binary.yaml`

**输出指标**：macro AUC, macro F1, key-slice top1, key-slice ±1

---

### Stage 1：冻结 grounded backbone + SFT

**目标**：验证 10-token grounded SFT 是否有效

**冻结策略**：
- **MRI-CV (含 grounding heads)**：全部冻结
- **VisualProjector**：全部可训练
- **Qwen2.5-7B-Instruct**：基座冻结，LoRA 适配器可训练
  - LoRA: r=16, alpha=32, target=q/k/v/o_proj

**Loss**：纯 LM loss（`keyslice_alpha = 0`）

**第一版只跑**：`label_binary` + `diagnosis_chain`

**Config**：`configs/sft_stage1_grounded.yaml`

| 参数 | 值 |
|------|-----|
| batch_size | 2/GPU × 8GPU × grad_accum=4 = 64 |
| learning_rate | 2e-4 |
| scheduler | cosine, warmup 3% |
| max_epochs | 5 |
| precision | bf16 |
| 并行策略 | 当前 MVP: torchrun (DDP) |

---

### Stage 2：部分解冻 + key-slice 联合 loss

**目标**：解冻 Swin 最后 stage，让视觉特征适应 LLM 需求

**冻结策略**：
- **MRI-CV 最后 stage + grounding heads**：解冻（lr × 0.1）
- **MRI-CV 其余部分**：冻结
- **VisualProjector**：可训练
- **Qwen LoRA**：可训练

**Loss**：`L = L_LM + 0.3 * L_keyslice`

**Config**：`configs/sft_stage2_grounded.yaml`

| 参数 | 值 |
|------|-----|
| learning_rate | 5e-5 |
| MRI-CV unfrozen lr | 5e-6 (× 0.1) |
| max_epochs | 3 |
| keyslice_alpha | 0.3 |
| resume_from | Stage 1 best checkpoint |

---

### Stage 3：全量微调（占位）

解冻所有参数，极低学习率 (1e-5)，1-2 epochs。当前仅占位。

---

## 第一版诊断链字段定义

```json
{
  "labels": {"SST": 1, "IST": 0, ...},
  "evidence": {"SST": {"positive": [...], "negative": [...]}, ...},
  "anchor_sequence": {"SST": "coronal_PD", ...},
  "key_slice": {"SST": 12, "IST": null, ...},
  "structured_findings": ["影像表现句子1", ...],
  "structured_impression": ["诊断印象句子1", ...]
}
```

**注意**：roi_box 已从第一版移除，将在第二版作为 ROI head 加入。

---

## 第一版不做的内容

1. 不让 LLM 直接生成 dense mask
2. 不依赖外部 nnUNet 预处理作为推理输入
3. 不第一版压入 ROI / bbox / mask
4. 不以 ResNet50 作为 grounded 主线
5. 不从诊断链中删除 key_slice

---

## 里程碑

| 阶段 | 完成标准 |
|------|----------|
| 数据就绪 | label_binary + diagnosis_chain JSONL + train/val 划分 |
| Grounded backbone | g3_grounded_swin_binary 训练完成，key-slice ±1 > 95% |
| Stage 1 冒烟 | 1 GPU + 10 条样本，loss 正常下降 |
| Stage 1 完整 | 8 GPU 训练，val JSON parse > 80%, label accuracy > 70% |
| Stage 2 | 部分解冻 + L_keyslice，指标不退化 |

---

## 共享常量

所有常量统一在 `utils/constants.py` 管理：
- `DISEASES`: 7 病种列表
- `SEQUENCE_ORDER`: 5 序列顺序
- `DISEASE_ANCHOR_SEQ`: 病种 → 锚定序列映射
- `DISEASE_BRANCH_MAP`: SliceHead 分支路由
- `DISEASE_TO_BRANCH`: 病种 → 分支反向映射

---

## 当前实现状态

### 已实现
- `torchrun` 多 GPU 训练（标准 PyTorch DDP）
- 梯度累积（`gradient_accumulation_steps`）
- bf16 混合精度 + Gradient checkpointing（LLM）
- `DiseaseSpecificSliceHeads`（3 路 branch SliceHead）
- `SoftAttentionLocalTokenPooler`（可微 local token 提取）
- 10-token VisualProjector
- Stage 2 `keyslice_alpha` 联合 loss
- 共享常量模块 `utils/constants.py`

### 尚未实现（后续扩展）
- ROI head / bbox regression
- Dense mask head
- DeepSpeed ZeRO-2
- RL 精修（诊断链一致性奖励）
- Stage 3 训练代码
