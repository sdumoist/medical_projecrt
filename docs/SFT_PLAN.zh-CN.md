# 肩关节 MRI 结构化 SFT 三阶段训练路线

## 总目标

在现有 MRI 视觉分类前端（ShoulderCoPASModel）基础上，通过结构化 SFT 将视觉特征接入 Qwen2.5-7B，实现以下中间诊断链的稳定输出：

1. 七病种标签（label_binary）
2. 完整诊断链（diagnosis_chain）：标签 + 证据 + 锚定序列 + 关键层面 + ROI
3. 结构化 findings（structured_findings）
4. 结构化 impression（structured_impression）

当前阶段**不以自由报告生成为主目标**。

---

## 架构总览

```
MRI [B, 5, 1, Z, H, W]
  → ShoulderCoPASModel (5 编码器 + 3 分支 CoPlane/CrossModal 注意力)
  → sag_feat [B,C], cor_feat [B,C], axi_feat [B,C]
  → stack → [B, 3, C]
  → VisualProjector (2 层 MLP)
  → visual_tokens [B, 3, H]        (H = Qwen2.5-7B hidden_size = 3584)
  → inputs_embeds 拼接
  → Qwen2.5-7B + LoRA
  → 结构化 JSON 文本输出
```

关键设计：
- **3 个 visual tokens**：每个 token 对应 sag/cor/axi 分支，保留多视角结构
- **VisualProjector**：每个 branch feature 独立过同一个 `Linear(C,H) → GELU → Linear(H,H)`
- **Token 注入**：使用 `inputs_embeds` 拼接，不依赖 tokenizer 特殊词

---

## Stage 1：冻结版（第一版主目标）

### 目标
验证结构化 SFT 是否有效：冻结 MRI-CV，只训练 projector + LLM 轻量适配。

### 冻结策略
- **MRI-CV**：全部冻结（requires_grad=False）
- **VisualProjector**：全部可训练
- **Qwen2.5-7B**：基座冻结，LoRA 适配器可训练
  - LoRA: r=16, alpha=32, target=q_proj/k_proj/v_proj/o_proj
  - 约 33M 可训练参数

### Loss
- **纯 LM loss**：交叉熵 on output tokens，instruction 部分 label=-100

### 训练超参
| 参数 | 值 |
|------|-----|
| batch_size | 2/GPU × 8GPU × grad_accum=4 = 64 |
| learning_rate | 2e-4 |
| scheduler | cosine, warmup 3% |
| max_epochs | 5 |
| precision | bf16 |
| 并行策略 | DeepSpeed ZeRO-2 |
| gradient_checkpointing | LLM only |

### 任务优先级
1. `label_binary` — 最先跑通
2. `diagnosis_chain` — 主任务
3. `structured_findings`
4. `structured_impression`

### 显存估计
- Qwen2.5-7B bf16: ~14GB
- MRI-CV (冻结): ~150MB
- Projector + LoRA: ~300MB
- 优化器状态 + 激活: ~10-15GB
- **总计 ~25-30GB/GPU**，8×A800-80GB 充裕

---

## Stage 2：部分解冻

### 目标
解冻 MRI-CV 最后一个 stage，让视觉特征能适应语言模型的需求。

### 冻结策略
- **MRI-CV 最后 stage**：解冻（lr × 0.1）
  - ResNet: `encoder.layer4`
  - Swin: `encoder.stages[-1]`
- **MRI-CV 其余部分**：冻结
- **VisualProjector**：可训练
- **Qwen LoRA**：可训练（r=16，与 Stage 1 一致）

### Loss
- **LM loss only**（与 Stage 1 相同）
- **TODO**：后续新增 differentiable auxiliary heads 后，可扩展为联合 loss：
  - `L = L_LM + λ₁ L_label + λ₂ L_keyslice`
  - 需要在 projector 后挂 label_head 和 keyslice_head

### 训练超参
| 参数 | 值 |
|------|-----|
| learning_rate | 5e-5 |
| MRI-CV unfrozen lr | 5e-6 (× 0.1) |
| max_epochs | 3 |
| resume_from | Stage 1 best checkpoint |

---

## Stage 3：全量微调（占位）

### 目标
解冻所有参数，作为最终对照实验。当前阶段仅写配置和占位代码，不要求跑通。

### 冻结策略
- 所有模块可训练
- 极低学习率 (1e-5)，短训练 (1-2 epochs)

---

## 明确禁止的做法

1. 不要一开始就把自由报告生成作为主任务
2. 不要一开始就全量微调
3. 不要绕开现有结构化 JSON 重写数据格式
4. 不要破坏现有 CV 主训练线（train.py / models/ 零修改）
5. 不要只做文本 loss，后续要保留视觉锚点任务接口

---

## 里程碑

| 阶段 | 完成标准 |
|------|----------|
| 数据就绪 | 4 类 JSONL + train/val/test 划分 + 统计报告 |
| Stage 1 冒烟 | 1 GPU + 10 条样本，loss 正常下降 |
| Stage 1 完整 | 8 GPU 训练，label_binary + diagnosis_chain，val JSON parse > 80% |
| Stage 2 | 部分解冻，验证指标不退化 |
