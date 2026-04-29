# 10. 完整研究计划（2026-04-30）

## 一、研究目标与论文贡献

### 核心问题
现有肩关节 MRI 诊断系统存在三个缺陷：
1. 单序列输入，忽略多序列互补信息
2. 只输出分类标签，无法生成可解释的诊断推理链
3. 监督学习优化代理目标（CE loss），与临床诊断准确率存在 gap

### 三大贡献（拟投：MedIA / TMI / MICCAI）

| 贡献 | 内容 |
|---|---|
| C1 | 多序列 3D MRI 融合编码器 + 疾病级空间定位（7病同时分类+定位） |
| C2 | 10-token 视觉接地方案将 MRI 特征注入 LLM，实现结构化报告生成与思维链 |
| C3 | GC-GRPO：定位条件组相对策略优化，过程奖励驱动诊断推理 |

---

## 二、当前进展（截至 2026-04-30）

| 模块 | 状态 | 关键结果 |
|---|---|---|
| MRI-CV Swin3D + Grounded | ✅ 完成 | AUC=0.675, key-slice ±1=97.6% |
| MRI-CV MedNet R34（无定位）| ✅ 完成 | AUC=0.746 |
| MRI-CV MedNet 系列 | ✅ 完成 | R18=0.590 optF1, R34=0.599 optF1 |
| SFT DDP Bug 修复 | ✅ 完成 | use_reentrant=False + _set_static_graph |
| SFT Stage 1 | 🔄 训练中 | 全量 8卡，11662 samples |
| SFT Stage 2 | ⏳ 待做 | — |
| GC-GRPO | ⏳ 待做 | — |

---

## 三、实验路线

### Phase 0：补齐 Backbone 实验（并行，Week 1-2）

**目标**：找到 AUC 和定位能力兼具的最佳 backbone。

#### 实验 0-A：MedNet R34 + Grounded（最高优先级）
```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/mricv_medicalnet_resnet34_grounded.yaml
```

判断标准：
- AUC ≥ 0.73 且 key-slice ±1 ≥ 95% → 作为正式 SFT backbone
- AUC < 0.70 → grounding 伤害分类，退回 3-token 方案（仅用全局特征）

#### 实验 0-B：Swin3D 无 Grounded 消融（可选）
验证 Swin3D AUC=0.675 是架构问题还是 grounding 问题。

#### Backbone 对照表

| 实验 | AUC | key-slice ±1 | 备注 |
|---|---|---|---|
| Swin3D + Grounded（已有）| 0.675 | 97.6% | 当前 SFT baseline |
| MedNet R34（已有）| 0.746 | — | 分类上限参考 |
| MedNet R34 + Grounded | ? | ? | **目标 backbone** |
| Swin3D 无 Grounding | ? | — | 消融对照 |

---

### Phase 1：SFT 完整训练（Week 2-3）

#### 模型架构
```
MRI [B, 5, 20, 448, 448]
  → ShoulderCoPASModel（Phase 0 最佳 backbone）
  → sag_feat, cor_feat, axi_feat  [B, C]      # 3 个分支全局 token
  → local_tokens                  [B, 7, C]   # 7 个疾病局部 token
  → concat → [B, 10, C]
  → VisualProjector（Linear→GELU→Linear）
  → [B, 10, 3584]  注入 Qwen2.5-7B inputs_embeds
  → Qwen2.5-7B + LoRA（r=16, q/k/v/o_proj）
  → 结构化 JSON 输出
```

#### Stage 1（进行中）
- 冻结：MRI-CV 完全冻结
- 训练：VisualProjector + LoRA
- Loss：L_LM
- lr=2e-4，epochs=5，effective batch=64
- 任务：label_binary + diagnosis_chain

```bash
torchrun --nproc_per_node=8 sft/train_sft.py \
    --config configs/sft_stage1_grounded_clean.yaml
```

#### Stage 2（Stage 1 完成后）
- 解冻：backbone last_stage（lr×0.1）
- Loss：L_LM + 0.3×L_keyslice
- lr=5e-5，epochs=3

```bash
torchrun --nproc_per_node=8 sft/train_sft.py \
    --config configs/sft_stage2_grounded.yaml \
    --resume outputs_clean/sft_experiments/sft_stage1_grounded_clean/best_checkpoint.pt
```

#### SFT 评估目标

| 指标 | 目标值 |
|---|---|
| label_binary parse_success | ≥ 0.99 |
| label_binary macro_F1 | ≥ 0.50 |
| diagnosis_chain field_complete | ≥ 0.90 |

---

### Phase 2：GC-GRPO（核心创新，Week 3-4）

#### 动机
标准 GRPO 将 G 个 rollout 扁平归一化，模型可通过"猜对标签但定位错误"
获得高 reward，学到结果捷径而非真正的推理路径。

GC-GRPO 引入过程奖励：正确定位后再正确诊断，获得额外组间奖励。

#### 算法：Grounding-Conditioned GRPO

```
标准 GRPO：
  advantage_i = (R_i - mean(R_1..G)) / std(R_1..G)

GC-GRPO：
  G 个 rollout 按 key-slice 预测分为两组：
    G+ = {正确预测 key-slice 的 rollout}
    G- = {定位错误的 rollout}

  Step 1 组内归一化（标准 GRPO）：
    advantage_intra_i = (R_i - mean(R_group)) / (std(R_group) + ε)

  Step 2 组间奖励（创新点）：
    inter_bonus = mean(R_G+) - mean(R_G-)
    advantage_inter_i = α × inter_bonus  （仅 G+ 组获得）

  Step 3 最终 advantage：
    advantage_i = advantage_intra_i + advantage_inter_i
```

#### Reward 函数

```
R_total = R_format × R_content

R_format:
  1.0  JSON 合法且含所有必要字段
  0.0  解析失败

R_content（label_binary）：
  = macro_F1（7病，masked 排除）

R_content（diagnosis_chain）：
  = 0.4 × label_F1
  + 0.3 × evidence_semantic（sentence embedding 相似度）
  + 0.2 × grounding_acc（key-slice 位置描述准确性）
  + 0.1 × field_complete

GC-GRPO 组间 bonus（G+ 组额外获得）：
  + alpha × (mean(R_G+) - mean(R_G-))
```

#### 训练配置
```yaml
rl:
  algorithm: gc_grpo
  num_generations: 4      # G=4，确保两组都有样本
  alpha: 0.5              # 组间奖励权重
  temperature: 0.9
  kl_beta: 0.01
  max_new_tokens: 512
training:
  batch_size: 1
  max_epochs: 2
  learning_rate: 5e-6     # 比 SFT 低一个数量级
```

```bash
# Smoke test
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python rl/train_grpo.py \
    --config configs/rl_gcgrpo_stage1_clean.yaml --max_samples 20

# 正式训练
torchrun --nproc_per_node=8 rl/train_grpo.py \
    --config configs/rl_gcgrpo_stage1_clean.yaml
```

---

### Phase 3：消融实验（Week 4-5）

| 实验组 | 配置 | 目的 |
|---|---|---|
| A：标准 GRPO | alpha=0，规则 reward | baseline |
| B：GC-GRPO (α=0.3) | 弱组间奖励 | 超参消融 |
| **C：GC-GRPO (α=0.5)** | **主实验** | — |
| D：GC-GRPO (α=1.0) | 强组间奖励 | 超参消融 |
| E：+ disease-decoupled advantage | 疾病解耦 | 叠加创新 |
| F：+ semantic evidence reward | embedding 替代关键词 | reward 升级 |

#### 疾病解耦 Advantage（实验 E）
```
7 种疾病难度差异大（SST opt-F1=0.92 vs SSC opt-F1=0.37），
扁平 macro F1 导致简单疾病梯度淹没难疾病。

R_total = Σ_d  w_d × GRPO_advantage(R_d)
w_d = (1 - current_F1_d)  # 越难权重越大，动态调整
```

---

## 四、评估体系

### 分类性能
| 指标 | 说明 |
|---|---|
| Macro AUC | 7病平均 AUC（主指标） |
| Macro opt-F1 | 最优阈值 F1 |
| Per-disease AUC/F1 | 逐病报告 |

### 定位性能
| 指标 | 说明 |
|---|---|
| Key-slice top-1 | 预测切片 == GT |
| Key-slice ±1 | 预测切片在 GT ±1 内 |

### 生成质量
| 指标 | 说明 |
|---|---|
| Parse success | JSON 合法率 |
| Label accuracy | 生成标签与 GT 一致率 |
| Evidence F1 | sentence embedding 相似度 |
| VLM Judge score | GPT-4o 离线评分（不参与训练） |

---

## 五、预期结果

| 模型 | Macro AUC | Macro opt-F1 | 报告 | 思维链 |
|---|---|---|---|---|
| CoPAS baseline | 0.730 | 0.580 | ❌ | ❌ |
| MedNet R34 | 0.746 | — | ❌ | ❌ |
| + SFT Stage1 | ~0.68 | ~0.55 | ✅ | ✅ |
| + SFT Stage2 | ~0.70 | ~0.58 | ✅ | ✅ |
| + 标准 GRPO | ~0.72 | ~0.60 | ✅ | ✅ |
| **+ GC-GRPO（目标）** | **~0.75+** | **~0.63+** | ✅ | ✅ |

---

## 六、时间线

```
Week 1-2   Phase 0：MedNet R34+Grounded 实验，确定最终 backbone
Week 2-3   Phase 1：SFT Stage 1（进行中）+ Stage 2
Week 3-4   Phase 2：实现 GC-GRPO，smoke test，正式训练
Week 4-5   Phase 3：消融实验，整理结果
Week 5-6   写作：方法 + 实验 + 分析
```

---

## 七、风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| R34+Grounded AUC 明显下降 | 中 | 用 3-token SFT（仅全局特征），牺牲 spatial token |
| GRPO 训练不稳定 | 中 | 降低 lr 至 1e-6，增大 G 至 8 |
| G+ 组为空（全部定位错误）| 低 | fallback 到标准 GRPO advantage |
| 显存 OOM（G 份 logits）| 中 | num_generations 从 4 降到 2 |

---

## 八、已解决的工程问题

| 问题 | 解决方案 |
|---|---|
| DDP + LoRA + gradient_checkpointing crash | `use_reentrant=False` + `_set_static_graph()` |
| validate 在所有 rank 上跑 generate 导致通信超时 | validate 只在 rank 0 执行，前后加 `dist.barrier()` |
| 显存随 validation generation 持续增长 | validate 结束后 `torch.cuda.empty_cache()` |
| MRI-CV checkpoint 路径未软链 | `ln -s` 到 config 指定路径 |
| DDP error traceback 不可见 | `__main__` 加 try/except 写 `/tmp/sft_rank{N}_error.log` |
