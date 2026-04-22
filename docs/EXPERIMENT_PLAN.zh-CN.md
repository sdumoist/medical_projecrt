# 肩关节 MRI 完整实验计划表（供 Claude Code 执行）

## 1. 文档目标

本文件用于指导 Claude Code 在当前仓库内补齐肩关节 MRI 项目的分类、localizer 与后续 VLM 对比实验工程。

当前原则如下：

1. 第一阶段只比较 **肩关节 MRI 分类 / localizer**，不把胸片报告模型直接混入主表。
2. 第二阶段再比较 **MRI token + 医学大模型 / VLM**。
3. 第一阶段主表至少覆盖 **CNN + Transformer**。Mamba 作为扩展实验，不作为第一优先级必选项。
4. 所有实验统一记录：
   - macro AUC
   - macro F1
   - macro optimal-threshold F1
   - per-disease AUC / F1 / Recall / Precision
   - 若为 localizer 模型，额外记录 key-slice 与 ROI 指标

---

## 2. 当前建议的论文拆分

| 阶段 | 论文主题 | 主问题 | 主表模型 |
|---|---|---|---|
| Phase 1 | 多序列肩关节 MRI 分类与显式定位 | 多序列融合与 localizer 是否有效 | DenseNet / ResNet / CoPAS / Transformer / localizer |
| Phase 2 | MRI token + VLM 结构化诊断链与报告生成 | 中间诊断步骤能否提升可解释报告生成 | Qwen2.5-VL / RadFM / M3D-LaMed / LLaVA-Med |

---

## 3. 第一阶段主实验表

### 3.1 必做主表

| 实验编号 | 名称 | 类型 | 当前状态 | 是否必做 | 说明 |
|---|---|---|---|---|---|
| A1 | G1 DenseNet121 | CNN 分类 | 已有 | 是 | 作为稳定的 3D CNN 基线 |
| A2 | G2 ResNet18 / 50 | CNN 分类 | 已有 | 是 | 作为更标准的 3D 医学预训练基线 |
| A3 | CoPAS | 多分支融合分类 | 已有 | 是 | 作为多序列主副分支融合基线 |
| A4 | G2L ResNet + localizer | CNN + localizer | 已有 | 是 | 当前推荐主方法 |
| A5 | G3 Swin-based classifier | Transformer 分类 | 待补 | 是 | 用于回答“只比 CNN 是否不够” |
| A6 | G3L Swin + localizer | Transformer + localizer | 待补 | 是 | 用于验证 localizer 是否对 Transformer 同样有效 |

### 3.2 建议扩展表

| 实验编号 | 名称 | 类型 | 当前状态 | 是否必做 | 说明 |
|---|---|---|---|---|---|
| B1 | G1L DenseNet + localizer | CNN + localizer | 已有 | 否 | 可作为 DenseNet 路线补充，但当前不建议做主力 |
| B2 | CoPAS + CB-ASL | 多分支融合分类 | 已有 | 是 | 必须与 CoPAS original 对照 |
| B3 | MedMamba-style baseline | Mamba / SSM | 待补 | 否 | 放入扩展实验或补充材料 |

---

## 4. 第二阶段预研表

> 这一部分不要求立刻完成，只做路线占位与接口预留。

| 实验编号 | 名称 | 类型 | 优先级 | 说明 |
|---|---|---|---|---|
| C1 | MRI-CV + projector + Qwen2.5-VL | 目标方法 | 高 | 当前长期主线 |
| C2 | RadFM | 放射基础模型对照 | 高 | 作为 radiology foundation model 对照 |
| C3 | M3D-LaMed | 3D medical MLLM | 中 | 作为 3D 医学多模态对照 |
| C4 | LLaVA-Med | 通用 biomedical VLM | 中 | 作为通用医学 VLM 弱对照 |
| C5 | BiomedCLIP front-end | 图文表征对照 | 高 | 作为强表征基线或教师模型 |

---

## 5. Claude Code 需要在仓库中补齐的内容

### 5.1 配置文件

需要新增以下配置文件：

| 路径 | 用途 |
|---|---|
| `configs/g3_swin_binary.yaml` | Transformer 二分类基线 |
| `configs/g3_swin_ternary.yaml` | Transformer 三分类基线 |
| `configs/g3l_swin_binary.yaml` | Transformer + localizer 二分类 |
| `configs/g3l_swin_ternary.yaml` | Transformer + localizer 三分类 |
| `configs/ablation_copas_cbasl.yaml` | CoPAS loss 对照统一入口，可选 |
| `configs/optional_mamba_binary.yaml` | Mamba 扩展实验，可选 |

### 5.2 模型代码

需要新增或补齐：

| 路径 | 操作 |
|---|---|
| `models/encoders.py` | 增加 Swin-based 3D encoder 工厂入口 |
| `models/multiseq_model.py` | 确保 Transformer encoder 可直接接入当前多序列融合主线 |
| `models/localizer_branch.py` | 检查是否与 Transformer 输出特征维度兼容 |
| `utils/metrics.py` | 统一 fixed 0.5 / optimal threshold 指标接口 |
| `train.py` | 确保所有 backbone 都复用同一套日志、保存与评估逻辑 |

### 5.3 脚本与实验工程

建议新增：

| 路径 | 用途 |
|---|---|
| `scripts/run_phase1_baselines.sh` | 统一启动 A1~A6 实验 |
| `scripts/run_copas_ablation.sh` | CoPAS original vs cbasl |
| `scripts/summarize_phase1_results.py` | 汇总 csv/jsonl 成总表 |
| `scripts/export_best_thresholds.py` | 导出每病种最佳阈值 |
| `docs/RESULT_TABLE_TEMPLATE.zh-CN.md` | 论文结果表模板 |

---

## 6. 第一阶段具体实验矩阵

### 6.1 分类主表

| 组别 | 配置名 | 输入 | 输出 | 主指标 | 备注 |
|---|---|---|---|---|---|
| A1 | `g1_densenet_binary.yaml` | 五序列，`cache_cls` | 7 病种二分类 | macro AUC / macro F1 | 稳定 CNN 基线 |
| A2 | `g2_resnet_binary.yaml` | 五序列，`cache_cls` | 7 病种二分类 | macro AUC / macro F1 | MedicalNet 路线 |
| A3 | `copas/train.py --loss_type original` | 五序列，`cache_cls` | 7 病种二分类 | macro AUC / macro F1 / macro opt-F1 | 原始 CoPAS |
| A4 | `copas/train.py --loss_type cbasl` | 五序列，`cache_cls` | 7 病种二分类 | macro AUC / macro F1 / macro opt-F1 | 长尾对照 |
| A5 | `g3_swin_binary.yaml` | 五序列，`cache_cls` | 7 病种二分类 | macro AUC / macro F1 | Transformer 基线 |

### 6.2 localizer 主表

| 组别 | 配置名 | 输入 | 输出 | 主指标 | localizer 指标 | 备注 |
|---|---|---|---|---|---|---|
| L1 | `g2l_resnet_binary.yaml` | 五序列，`cache_loc` | 分类 + key-slice | macro AUC / macro F1 | key-slice acc | 当前推荐主方法 |
| L2 | `g3l_swin_binary.yaml` | 五序列，`cache_loc` | 分类 + key-slice | macro AUC / macro F1 | key-slice acc | Transformer + localizer |
| L3 | `g1l_densenet_binary.yaml` | 五序列，`cache_loc` | 分类 + key-slice | macro AUC / macro F1 | key-slice acc | 作为补充，不做主结论 |

### 6.3 ROI / 上游定位链条对照

| 组别 | 上游模块 | 下游分类器 | 主要回答的问题 |
|---|---|---|---|
| R1 | nnU-Net | ResNet / DenseNet | 显式 ROI 是否优于纯全局分类 |
| R2 | UNETR 或 Swin UNETR | ResNet / DenseNet | Transformer 分割器做上游定位是否更好 |

---

## 7. 每组实验必须保存的结果

每个实验目录至少保存：

1. `metrics_epoch.csv`
2. `metrics_epoch.jsonl`
3. `best_model.pt`
4. `last_model.pt`
5. `best_thresholds.json`
6. `config_resolved.yaml`
7. `log.txt`

如果是 localizer 模型，还需要：

8. `key_slice_metrics.json`
9. `roi_metrics.json`（若有 ROI）
10. `vis_samples/` 若干可视化案例

---

## 8. 指标统一规范

### 8.1 分类指标

所有分类实验统一输出：

- `val_macro_auc`
- `val_macro_f1`
- `val_macro_opt_f1`
- 每病种：
  - `auc`
  - `f1`
  - `recall`
  - `precision`
  - `opt_thr`
  - `opt_f1`

### 8.2 localizer 指标

建议统一输出：

- key-slice top-1 accuracy
- key-slice ±1 命中率
- ROI 命中率
- ROI IoU 或 box overlap（若已实现）

### 8.3 数据分层分析

至少按以下子集做结果分析：

- overall
- 少数类病种：IST / SSC / LHBT / IGHL / GHOA
- uncertain 样本映射为 0 后的表现
- high / medium `quality_flag` 分层表现（若数据读取链已支持）

---

## 9. Claude Code 的具体执行顺序

### Step 1
补齐文档和配置骨架：

- 新增本文件中列出的 G3 / G3L 配置文件
- 确保 README 能链接到本计划

### Step 2
补 Transformer baseline：

- 在 `models/encoders.py` 中加入 Swin-based 3D encoder
- 确保 `train.py` 能无缝读取新 encoder 名称
- 跑通 `g3_swin_binary.yaml`

### Step 3
补 Transformer + localizer：

- 让 G3L 可以读取 `cache_loc`
- 验证 `slice_logits`、`key_slices`、localizer loss 都工作正常
- 跑通 `g3l_swin_binary.yaml`

### Step 4
统一指标和阈值导出：

- 所有主实验都要保存 `best_thresholds.json`
- 汇总脚本中自动生成主表和病种分表

### Step 5
补 CoPAS 对照表：

- 跑 `original`
- 跑 `cbasl`
- 输出统一表格，重点分析少数类提升

### Step 6
可选扩展：

- 加入 MedMamba-style baseline
- 仅作为扩展表，不阻塞 Phase 1 主结论

---

## 10. 论文主结论预期

Phase 1 预期回答以下问题：

1. 多序列肩关节 MRI 分类中，CoPAS 风格融合是否优于普通 CNN 基线。
2. 显式 localizer 是否能稳定提升少数类病种表现。
3. 这种提升是否不仅发生在 CNN，也能在 Transformer backbone 上成立。
4. 长尾 loss 是否能提高少数类的 recall / F1。

---

## 11. 完成标准

当以下条件全部满足时，可视为 Phase 1 工程完成：

- [ ] G1 / G2 / G2L / CoPAS original / CoPAS cbasl 已完整跑通
- [ ] G3 / G3L 已实现并至少完成一轮正式实验
- [ ] 每组实验都有 csv、jsonl、best_model、best_thresholds
- [ ] 有统一的汇总脚本输出总表
- [ ] 有至少一份面向论文的主表 Markdown 模板
- [ ] README 中已能找到本实验计划文档

---

## 12. 不建议的做法

1. 不要把胸片报告生成模型直接塞进当前肩 MRI 分类主表。
2. 不要在第一阶段把所有 VLM 与所有 CNN 混成一个比较表。
3. 不要只看 macro AUC，不看 per-disease F1 与最佳阈值表现。
4. 不要让 Mamba baseline 阻塞 Transformer baseline 的落地。

---

## 13. 给 Claude Code 的一句话任务说明

请以本文件为执行蓝图，在当前仓库中补齐 Phase 1 所需的 Transformer baseline、localizer 实验配置、统一指标输出、阈值导出和结果汇总脚本，并保证所有新增实验可复现实验目录结构与结果文件规范。