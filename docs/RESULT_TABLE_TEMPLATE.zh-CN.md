# Phase 1 实验结果汇总表模板

> 由 `scripts/summarize_phase1_results.py` 自动填充，或手动录入。

## 分类主表

| 组别 | 配置名 | Encoder | Macro AUC | Macro F1 | Macro Opt-F1 | 备注 |
|---|---|---|---|---|---|---|
| A1 | g1_densenet_binary | DenseNet121 3D | | | | CNN 基线 |
| A2 | g2_resnet_binary | ResNet18 3D | | | | MedicalNet |
| A3 | CoPAS_orig | ResNet18 CoPAS | | | | CoPAS original |
| A4 | CoPAS_cbasl | ResNet18 CoPAS | | | | CoPAS cbasl |
| A5 | g3_swin_binary | Swin3D Tiny | | | | Transformer 基线 |

## Localizer 主表

| 组别 | 配置名 | Encoder | Macro AUC | Macro F1 | Macro Opt-F1 | KS Top1 | KS ±1 | 备注 |
|---|---|---|---|---|---|---|---|---|
| L1 | g2l_resnet_binary | ResNet18 3D | | | | | | ResNet + localizer |
| L2 | g3l_swin_binary | Swin3D Tiny | | | | | | Swin + localizer |

## 病种 AUC 分表

| 组别 | SST | IST | SSC | LHBT | IGHL | RIPI | GHOA |
|---|---|---|---|---|---|---|---|
| A1 | | | | | | | |
| A2 | | | | | | | |
| A3 | | | | | | | |
| A4 | | | | | | | |
| A5 | | | | | | | |
| L1 | | | | | | | |
| L2 | | | | | | | |

## 病种 Opt-F1 分表

| 组别 | SST | IST | SSC | LHBT | IGHL | RIPI | GHOA |
|---|---|---|---|---|---|---|---|
| A1 | | | | | | | |
| A2 | | | | | | | |
| A3 | | | | | | | |
| A4 | | | | | | | |
| A5 | | | | | | | |
| L1 | | | | | | | |
| L2 | | | | | | | |

## 病种最佳阈值

| 组别 | SST | IST | SSC | LHBT | IGHL | RIPI | GHOA |
|---|---|---|---|---|---|---|---|
| A1 | | | | | | | |
| A2 | | | | | | | |
| A3 | | | | | | | |
| A4 | | | | | | | |
| A5 | | | | | | | |
| L1 | | | | | | | |
| L2 | | | | | | | |

## 说明

- **Macro AUC / F1**: 7 病种的宏平均
- **Macro Opt-F1**: 每病种搜索最优阈值（0.05–0.95）后的 F1 宏平均
- **KS Top1**: key-slice 精确命中率
- **KS ±1**: key-slice ±1 切片命中率
- 所有指标来自验证集 best epoch
