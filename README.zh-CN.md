# Shoulder MRI Multi-Sequence Experiments

[English](./README.en.md)

## 项目简介

这是一个面向肩关节五序列 MRI 的研究型实验框架，目标不是只做一个黑盒分类器，而是逐步构建这样一条链路：

**多序列 MRI → 视觉前端 → 粗定位先验 → 医学 token → 大模型推理 → structured findings / impression / report**

当前仓库主要覆盖以下部分：

- 五序列 MRI 视觉前端训练
- 二分类与三分类两套标签策略
- CoPAS 风格主副序列融合
- 基于 nnUNet 粗分割 mask 的 key-slice / ROI 引导
- 医学 token 导出
- 为后续 Qwen 的 SFT / RL 预留接口

本仓库当前重点是 **视觉前端与实验工程**，不是最终报告生成系统。

---

## 五序列输入

每个病例默认包含以下五个序列：

- axial_PD
- coronal_PD
- sagittal_PD
- coronal_T2WI
- sagittal_T1WI

---

## 七个目标病种

- **SST**：冈上肌腱损伤
- **IST**：冈下肌腱损伤
- **SSC**：肩胛下肌腱损伤
- **LHBT**：肱二头肌长头腱损伤
- **IGHL**：盂肱下韧带 / 腋囊相关异常
- **RIPI**：肩袖间隙异常
- **GHOA**：盂肱关节退行性变

---

## 病种与主序列先验

当前使用如下病种先验主序列：

| 病种 | 缩写 | 主序列 |
|---|---|---|
| 肩袖间隙异常 | RIPI | sagittal_PD |
| 腋囊 / IGHL 异常 | IGHL | coronal_PD |
| 冈上肌腱损伤 | SST | coronal_PD |
| 肩关节退行性变 | GHOA | coronal_PD |
| 冈下肌腱损伤 | IST | axial_PD |
| 肩胛下肌腱损伤 | SSC | axial_PD |
| 肱二头肌长头腱损伤 | LHBT | coronal_PD |

这些先验将用于：

- 多序列主副融合设计
- nnUNet 粗分割 mask 对应的 anchor sequence
- 后续 Qwen 结构化推理链监督

---

## 标签体系

### 原始标签值

结构化 JSON 中，每个病种原始标签可能为：

- `1` 明确阳性
- `0` 阴性
- `2` 可疑
- `-1` 术后不可可靠映射

### label_status

同时保留更细的标签状态：

- `explicit_positive`
- `explicit_negative`
- `implicit_negative`
- `uncertain`
- `postop_unmappable`

---

## 当前支持的训练模式

### 二分类模式

映射规则如下：

- `1 -> 1`
- `0 -> 0`
- `2 -> 0`
- `-1 -> mask`

### 三分类模式

映射规则如下：

- `0 -> negative`
- `2 -> uncertain`
- `1 -> positive`
- `-1 -> mask`

### 设计原则

- `metadata_master.csv` 只保存原始标签与原始状态
- binary / ternary 的映射在运行时由 `label_mapper.py` 完成
- 训练时同时保留 raw label、train label、train mask

---

## 数据组成

本项目默认围绕三类数据组织：

### 1. 五序列 MRI 图像
一个病例一个目录，包含五个标准序列。

### 2. 结构化 JSON 标签
一个病例一个 JSON 文件，包含 labels、label_status、evidence_text、negative_evidence、structured_findings、structured_impression 等字段。

### 3. nnUNet 粗分割 mask
用于提供弱监督定位先验，不作为主线中的精细分割真值。当前主要用途包括：

- key-slice 提取
- ROI bbox 提取
- local branch 引导
- lesion-focused token 构造

---

## metadata_master.csv 的作用

`metadata_master.csv` 是整个项目的主索引表。
它的作用是把分散在不同目录中的数据统一到一行一个病例的结构里。

通常会关联以下信息：

- exam_id
- 五序列图像路径
- JSON 路径
- 原始标签
- 标签状态
- source_summary
- nnUNet mask 路径
- split 信息

这样训练脚本不用反复扫目录，数据读取、调试和复现都更稳定。

---

## 项目结构

```text
project/
├── configs/
│   ├── g1_densenet_binary.yaml
│   ├── g1_densenet_ternary.yaml
│   ├── g2_resnet_binary.yaml
│   ├── g2_resnet_ternary.yaml
│   ├── g1l_densenet_binary.yaml
│   ├── g1l_densenet_ternary.yaml
│   ├── g2l_resnet_binary.yaml
│   └── g2l_resnet_ternary.yaml
├── data/
│   ├── build_index.py
│   ├── json_parser.py
│   ├── label_mapper.py
│   ├── mask_index.py
│   └── shoulder_dataset.py
├── models/
│   ├── encoders.py
│   ├── fusion_copas.py
│   ├── localizer_branch.py
│   ├── heads.py
│   └── multiseq_model.py
├── utils/
│   ├── io.py
│   ├── losses.py
│   ├── metrics.py
│   ├── export_tokens.py
│   ├── vis.py
│   └── seed.py
├── scripts/
├── train.py
├── infer.py
└── outputs/
```

### 各目录说明

#### `configs/`

实验配置文件，包括：

* G1 / G2
* binary / ternary
* localizer 版与无 localizer 版

#### `data/`

负责：

* 生成 metadata
* 解析 JSON
* 标签映射
* mask 索引
* Dataset 构建

#### `models/`

负责：

* 3D encoder
* CoPAS 风格融合
* 分类 head
* localizer branch
* 完整多序列模型封装

#### `utils/`

负责：

* NIfTI 与 mask 读写
* 损失函数
* 指标计算
* token 导出
* 可视化
* 随机种子

#### `outputs/`

保存：

* metadata
* 实验结果
* 模型权重
* token 导出结果

---

## 当前实验主线

### G1

**DenseNet + CoPAS-style fusion**

作为稳定的多序列 CNN 基线。

### G2

**ResNet(MedicalNet) + CoPAS-style fusion**

作为更强的 3D 医学预训练前端基线。

### G1-L / G2-L

在 G1 / G2 基础上加入 nnUNet 粗 mask 引导的 localizer 分支。

---

## 训练流程

### 1. 构建 metadata

```bash
python data/build_index.py
```

### 2. 训练模型

例如运行 G1 二分类：

```bash
python train.py --config configs/g1_densenet_binary.yaml
```

### 3. 推理与评估

```bash
python infer.py --config configs/g1_densenet_binary.yaml
```

### 4. 导出 token

训练完成后导出：

* sequence-level features
* fused features
* sequence weights
* 可选 local features
* 可选 key slices
* 可选 ROI boxes

这些 token 后续将用于 Qwen 的 SFT / RL。

---

## token 导出的意义

本项目并不把视觉前端只看作"分类器"。
更重要的是让前端输出可复用的 **医学 token**，用于后续更强的结构化推理。

默认导出的信息可以包括：

* `seq_feats`
* `fused_feat`
* `seq_weights`
* 可选 `local_feats`
* 可选 `key_slices`
* 可选 `roi_boxes`

---

## 为什么要引入粗分割 mask

很多肩关节病种并不是大范围全局异常，而是局部结构问题。
因此仅用全局 feature 不一定足够。

nnUNet 粗分割 mask 的作用不是提供绝对精确的边界，而是提供：

* 哪些切片更关键
* 哪些局部区域更值得看
* 哪些局部特征更适合形成 lesion token

这一步可以让系统从纯黑盒分类向带中间定位步骤的诊断链推进。

---

## 长期目标

本项目的长期目标是构建完整的多阶段肩关节 MRI 智能诊断流程：

**多序列 MRI → 视觉前端 → 粗定位先验 → 医学 token → Qwen 推理 → structured findings / impression / report**

当前仓库主要完成前半段，也就是视觉前端与 token 化部分。

---

## 计划扩展

后续计划包括：

* 更严格的多标签数据划分
* localizer-guided 融合实验
* Triad encoder 支持
* Decipher-MR encoder 支持
* key-slice / ROI token 更细粒度导出
* Qwen 的结构化诊断链 SFT
* 基于可验证中间步骤的 RL

---

## 使用说明

本仓库仅用于研究实验。

请注意：

* 不提供医疗建议
* 不要提交私有患者数据
* 所有数据路径和权重路径请本地配置
* 发布开源版本时请确认数据和权重的合规性

---

## License

请根据你的发布方式自行添加，例如：

* MIT
* Apache-2.0
* CC BY-NC 4.0

---

## Contact

如有问题或合作意向，请通过 GitHub issue 或仓库维护者联系方式联系。
