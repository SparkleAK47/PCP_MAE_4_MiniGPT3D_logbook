
# MiniGPT-3D 复现与点云编码器替代研究

本项目是对 [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) 的完整复现，并系统研究了使用**低成本自监督点云编码器（PCP-MAE / Point-MAE）替代 ULIP-2 预训练 Point-BERT** 的可行性与效果。

核心工作包括：Q-Former 权重修复、PCP-MAE 在 Objaverse 和 ShapeNet55-34 上的预训练、多编码器变体（V1/V2/Point-MAE）的对比实验、解冻/冻结消融实验、编码器质量诊断，以及完整的 MiniGPT-3D 四阶段训练与多任务评估。

> **实验日志（训练日志、评估结果等）已开源至**：https://github.com/SparkleAK47/PCP_MAE_4_MiniGPT3D_logbook

---

## 目录

- [1. 编码器变体一览](#1-编码器变体一览)
- [2. 评估结果汇总](#2-评估结果汇总)
  - [2.1 开放词汇分类 (Objaverse)](#21-开放词汇分类-objaverse)
  - [2.2 ModelNet40 闭集分类](#22-modelnet40-闭集分类)
  - [2.3 物体描述生成](#23-物体描述生成)
- [3. 编码器质量诊断](#3-编码器质量诊断)
- [4. 关键发现与结论](#4-关键发现与结论)
- [5. 关键技术问题](#5-关键技术问题)
- [6. 仓库结构](#6-仓库结构)
- [7. 环境与协作](#7-环境与协作)

---

## 1. 编码器变体一览

| 代号 | 权重文件 | 预训练方式 | 预训练数据 | 架构 | cls 来源 | 备注 |
|------|----------|------------|------------|------|----------|------|
| **Baseline** | `point_model.pth` | ULIP-2 / Point-BERT（官方） | Objaverse | PointTransformer (self-attn) | 原生 | MiniGPT-3D 原始编码器 |
| **V1** | `point_model_pcpmae.pth` | PCP-MAE + MaskTransformer | Objaverse 660K | MaskTransformer (cross-attn) | **无 cls** | 预训练与下游推理编码器实现不同 |
| **V1 hybrid** | `point_model_hybrid.pth` | V1 骨干 + Baseline cls | Objaverse 660K | MaskTransformer | 从 Baseline 拷贝 | 补充 cls 后供 MiniGPT-3D 加载 |
| **V2** | `point_model_pcp_v2.pth` | PCP-MAE + PointTransformer | Objaverse 660K | PointTransformer (self-attn) | 预训练原生 | 预训练与下游使用一致的 PointTransformer |
| **Point-MAE** | `point_model_pointmae.pth` | 纯 Point-MAE（`ita=0`） | Objaverse 660K | PointTransformer (self-attn) | 预训练原生 | 关闭 PCP 中心预测分支，与 V2 对照 |
| **ShapeNet55-34** | `pcpmae_ShapeNet_fixed.pth` | PCP-MAE + MaskTransformer | ShapeNet55-34 | MaskTransformer (cross-attn) | 随机初始化（修复补充） | 需 3ch→6ch 修复 |
| **Random** | `point_model_random.pth` | 随机初始化（无预训练） | — | PointTransformer (self-attn) | 随机初始化 | 消融实验：排除解冻训练的混淆 |

### 实验对照关系

| 对比维度 | 实验组 |
|----------|--------|
| 预训练任务有效性 | PCP-MAE (V2) vs Point-MAE vs Random |
| 编码器架构 | MaskTransformer (V1) vs PointTransformer (V2) |
| 预训练数据域 | Objaverse (V1/V2) vs ShapeNet55-34 |
| 冻结 vs 解冻 | 每个变体均做了 `freeze_pc: True` 与 `freeze_pc: False`对比 |

---

## 2. 评估结果汇总

所有主观评估使用阿里云百炼 **Qwen-Flash** API 进行打分。评估物为 MiniGPT-3D 的 200 个 Objaverse 描述样本和 2468 个 ModelNet40 测试样本。

### 2.1 开放词汇分类 (Objaverse)

| 实验 | Prompt 0 ("What is this?") | Prompt 1 ("This is an object of ") | 备注 |
|------|---------------------------|-------------------------------------|------|
| **Baseline** (Point-BERT 官方) | **67.00%** | **66.00%** | 上限参考 |
| V1 hybrid (freeze) | 47.50% | 51.00% | 梯度修复后训练 |
| V1 hybrid **unfreeze** | **59.00%** | **59.00%** | 解冻后大幅提升 |
| V2 (freeze) | 39.00% | 44.00% | PointTransformer 架构 |
| V2 **unfreeze** | 52.50% | 52.50% | |
| Point-MAE (freeze) | 36.00% | 42.00% | 消融：无中心预测 |
| ShapeNet55-34 (freeze) | 49.50% | 50.50% | |
| ShapeNet55-34 **unfreeze** | 54.50% | 52.50% | |
| Random **unfreeze** | 52.50% | 52.50% | 无预训练，仅解冻训练 |

> **说明**：「V1 (bad loss)」——梯度阻断 Bug 期间训练的第一版 V1 权重，Prompt 0 仅 49%，不在上表中；上表 V1 均为梯度修复后的训练结果。旧版 `bad_pcpmae/` 目录下的结果为历史遗留。

### 2.2 ModelNet40 闭集分类

| 实验 | Prompt 0 (acc / clean) | Prompt 1 (acc / clean) | 备注 |
|------|------------------------|------------------------|------|
| **Baseline** (Point-BERT 官方) | **61.35%** / **63.83%** | **59.08%** / **61.06%** | 上限参考 |
| V1 hybrid (freeze) | 10.98% / 14.62% | 13.17% / 15.77% | |
| V1 hybrid **unfreeze** | ~13.29% / **18.41%** | ~13.41% / **17.25%** | |
| V2 (freeze) | 10.21% / 12.72% | 9.93% / 12.55% | |
| V2 **unfreeze** | 10.05% / 12.50% | 9.85% / 12.22% | |
| Point-MAE (freeze) | 11.10% / 14.14% | 12.20% / **15.71%** | |
| ShapeNet55-34 (freeze) | ~10.45% / 12.84% | ~10.70% / 12.89% | |
| ShapeNet55-34 **unfreeze** | 12.16% / **16.94%** | 12.03% / **15.65%** | |
| Random **unfreeze** | 10.45% / 13.67% | 9.16% / 10.74% | |

> ModelNet40 闭集分类是所有替代编码器与 Baseline 差距最大的任务（~10-13% vs ~61%），说明该任务高度依赖 ULIP-2 的多模态（图像-文本-点云）预训练语义对齐，纯几何自监督预训练难以弥补。

### 2.3 物体描述生成

| 实验 | Average Score | 备注 |
|------|---------------|------|
| **Baseline** (Point-BERT 官方) | **53.37** | 上限参考 |
| V1 hybrid (freeze) | 36.24 | |
| V1 hybrid **unfreeze** | **49.63** | 最接近 Baseline |
| V2 (freeze) | 25.56 | |
| V2 **unfreeze** | 43.94 | |
| Point-MAE (freeze) | 33.46 | |
| ShapeNet55-34 (freeze) | 29.19 | |
| ShapeNet55-34 **unfreeze** | 46.58 | |
| Random **unfreeze** | 44.45 | |

---

## 3. 编码器质量诊断

在接入 MiniGPT-3D 四阶段训练之前，使用 `point_model_VS_hybrid.py` 脚本对编码器权重进行特征空间诊断（基于 ModelNet40 测试集的 2468 个样本）。

### 3.1 V1 (point_model_pcpmae.pth) — 无 cls

```
=== Same-sample cosine (A vs B) ===
  cls         : mean=0.0541   (几乎正交)
  global      : mean=0.3917
  router      : mean=0.2711
  patch_mean  : mean=0.0026

=== kNN accuracy on ModelNet40 ===
  cls         : A=73.28%  B=60.93%  (gap=-12.35%)
  global      : A=79.76%  B=76.32%  (gap=-3.44%)
  router      : A=79.15%  B=72.27%  (gap=-6.88%)
```

**诊断**：缺少 `cls_token`/`cls_pos`，cls cosine ≈ 0.03 说明与 Baseline 的 cls 输出几乎正交。global 特征的 gap 最小（3.44%），说明编码器学到了部分几何结构。

### 3.2 V1 hybrid (point_model_hybrid.pth) — 拷贝 cls

```
=== Same-sample cosine (A vs B) ===
  cls         : mean=0.0345
  global      : mean=0.3916
  router      : mean=0.2617

=== kNN accuracy on ModelNet40 ===
  cls         : A=72.27%  B=62.35%  (gap=-9.92%)
  global      : A=80.77%  B=77.33%  (gap=-3.44%)

=== Intra-class vs Inter-class cosine ===
  Encoder A cls: intra=0.8465, inter=0.7307, margin=0.1158
  Encoder B cls: intra=0.8352, inter=0.7494, margin=0.0858
```

**诊断**：拷贝 cls 参数未能解决 cls 输出正交的问题（cosine 仍 ≈ 0.03）。这证明 cls 输出由整网 self-attention 决定，backbone 不同时同一 cls 参数产生完全不同的输出。margin 从 0.116 降至 0.086，说明类间可分性有所下降。

### 3.3 V2 (point_model_pcp_v2.pth) — PointTransformer

```
=== Same-sample cosine (A vs B) ===
  cls         : mean=0.0289
  global      : mean=0.4722
  router      : mean=0.2998

=== kNN accuracy on ModelNet40 ===
  cls         : A=73.48%  B=63.56%  (gap=-9.92%)
  global      : A=80.16%  B=74.49%  (gap=-5.67%)
  router      : A=78.95%  B=70.85%  (gap=-8.10%)

=== Intra-class vs Inter-class cosine ===
  Encoder B cls: intra=0.9162, inter=0.8293, margin=0.0869
```

**诊断**：V2 的 global cosine 高于 V1（0.47 vs 0.39），说明 PointTransformer 结构保留了一定的特征空间一致性。但 cls 空间几乎正交的问题仍然存在，且 margin（0.087）仍然偏低。

### 3.4 诊断结论

- 所有替代编码器与 Baseline 的 **cls 特征空间均几乎正交**（cosine ≤ 0.05），cls 受 backbone 整网影响，不是简单拷贝参数能对齐的。
- **global 特征是差距最小的**（kNN gap 仅 3-6%，cosine 0.39-0.47），说明自监督预训练确实学到了有判别力的全局几何特征。
- 替代编码器的 intra-inter margin 偏低（0.086 vs Baseline 0.116），类间可分性弱于 ULIP-2 多模态预训练，这直接解释了它们在 ModelNet40 闭集分类上的惨淡表现。

---

## 4. 关键发现与结论

### 4.1 主要发现

1. **替代编码器在冻结状态下与 Baseline 差距显著**：开放词汇分类上 V1 最高达到 51%（Baseline 67%），ModelNet40 分类上所有替代编码器仅 10-13%（Baseline ~61%），描述生成 V1 最高 36（Baseline 53）。

2. **解冻编码器训练（unfreeze）是缩小差距的关键**：V1 unfreeze 在开放词汇分类上达到 59%（接近 Baseline 67%），描述生成达到 49.63（接近 Baseline 53.37）。

3. **V1（MaskTransformer, cross-attention）优于 V2（PointTransformer, self-attention）**：在冻结和解冻两种设置下 V1 均全面优于 V2，说明 cross-attention 架构在预训练中能学到更丰富的表征。

4. **PCP-MAE 的中心预测任务提供了有效增益**：V2（PCP-MAE）vs Point-MAE（`ita=0`）在开放词汇分类上为 39% vs 36%，描述生成为 25.56 vs 33.46（Point-MAE 更优），但在 ModelNet40 上 Point-MAE 略有优势（14.14% vs 12.72% clean accuracy）。

5. **ShapeNet55-34 预训练效果与 Objaverse 可比**：尽管 ShapeNet55-34 数据量远小于 Objaverse（~50K vs ~660K），其预训练编码器在 unfreeze 后达到与 Objaverse V1 相近的性能（开放词汇分类 54.5% vs 59%），说明数据质量可能比数据量更重要。

6. **预训练权重 vs 随机权重的增益有限**：随机权重 unfreeze 在开放词汇分类上达到 52.50%，与 V2 unfreeze 相同，说明当前预训练策略带来的额外知识增益在解冻训练场景下不够显著。但在 V1 冻结场景下，预训练权重的增益是明确的（47.5% vs 难以训练）。

### 4.2 结论

- **纯几何自监督预训练（PCP-MAE / Point-MAE）可以作为 ULIP-2 多模态预训练的低成本替代方案**，但需要在下游任务中解冻编码器进行端到端微调。冻结状态下的性能与多模态预训练仍有显著差距。
- **V1（MaskTransformer + hybrid cls）是当前最佳的替代方案**，unfreeze 后在开放词汇分类和描述生成上最接近 Baseline。
- ModelNet40 闭集分类是所有替代编码器的软肋，这可能需要引入语言或视觉模态的预训练信号才能解决。

---

## 5. 关键技术问题

### 5.1 Q-Former 权重加载错误

首次完成 4 个 stage 训练后，模型输出乱码。根本原因是 `minigpt4/models/minigpt_v2.py` 中 Q-Former 权重加载路径被错误指向 TinyGPT-V 的不匹配权重。修复后模型恢复正常描述能力。

> 详细排查过程见 [`qformer_weight_fix.md`](./qformer_weight_fix.md)。

### 5.2 PCP-MAE 预训练中梯度无法回传

启用 AMP 后 loss1 长期卡在 1000 附近不下降。通过 git diff 定位到 `models/PCP_MAE.py` 中 Chamfer 损失计算被错误包裹在 `with torch.no_grad():` 块内，阻断了梯度回传。删除该块后 loss 恢复正常下降。

> 完整训练日志与修复过程见 [`pcp_mae_pretraining_log.md`](./pcp_mae_pretraining_log.md)。

### 5.3 AMP 数值不稳定性

训练后期出现 `NaN or Inf` 错误，原因是 AMP (float16) 与 Transformer 操作不兼容。通过切换至 bfloat16、强制 Chamfer 计算在 FP32 下执行、增加梯度裁剪等措施解决。

### 5.4 Qwen API 模型下线

原项目使用的 Qwen2-72B-Instruct API 已下线。在 `evaluator_opensource_llm_QwenAPI.py` 中适配了阿里云百炼 Qwen-Flash 模型，使评估可正常进行。

> 详细适配过程见 [`evaluation_script_adaptation.md`](./evaluation_script_adaptation.md)。

---

## 6. 仓库结构

```
├── README.md                         # 本文件 — 项目总览与评估结果汇总
├── environment_setup.md              # 服务器与环境搭建细节
├── qformer_weight_fix.md             # Q-Former 权重修复全过程
├── pcp_mae_pretraining_log.md        # PCP-MAE 预训练详细日志（V1/V2/Point-MAE/ShapeNet55-34）
├── evaluation_script_adaptation.md   # 评估脚本修改与 API 适配
├── random_weight.sh                  # 随机权重生成脚本
├── results/                          # 评估结果与训练日志
│   ├── MiniGPT-3D_evaluate/          # 各实验的评估 JSON（分目录）
│   │   ├── official/                 # Baseline（官方 Point-BERT）
│   │   ├── hybrid/                   # V1 hybrid（freeze）
│   │   ├── hybrid-with-objaverse_unfreeze/  # V1 unfreeze
│   │   ├── v2/                       # V2（freeze）
│   │   ├── v2_unfreeze/              # V2 unfreeze
│   │   ├── pointmae/                 # Point-MAE（freeze）
│   │   ├── ShapeNet/                 # ShapeNet55-34（freeze）
│   │   ├── ShapeNet_unfreeze/        # ShapeNet55-34 unfreeze
│   │   ├── random_unfreeze/          # Random unfreeze
│   │   ├── own/                      # 第一次成功复现
│   │   ├── bad_pcpmae/               # 梯度阻断期间的旧版 V1
│   │   └── bad/                      # Q-Former 权重错误期间的评估
│   └── PCP-MAE-train-log/            # PCP-MAE 各变体训练日志
│       ├── V1_objaverse/             # V1 训练日志
│       ├── V2_objaverse/             # V2 训练日志
│       ├── ShapeNet/                 # ShapeNet55-34 训练日志
│       └── pointmae/                 # Point-MAE 训练日志
└── LICENSE
```

### 相关代码仓库

| 仓库 | 说明 | GitHub |
|------|------|--------|
| MiniGPT-3D 复现修改版 | 下游训练与评测中心 | [SparkleAK47/MiniGPT-3D_Re_Encoder](https://github.com/SparkleAK47/MiniGPT-3D_Re_Encoder) |
| PCP-MAE（ShapeNet55-34 分支） | PCP-MAE + MaskTransformer，ShapeNet55-34 数据集 | [SparkleAK47/PCP_MAE_for_MiniGPT3D](https://github.com/SparkleAK47/PCP_MAE_for_MiniGPT3D) |
| PCP-MAE-with-Objaverse | PCP-MAE V1/V2/Point-MAE，Objaverse 数据集 | [SparkleAK47/PCP-MAE-f-M-3D-w-Objaverse](https://github.com/SparkleAK47/PCP-MAE-f-M-3D-w-Objaverse) |
| 实验日志 | 训练日志、评估结果等 | [SparkleAK47/PCP_MAE_4_MiniGPT3D_logbook](https://github.com/SparkleAK47/PCP_MAE_4_MiniGPT3D_logbook) |

---

## 7. 环境与协作

项目先后使用海南试验区公用电脑（RTX 4090，已停用）和自建服务器（最终使用 RTX 3090）。操作系统为 Ubuntu 24.04，通过 ZeroTier 构建虚拟局域网实现跨校区远程连接。权限通过 `3d_team` 用户组和 SSH 密钥统一管理。

> 详细环境搭建、conda 安装（micromamba）、远程访问配置请参考 [`environment_setup.md`](./environment_setup.md)。

---

## License

本仓库文档与脚本采用 MIT License。相关代码仓库的 License 见各自仓库。
