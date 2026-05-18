
# MiniGPT-3D 复现与编码器替换实验

本项目是对 MiniGPT-3D 的完整复现，并尝试使用 PCP‑MAE 替换其点云编码器（Point‑BERT）以提升性能。核心工作包括：**Q‑Former 权重修复**、**PCP‑MAE 预训练与集成**、**多任务评估**。

> **当前状态**：多条任务线并行推进中，详见 [进行中任务](#进行中任务)。

## 目录
- [关键技术问题与解决](#关键技术问题与解决)
  - [Q-Former 权重加载错误](#q-former-权重加载错误)
  - [PCP‑MAE 预训练中梯度无法回传](#pcp-mae-预训练中梯度无法回传)
  - [Objaverse 数据集 NaN 问题](#objaverse-数据集-nan-问题)
- [环境与协作 (简要)](#环境与协作)
- [评估结果 (节选)](#评估结果)
- [进行中任务](#进行中任务)
- [仓库结构](#仓库结构)

---

## 关键技术问题与解决

### Q-Former 权重加载错误

**现象**：首次完成 4 个 stage 训练后，模型输出乱码，点云占位符 `<PC><PointCloudHere></PC>` 未被替换，对话模板出现重复 `[INST]` 标签。

**定位过程**：
1. 验证 LLM 主体权重：通过比较加载后的参数与官方 Phi‑2 权重的余弦相似度（结果为 1.0），排除随机初始化。
2. 怀疑占位符处理逻辑，在 `minigpt4/models/minigpt_base.py` 的 `generate` 函数中添加调试代码，发现占位符作为普通文本被 tokenizer 编码。
3. 重新克隆官方仓库至 `MiniGPT-3D-clean`，使用 `diff -r MiniGPT-3D MiniGPT-3D-clean --exclude=".git"` 比对当前代码与原始代码。

**根本原因**：  
`minigpt4/models/minigpt_v2.py` 中 Q‑Former 权重加载路径被错误修改：
```python
# 当前错误路径（指向不匹配的 TinyGPT-V 权重）
url_or_filename="./params_weight/TinyGPT_V_stage_3/TinyGPT-V_for_Stage3.pth"

# 官方正确路径
url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
```

**解决**：  
直接使用官方 URL 时因网络问题加载失败（`EOFError`），改用 `wget` 下载至本地 `/data/workspace/MiniGPT-3D/sfr-vision-language-research^LAVIS/blip2_pretrained_flant5xxl.pth`，并修改代码指向本地文件。重新评估后模型描述恢复正常（见 `results/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2_evaluated_qwen-flash.json`等等）。

> 详细排查过程见 [`qformer_weight_fix.md`](./qformer_weight_fix.md)。

---

### PCP‑MAE 预训练中梯度无法回传

**背景**：  
为用 PCP‑MAE 替代 Point‑BERT，需在 Objaverse 数据集上预训练 PCP‑MAE 编码器。训练初期 loss 快速下降（epoch 0 时 loss1 从 1100 降至 64），但启用 AMP（自动混合精度）并处理相关报错后，loss1 长期卡在 1000 附近不下降，后期出现 NaN。

**定位方法**：  
通过 `git log --oneline` 查看快照，并使用 `git diff` 对比出现性能退化的前后版本代码。发现 `models/PCP_MAE.py` 的 `forward` 方法中 Chamfer 损失计算被错误包裹在 `with torch.no_grad():` 块内：

```python
with torch.no_grad():
    rebuild_points_fp32 = rebuild_points.float()
    gt_coords_fp32 = gt_coords.float()
    loss1 = self.loss_func(rebuild_points_fp32, gt_coords_fp32)
```

该块阻断了梯度从损失回传至编码器和解码器，导致模型无法学习点云重建。

**解决**：  
删除 `with torch.no_grad():`，仅保留类型转换。修改后 loss1 恢复正常下降趋势。

> 完整训练日志与修复过程见 [`pcp_mae_pretraining_log.md`](./pcp_mae_pretraining_log.md)。

---

### Objaverse 和 ShapeNet 数据集 NaN 问题

修正梯度回传后，训练至 epoch 15 左右再次出现 `NaN or Inf`。分析原因为：
- AMP (float16) 与 Transformer 的某些操作不兼容（大矩阵乘法、Softmax 可能溢出）。
- 梯度爆炸，即使 Chamfer 计算转为 FP32，解码器其他部分仍在 FP16 反向传播。

**应对措施**：
- 将 AMP 数据类型切换为 `torch.bfloat16`（RTX 3090 支持），代码中所有 `scaler` 相关行同步修改。
- 强制 Chamfer 距离计算始终在 FP32 下执行。
- 增加梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)。

调整后预训练顺利完成，得到 `point_model_pcpmae.pth`。

---

## 环境与协作

项目先后使用试验区公用电脑（RTX 4090）和自建服务器（最终使用 RTX 3090）。操作系统为 Ubuntu 24.04，通过 ZeroTier 构建虚拟局域网实现跨校区远程连接。权限通过 `3d_team` 用户组和 SSH 密钥统一管理。  
详细环境搭建、conda 安装（micromamba）、远程访问配置等请参考 [`environment_setup.md`](./environment_setup.md)。

---

## 评估结果 (节选)

| 任务 | 指标 | 原编码器 (Point-BERT) | PCP‑MAE 编码器 (初步) |
|------|------|------------------------|------------------------|
| 开放词汇分类 (Objaverse) | Accuracy (GPT‑Eval) | 67.0% (Prompt 0) | 49.0% (← 梯度修复前训练) |
| ModelNet40 闭集分类 | Accuracy (GPT‑Eval) | 61.35% (Prompt 0) | 11.91% (同上) |
| 物体描述生成 | Avg Score (GPT‑Eval) | 53.57 | 40.08 (同上) |

*注：PCP‑MAE 结果对应梯度回传修复前、仅使用 Objaverse 数据集预训练 33 个 epoch 的权重。重新预训练后的评估正在进行中。*

> 完整评估数据与脚本修改见 [`evaluation_script_adaptation.md`](./evaluation_script_adaptation.md) 及 `results/` 目录。

---

## 进行中任务

以下任务并行开展，本文档将随进展更新：

- **重新预训练（梯度修复后）→ MiniGPT‑3D 点云描述任务评估**  
  状态：已完成预训练，下游 4 个 stage 重新训练中。
- **使用 ShapeNet55‑34 数据集预训练 PCP‑MAE → 点云描述任务评估**  
  状态：ShapeNet55‑34 预训练已完成（loss 收敛正常），已启动 MiniGPT‑3D 训练。
- **多种下游任务公平对比**  
  计划对比编码器：  
  - 官方 Point‑BERT 权重  
  - 我们预训练的 PCP‑MAE（Objaverse）  
  - 我们预训练的 PCP‑MAE（ShapeNet55‑34）  
  评估项目：物体分类、少样本分类、点云描述等。  
  状态：正在搭建 Point‑BERT 原项目评估环境（[`downstream_evaluation_plan.md`](./downstream_evaluation_plan.md)）。

---

## 仓库结构

```
├── README.md                  # 本文件
├── environment_setup.md       # 服务器与环境搭建细节
├── qformer_weight_fix.md      # Q-Former 权重修复全过程
├── evaluation_script_adaptation.md  # 评估脚本修改与结果记录
├── pcp_mae_pretraining_log.md # PCP‑MAE 预训练详细日志
├── downstream_evaluation_plan.md # 下游任务评估计划 undone
├── results/                   # 评估结果 JSON、图表等
│   └── ...
```

相关代码仓库：
- MiniGPT-3D 复现修改版：[SparkleAK47/MiniGPT-3D_Re_Encoder](https://github.com/SparkleAK47/MiniGPT-3D_Re_Encoder)
- PCP‑MAE 适配 ShapeNet55‑34：[SparkleAK47/PCP_MAE_for_MiniGPT3D](https://github.com/SparkleAK47/PCP_MAE_for_MiniGPT3D)
- PCP‑MAE 适配 Objaverse：[SparkleAK47/PCP-MAE-f-M-3D-w-Objaverse](https://github.com/SparkleAK47/PCP-MAE-f-M-3D-w-Objaverse)


