
# PCP-MAE 预训练详细日志

本文档记录 PCP-MAE 点云编码器在 **Objaverse 660K** 和 **ShapeNet55-34** 两个数据集上的完整预训练过程，涵盖 V1（MaskTransformer）、V2（PointTransformer）和 Point-MAE 消融三个变体。内容涉及环境编译、数据集适配、显存优化、关键梯度回传 Bug 的定位与修复，以及各变体的训练曲线与收敛情况。

最终目标：获得可替换 MiniGPT-3D 中原 Point-BERT 的编码器权重，并在下游四阶段训练中进行系统评估。

---

## 目录

- [1. 环境搭建](#1-环境搭建)
- [2. 数据集适配（Objaverse .npy）](#2-数据集适配objaverse-npy)
- [3. 显存优化](#3-显存优化)
- [4. V1：PCP-MAE + MaskTransformer（Objaverse）](#4-v1pcp-mae--masktransformerobjaverse)
  - [4.1 初次预训练与梯度阻断 Bug](#41-初次预训练与梯度阻断-bug)
  - [4.2 修复后重新预训练（hybrid-with-objaverse）](#42-修复后重新预训练hybrid-with-objaverse)
- [5. V2：PCP-MAE + PointTransformer（Objaverse）](#5-v2pcp-mae--pointtransformerobjaverse)
- [6. Point-MAE 消融（Objaverse）](#6-point-mae-消融objaverse)
- [7. ShapeNet55-34 预训练（对比实验）](#7-shapenet55-34-预训练对比实验)
- [8. 权重提取与集成](#8-权重提取与集成)
- [9. 预训练总结](#9-预训练总结)

---

## 1. 环境搭建

### 1.1 依赖版本
PCP-MAE 的 README 中依赖描述自相矛盾：一方面要求 PyTorch >=1.7.0 <1.11.0，另一方面给出的安装指令却指定 PyTorch 2.0.1。经实际尝试，按低版本安装会遇到大量兼容性问题，最终采用作者提供的指令：

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
```

### 1.2 CUDA Toolkit
conda-forge 缺少 `cudatoolkit=11.8`，使用 nvidia 官方 channel 安装：

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

### 1.3 扩展编译

#### chamfer_dist 依赖
安装时提示 `No module named 'pkg_resources'`，原因为 setuptools 版本过高，降级解决：

```bash
pip install 'setuptools<81'
```

#### PointNet++ 算子编译
在构建 `pointnet2_ops_lib` 时，即使当前 shell 可导入 torch，构建过程仍找不到 torch 模块。解决方法：

```bash
pip install -e . --no-build-isolation
```

#### 编译器兼容性
PCP-MAE 部分 C++ 扩展依赖较旧的 GCC 版本。系统中默认的 GCC 13 不兼容，降级为 GCC-10 和 G++-10：

```bash
export CC=gcc-10
export CXX=g++-10
```

同时，为避免 conda 自带的链接器与系统链接器冲突，临时移走 conda 的 `compiler_compat` 和 `bin/ld`：

```bash
mv $CONDA_PREFIX/compiler_compat $CONDA_PREFIX/compiler_compat.bak
mv $CONDA_PREFIX/bin/ld $CONDA_PREFIX/bin/ld_conda_backup
mv $CONDA_PREFIX/bin/ld.gold $CONDA_PREFIX/bin/ld.gold_backup 2>/dev/null
```

编译成功后恢复：
```bash
mv $CONDA_PREFIX/compiler_compat.bak $CONDA_PREFIX/compiler_compat
mv $CONDA_PREFIX/bin/ld_conda_backup $CONDA_PREFIX/bin/ld
```

#### 其他环境问题
- `pip install -r requirements.txt` 因 `open3d==0.9` 无法找到分发包而报错，暂时注释该依赖。
- conda 环境挂载点空间不足，将 `/opt/miniconda3` 迁移至 `/data/miniconda3` 并创建软链接。操作时需先移走目标目录，否则会自动创建错误链接。

### 1.4 模型配置对齐 MiniGPT-3D

为使 PCP-MAE 预训练的编码器能被 MiniGPT-3D 直接加载，需将 PCP-MAE 的模型配置强制对齐：
- 输入：8192 点 × 6 维（xyz+rgb）
- group_size=32, num_group=512
- trans_dim=384, depth=12, num_heads=6

修改文件：`/data/workspace/PCP-MAE/cfgs/pretrain/base.yaml`

同时将 MiniGPT-3D 的 PointTransformer 及其依赖完整复制到 PCP-MAE 的 `models/pointbert_mg/` 目录下（包括 `point_encoder.py`、`dvae.py`、`misc.py`、`logger.py`、`checkpoint.py`）。

---

## 2. 数据集适配（Objaverse .npy）

### 2.1 数据集划分
使用 MiniGPT-3D 提供的 Objaverse `.npy` 文件（共 661,575 个样本）。随机打乱后划分：
- 训练集：前 595,418 个
- 测试集：剩余 66,157 个

### 2.2 新增数据集类
在 `/data/workspace/PCP-MAE/datasets/` 下创建 `objaverse_npy.py`，实现从 `.npy` 文件读取点云并返回规范化的张量。

### 2.3 配置文件
新建 `cfgs/dataset_configs/ObjaverseNPY.yaml`，指定数据路径、点云点数（8192）、输入维度等参数。

### 2.4 导入逻辑修复
原项目 `datasets/__init__.py` 与 `datasets/build.py` 职责不清晰，导致自定义数据集无法正确注册。解决方法：
- `__init__.py` 负责导入子模块
- `build.py` 定义全局字典 `DATASETS` 和 `build_dataset_from_cfg` 函数

### 2.5 数据变换
修改 `datasets/data_transforms.py` 中的旋转变换，适配 Objaverse 点云的数值范围。

---

## 3. 显存优化

使用 RTX 3090（24 GB）训练时频繁发生 CUDA OOM。

### 措施
1. 大幅降低 batch size，并通过梯度累积维持等效总 batch size。
   - 初始配置：`bs: 2`, `total_bs: 64`, `step_per_update: 32`
2. 在 `PCP_MAE.py` 的 `MaskTransformer.forward` 和 `TransformerDecoder.forward` 中对特定循环使用 `torch.utils.checkpoint`，以时间换显存。
3. 后期启用 AMP（自动混合精度）后，逐步加大 batch size，V1 最终稳定参数为：
   - `bs: 8`, `total_bs: 64`, `step_per_update: 8`

### V2 / Point-MAE 显存优化
V2 使用与 MiniGPT-3D 一致的 PointTransformer（self-attention + cls token），显存消耗与 V1 相近。同样采用 bfloat16 AMP + 梯度检查点 + 梯度累积的组合方案，batch size 保持 `bs: 8, total_bs: 64, step_per_update: 8`。

---

## 4. V1：PCP-MAE + MaskTransformer（Objaverse）

V1 使用 PCP-MAE 原始的 MaskTransformer（cross-attention，visible + mask 双分支），预训练时无 cls token。配置文件为 `cfgs/pretrain/base.yaml`，`encoder_type=mask_transformer`。

### 4.1 初次预训练与梯度阻断 Bug

#### 早期训练快速下降
在最初未受干扰的训练中，loss 下降正常。日志 `20260501_100212.log` 显示 Epoch 0 时 loss 即出现明显下降：

```
[Epoch 0/300][Batch 1/9303] Losses = ['1135.8078', '26.3413'] lr = 0.000001
...
[Epoch 0/300][Batch 4881/9303] Losses = ['64.4053', '8.6042'] lr = 0.000001
```

此时 loss1 已从 1100+ 降至 64，表明模型学习正常。

#### 启用 AMP 后异常
为提高训练速度，启用 AMP（自动混合精度），并处理了 Chamfer 距离计算的报错。然而，后续训练日志（如 `20260501_160920.log`）中出现 loss1 停滞在约 1000 附近的现象，且后期出现 `NaN or Inf found in input tensor.` 错误。此阶段日志示例：

```
[Epoch 0/300][Batch 1/9303] Losses = ['1107.1138', '24.2080'] lr = 0.000001
...
[Epoch 33/300][Batch 1861/9303] Losses = ['994.6120', '1.6394'] lr = 0.004861
```

loss1 始终在 1000 左右波动，无法下降。当时误认为这是数据量过大、模型未充分收敛所致，后经对比实验发现该现象不合理（见第 7 节）。

#### 定位方法
后续使用 ShapeNet55-34 数据集预训练时，loss 下降正常，怀疑 Objaverse 训练中存在特定 Bug。通过 `git log --oneline` 查看快照历史，用 `git diff` 对比不同代码版本，定位到 `models/PCP_MAE.py` 中一处为处理 AMP 报错而引入的修改。

#### 问题代码
在 `forward` 方法中，Chamfer 重建损失被错误包裹在 `with torch.no_grad():` 块内：

```python
with torch.no_grad():
    rebuild_points_fp32 = rebuild_points.float()
    gt_coords_fp32 = gt_coords.float()
    loss1 = self.loss_func(rebuild_points_fp32, gt_coords_fp32)
```

该上下文管理器阻断了梯度从损失回传至编码器和解码器，导致网络无法学习点云重建，loss1 维持在高位。

#### 修复
删除 `with torch.no_grad():`，仅保留类型转换：

```python
rebuild_points_fp32 = rebuild_points.float()
gt_coords_fp32 = gt_coords.float()
loss1 = self.loss_func(rebuild_points_fp32, gt_coords_fp32)
```

### 4.2 修复后重新预训练（hybrid-with-objaverse）

#### 训练恢复正常
修复后重新启动 Objaverse 预训练，loss1 开始正常下降。日志 `20260515_154850.log` 显示：

```
[Epoch 0/300][Batch 1/9303] Losses = ['1095.4763', '24.3517'] lr = 0.000001
...
[Epoch 0/300][Batch 3201/9303] Losses = ['3.0734', '15.4092'] lr = 0.000001
```

在第一个 epoch 内，loss1 即从 1095 降至个位数。

#### 完整训练
从 2026-05-15 15:48 训练至 2026-05-17 09:16（约 41 小时），共计 24 个 epoch：

```
开始：  [Epoch 0/300][Batch 1/9303]      Losses = ['1095.4763', '24.3517'] lr = 0.000001
结束：  [Epoch 24/300][Batch 5581/9303]   Losses = ['0.5734', '1.4072']    lr = 0.004928
```

- Loss1：1095 → 0.57（Chamfer 重建损失，点云几何）
- Loss2：24.35 → 1.41（中心预测损失）

训练在第 24 个 epoch 时手动中断（Ctrl+C），loss 趋势表明仍有下降空间。该权重随后被导出为 `point_model_pcpmae.pth`（不含 cls_token/cls_pos），并通过 hybrid 合并 Baseline 的 cls 参数生成 `point_model_hybrid.pth`，用于 MiniGPT-3D 下游训练与评估。

---

## 5. V2：PCP-MAE + PointTransformer（Objaverse）

### 5.1 动机
V1 使用 MaskTransformer（cross-attention），与 MiniGPT-3D 下游使用的 PointTransformer 实现不同，导出的权重缺少 `cls_token`/`cls_pos`，需要 hybrid 合并。V2 直接在预训练中使用与 MiniGPT-3D 完全一致的 PointTransformer 结构，尝试解决该问题。

### 5.2 模型修改
在 `/data/workspace/PCP-MAE_with_Objaverse/models/PCP_MAE.py` 中新增 `PointTransformerMAEEncoder` 类，保留原 MaskTransformer 不动（便于对照）。修改 PCP_MAE 构造函数，通过 `encoder_type` 配置项切换。

配置文件：`cfgs/pretrain/base_minigpt_encoder.yaml`（`encoder_type: point_transformer`）

### 5.3 训练过程
训练启动于 2026-05-25，因故多次中断续训：

**第一段**（2026-05-25 14:33 → 2026-05-26 14:26，约 24 小时）：
```
开始：  [Epoch 0/300][Batch 1/9303]     Losses = ['815.3238', '34.5183'] lr = 0.000001
结束：  [Epoch 16/300][Batch 201/9303]  Losses = ['1.1559', '0.0025']    lr = 0.000099
```

**第二段**（2026-05-26 14:59 → 2026-05-27 10:41，约 20 小时）：
```
开始：  [Epoch 16/300][Batch 1/9303]     Losses = ['0.9576', '0.0023'] lr = 0.000099
结束：  [Epoch 29/300][Batch 2421/9303]  Losses = ['0.8027', '0.0088'] lr = 0.000098
```

**第三段**（2026-05-28 10:54 → 2026-05-28 15:26，约 4.5 小时）：
```
开始：  [Epoch 29/300][Batch 1/9303]    Losses = ['0.7592', '0.0183'] lr = 0.000098
结束：  [Epoch 32/300][Batch 81/9303]   Losses = ['0.8235', '0.0462'] lr = 0.000097
```

- Loss1：815 → 0.82（Chamfer 重建损失）
- Loss2：34.5 → 0.046（中心预测损失，收敛极低）

V2 的 Loss2（中心预测损失）收敛到远低于 V1（0.046 vs 1.4），这可能与 PointTransformer 的 self-attention 结构更适合该任务有关。但 Loss1 的收敛水平与 V1 相近（0.82 vs 0.57）。

### 5.4 权重导出
由于 V2 的预训练编码器与 MiniGPT-3D 下游使用的 PointTransformer 结构完全一致，checkpoint 中已包含 `cls_token`/`cls_pos`。使用专用导出脚本或通用提取脚本即可导出：

```bash
python tools/export_minigpt_encoder.py \
  --pcp-ckpt experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/ckpt-last.pth \
  --out point_model_pcp_v2.pth
```

导出后的权重可直接被 MiniGPT-3D 加载，无需 hybrid 合并。

---

## 6. Point-MAE 消融（Objaverse）

### 6.1 实验目的
PCP-MAE = Point-MAE + 中心预测（Center Prediction）。设置 `ita: 0` 可禁用中心预测分支，产生纯 Point-MAE 基线，用于衡量中心预测任务的增益。

### 6.2 配置
配置文件：`cfgs/pretrain/ablation_point_mae.yaml`（`encoder_type: point_transformer`, `ita: 0.0`）

除 `ita=0` 外，所有配置与 V2 完全相同。

### 6.3 训练
训练日志：`20260612_104057.log`

与 V2 类似，使用了 bfloat16 AMP + 梯度检查点 + 梯度累积的组合方案。由于 `ita=0` 使 Loss2 始终为 0，训练仅由 Chamfer 重建损失驱动。

训练稳定收敛，获得了 `point_model_pointmae.pth` 权重。

### 6.4 与 V2（PCP-MAE）的对比
在下游 MiniGPT-3D 评估中（均为冻结编码器）：

| 任务 | V2 (PCP-MAE) | Point-MAE (ita=0) | 差异 |
|------|-------------|-------------------|------|
| 开放词汇分类 Prompt 0 | 39.00% | 36.00% | +3.00% (PCP 更优) |
| 开放词汇分类 Prompt 1 | 44.00% | 42.00% | +2.00% (PCP 更优) |
| ModelNet40 Prompt 0 (clean) | 12.72% | 14.14% | -1.42% (Point-MAE 更优) |
| ModelNet40 Prompt 1 (clean) | 12.55% | 15.71% | -3.16% (Point-MAE 更优) |
| 描述生成 Avg Score | 25.56 | 33.46 | -7.90 (Point-MAE 更优) |

结果显示中心预测任务在开放词汇分类上有小幅增益，但在 ModelNet40 分类和描述生成上 Point-MAE 反而更优。这可能是由于 PCP 的中心预测任务在 Objaverse 数据量级下尚未充分训练（V2 仅训练了 32 epoch）。

---

## 7. ShapeNet55-34 预训练（对比实验）

### 7.1 实验目的
使用 ShapeNet55-34 数据集（仅 3 维 xyz，约 50K 样本）预训练 PCP-MAE（MaskTransformer），作为与 Objaverse 660K 数据集的对比实验。

该实验在 PCP-MAE 仓库的 `main` 分支进行（独立的 ShapeNet55-34 训练代码）。

### 7.2 显存优化
ShapeNet55-34 数据集样本量较小（约 50K），但单个点云大小与 Objaverse 相同（8192 点），同样遇到 OOM 问题。复用之前的优化方案：
- bfloat16 AMP
- 梯度检查点（Gradient Checkpointing）
- 梯度累积（`total_bs=32, step_per_update=4`）

### 7.3 NaN 问题与修复
训练至 Epoch 38 左右，出现与 V1 早期类似的 `NaN or Inf` 错误。分析原因：
- AMP (float16) 与 Transformer 的某些操作不兼容（大矩阵乘法、Softmax、LayerNorm 可能溢出）
- 梯度爆炸：即使 Chamfer 计算转为 FP32，解码器其他部分仍在 FP16，反向传播时梯度可能变成 Inf/NaN

应对措施：
- 将 AMP 数据类型从 `float16` 切换为 `bfloat16`（RTX 3090 支持），移除所有 `scaler` 相关代码
- 加强梯度裁剪（`torch.nn.utils.clip_grad_norm_`）
- 恢复 `pred_pos_proj` 的中间激活与归一化，为重建点云添加 `torch.clamp` 防止极端值

调整后训练稳定完成。

### 7.4 训练结果
```
开始：  [Epoch 0/300][Batch 1/3279]     Losses = ['752.1091', '36.9058'] lr = 0.000001
...
结束：  [Epoch 71/300][Batch 1761/3279]  Losses = ['0.8364', '4.2537']    lr = 0.000436
```

- Loss1：752 → 0.84（收敛良好）
- Loss2：36.9 → 4.25（中心预测损失，高于 Objaverse V1 的 1.41，可能与数据量较小有关）

### 7.5 关键洞察
ShapeNet55-34 上 loss 正常收敛（Loss1 降至 0.79），而此前 Objaverse 上 Loss1 长期卡在 1000 附近。这一对比反向证实了 Objaverse V1 训练中 `torch.no_grad()` 是导致异常的根因，而非数据集差异。

---

## 8. 权重提取与集成

### 8.1 V1 权重提取（MaskTransformer，无 cls）
从 `ckpt-last.pth` 中手动提取 `MAE_encoder` 的骨干权重（`encoder`, `reduce_dim`, `pos_embed`, `blocks`, `norm` 等），过滤掉 `pred_head`、`ita` 等预测头键。保存为 `point_model_pcpmae.pth`。

由于 V1 使用 MaskTransformer（cross-attention），预训练时无 `cls_token`/`cls_pos`。通过 hybrid 合并脚本从 Baseline 拷贝 cls 参数：

```python
new['base_model']['cls_token'] = original['base_model']['cls_token']
new['base_model']['cls_pos'] = original['base_model']['cls_pos']
```

生成 `point_model_hybrid.pth`。但诊断结果显示，拷贝 cls 参数无法使 cls 特征空间对齐（cosine 仍 ≈ 0.03），cls 输出由整网 self-attention 决定，backbone 不同时同一 cls 参数产生完全不同的输出。

### 8.2 V2 / Point-MAE 权重提取（PointTransformer，含 cls）
V2 和 Point-MAE 使用与 MiniGPT-3D 一致的 PointTransformer 结构，checkpoint 中已包含 `cls_token`/`cls_pos`。

使用通用提取脚本：
```bash
python ckpt_extract.py \
  --ckpt experiments/.../ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pcp_v2.pth
```

导出格式为 `{'base_model': {...}}`，包含所有 encoder 骨干键、`cls_token`、`cls_pos`，与 MiniGPT-3D 的 `load_checkpoint()` 直接兼容。

### 8.3 ShapeNet55-34 权重修复
ShapeNet 数据仅 3 维 xyz，而 MiniGPT-3D 期望 6 维 xyz+rgb。需使用 `fix_pcpmae_shapenet.py` 修复：
1. `encoder.first_conv.0.weight`：`[128, 3, 1]` → `[128, 6, 1]`（RGB 通道补零）
2. 补充 `cls_token` / `cls_pos`（随机初始化）

### 8.4 集成至 MiniGPT-3D
修改 `minigpt4/models/base_model.py` 中的 `init_pc_encoder` 方法，使其在初始化时加载对应的编码器权重，替换原有 Point-BERT 权重。后续进行 MiniGPT-3D 的四阶段训练与评估。

---

## 9. 预训练总结

### 各变体训练概况

| 变体 | 数据集 | 架构 | Epoch | Loss1 (始→终) | Loss2 (始→终) | 训练时间 |
|------|--------|------|-------|---------------|---------------|----------|
| V1 | Objaverse 660K | MaskTransformer | 24 (中断) | 1095 → 0.57 | 24 → 1.41 | ~41 小时 |
| V2 | Objaverse 660K | PointTransformer | 32 (中断) | 815 → 0.82 | 34.5 → 0.046 | ~48 小时 |
| Point-MAE | Objaverse 660K | PointTransformer | — | 收敛正常 | 恒为 0 (ita=0) | — |
| ShapeNet55-34 | ShapeNet55-34 | MaskTransformer | 71 (中断) | 752 → 0.84 | 36.9 → 4.25 | ~22 小时 |

### 关键教训
1. **AMP 与 no_grad 的交互**：为处理 AMP 报错而引入的 `torch.no_grad()` 阻断了 Chamfer 损失的梯度回传，是 V1 早期训练失败的根本原因。AMP 下 Chamfer 计算需转 FP32，但不能包裹在 no_grad 块中。
2. **bfloat16 优于 float16**：在 RTX 3090 上，bfloat16 比 float16 更稳定，有效避免了 Transformer 操作中的 NaN 问题。
3. **Objaverse 数据量巨大**：每个 epoch 约 2 小时（9303 batches），完整 300 epoch 预计需要 25 天。所有变体均未跑完完整的 300 epoch，下游评估基于提前中断的 checkpoint。
4. **V2 与 Point-MAE 的 Loss2 差异巨大**：V2 的 Loss2 收敛到 0.046，远低于 V1 的 1.41 和 ShapeNet 的 4.25，可能与 PointTransformer 的 self-attention 结构更适合中心预测任务有关。

---

> 下游 MiniGPT-3D 四阶段训练与评估的完整结果见 [README.md](./README.md)。
