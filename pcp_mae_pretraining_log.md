
# PCP‑MAE 预训练详细日志

本文档记录 PCP‑MAE 点云编码器的预训练过程，涵盖环境编译、数据集适配、显存优化、关键梯度回传 Bug 的定位与修复，以及最终的预训练完成。预训练旨在获得可替换 MiniGPT‑3D 中原 Point‑BERT 的编码器权重。

## 1. 环境搭建

### 1.1 依赖版本
PCP‑MAE 的 README 中依赖描述自相矛盾：一方面要求 PyTorch >=1.7.0 <1.11.0，另一方面给出的安装指令却指定 PyTorch 2.0.1。经实际尝试，按低版本安装会遇到大量兼容性问题，最终采用作者提供的指令：

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
```

### 1.2 CUDA Toolkit
conda‑forge 缺少 `cudatoolkit=11.8`，使用 nvidia 官方 channel 安装：

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
PCP‑MAE 部分 C++ 扩展依赖较旧的 GCC 版本。系统中默认的 GCC 13 不兼容，降级为 GCC‑10 和 G++‑10：

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

## 2. 数据集适配（Objaverse .npy）

### 2.1 数据集划分
使用 MiniGPT‑3D 提供的 Objaverse `.npy` 文件（共 661,575 个样本）。随机打乱后划分：
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

## 3. 显存优化

使用 RTX 3090（24 GB）训练时频繁发生 CUDA OOM。

### 措施
1. 大幅降低 batch size，并通过梯度累积维持等效总 batch size。
   - 配置：`bs: 2`, `total_bs: 64`, `step_per_update: 32`
2. 在 `PCP_MAE.py` 的 `MaskTransformer.forward` 和 `TransformerDecoder.forward` 中对特定循环使用 `torch.utils.checkpoint`，以时间换显存。
3. 后期启用 AMP（自动混合精度）后，逐步加大 batch size，最终稳定参数为：
   - `bs: 8`, `total_bs: 64`, `step_per_update: 8`

## 4. 初次 Objaverse 预训练（梯度阻断前）

### 4.1 早期训练快速下降
在最初未受干扰的训练中，loss 下降正常。日志 `20260501_100212.log` 显示 Epoch 0 时 loss 即出现明显下降：

```
[Epoch 0/300][Batch 1/9303] Losses = ['1135.8078', '26.3413'] lr = 0.000001
...
[Epoch 0/300][Batch 4881/9303] Losses = ['64.4053', '8.6042'] lr = 0.000001
```

此时 loss1 已从 1100+ 降至 64，表明模型学习正常。

### 4.2 启用 AMP 后异常
为提高训练速度，启用 AMP（自动混合精度），并处理了 Chamfer 距离计算的报错。然而，后续训练日志（如 `20260501_160920.log`）中出现 loss1 停滞在约 1000 附近的现象，且后期出现 `NaN or Inf found in input tensor.` 错误。此阶段日志示例：

```
[Epoch 0/300][Batch 1/9303] Losses = ['1107.1138', '24.2080'] lr = 0.000001
...
[Epoch 33/300][Batch 1861/9303] Losses = ['994.6120', '1.6394'] lr = 0.004861
```

loss1 始终在 1000 左右波动，无法下降。当时误认为这是数据量过大、模型未充分收敛所致，后经对比实验发现该现象不合理（见第 6 节）。

## 5. 关键 Bug：梯度无法回传

### 5.1 定位方法
后续使用 ShapeNet55‑34 数据集预训练时，loss 下降正常（见第 6 节），怀疑 Objaverse 训练中存在特定 Bug。通过 `git log --oneline` 查看快照历史，用 `git diff` 对比不同代码版本，定位到 `models/PCP_MAE.py` 中一处为处理 AMP 报错而引入的修改。

### 5.2 问题代码
在 `forward` 方法中，Chamfer 重建损失被错误包裹在 `with torch.no_grad():` 块内：

```python
with torch.no_grad():
    rebuild_points_fp32 = rebuild_points.float()
    gt_coords_fp32 = gt_coords.float()
    loss1 = self.loss_func(rebuild_points_fp32, gt_coords_fp32)
```

该上下文管理器阻断了梯度从损失回传至编码器和解码器，导致网络无法学习点云重建，loss1 维持在高位。

### 5.3 修复
删除 `with torch.no_grad():`，仅保留类型转换：

```python
rebuild_points_fp32 = rebuild_points.float()
gt_coords_fp32 = gt_coords.float()
loss1 = self.loss_func(rebuild_points_fp32, gt_coords_fp32)
```

## 6. 修复后重新预训练（Objaverse）

### 6.1 训练恢复正常
修复后重新启动 Objaverse 预训练，loss1 开始正常下降。日志 `20260515_154850.log` 显示：

```
[Epoch 0/300][Batch 3201/9303] Losses = ['3.0734', '15.4092'] lr = 0.000001
```

在第一个 epoch 内，loss1 即从 1095 降至个位数，后续继续稳定下降。


## 7. ShapeNet55‑34 预训练（对比实验）

为排除数据集差异对 loss 收敛的影响，同时进行 ShapeNet55‑34 数据集的预训练。

### 7.1 显存优化复用
同样遇到 OOM，复用之前的优化方案：AMP + bfloat16、梯度检查点、梯度累积。

### 7.2 NaN报错
修复后训练至 Epoch 15 左右，同样出现 `NaN or Inf`。此时模型检查点本身不含 NaN/Inf，数据样本亦无异常。分析原因为 AMP (float16) 与 Transformer 的数值不稳定，即便 Chamfer 计算已转 FP32，解码器其他部分仍可能产生梯度溢出。

应对措施：
- 将 AMP 数据类型从 `float16` 切换为 `bfloat16`（RTX 3090 支持），移除所有 `scaler` 相关代码。
- 加强梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)。
- 恢复 `pred_pos_proj` 的中间激活与归一化，为重建点云添加 `torch.clamp` 防止极端值。

调整后训练稳定完成，获得 Objaverse 预训练权重 `point_model_pcpmae.pth`。

### 7.2 训练中……
ShapeNet55‑34 上 loss 正常收敛：
- Loss1 从初始的 771.9 降至 Epoch 38 的 0.79
- Loss2 从 37.27 降至 5.91

该结果反向证实了此前 Objaverse 训练中 loss 停滞在 1000 是异常，也验证了 `torch.no_grad()` 为根本原因。

## 8. 权重提取与集成

### 8.1 权重转换脚本
编写 `ckpt_extract.py`，遍历检查点文件，过滤所有键名中包含 `pred_head` 或 `ita` 的条目，仅保留骨干网络权重。提取后的权重保存为 `point_model_pcpmae.pth`，其结构与 MiniGPT‑3D 的点云编码器 `PointTransformer` 相匹配。

### 8.2 集成至 MiniGPT‑3D
修改 `minigpt4/models/base_model.py` 中的 `init_pc_encoder` 方法，使其在初始化时加载 `point_model_pcpmae.pth`，替换原有 Point‑BERT 权重。后续将进行 MiniGPT‑3D 的 4 个 stage 微调与评估。

---

> **进行中**：使用修复后的 Objaverse 预训练权重以及 ShapeNet55‑34 预训练权重，分别进行 MiniGPT‑3D 下游任务评估。详见 [下游任务评估计划](./downstream_evaluation_plan.md)。
