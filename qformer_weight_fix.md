# Q-Former 权重加载错误定位与修复

在 MiniGPT-3D 复现过程中，完成 4 个 stage 的训练后，模型输出异常，本文件记录该问题的完整排查与修复过程。

## 1. 问题现象

- 运行 Gradio Demo 时，模型输出为乱码，无法理解点云内容。
- 评估脚本结果同样无效，物体描述无意义。
- 初步检查发现：
  - 对话模板中出现重复的 `[INST]` 标签。
  - 点云占位符 `<PC><PointCloudHere></PC>` 在传入 tokenizer 前未被替换为实际点云 token，而是作为普通文本被编码为 token ID。

## 2. 排查过程

### 2.1 排除 LLM 权重随机初始化
在 `pointllm/eval/eval_objaverse.py` 的模型初始化后插入验证代码，比较加载的模型参数中与官方 Phi-2 名称相同的参数。输出嵌入向量余弦相似度为 `1.000000`，确认 LLM 主体权重已成功加载，并非随机初始化。

### 2.2 检查占位符处理逻辑
在 `minigpt4/models/minigpt_base.py` 的 `generate` 函数中添加调试输出，打印 tokenized 后的文本。发现占位符未被处理，且对话模板存在重复标签。同时发现 `get_context_emb` 方法未能正确识别占位符，但其内部逻辑复杂，直接修改风险较高，转而怀疑上游特征提取出现问题。

### 2.3 比对当前代码与官方源码
- 使用 `git` 保存当前工作目录快照。
- 克隆官方 MiniGPT-3D 仓库至 `/data/workspace/MiniGPT-3D-clean`，确保处于 `origin/main` 一致状态。
- 执行 `diff -r MiniGPT-3D MiniGPT-3D-clean --exclude=".git"` 进行逐文件比对。

比对结果中找到影响模型性能的关键差异：
**文件**：`minigpt4/models/minigpt_v2.py`  
**Q-Former 权重加载路径**被修改。

```python
# 当前代码（错误）
url_or_filename="./params_weight/TinyGPT_V_stage_3/TinyGPT-V_for_Stage3.pth"

# 官方原代码
url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
```

## 3. 问题根因

- 当前代码指向了 TinyGPT-V 项目的权重文件，该权重与 MiniGPT-3D 所需的 Q-Former 不匹配，导致点云特征无法被正确提取。
- 点云占位符未替换、标签重复等异常，实质是 Q-Former 无法抽取出有效特征所引起的连锁反应，并非模板代码本身错误。

## 4. 修复过程

### 4.1 直接修改路径报错
将路径改回官方 URL 后，运行时报错：
```
EOFError: Ran out of input
```
原因是从 Google Storage 下载权重失败，生成的文件损坏。

### 4.2 本地下载权重
使用 `wget` 独立下载该权重文件：
```bash
wget -P /data/workspace/MiniGPT-3D/sfr-vision-language-research^LAVIS/ \
  "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
```
然后将 `url_or_filename` 改为本地路径：
```python
url_or_filename="/data/workspace/MiniGPT-3D/sfr-vision-language-research^LAVIS/blip2_pretrained_flant5xxl.pth"
```

### 4.3 验证
修改后重新加载模型并评估，模型能够正确描述 3D 物体，样例输出：
```json
{
    "object_id": "9d34f25cb4744435a087099c6d273d04",
    "ground_truth": "A gift box.",
    "model_output": "this is a 3d model of a cartoon-style, blue and white box."
}
```
确认 Q-Former 权重加载正常，模型功能恢复。

## 5. 附带说明

- 之前观察到的权重加载时的 `weights were not initialized` 警告，对于采用 LoRA 等微调技术的模型是常见且无害的。
- 由于首次训练时 Q-Former 权重错误，该阶段的评估结果无效。重新训练后方可得到有效评估数据。


