# 评估脚本适配：Qwen API 模型替换

MiniGPT-3D 原项目使用 Qwen2-72B-Instruct API 对生成结果做开放式评估。在复现过程中该接口不再可用，需要对评估脚本进行修改以适配当前可调用的模型。

## 1. 问题

运行 `evaluator_opensource_llm_QwenAPI.py` 时，所有评估结果均为空：

```json
{
  "accuracy": "0.00%",
  "total_predictions": 0,
  "correct_predictions": 0,
  "invalid_responses": 0,
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "GPT_cost": 0.0,
  "results": []
}
```

检查后发现阿里云百炼平台（bailian.console.aliyun.com）已无法搜索到 `qwen2-72b-instruct`。该模型及其相关版本已下线（Qwen2.5-VL-32B-Instruct 也于 2026 年 5 月 13 日下线）。

## 2. 代码定位

在 `evaluator_opensource_llm_QwenAPI.py` 中发现多处硬编码：

```python
model_name = "qwen2-72b-instruct"
```

同时，定价参数也固定为原模型的标准，无法直接切换到其他模型。

## 3. 修改内容

- 将硬编码的模型名改为可配置参数，并补充当前可用的模型 ID（`qwen-flash`、`qwen-plus` 等）。
- 补充新模型的输入/输出 token 单价，使成本统计保持准确。
- 调整模型调用方式，使其能够动态适配不同模型（原脚本仅针对单一模型做了参数固定）。
- 由于替换后的模型（如 Qwen-Flash）指令遵循能力有差异，在部分 prompt 模板中添加了更明确的格式约束，以降低无效输出比例。

## 4. 验证结果（中间快照）

修改后使用**修复了 Q-Former 权重但尚未重新训练**的模型进行了传统评估和开源模型评估，验证 API 适配正常工作。

> **注意**：以下数值来自中间快照，不代表最终实验结果。完整评估结果见 [README.md](./README.md)。

### 4.1 传统评估（与论文基本一致）

- 物体描述（Objaverse Captioning）：BLEU-4 0.5122，METEOR 14.4264，SBERT 相似度 48.78
- 开放词汇分类（Objaverse, Prompt "What is this?"）：BLEU-4 1.1807，METEOR 16.2078，SBERT 53.11

传统评估不依赖外部 LLM，结果直接由指标计算得出，与官方参考值无明显偏差。

### 4.2 开源模型评估（Qwen-Flash）

开放词汇分类主观评估（Prompt 0）：
- Accuracy: 66.00%
- Total predictions: 200
- Cost: 0.022 CNY

ModelNet40 闭集分类主观评估（Prompt 0）：
- Accuracy: 60.82%
- Clean accuracy: 62.66%
- Total predictions: 2468
- Cost: 0.51 CNY

物体描述生成主观评估：
- Average score: 53.57
- Total predictions: 200
- Cost: 0.022 CNY

评估过程没有出现空结果，说明 API 适配正常。

## 5. 后续使用

API 适配完成后，Qwen-Flash 被用于所有后续实验的主观评估，包括：
- Baseline（官方 Point-BERT）的完整评估
- V1 / V1 hybrid / V1 unfreeze
- V2 / V2 unfreeze
- Point-MAE 消融
- ShapeNet55-34 / ShapeNet55-34 unfreeze
- Random unfreeze 消融

所有评估均使用相同的 `--model_type qwen-flash` 参数，确保可比性。

> 完整评估数据见 [README.md](./README.md) 及 `results/MiniGPT-3D_evaluate/` 目录下的各实验子目录。
