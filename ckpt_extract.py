import torch
from collections import OrderedDict

src = torch.load('/data/workspace/PCP-MAE/experiments/base/pretrain/pcpmae_pretrain/ckpt-last.pth', map_location='cpu')

# 提取模型参数
if 'base_model' in src:
    state_dict = src['base_model']
elif 'model' in src:
    state_dict = src['model']
elif 'state_dict' in src:
    state_dict = src['state_dict']
else:
    raise KeyError("No recognized model state key found in checkpoint")

new_state = OrderedDict()
for k, v in state_dict.items():
    # 去除多卡训练带来的 'module.' 前缀
    if k.startswith('module.'):
        k = k[7:]
    
    if k.startswith('MAE_encoder.'):
        k_body = k[len('MAE_encoder.'):]
        
        if k_body.startswith('encoder.'):
            new_state[k_body] = v
        elif k_body.startswith('reduce_dim.'):
            new_state[k_body] = v
        elif k_body.startswith('pos_embed.'):
            new_state[k_body] = v
        elif k_body.startswith('blocks.'):
            new_state[k_body] = v
        elif k_body.startswith('norm.'):
            new_state['norm.' + k_body.split('.', 1)[1]] = v

# 保存为 base_model 格式
torch.save({'base_model': new_state}, 'point_model_pcpmae.pth')
print('Saved with key "base_model", number of weights:', len(new_state))