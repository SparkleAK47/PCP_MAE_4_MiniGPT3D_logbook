cd /data/workspace/MiniGPT-3D && /opt/miniconda3/envs/minigpt_3d/bin/python -c "
import sys
sys.path.insert(0, '.')
import torch
from collections import OrderedDict
from minigpt4.models.pointbert.point_encoder import PointTransformer
from easydict import EasyDict

config = EasyDict({
    'trans_dim': 384,
    'depth': 12,
    'drop_path_rate': 0.1,
    'cls_dim': 40,
    'num_heads': 6,
    'group_size': 32,
    'num_group': 512,
    'encoder_dims': 256,
    'point_dims': 6,
})

model = PointTransformer(config, use_max_pool=False)
state_dict = model.state_dict()

print(f'Total keys: {len(state_dict)}')
for k in sorted(state_dict.keys())[:10]:
    print(f'  {k}: shape={state_dict[k].shape}')
print('  ...')

# Save in same format as exported PCP checkpoint
out_path = 'params_weight/pc_encoder/point_model_random.pth'
torch.save({'base_model': state_dict}, out_path)
print(f'Saved -> {out_path}')
" 2>&1