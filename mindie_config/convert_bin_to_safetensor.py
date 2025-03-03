import json
import torch
from safetensors import safe_open

from safetensors.torch import save_file

# filename = '/mnt/nas_acr89/nanxiao/mm'
# tensors = {}
# for i in range(1000):
#     tensors[str(i)] = torch.rand(256,5120,dtype=torch.bfloat16,device='cuda:0')
# save_file(tensors, f"{filename}/embs.safetensors", metadata={'format': 'pt'})

# with safe_open(f"{filename}/embs.safetensors", framework="pt", device=0) as f:
#     for ok in f.keys():
#         pass

src_dir = '/mnt/eval/moe-lite/jingyue/moe_lite_0220_dpo_sub_qwen_judge_aug_0220len30-bailing'
dst_dir = '/tmp/moe_lite_0220_dpo_sub_qwen_judge_aug_0220len30-bailing-safetensor'
total_size = 0
sd = torch.load(f'{src_dir}/pytorch_model.bin',weights_only=True)
n_shard = 4
block_size = 8
weight_map = {}
for i in range(n_shard):
    ts = str(100000+n_shard)[1:]
    cs = str(100000+i+1)[1:]
    tensors = {}
    filename = f'model-{cs}-of-{ts}.safetensors'
    for k,v in sd.items():
        try:
            layer_idx = int(k.split('layers.')[1].split('.')[0])
            block_idx = layer_idx//block_size
        except:
            block_idx = n_shard-1
        if block_idx != i:
            continue 
        print(k,v.shape,v.dtype)
        weight_map[k] = filename 
        total_size += v.numel()*v.element_size()
        tensors[k] = v.contiguous()
    save_file(tensors, f"{dst_dir}/{filename}", metadata={'format': 'pt'})


meta = {
    "metadata": {
        "total_size": total_size
    },
    "weight_map": dict(sorted(weight_map.items(), key=lambda x:x[1]+x[0] ))
}


with open(f'{dst_dir}/model.safetensors.index.json', 'w') as f:
    json.dump(meta, f,indent=4)
