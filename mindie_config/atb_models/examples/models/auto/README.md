# README
- AutoRunner提供统一API，实现模型NPU迁移
## 目标
```python
from examples.models.auto.runner_auto import AutoRunner
inputs = [{'text':''},{'image':''}] #user_input
runner = AutoRunner.from_pretrained(model_name_or_path,**kwargs)
output = model.infer(inputs,**kwargs)
```
## 说明
- AutoRunner支持LLAMA、CLIP等模型
- model_name_or_path为本地模型路径，暂不支持云端下载
- 不同模型runner初始化参数以及infer参数可能不一致，关于样例中runner_args以及infer_params的详细配置参考models文件夹下不同模型的使用说明
- 将使用样例中代码复制到python文件中运行即可，如需脚本运行，参考models文件夹下不同模型的脚本环境变量配置
## 使用样例
- LLAMA
```python
from examples.models.auto.runner_auto import AutoRunner
import os

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
batch_size = 8

runner_args = {
    "model_name":"llama",
    "rank": rank,
    "world_size": world_size,
    "local_rank": local_rank,
    "max_batch_size":batch_size,
    "max_input_length":1024,
    "max_output_length":20,
    "max_prefill_tokens":-1,
    "max_batch_size":1,
    "block_size":128,
    "is_flash_model":True,
}

model_path = "/data/datasets/llama2-7b/"
runner = AutoRunner.from_pretrained(model_path, **runner_args)
runner.warm_up()
infer_params = {
    "inputs": ["What's deep learning?"],
    "batch_size": 1,
    "max_output_length": 65,
    "ignore_eos": True,
    "is_chat_model": False
}
generate_texts, token_nums, _ = runner.infer(**infer_params)
print(generate_texts)
```

- CLIP
```python
from examples.models.auto.runner_auto import AutoRunner
import os

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
batch_size = 8

runner_args = {
    "model_name":"clip",
    "rank": rank,
    "world_size": world_size,
    "local_rank": local_rank,
    "max_batch_size":batch_size
}

model_path = "/data/datasets/chinese-clip-vit-base-patch16/"
runner = AutoRunner.from_pretrained(model_path, **runner_args)

label_texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
mm_inputs = [[{"image": ["./examples/models/clip/pokemon.jpeg"]}, {"text": label_texts}]]
probs, labels, time = runner.infer(mm_inputs,batch_size)
print(labels)

```
- QWEN_VL

```python
from examples.models.auto.runner_auto import AutoRunner
import os

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
batch_size = 8

runner_args = {
    "model_name":"qwen_vl",
    "rank": rank,
    "world_size": world_size,
    "local_rank": local_rank,
    "max_batch_size":batch_size,
    "max_input_length":1024,
    "max_output_length":65,
    "max_prefill_tokens":-1,
    "max_batch_size":1,
    "block_size":128,
    "is_flash_model":True,
}

model_path = "/data/datasets/qwen-vl/"
runner = AutoRunner.from_pretrained(model_path, **runner_args)
mm_inputs = [[{"image": "./examples/models/clip/pokemon.jpeg"}, {"text": "Generate the caption in English with grounding:"}]]

runner.warm_up()
generate_text_list, token_num_list, e2e_time = runner.infer(mm_inputs,65)

```

- LLAVA
```python
from examples.models.auto.runner_auto import AutoRunner
import os
from dataclasses import dataclass
from typing import List

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
batch_size = 8

runner_args = {
    "model_name":"llava",
    "rank": rank,
    "world_size": world_size,
    "local_rank": local_rank,
    "max_batch_size":batch_size,
    "max_input_length":1024,
    "max_output_length":65,
    "max_prefill_tokens":-1,
    "max_batch_size":1,
    "block_size":128,
    "is_flash_model":True,
}

@dataclass
class InputAttrs:
    input_texts_for_image:List | None
    input_texts_for_video:List | None
    image_or_video_path:str | None

infer_params = {
    "inputs": InputAttrs(["USER: <image>\nDescribe this image in detail. ASSISTANT:"],
                         ["USER: <video>\nDescribe this video in detail. ASSISTANT:"],
                        "./test_images_dir/"),
    "batch_size": 1,
    "max_output_length": 65,
    "ignore_eos": True,
}


runner.warm_up()
generate_text_list, token_num_list, e2e_time = runner.infer(**infer_params)

```