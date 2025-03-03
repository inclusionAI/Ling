# coding=utf-8
# Copyright 2023 The Bigcode team and HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 导入相关依赖
import os
import json
import torch
import torch.utils.data
from atb_llm.utils.file_utils import standardize_path, check_path_permission, safe_open, safe_readlines
from atb_llm.utils.log import logger
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained, safe_get_tokenizer_from_pretrained
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig # 导入量化配置接口
from examples.convert.model_slim.quantifier import parse_arguments
from examples.convert.convert_utils import copy_tokenizer_files, modify_config


#获取校准数据函数定义
def get_calib_dataset(_tokenizer, _calib_list):
    calib_dataset = []
    for calib_data in _calib_list:
        inputs = _tokenizer([calib_data], return_tensors='pt').to('cpu')
        logger.info(inputs)
        calib_dataset.append([inputs.data['input_ids'], None, inputs.data['attention_mask']])
    return calib_dataset


# 修改config.json中的model_type
def change_model_type(model_dir, model_type):
    config_file = os.path.join(model_dir, 'config.json')
    with safe_open(config_file, 'r', encoding='utf-8') as fr:
        config_data = json.load(fr)
    config_data['model_type'] = model_type
    with safe_open(config_file, "w", override_flags=os.O_WRONLY | os.O_CREAT, encoding='utf-8') as fw:
        fw.truncate()
        json.dump(config_data, fw, indent=4)

# for local path
args = parse_arguments()
model_path = standardize_path(args.model_path)
check_path_permission(model_path)
logger.info('changing model_type in config.json...')
change_model_type(model_path, 'gpt_bigcode')
logger.info('changing done!')
tokenizer = safe_get_tokenizer_from_pretrained(model_path)
model = safe_get_model_from_pretrained(model_path, torch_dtype=torch.float32)
logger.info("loading success!")
logger.info("start quant...")

# 准备校准数据，请根据实际情况修改
calib_list = []
with safe_open('humaneval_python.txt', 'r') as file:
    for line in safe_readlines(file):
        calib_list.append(line.strip())
#校准数据获取
dataset_calib = get_calib_dataset(tokenizer, calib_list) 

# 量化配置
# 配置回退层数
disable_names = [
    # "transformer.h.0.mlp.c_proj",
    # "transformer.h.1.attn.c_attn",
    # "transformer.h.1.mlp.c_fc",
    # "transformer.h.1.mlp.c_proj",
    # "transformer.h.2.attn.c_attn",
    # "transformer.h.2.mlp.c_proj",
    # "transformer.h.3.attn.c_attn",
    # "transformer.h.3.mlp.c_proj",
    # "transformer.h.4.attn.c_attn",
    # "transformer.h.4.mlp.c_proj",
    # "transformer.h.11.attn.c_attn",
    # "transformer.h.12.mlp.c_fc",
    # "transformer.h.13.mlp.c_fc",
    # "transformer.h.14.mlp.c_fc",
    # "transformer.h.15.mlp.c_fc",
    # "transformer.h.16.mlp.c_fc", 
    # "transformer.h.17.mlp.c_fc",
    # "transformer.h.18.mlp.c_fc",
    # "transformer.h.19.mlp.c_fc",
    # "transformer.h.20.mlp.c_fc",
    # "transformer.h.21.mlp.c_fc",
    # "transformer.h.39.attn.c_attn",
    # "transformer.h.39.mlp.c_fc",
    # "transformer.h.39.mlp.c_proj",
    # "lm_head"
]

# 配置量化参数，并返回量化配置实例
quant_config = QuantConfig(disable_names=disable_names, w_bit=8, dev_type='cpu', 
                            act_method=3, pr=1.0, mm_tensor=False)
# 输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')

#执行量化
calibrator.run()

if os.path.exists(args.save_directory):
    save_directory = standardize_path(args.save_directory)
    check_path_permission(save_directory)
else:
    os.makedirs(args.save_directory, exist_ok=True)
    save_directory = standardize_path(args.save_directory)
# save()保存模型量化参数
calibrator.save(save_directory, save_type=["safe_tensor"])
logger.info("quant weight saved successfully")

logger.info('changing back model_type in config.json')
change_model_type(model_path, 'starcoder')
logger.info('changing done!')
modify_config(model_path, save_directory, torch.float16, 'w8a8')
copy_tokenizer_files(model_path, save_directory)
logger.info('All done!')