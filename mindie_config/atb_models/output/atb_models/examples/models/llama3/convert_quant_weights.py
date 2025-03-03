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


from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

from atb_llm.models.llama.config_llama import LlamaConfig
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from examples.convert.model_slim.get_calibration_dataset import load_jsonl
from examples.convert.model_slim.quantifier import parse_arguments, Quantifier
from examples.convert.convert_utils import copy_tokenizer_files, modify_config





if __name__ == "__main__":
    args = parse_arguments()
    rank = ENV.rank
    model_path = args.model_path
    config = LlamaConfig.from_pretrained(model_path)
    config_path = os.path.join(model_path, "config.json")

    disable_names = []
    with file_utils.safe_open(config_path, 'r') as f:
        config_data = json.load(f)
        num_layers = config.num_hidden_layers
        #Llama3 70B 需要进行前5层回退
        if "torch_dtype" in config_data and config_data["torch_dtype"] == "bfloat16":
            disable_names = [f"model.layers.{layer}.mlp.down_proj" for layer in range(num_layers)]
            disable_names.append("lm_head")
        elif "hidden_size" in config_data and config_data["hidden_size"] == 8192:
            if config_data["max_position_embeddings"] == 131072:
                disable_names = [f"model.layers.{layer}.mlp.down_proj" for layer in range(num_layers)]
            else:
                disable_names.append("model.layers.0.mlp.down_proj")
                disable_names.append("model.layers.1.mlp.down_proj")
                disable_names.append("model.layers.2.mlp.down_proj")
                disable_names.append("model.layers.3.mlp.down_proj")
                disable_names.append("model.layers.4.mlp.down_proj")
            disable_names.append("model.layers.0.self_attn.q_proj")
            disable_names.append("model.layers.1.self_attn.q_proj")
            disable_names.append("model.layers.2.self_attn.q_proj")
            disable_names.append("model.layers.3.self_attn.q_proj")
            disable_names.append("model.layers.4.self_attn.q_proj")
            disable_names.append("model.layers.0.self_attn.k_proj")
            disable_names.append("model.layers.1.self_attn.k_proj")
            disable_names.append("model.layers.2.self_attn.k_proj")
            disable_names.append("model.layers.3.self_attn.k_proj")
            disable_names.append("model.layers.4.self_attn.k_proj")
            disable_names.append("model.layers.0.self_attn.v_proj")
            disable_names.append("model.layers.1.self_attn.v_proj")
            disable_names.append("model.layers.2.self_attn.v_proj")
            disable_names.append("model.layers.3.self_attn.v_proj")
            disable_names.append("model.layers.4.self_attn.v_proj")
            disable_names.append("model.layers.0.self_attn.o_proj")
            disable_names.append("model.layers.1.self_attn.o_proj")
            disable_names.append("model.layers.2.self_attn.o_proj")
            disable_names.append("model.layers.3.self_attn.o_proj")
            disable_names.append("model.layers.4.self_attn.o_proj")
            disable_names.append("model.layers.0.mlp.gate_proj")
            disable_names.append("model.layers.1.mlp.gate_proj")
            disable_names.append("model.layers.2.mlp.gate_proj")
            disable_names.append("model.layers.3.mlp.gate_proj")
            disable_names.append("model.layers.4.mlp.gate_proj")
            disable_names.append("model.layers.0.mlp.up_proj")
            disable_names.append("model.layers.1.mlp.up_proj")
            disable_names.append("model.layers.2.mlp.up_proj")
            disable_names.append("model.layers.3.mlp.up_proj")
            disable_names.append("model.layers.4.mlp.up_proj")
            disable_names.append("lm_head")
        elif args.a_bit != 16:
            disable_names = [f"model.layers.{layer}.mlp.down_proj" for layer in range(num_layers)]
            disable_names.append("lm_head")
     
    anti_outlier_config = None
    if args.anti_method == 'm3':
        anti_outlier_config = AntiOutlierConfig(a_bit=args.a_bit, w_bit=args.w_bit, 
            anti_method=args.anti_method, w_sym=args.w_sym,
            dev_type=args.device_type, dev_id=rank)
    elif args.anti_method:
        anti_outlier_config = AntiOutlierConfig(anti_method=args.anti_method)

    quant_config = QuantConfig(
        a_bit=args.a_bit,
        w_bit=args.w_bit,
        disable_names=disable_names,
        act_method=args.act_method,
        w_sym=args.w_sym,
        mm_tensor=False,
        dev_type=args.device_type,
        dev_id=rank,
        pr=1.0,
        fraction=args.fraction,
        co_sparse=args.co_sparse,
        do_smooth=args.do_smooth,
        use_sigma=args.use_sigma,
        sigma_factor=args.sigma_factor,
        is_lowbit=args.is_lowbit,
        use_kvcache_quant=args.use_kvcache_quant,
        open_outlier=args.open_outlier,
        group_size=args.group_size
    )
    if args.use_kvcache_quant:
        quant_config = quant_config.kv_quant(kv_sym=True)
    if args.use_reduce_quant:
        quant_config = quant_config.simulate_tp(tp_size=args.tp_size, \
            enable_communication_quant=True, enable_per_device_quant=False)

    # 默认无校准数据集
    calibration_dataset = None
    # 若存在calib_file，则使用calib_file作为校准数据集
    if args.calib_file:
        calibration_dataset = load_jsonl(args.calib_file)
    calibration_dataset = calibration_dataset
    quant_weight_generator = Quantifier(
        model_path, quant_config, anti_outlier_config,
        device_type=args.device_type, tokenizer_args={"padding_side": "left"}
    )
    quant_weight_generator.tokenizer.pad_token_id = 0

    tokenized_data = None
    if calibration_dataset is not None:
        tokenized_data = quant_weight_generator.get_tokenized_data(calibration_dataset)

    quant_weight_generator.convert(tokenized_data, args.save_directory, args.disable_level)
    #为适配工具稀疏量化传入w_bit=4,a_bit=8暂时修改quant_type
    quant_type = f"w{args.w_bit}a{args.a_bit}"
    is_sparseCompress = args.w_bit == 4 and args.a_bit == 8 and (args.co_sparse or args.is_lowbit)
    if is_sparseCompress:
        quant_type = "w8a8s"
    modify_config(
        model_path, args.save_directory, config.torch_dtype,
        quant_type, args
    )
    copy_tokenizer_files(model_path, args.save_directory)
