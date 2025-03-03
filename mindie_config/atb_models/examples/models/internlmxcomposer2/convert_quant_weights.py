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
import shutil


from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

from atb_llm.models.base.model_utils import safe_get_model_from_pretrained, safe_get_config_from_pretrained
from atb_llm.utils import file_utils
from atb_llm.utils.env import ENV
from examples.convert.model_slim.get_calibration_dataset import load_jsonl
from examples.convert.model_slim.quantifier import parse_arguments, Quantifier
from examples.convert.convert_utils import modify_config


MAX_TOKENIZER_FILE_SIZE = 1024 * 1024 * 1024
SIXTEEN = 16


def need_saved_file(file_name):
    if 'tokenizer' in file_name:
        return file_name
    if 'tokenization' in file_name:
        return file_name
    if 'special_token_map' in file_name:
        return file_name
    if 'generation' in file_name:
        return file_name
    if 'configuration' in file_name:
        return file_name
    return False


def copy_process_files(model_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for filename in file_utils.safe_listdir(model_dir):
        if need_saved_file(filename):
            src_filepath = os.path.join(model_dir, filename)
            src_filepath = file_utils.standardize_path(src_filepath)
            file_utils.check_file_safety(src_filepath, 'r', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            dest_filepath = os.path.join(dest_dir, filename)
            dest_filepath = file_utils.standardize_path(dest_filepath)
            file_utils.check_file_safety(dest_filepath, 'w', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            shutil.copyfile(src_filepath, dest_filepath)


def get_disable_names(a_bit, model_path, num_hidden_layers, trust_remote_code):
    disable_names = []
    if a_bit != SIXTEEN:
        # W8A16, W4A16没有回退层
        num_layers = num_hidden_layers
        disable_names = [f"model.layers.{layer}.feed_forward.w2" for layer in range(num_layers)]
        disable_names.append("output")
    
    model = safe_get_model_from_pretrained(
        model_path, trust_remote_code=trust_remote_code, device_map="cpu"
    )
    # 回退lora
    for name, _ in model.named_modules():
        if "plora_" in name.lower():
            disable_names.append(name)
    
    disable_names.append("vision_proj.0")
    disable_names.append("vision_proj.2")
    disable_names.append("vit.vision_tower.vision_model.embeddings.patch_embedding")
    vit_num_layers = 24
    disable_names_kproj = [f"vit.vision_tower.vision_model.encoder.layers.{layer}.self_attn.k_proj" 
                            for layer in range(vit_num_layers)]
    disable_names_qproj = [f"vit.vision_tower.vision_model.encoder.layers.{layer}.self_attn.q_proj" 
                            for layer in range(vit_num_layers)]
    disable_names_vproj = [f"vit.vision_tower.vision_model.encoder.layers.{layer}.self_attn.v_proj"
                            for layer in range(vit_num_layers)]
    disable_names_oproj = [f"vit.vision_tower.vision_model.encoder.layers.{layer}.self_attn.out_proj" 
                            for layer in range(vit_num_layers)]
    disable_names_fc1 = [f"vit.vision_tower.vision_model.encoder.layers.{layer}.mlp.fc1"
                            for layer in range(vit_num_layers)]
    disable_names_fc2 = [f"vit.vision_tower.vision_model.encoder.layers.{layer}.mlp.fc2"
                            for layer in range(vit_num_layers)]
    disable_names.extend(disable_names_kproj)
    disable_names.extend(disable_names_qproj)
    disable_names.extend(disable_names_vproj)
    disable_names.extend(disable_names_oproj)
    disable_names.extend(disable_names_fc1)
    disable_names.extend(disable_names_fc2)
    return disable_names

if __name__ == "__main__":
    args = parse_arguments()
    rank = ENV.rank
    config = safe_get_config_from_pretrained(args.model_path)

    disable_names_without_lora = get_disable_names(args.a_bit,
                                                   args.model_path,
                                                   config.num_hidden_layers,
                                                   args.trust_remote_code)
    anti_outlier_config = None
    if args.anti_method == 'm3':
        anti_outlier_config = AntiOutlierConfig(a_bit=args.a_bit, w_bit=args.w_bit, 
            anti_method=args.anti_method, w_sym=args.w_sym, dev_type=args.device_type)
    elif args.anti_method:
        anti_outlier_config = AntiOutlierConfig(anti_method=args.anti_method)

    quant_config = QuantConfig(
        a_bit=args.a_bit,
        w_bit=args.w_bit,
        disable_names=disable_names_without_lora,
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

    calibration_dataset = None
    if args.calib_file:
        calibration_dataset = load_jsonl(args.calib_file)
    quant_weight_generator = Quantifier(
        args.model_path, quant_config, anti_outlier_config,
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
        args.model_path, args.save_directory, config.torch_dtype,
        quant_type, args
    )
    copy_process_files(args.model_path, args.save_directory)
