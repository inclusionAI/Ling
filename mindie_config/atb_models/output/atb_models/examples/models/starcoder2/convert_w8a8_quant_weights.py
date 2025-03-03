# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from atb_llm.utils.log import logger
from atb_llm.models.starcoder2.config_starcoder2 import Starcoder2Config
from atb_llm.utils.file_utils import safe_open, safe_readlines
from examples.convert.model_slim.quantifier import parse_arguments, Quantifier
from examples.convert.convert_utils import copy_tokenizer_files, modify_config


def get_calib_dataset(_tokenizer, _calib_list):
    calib_dataset = []
    for calib_data in _calib_list:
        inputs = _tokenizer([calib_data], return_tensors='pt')
        logger.info(inputs)
        calib_dataset.append([inputs.data['input_ids'], None, inputs.data['attention_mask']])
    return calib_dataset

if __name__ == "__main__":
    args = parse_arguments()
    disable_names = []
    quant_conf = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        act_method=args.act_method,
        pr=1.0,  # randseed
        nonuniform=False,
        w_sym=args.w_sym,
        mm_tensor=False,
        co_sparse=args.co_sparse,
        fraction=args.fraction,
        sigma_factor=args.sigma_factor,
        use_sigma=args.use_sigma,
        is_lowbit=args.is_lowbit,
        do_smooth=args.do_smooth,
        use_kvcache_quant=args.use_kvcache_quant
    )

    quantifier = Quantifier(args.model_path, quant_conf)
    quantifier.tokenizer.pad_token_id = 0
    calib_list = []
    with safe_open('humaneval_python.txt', 'r') as file:
        for line in safe_readlines(file):
            calib_list.append(line.strip())
    dataset_calib = get_calib_dataset(quantifier.tokenizer, calib_list) 

    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory, exist_ok=True)

    quantifier.convert(dataset_calib, args.save_directory, args.disable_level)
    quant_type = f"w{args.w_bit}a{args.a_bit}"
    auto_config = Starcoder2Config.from_pretrained(args.model_path)
    modify_config(args.model_path, args.save_directory, auto_config.torch_dtype,
                  quant_type, args)
    copy_tokenizer_files(args.model_path, args.save_directory)