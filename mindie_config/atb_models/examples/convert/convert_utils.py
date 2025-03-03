# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os.path
import shutil
from atb_llm.utils import file_utils

MAX_TOKENIZER_FILE_SIZE = 1024 * 1024 * 1024


def copy_tokenizer_files(model_dir, dest_dir):
    model_dir = file_utils.standardize_path(model_dir, check_link=False)
    file_utils.check_path_permission(model_dir)
    if os.path.exists(dest_dir):
        dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
        file_utils.check_path_permission(dest_dir)
    else:
        os.makedirs(dest_dir, exist_ok=True)
        dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
    for filename in file_utils.safe_listdir(model_dir):
        need_move = False
        file_names = ['tokenizer', 'tokenization', 'special_tokens_map', 'generation', 'configuration']
        for f in file_names:
            if f in filename:
                need_move = True
                break
        if need_move:
            src_filepath = os.path.join(model_dir, filename)
            src_filepath = file_utils.standardize_path(src_filepath, check_link=False)
            file_utils.check_file_safety(src_filepath, 'r', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            dest_filepath = os.path.join(dest_dir, filename)
            dest_filepath = file_utils.standardize_path(dest_filepath, check_link=False)
            file_utils.check_file_safety(dest_filepath, 'w', max_file_size=MAX_TOKENIZER_FILE_SIZE)
            shutil.copyfile(src_filepath, dest_filepath)


def modify_config(model_dir, dest_dir, torch_dtype, quantize_type, args=None):
    model_dir = file_utils.standardize_path(model_dir, check_link=False)
    file_utils.check_path_permission(model_dir)
    src_config_filepath = os.path.join(model_dir, 'config.json')
    with file_utils.safe_open(src_config_filepath, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    data['torch_dtype'] = str(torch_dtype).split(".")[1]
    data['quantize'] = quantize_type
    if args is not None:
        quantization_config = {
            # 当is_lowbit为True，open_outlier为False时，group_size生效
            'group_size': args.group_size if args.is_lowbit and not args.open_outlier else 0,
            'kv_quant_type': "C8" if args.use_kvcache_quant else None,
            "fa_quant_type": "FAQuant" if args.use_fa_quant else None,
            'w_bit': args.w_bit,
            'a_bit': args.a_bit,
            'dev_type': args.device_type,
            'fraction': args.fraction,
            'act_method': args.act_method,
            'co_sparse': args.co_sparse,
            'anti_method': args.anti_method,
            'disable_level': args.disable_level,
            'do_smooth': args.do_smooth,
            'use_sigma': args.use_sigma,
            'sigma_factor': args.sigma_factor,
            'is_lowbit': args.is_lowbit,
            'mm_tensor': False,
            'w_sym': args.w_sym,
            'open_outlier': args.open_outlier,
            'is_dynamic': args.is_dynamic
        }
        if args.use_reduce_quant:
            quantization_config.update({"reduce_quant_type": "per_channel"})
        data['quantization_config'] = quantization_config
    dest_dir = file_utils.standardize_path(dest_dir, check_link=False)
    file_utils.check_path_permission(dest_dir)
    dest_config_filepath = os.path.join(dest_dir, 'config.json')
    with file_utils.safe_open(dest_config_filepath, 'w', encoding='utf-8', is_exist_ok=False) as fw:
        json.dump(data, fw, indent=4)
