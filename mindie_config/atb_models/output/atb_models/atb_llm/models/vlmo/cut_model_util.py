# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import logging as logger
import os

import torch
from vlmo.modules.vlmo_file_check import file_check
pwd = os.path.realpath(os.path.dirname(__file__))


# cut_row_keys: dim 0  cut_col_keys: dim 1  nn.linear: x*A.T
def cut_weights(model, world_size, cut_row_keys, cut_col_keys):
    logger.info('***********Cutting weights***********')
    if cut_row_keys is None:
        cut_row_keys = ['fc1', 'attn']
    if cut_col_keys is None:
        cut_col_keys = ['fc2', 'proj']
    _state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in model.items():
        cut_tensor_list = []
        if len(key.split('.')) > 3 :
            key_group = key.split('.')[-3]
            key_short = key.split('.')[-2]
            key_type = key.split('.')[-1]
            if key_short == 'qkv':  # attention
                split_linear_size = 3  # q k v linear
                full_q_weights, full_k_weights, full_v_weights = torch.chunk(tensor, split_linear_size, dim=0)
                cut_q_weights = torch.chunk(full_q_weights, world_size, dim=0)
                cut_k_weights = torch.chunk(full_k_weights, world_size, dim=0)
                cut_v_weights = torch.chunk(full_v_weights, world_size, dim=0)
                for i in range(world_size):
                    cut_tensor_list.append(torch.concat((cut_q_weights[i], cut_k_weights[i], cut_v_weights[i]), dim=0))
                # break
            else:
                if key_short in cut_row_keys:
                    cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
                elif key_short in cut_col_keys and key_group != 'patch_embed':
                    if key_type == "weight":
                        cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
                    elif key_type == "bias" :
                        cut_tensor_list = [tensor] * world_size
                else:
                    cut_tensor_list = [tensor] * world_size
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            _state_dict_list[i][key] = cut_tensor_list[i]
    return _state_dict_list


if __name__ == "__main__":
    logger.info('********0*********')
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="./vlmo",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='./vlmo/part_model',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
        type=int
    )
    parser.add_argument(
        "--cut_row_keys",
        default=['fc1', 'attn'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default=['fc2', 'proj'],
        help="cut_col_keys",
    )
    logger.info('********init finished*********')

    args = parser.parse_args()
    args.world_size = int(args.world_size)

    logger.info('********got path*********')

    # step 4: cut weight
    load_path = os.path.join(args.input_path, '/vlmo_base_patch16_480_vqa.pt')
    load_path = file_check(load_path)
    state_dict_list = cut_weights(torch.load(load_path, map_location='cpu', weights_only=True),
        args.world_size, args.cut_row_keys, args.cut_col_keys)

    # step 5: create new model config, add the world size parameter,
    # the model size will be cut according to the world size in the model file
    logger.info('********cut done**********')


    # step 6: create new model according to the new model config
    save_path = args.output_path
    for j in range(args.world_size):
        target_dir = os.path.realpath(os.path.join(args.output_path, str(j)))
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "cut_VQA_weights.pt")
        target_path = file_check(target_path)
        torch.save(state_dict_list[j], target_path)
    logger.info('Tensor parallelism weights have been successfully saved.')
