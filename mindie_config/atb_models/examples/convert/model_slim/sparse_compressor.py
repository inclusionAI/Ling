# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import torch
from atb_llm.runner.model_runner import ModelRunner
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.file_utils import MAX_PATH_LENGTH
from atb_llm.models.base.model_utils import unwrap_model_state_dict
from atb_llm.utils.argument_utils import ArgumentParser, StringArgumentValidator

from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
from examples.convert.convert_utils import copy_tokenizer_files, modify_config


class SparseCompressor:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.save_directory = kwargs.get('save_directory', None)
        if self.save_directory is not None:
            self.save_directory = file_utils.standardize_path(self.save_directory, check_link=False)
            file_utils.check_path_permission(self.save_directory)
        self.multiprocess_num = kwargs.get('multiprocess_num', 8)
        self.save_split_w8a8s_dir = kwargs.get('save_split_w8a8s_dir', None)
        if self.save_split_w8a8s_dir is not None:
            self.save_split_w8a8s_dir = file_utils.standardize_path(self.save_split_w8a8s_dir, check_link=False)
            file_utils.check_path_permission(self.save_split_w8a8s_dir)

        self.trust_remote_code = kwargs.get('trust_remote_code', False)

        self.model = ModelRunner(
            self.model_path, rank=self.rank, world_size=self.world_size,
            trust_remote_code=self.trust_remote_code)
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.model.load_weights()

        self.device = self.model.device
        self.max_memory = NpuHbmInfo.get_hbm_capacity(self.rank, self.world_size, self.model.soc_info.need_nz)
        self.init_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, '
                                          f'init_memory(GB): {self.init_memory / (1024 ** 3)}')

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None

        if self.save_split_w8a8s_dir is not None:
            self.model.save_pretrained(save_directory=self.save_split_w8a8s_dir,
                                       safe_serialization=True)
            if self.rank == 0:
                modify_config(model_path, self.save_split_w8a8s_dir, torch.float16, 'w8a8s')
                copy_tokenizer_files(model_path, self.save_split_w8a8s_dir)

    def compress(self):
        model_dict = unwrap_model_state_dict(self.model.model.state_dict())
        quant_desc = self.model.model.generate_description()
        compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True,
                                         record_detail_root=self.save_directory,
                                         multiprocess_num=self.multiprocess_num)
        compressor = Compressor(compress_config, weight=model_dict, quant_model_description=quant_desc)
        compressor.run()
        part_save_directory = os.path.join(self.save_directory, f'part{self.rank}-of-{self.world_size}')
        os.makedirs(part_save_directory, exist_ok=True)
        compressor.export_safetensors(part_save_directory)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path",
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--save_directory', type=str, required=True,
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--multiprocess_num', type=int, default=8)
    parser.add_argument('--save_split_w8a8s_dir', type=str, default=None)
    parser.add_argument('--trust_remote_code', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    rank = ENV.rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        **vars(args)
    }

    model_path = args.model_path
    save_directory = args.save_directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    sparse_compressor = SparseCompressor(**input_dict)

    sparse_compressor.compress()

    if rank == 0:
        modify_config(model_path, save_directory, torch.float16, 'w8a8sc')
        copy_tokenizer_files(model_path, save_directory)
