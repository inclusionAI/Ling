# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import copy
import math
import os
import time
import json

import torch
import torch_npu
import numpy as np
from atb_llm.utils import argument_utils
from atb_llm.runner.model_runner import ModelRunner
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, standardize_path, check_file_safety, safe_listdir
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import file_utils
from atb_llm.utils import shm_utils
from atb_llm.utils.shm_utils import encode_shape_to_int64, encode_shm_name_to_int64, create_shm
from atb_llm.models.qwen.vl.data_preprocess_qwen_vl import qwen_vl_image_preprocess
from examples.multimodal_runner import parser
from examples.multimodal_runner import path_validator, bool_validator
from examples.server.cache import CacheConfig, ModelConfig, CacheManager
from examples.server.generate import decode_token, generate_req
from examples.server.request import request_from_token

_IMAGE_START_ID = 151857
_IMAGE_PLACE_HOLDER = 256


def replace_crlf(mm_input):
    result = []
    for mm_in in mm_input:
        res = {}
        for k, v in mm_in.items():
            input_text_filter = v
            input_text_filter = input_text_filter.replace('\n', ' ').replace('\r', ' ').replace('\f', ' ')
            input_text_filter = input_text_filter.replace('\t', ' ').replace('\v', ' ').replace('\b', ' ')
            input_text_filter = input_text_filter.replace('\u000A', ' ').replace('\u000D', ' ').replace('\u000C', ' ')
            input_text_filter = input_text_filter.replace('\u000B', ' ').replace('\u0008', ' ').replace('\u007F', ' ')
            input_text_filter = input_text_filter.replace('\u0009', ' ').replace('    ', ' ')
            res[k] = input_text_filter.replace("\n", "_").replace("\r", "_")
        result.append(res)
    return result


def validate_path_new(value: str):
    check_file_safety(standardize_path(value), mode='a')


class PARunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get("rank", "0")
        self.local_rank = kwargs.get("local_rank", self.rank)
        self.world_size = kwargs.get("world_size", "1")

        self.model_path = kwargs.get("model_path", None)
        self.input_text = kwargs.get("input_text", None)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", None)
        self.max_input_length = kwargs.get("max_input_length", None)
        self.max_prefill_tokens = kwargs.get("max_prefill_tokens", None)
        self.max_output_length = kwargs.get("max_output_length", None)
        self.is_flash_model = kwargs.get("is_flash_model", None)
        self.max_batch_size = kwargs.get("max_batch_size", None)
        self.shm_name_save_path = kwargs.get("shm_name_save_path", None)
        self.trust_remote_code = kwargs.get("trust_remote_code", None)
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (
                self.max_input_length + self.max_output_length
            )

        self.block_size = kwargs.get("block_size", None)

        self.model = ModelRunner(
            self.model_path,
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            max_position_embeddings=self.max_position_embeddings,
            trust_remote_code=self.trust_remote_code
        )
        self.tokenizer_wrapper = TokenizerWrapper(self.model_path, \
                                                  trust_remote_code=self.trust_remote_code)
        self.tokenizer = self.tokenizer_wrapper.tokenizer
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.kv_quant_type = self.model.kv_quant_type
        self.model.load_weights()

        self.device = self.model.device
        self.model_config = ModelConfig(
            self.model.num_heads,
            self.model.num_kv_heads,
            self.model.config.num_key_value_heads \
                if hasattr(self.model.config, "num_key_value_heads") \
                else self.model.num_kv_heads,
            self.model.k_head_size,
            self.model.v_head_size,
            self.model.num_layers,
            self.model.device,
            self.model.dtype,
            self.model.soc_info,
            self.kv_quant_type,
        )

        self.max_memory = NpuHbmInfo.get_hbm_capacity(
            self.local_rank, self.world_size, self.model.soc_info.need_nz
        )
        self.init_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.model.soc_info.need_nz
            )
        )
        print_log(
            self.rank,
            logger.info,
            f"hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, "
            f"init_memory(GB): {self.init_memory / (1024 ** 3)}",
        )

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None

    def __repr__(self):
        return (
            f"PARunner(model_path={self.model_path}, "
            f"input_text={self.input_text}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"max_input_length={self.max_input_length}, "
            f"max_output_length={self.max_output_length}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"is_flash_model={self.is_flash_model}, "
            f"max_batch_size={self.max_batch_size}, "
            f"dtype={self.dtype}, "
            f"block_size={self.block_size}, "
            f"model_config={self.model_config}, "
            f"max_memory={self.max_memory}, "
        )

    def warm_up(self):
        all_input_length = self.max_batch_size * self.max_input_length
        input_ids_list = (
                [_IMAGE_START_ID]
                + [_IMAGE_START_ID + 2] * _IMAGE_PLACE_HOLDER
                + [_IMAGE_START_ID + 1]
                + [1] * (all_input_length - _IMAGE_PLACE_HOLDER - 2)
        )
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).to(self.device)
        image_pixel = qwen_vl_image_preprocess("")
        image_pixel = image_pixel[None, :]
        shm = create_shm(image_pixel.nbytes, self.shm_name_save_path)
        shared_array = np.ndarray(image_pixel.shape, dtype=np.float32, buffer=shm.buf)
        shared_array[:] = image_pixel

        shm_name = encode_shm_name_to_int64(shm.name)
        shape_value = encode_shape_to_int64(image_pixel.shape)
        input_ids[1] = shm_name
        input_ids[2] = shape_value
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = (
                math.ceil(
                    (self.max_input_length + self.max_output_length) / self.block_size
                )
                * self.max_batch_size
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        self.model.postprocessor.max_new_tokens = 2
        generate_req(
            [request_from_token(input_ids, self.max_output_length, self.block_size)],
            self.model,
            self.max_batch_size,
            self.max_prefill_tokens,
            self.cache_manager,
        )
        self.warm_up_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.model.soc_info.need_nz
            )
        )
        print_log(
            self.rank,
            logger.info,
            f"warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}",
        )
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(
        self,
        mm_inputs,
        max_output_length,
        is_chat_model,
        shm_name_save_path
    ):
        print_log(
            self.rank, logger.info, "---------------begin inference---------------"
        )

        input_ids = self._build_model_inputs(mm_inputs, is_chat_model, shm_name_save_path)

        req_list = [
            request_from_token(
                input_ids[i],
                max_output_length,
                self.block_size,
                req_idx=i,
            )
            for i in range(len(mm_inputs))
        ]

        print_log(
            self.rank, logger.debug, f"req_list[0].input_ids: {req_list[0].input_ids}"
        )

        if not self.cache_manager:
            cache_block_size = (
                self.block_size * self.model.num_kv_heads * self.model.head_size
            )
            dtype_size = CacheManager.get_dtype_size(self.dtype)
            total_cache_size = self.model.num_layers * cache_block_size * 2 * dtype_size

            max_memory = (
                ENV.memory_fraction * self.max_memory
            )
            free_memory = (
                max_memory
                - ENV.reserved_memory_gb * (1 << 30)
                - (
                    self.warm_up_memory
                    if self.warm_up_memory != 0
                    else self.init_memory
                )
            )
            print_log(
                self.rank,
                logger.info,
                f"infer max_memory(GB): {max_memory / (1024 ** 3): .2f}, "
                f"warm_up_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}, "
                f"free_memory(GB): {free_memory / (1024 ** 3): .2f}",
            )

            num_blocks = int(free_memory // total_cache_size)
            if num_blocks <= 0:
                raise ValueError("num_blocks must be positive!")
            print_log(
                self.rank,
                logger.info,
                f"num_blocks: {num_blocks}, free_memory: {free_memory}",
            )
            cache_config = CacheConfig(num_blocks, self.block_size)
            self.cache_manager = CacheManager(cache_config, self.model_config)

        if ENV.benchmark_enable:
            self.model.postprocessor.max_new_tokens = 2
            req_list_dummy = copy.deepcopy(req_list)
            generate_req(
                req_list_dummy,
                self.model,
                self.max_batch_size,
                self.max_prefill_tokens,
                self.cache_manager,
            )

        self.model.postprocessor.max_new_tokens = max_output_length
        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            torch.npu.synchronize()
            e2e_start = time.time()
            generate_req(
                req_list,
                self.model,
                self.max_batch_size,
                self.max_prefill_tokens,
                self.cache_manager,
            )
            _, _ = decode_token(req_list, self.tokenizer)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            print_log(self.rank, logger.debug, "enter profiling")
            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                l2_cache=False,
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    profiling_path
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ) as _:
                generate_req(
                    req_list,
                    self.model,
                    self.max_batch_size,
                    self.max_prefill_tokens,
                    self.cache_manager,
                )
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start

        generate_text_list, token_num_list = decode_token(req_list, self.tokenizer)
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return generate_text_list, token_num_list, e2e_time

    def _build_model_inputs(self, inputs, is_chat_model=False, shm_name_save_path=None):
        mm_inputs, input_conversations = [], []

        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], list) and inputs[0]:
                first_dict = inputs[0][0]
                if isinstance(first_dict, dict):
                    if "role" in first_dict:
                        input_conversations = inputs
                    else:
                        mm_inputs = inputs
            
        if not (mm_inputs or input_conversations):
            raise ValueError(f"The inputs of `PARunner.infer` must be as List[List[Dict]]."
                             f" Now the input of image or video and text is not acceptable or is empty.")
        
        if is_chat_model:
            if input_conversations:
                input_ids = self.model.build_inputs(input_conversations, shm_name_save_path=shm_name_save_path)
            elif mm_inputs:
                input_conversations = [[{"role": "user", "content": i}] for i in mm_inputs]
                input_ids = self.model.build_inputs(input_conversations, shm_name_save_path=shm_name_save_path)
            else:
                print_log(self.rank, logger.warning, "Neither conversations nor mm_inputs exist, "
                                                     "'chat' parameter is not effective.")
        elif mm_inputs:
            input_ids = [self.tokenizer_wrapper.tokenize(single_input, shm_name_save_path=shm_name_save_path)
                for single_input in mm_inputs]
            # token长度校验
            for item in input_ids:
                if len(item) > self.max_input_length:
                    print_log(self.rank, logger.warning,
                              "Num of tokens in input is larger than max_input_length. "
                              "Please shorten input to avoid out of memory.")
        return input_ids


def parse_ids(list_str):
    return [int(item) for item in list_str.split(",")]


def parse_arguments():
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    quantize_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000, allow_none=True)
    inputfile_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096, allow_none=True)
    savepath_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096)
    savepath_validator.validate_path = validate_path_new
    savepath_validator.create_validation_pipeline()
    parser_qwenvl = parser
    parser_qwenvl.add_argument(
        "--input_text", default="Generate the caption in English with grounding:",
        validator=string_validator,
    )
    parser_qwenvl.add_argument("--input_image", default="/image/path", validator=path_validator)
    parser_qwenvl.add_argument(
        "--input_file",
        type=str,
        help="CSV or Numpy file containing tokenized input. Alternative to text input.",
        default=None,
        validator=inputfile_validator,
    )
    parser_qwenvl.add_argument("--quantize", type=str, default=None, validator=quantize_validator)

    parser_qwenvl.add_argument("--is_flash_model", action="store_false", validator=bool_validator)
    parser_qwenvl.add_argument('--is_chat_model', action="store_true", validator=bool_validator)
    parser_qwenvl.add_argument("--shm_name_save_path", 
        type=str,
        help='This path is used to temporarily store the shared '
             'memory addresses that occur during the inference process.',
        default='./shm_name.txt',
        validator=savepath_validator)
    parser_qwenvl.add_argument(
        '--conversation_file',
        type=str,
        help='This parameter is used to input multi-turn dialogue information in the form '
             'of a jsonl file, with each line in the format of a List[Dict]. Each dictionary '
             '(Dict) must contain at least two fields: "role" and "content". Each "content" '
             'must be List[Dict[str, str]]',
        default=None,
        validator=inputfile_validator)
    parser_qwenvl.add_argument(
        "--dataset_path", help="precision test dataset path", default=None, validator=inputfile_validator
    )
    parser_qwenvl.add_argument(
        "--results_save_path",
        help="precision test result path",
        default="./npu_coco_rst.json",
        validator=savepath_validator
    )

    return parser_qwenvl.parse_args()


def deal_dataset(dataset_path):
    input_images = []
    dataset_path = standardize_path(dataset_path)
    check_file_safety(dataset_path)
    images_list = safe_listdir(dataset_path)
    for img_name in images_list:
        image_path = os.path.join(dataset_path, img_name)
        input_images.append(image_path)
    input_texts = ["Generate the caption in English with grounding:"] * len(
        input_images
    )

    return input_images, input_texts


def main():
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        **vars(args),
    }

    # precision test
    npu_rst_dict = {}
    if args.dataset_path:
        dataset_images, dataset_texts = deal_dataset(args.dataset_path)

    pa_runner = PARunner(**input_dict)
    print_log(rank, logger.info, f"pa_runner: {pa_runner}")
    pa_runner.warm_up()

    if args.dataset_path:
        mm_inputs = [
            [{"image": i}, {"text": j}]
            for i, j in zip(dataset_images, dataset_texts)
        ]
    else:
        mm_inputs = [
            [
                {"image": args.input_image},
                {"text": args.input_text},
            ]
        ] * args.max_batch_size

    if args.is_chat_model and args.conversation_file:
        conversation_inputs = []
        with file_utils.safe_open(args.conversation_file, "r", encoding="utf-8") as f:
            for line in file_utils.safe_readlines(f):
                conversation_inputs.append(json.loads(line))
        mm_inputs = conversation_inputs

    generate_texts, token_nums, latency = pa_runner.infer(
        mm_inputs,
        args.max_output_length,
        args.is_chat_model,
        args.shm_name_save_path
    )

    if file_utils.is_path_exists(args.shm_name_save_path):
        try:
            shm_utils.release_shared_memory(args.shm_name_save_path)
        except Exception as e:
            print_log(rank, logger.error, f"release shared memory failed: {e}")
        try:
            os.remove(args.shm_name_save_path)
        except Exception as e:
            print_log(rank, logger.error, f"remove shared memory file failed: {e}")

    for i, generate_text in enumerate(generate_texts):
        inputs = mm_inputs
        if args.dataset_path:
            rst_key = dataset_images[i].split("/")[-1]
            npu_rst_dict[rst_key] = generate_text
        if i < args.max_batch_size:
            question = replace_crlf(inputs[i]) if not args.is_chat_model else conversation_inputs[i]

            print_log(rank, logger.info, f'Question[{i}]: {question}')
        print_log(rank, logger.info, f"Answer[{i}]: {generate_text}")
        print_log(rank, logger.info, f"Generate[{i}] token num: {token_nums[i]}")
        print_log(rank, logger.info, f"Latency: {latency}")

    if args.dataset_path:
        sorted_dict = dict(sorted(npu_rst_dict.items()))
        with safe_open(
            args.results_save_path,
            "w",
            override_flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        ) as f:
            json.dump(sorted_dict, f)
        print_log(
            rank, logger.info, "--------------npu precision test finish--------------"
        )


if __name__ == "__main__":
    main()
