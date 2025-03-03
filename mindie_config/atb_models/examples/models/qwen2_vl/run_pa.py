# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import argparse
import json
import math
import os
import time

import torch
import torch_npu
from PIL import Image
from transformers import AutoImageProcessor
from atb_llm.utils import argument_utils
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.models.qwen2_vl.router_qwen2_vl import process_shared_memory
from atb_llm.runner.model_runner import ModelRunner
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, is_path_exists
from atb_llm.utils.file_utils import standardize_path, check_file_safety, safe_listdir
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.shm_utils import decode_shape_from_int64, release_shared_memory
from examples.multimodal_runner import parser
from examples.multimodal_runner import path_validator, bool_validator
from examples.server.cache import CacheConfig, ModelConfig, CacheManager
from examples.server.generate import decode_token, generate_req
from examples.server.request import MultiModalRequest

VISION_START_TOKEN_ID = 151652
_IMAGE_TOKEN_ID = 151655
VISION_END_TOKEN_ID = 151653
_IMAGE_FEATURE_LENS = 64
IMAGE_THW_TOKEN_OFFSET = 3
SUPPORTED_IMAGE_MODE = "RGB"
PYTORCH_TENSOR = "pt"


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
    

def request_from_token_qwen2_vl(input_ids, max_out_length, block_size,
                                req_idx=0, adapter_id=None):
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
    position_ids = torch.arange(len(input_ids), dtype=torch.int64)
    if torch.any(torch.eq(input_ids, VISION_START_TOKEN_ID)):
        bos_pos = torch.where(torch.eq(input_ids, VISION_START_TOKEN_ID))[0]
        eos_pos = torch.where(torch.eq(input_ids, VISION_END_TOKEN_ID))[0]
        vision_num = bos_pos.shape[0]
        deltas = 0
        for i in range(vision_num):
            thw_shape_value = input_ids[bos_pos[i] + IMAGE_THW_TOKEN_OFFSET]
            thw_shape = decode_shape_from_int64(thw_shape_value)

            vision_feature_len = eos_pos[i] - bos_pos[i] - 1
            max_hw = max(thw_shape[1:])
            if thw_shape[0] > (max_hw // 2):
                deltas += vision_feature_len - thw_shape[0]
            else:
                deltas += vision_feature_len - max_hw // 2
        position_ids[-1] = position_ids[-1] - deltas

    request = MultiModalRequest(max_out_length, block_size, req_idx, input_ids, adapter_id, position_ids)
    return request


class PARunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get("rank", "0")
        self.local_rank = kwargs.get("local_rank", self.rank)
        self.world_size = kwargs.get("world_size", "1")

        self.enable_atb_torch = kwargs.get('enable_atb_torch', False)

        self.model_path = kwargs.get("model_path", None)
        self.input_text = kwargs.get("input_text", None)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", None)
        self.max_input_length = kwargs.get("max_input_length", None)
        self.max_prefill_tokens = kwargs.get("max_prefill_tokens", None)
        self.max_output_length = kwargs.get("max_output_length", None)
        self.is_flash_model = kwargs.get("is_flash_model", None)
        self.max_batch_size = kwargs.get("max_batch_size", None)
        self.shm_name_save_path = kwargs.get("shm_name_save_path", None)
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
            enable_atb_torch=self.enable_atb_torch,
            max_position_embeddings=self.max_position_embeddings,
        )
        self.tokenizer_wrapper = TokenizerWrapper(self.model_path)
        self.tokenizer = self.tokenizer_wrapper.tokenizer
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.kv_quant_type = self.model.kv_quant_type
        self.model.load_weights()

        self.device = self.model.device
        self.model_config = ModelConfig(
            self.model.num_heads,
            self.model.num_kv_heads,
            self.model.num_kv_heads,
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
            f"enable_atb_torch={self.enable_atb_torch}, "
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
                [VISION_START_TOKEN_ID]
                + [_IMAGE_TOKEN_ID] * _IMAGE_FEATURE_LENS
                + [VISION_END_TOKEN_ID]
                + [1] * (all_input_length - _IMAGE_FEATURE_LENS - 2)
        )
        image = Image.new(SUPPORTED_IMAGE_MODE, (224, 224), (255, 255, 255))
        warmup_image_processor = safe_from_pretrained(AutoImageProcessor, self.model_path)
        images_inputs = warmup_image_processor(images=image,
                                               videos=None,
                                               return_tensors=PYTORCH_TENSOR)
        image.close()
        input_ids_list[1:4] = process_shared_memory(
            images_inputs.pixel_values, self.shm_name_save_path, images_inputs.image_grid_thw
        )

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).to(self.device)
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
        max_output_length = 2
        self.model.postprocessor.max_new_tokens = max_output_length
        single_req = request_from_token_qwen2_vl(
            input_ids,
            max_output_length,
            self.block_size,
            req_idx=0,
        )
        generate_req([single_req], self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        if is_path_exists(self.shm_name_save_path):
            try:
                release_shared_memory(self.shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"release shared memory failed: {e}")
            try:
                os.remove(self.shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"remove shared memory file failed: {e}")
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
            shm_name_save_path,
    ):
        print_log(
            self.rank, logger.info, "---------------begin inference---------------"
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

        self.model.postprocessor.max_new_tokens = max_output_length

        req_list = []
        if not ENV.profiling_enable:
            torch.npu.synchronize()
            e2e_start = time.time()
            for i, mm_input in enumerate(mm_inputs):
                input_ids = self.tokenizer_wrapper.tokenize(mm_input, shm_name_save_path=shm_name_save_path)
                single_req = request_from_token_qwen2_vl(
                    input_ids,
                    max_output_length,
                    self.block_size,
                    req_idx=i,
                )
                req_list.append(single_req)
            generate_req(
                req_list,
                self.model,
                self.max_batch_size,
                self.max_prefill_tokens,
                self.cache_manager,
            )
            generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens=True)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            for i, mm_input in enumerate(mm_inputs):
                input_ids = self.tokenizer_wrapper.tokenize(mm_input, shm_name_save_path=shm_name_save_path)
                single_req = request_from_token_qwen2_vl(
                    input_ids,
                    max_output_length,
                    self.block_size,
                    req_idx=i,
                )
                req_list.append(single_req)
            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
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
            generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, skip_special_tokens=True)
        if is_path_exists(shm_name_save_path):
            try:
                release_shared_memory(shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"release shared memory failed: {e}")
            try:
                os.remove(shm_name_save_path)
            except Exception as e:
                print_log(self.rank, logger.error, f"remove shared memory file failed: {e}")
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return generate_text_list, token_num_list, e2e_time


def parse_arguments():
    store_true = 'store_true'
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    savepath_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096)
    quantize_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000, allow_none=True)
    savepath_validator.validate_path = validate_path_new
    savepath_validator.create_validation_pipeline()
    parser_qwenvl = parser
    inputfile_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096, allow_none=True)
    parser_qwenvl.add_argument(
        "--input_text",
        default="Describe the image.",
        validator=string_validator
    )
    parser_qwenvl.add_argument(
        "--input_image",
        default="",
        validator=path_validator
    )
    parser_qwenvl.add_argument(
        "--dataset_path",
        help="precision test dataset path",
        default="",
        validator=inputfile_validator
    )
    parser_qwenvl.add_argument(
        "--input_file",
        type=str,
        help="CSV or Numpy file containing tokenized input. Alternative to text input.",
        default=None,
        validator=inputfile_validator,
    )
    parser_qwenvl.add_argument("--shm_name_save_path",
                        type=str,
                        help='This path is used to temporarily store the shared '
                             'memory addresses that occur during the inference process.',
                        default='./shm_name.txt',
                        validator=savepath_validator)
    parser_qwenvl.add_argument(
        "--results_save_path",
        help="precision test result path",
        default="./results.json",
        validator=savepath_validator,
    )
    parser_qwenvl.add_argument("--quantize", type=str, default=None, validator=quantize_validator)
    parser_qwenvl.add_argument("--is_flash_model", action="store_false", validator=bool_validator)
    parser_qwenvl.add_argument('--enable_atb_torch', 
                               default=False, 
                               action=store_true, 
                               validator=bool_validator)

    return parser_qwenvl.parse_args()


def deal_dataset(dataset_path, text):
    input_images = []
    dataset_path = standardize_path(dataset_path)
    check_file_safety(dataset_path)
    images_list = safe_listdir(dataset_path)
    for img_name in images_list:
        image_path = os.path.join(dataset_path, img_name)
        input_images.append(image_path)
    input_texts = [text] * len(
        input_images
    )
    return input_images, input_texts


def is_video(file_name):
    ext = os.path.splitext(file_name)[1]
    ext = ext.lower()
    if ext in [".mp4", ".wmv", ".avi"]:
        return True
    return False


def is_image(file_name):
    ext = os.path.splitext(file_name)[1]
    ext = ext.lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return True
    return False


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

    pa_runner = PARunner(**input_dict)
    print_log(rank, logger.info, f"pa_runner: {pa_runner}")
    pa_runner.warm_up()

    npu_rst_dict = {}
    if args.dataset_path:
        dataset_images, dataset_texts = deal_dataset(args.dataset_path, args.input_text)
    if args.dataset_path:
        mm_inputs = []
        for dataset_image, dataset_text in zip(dataset_images, dataset_texts):
            if is_video(dataset_image):
                key = "video"
            elif is_image(dataset_image):
                key = "image"
            else:
                raise TypeError("The input field currently only needs to support 'image', 'video'.")
            single_inputs = [{key: dataset_image}, {"text": dataset_text}]
            mm_inputs.append(single_inputs)
    else:
        if is_video(args.input_image):
            key = "video"
        elif is_image(args.input_image):
            key = "image"
        else:
            raise TypeError("The input field currently only needs to support 'image', 'video'.")
        mm_inputs = [
                        [
                            {key: args.input_image},
                            {"text": args.input_text},
                        ]
                    ] * args.max_batch_size

    generate_texts, token_nums, latency = pa_runner.infer(
        mm_inputs,
        args.max_output_length,
        args.shm_name_save_path,
    )

    for i, generate_text in enumerate(generate_texts):
        inputs = mm_inputs
        if args.dataset_path:
            rst_key = dataset_images[i].split("/")[-1]
            npu_rst_dict[rst_key] = generate_text
        question = replace_crlf(inputs[i])

        print_log(rank, logger.info, f"Question[{i}]: {question}")
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
