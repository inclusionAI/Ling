# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import argparse
import math
import os
import time
import json
from dataclasses import dataclass
from typing import List
import numpy as np

import torch
import torch_npu
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log import logger, print_log
from examples.server.cache import CacheConfig, CacheManager
from examples.server.generate import decode_token, generate_req
from examples.server.request import request_from_text_and_image, MultiModalRequestParams
from examples.run_pa import PARunner, parse_ids


STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/bunny/bunny_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, FirstToken(s), ResponseTime(s), E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/bunny/predict_result.json"


@dataclass
class MultiModalRequestOut:
    req_list: List
    batch: int
    file_list: List
    input_texts: List


@dataclass
class InputAttrs:
    input_texts: List
    image_path: str


class MultiModalPARunner(PARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_attrs = InputAttrs(kwargs.get('input_texts', None),
                                      kwargs.get('image_path', None))
        self.predict_result = kwargs.get('prediction_result', False)
        self.performance = kwargs.get('performance', False)
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (
            self.max_input_length + self.max_output_length
            )
        self.warm_up_num_blocks = 0
        self.warm_up_memory = 0
        self.cache_manager = None

    def prepare_request(self, input_texts, image_path, batch_size, max_output_length, current_iter):
        file_list = os.listdir(image_path)
        if len(input_texts) == 1:
            input_texts = [input_texts[0] for _ in range(batch_size)]
            if len(file_list) == 1:
                image_list = [file_list[0] for _ in range(batch_size)]
            else:
                image_list = [file_list[current_iter * batch_size + single_batch]
                              for single_batch in range(batch_size)]
        else:
            if len(file_list) == 1:
                input_texts = [input_texts[single_batch] for single_batch in range(batch_size)]
                image_list = [file_list[0] for _ in range(batch_size)]
            elif len(input_texts) == len(file_list):
                input_texts = [input_texts[current_iter * batch_size + single_batch]
                               for single_batch in range(batch_size)]
                image_list = [file_list[current_iter * batch_size + single_batch]
                              for single_batch in range(batch_size)]
            else:
                raise RuntimeError("input_texts length must equal input_images length")

        input_image = zip(input_texts, image_list)
        req_list = [request_from_text_and_image(
                    self.tokenizer,
                    self.model,
                    MultiModalRequestParams(item[0],
                                            os.path.join(image_path, item[1]),
                                            None,
                                            max_output_length,
                                            self.block_size,
                                            req_idx=idx))
                    for idx, item in enumerate(input_image)]
        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
        return MultiModalRequestOut(req_list, batch_size, file_list, input_texts)

    def warm_up(self):
        input_texts = self.input_attrs.input_texts
        image_path = self.input_attrs.image_path
        print_log(self.rank, logger.info, f"BunnyModel:{self.model.model.config.model_type}")
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        file_list = os.listdir(image_path)
        req_list = [request_from_text_and_image(
                        self.tokenizer,
                        self.model,
                        MultiModalRequestParams(input_texts[0],
                                                os.path.join(image_path, file_list[0]),
                                                None,
                                                self.max_output_length,
                                                self.block_size,
                                                req_idx=single_batch))
                        for single_batch in range(self.max_batch_size)
                    ]
        self.model.postprocessor.max_new_tokens = 2
        generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'warmup_memory(GB): {np.divide(self.warm_up_memory, 1024 ** 3): .2f}')
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, inputs, batch_size, max_output_length, ignore_eos, **kwargs):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        input_texts = inputs.input_texts
        image_path = inputs.image_path
        if not self.cache_manager:
            if self.max_prefill_tokens == -1:
                self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
            cache_block_size = self.block_size * self.model.num_kv_heads * self.model.head_size
            dtype_size = CacheManager.get_dtype_size(self.dtype)
            total_cache_size = self.model.num_layers * cache_block_size * 2 * dtype_size
            # 1 << 30正好是1G
            max_memory = ENV.memory_fraction * self.max_memory \
                if not ENV.max_memory_gb else int(ENV.max_memory_gb) * (1 << 30)
            free_memory = max_memory - ENV.reserved_memory_gb * (1 << 30) - (
                self.warm_up_memory if self.warm_up_memory != 0 else self.init_memory)
            print_log(self.rank, logger.info,
                      f"infer max_memory(GB): {np.divide(max_memory, 1024 ** 3): .2f}, "
                      f"warm_up_memory(GB): {np.divide(self.warm_up_memory, 1024 ** 3): .2f}, "
                      f"free_memory(GB): {np.divide(free_memory, 1024 ** 3): .2f}")
            num_blocks = int(free_memory // total_cache_size)
            print_log(self.rank, logger.info, f"num_blocks: {num_blocks}, free_memory: {free_memory}")
            cache_config = CacheConfig(num_blocks, self.block_size)
            self.cache_manager = CacheManager(cache_config, self.model_config)
        self.model.postprocessor.max_new_tokens = max_output_length
        all_input_texts = []
        all_generate_text_list = []
        all_token_num_list = []
        e2e_time_all = 0
        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            # 统计首token
            self.model.postprocessor.max_new_tokens = 1
            fisrt_start = time.time()
            torch.npu.synchronize()
            if ignore_eos:
                self.model.postprocessor.eos_token_id = []
            multimodalrequestout = self.prepare_request(input_texts,
                                                        image_path,
                                                        batch_size,
                                                        1, 
                                                        1)
            req_list = multimodalrequestout.req_list
            batch = multimodalrequestout.batch
            file_list = multimodalrequestout.file_list
            input_texts = multimodalrequestout.input_texts
            generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
            _, _ = decode_token(req_list, self.tokenizer)
            torch.npu.synchronize()
            fisrt_end = time.time()
            first_token = fisrt_end - fisrt_start
            # 统计端到端时间消耗
            self.model.postprocessor.max_new_tokens = max_output_length
            torch.npu.synchronize()
            if ignore_eos:
                self.model.postprocessor.eos_token_id = []
            max_iters = math.ceil(np.divide(len(os.listdir(image_path)), self.max_batch_size))
            for current_iter in range(max_iters):
                e2e_start = time.time()
                multimodalrequestout = self.prepare_request(input_texts,
                                                            image_path,
                                                            batch_size,
                                                            max_output_length,
                                                            current_iter)
                req_list = multimodalrequestout.req_list
                batch = multimodalrequestout.batch
                file_list = multimodalrequestout.file_list
                input_texts = multimodalrequestout.input_texts
                print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
                print_log(self.rank, logger.info, f'current iter: {current_iter}')
                generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
                generate_text_list, token_num_list = decode_token(req_list, self.tokenizer)
                torch.npu.synchronize()
                e2e_end = time.time()
                e2e_time = e2e_end - e2e_start
                e2e_time_all += e2e_time
                all_input_texts.extend(input_texts)
                all_generate_text_list.extend(generate_text_list)
                all_token_num_list.extend(token_num_list)
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
                data_simplification=False
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config) as _:
                multimodalrequestout = self.prepare_request(input_texts,
                                                            image_path,
                                                            batch_size,
                                                            max_output_length,
                                                            current_iter=0)
                req_list = multimodalrequestout.req_list
                batch = multimodalrequestout.batch
                file_list = multimodalrequestout.file_list
                input_texts = multimodalrequestout.input_texts
                generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
            e2e_time_all += e2e_time
            all_input_texts.extend(input_texts)
        if self.predict_result:
            if self.local_rank == 0:
                image_answer_pairs = {}
                for text_index in range(len(all_input_texts)):
                    image_answer_pairs[file_list[text_index]] = all_generate_text_list[text_index]
                    image_answer_pairs = dict(sorted(image_answer_pairs.items()))
                if not os.path.exists(PRED_FILE):
                    with safe_open(PRED_FILE, "w") as f:
                        json.dump(image_answer_pairs, f)
                else:
                    with safe_open(PRED_FILE, "r") as f:
                        old_data = json.load(f)
                    old_data.update(image_answer_pairs)
                    old_data = dict(sorted(old_data.items()))
                    with safe_open(PRED_FILE, "w") as f:
                        json.dump(old_data, f)
        if self.performance:
            e2e_throughput = np.divide(batch * max_output_length, e2e_time_all + 1e-12)
            input_len = self.tokenizer([all_input_texts[0]], return_tensors="pt")["input_ids"].flatten().shape[0]
            output_len = all_token_num_list[0][1]
            e2e_time = e2e_time_all
            e2e_throughput = e2e_throughput
            if self.local_rank == 0:
                if not os.path.exists(PERF_FILE):
                    file_util = safe_open(PERF_FILE, 'a')
                    file_util.write(PERF_COLUMNS)
                else:
                    file_util = safe_open(PERF_FILE, 'a')
                file_util.write(f"{batch}, {input_len}, {output_len}, "
                                 f"{first_token}, {e2e_time}, {e2e_throughput}\n")
                file_util.close()
        if ENV.token_ids_save_enable:
            if self.local_rank == 0: 
                for idx, req in enumerate(req_list):
                    input_ids_save_filename = f"input_ids_{idx}.pth"
                    output_ids_save_filename = f"output_ids_{idx}.txt"

                    input_ids_save_path = os.path.join(ENV.token_ids_save_folder, input_ids_save_filename)
                    input_ids_save_path = file_utils.standardize_path(input_ids_save_path)
                    file_utils.check_file_safety(input_ids_save_path, 'w') # 检查输入
                    torch.save(req.input_ids.cpu(), input_ids_save_path)

                    output_path = os.path.join(ENV.token_ids_save_folder, output_ids_save_filename)
                    with safe_open(output_path, 'w', encoding='utf-8') as f:
                        f.write(' '.join(map(str, req_list[idx].out_token_list)))
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return all_generate_text_list, all_token_num_list, e2e_time_all


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        help="model and tokenizer path",
        default='/data/weights/bunny',
    )
    parser.add_argument(
        '--input_texts', type=str, nargs="+", default=["用500字详细描述图片"]
    )
    parser.add_argument('--image_path', type=str, default="/data/images/bunny")
    parser.add_argument('--prediction_result', action=STORE_TRUE)
    parser.add_argument('--performance', action=STORE_TRUE)
    parser.add_argument('--input_ids', type=parse_ids, nargs="+", default=None)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_output_length', type=int, default=256)
    parser.add_argument('--max_prefill_tokens', type=int, default=-1)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--quantize", type=str, default=None)
    parser.add_argument("--is_flash_model", action="store_false")
    parser.add_argument("--is_bf16", action="store_true")
    parser.add_argument(
        '--num_beams', type=int, help="Use beam search if num_beams >1", default=1
    )
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--ignore_eos', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    pa_runner = MultiModalPARunner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')

    infer_params = {
        "inputs": InputAttrs(args.input_texts, args.image_path),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }

    pa_runner.warm_up()
    generate_texts, token_nums, latency = pa_runner.infer(**infer_params)

    for i, generate_text in enumerate(generate_texts):
        length = len(args.input_ids) if args.input_ids else len(args.input_texts)
        if i < length:
            input_text_filter = args.input_texts[i]
            input_text_filter = input_text_filter.replace('\n', ' ').replace('\r', ' ').replace('\f', ' ')
            input_text_filter = input_text_filter.replace('\t', ' ').replace('\v', ' ').replace('\b', ' ')
            input_text_filter = input_text_filter.replace('\u000A', ' ').replace('\u000D', ' ').replace('\u000C', ' ')
            input_text_filter = input_text_filter.replace('\u000B', ' ').replace('\u0008', ' ').replace('\u007F', ' ')
            input_text_filter = input_text_filter.replace('\u0009', ' ').replace('    ', ' ')
            print_log(rank, logger.info, f'Question[{i}]: {input_text_filter}')
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
        print_log(rank, logger.info, f"Latency: {latency}")