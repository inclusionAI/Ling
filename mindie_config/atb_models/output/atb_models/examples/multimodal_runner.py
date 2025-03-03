# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import csv
import json
import math
import os
import time

import torch
import torch_npu
from transformers import AutoProcessor

from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.argument_utils import NumberArgumentValidator, StringArgumentValidator
from atb_llm.utils.argument_utils import BooleanArgumentValidator, ArgumentParser, ArgumentAction
from atb_llm.utils.multimodal_utils import MultimodalRequestOut, MultimodalInput, RunReturns
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import file_utils
from examples.run_pa import PARunner
from examples.server.request import request_from_multimodalinputs, MultiModalReqParams
from examples.server.cache import CacheManager, CacheConfig
from examples.server.generate import generate_req, decode_token

PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"


class MultimodalPARunner(PARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = None
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.skip_special_tokens = kwargs.get("skip_special_tokens", False)
        self.predict_result = kwargs.get('prediction_result', False)
        self.performance = kwargs.get('performance', False)
        self.max_prefill_tokens = kwargs.get('max_prefill_tokens', None)
        self.perf_file = kwargs.get('perf_file', None)
        self.pred_file = kwargs.get('pred_file', None)
        self.warm_up_num_blocks = 0
        self.warm_up_memory = 0
        self.cache_manager = None
        self.adapter_id = None
        self.warm_up_inputs = MultimodalInput(kwargs.get('input_texts', None),
                                           kwargs.get('image_path', None),
                                           kwargs.get('video_path', None),
                                           kwargs.get('audio_path', None))
        self.init_processor()
    
    def init_processor(self):
        try:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path, 
                                                  trust_remote_code=self.trust_remote_code)
        except AssertionError:
            self.processor = self.model.tokenizer

    def prepare_request(self, multimodalinputs, batch_size, max_output_length, current_iter):
        input_texts = multimodalinputs.input_texts
        image_path = multimodalinputs.image_path
        video_path = multimodalinputs.video_path
        audio_path = multimodalinputs.audio_path
        req_list = [request_from_multimodalinputs(
                    self.processor,
                    self.model,
                    MultiModalReqParams(input_texts[current_iter * batch_size
                                                        + single_batch] if input_texts else None,
                                        image_path[current_iter * batch_size
                                                        + single_batch] if image_path else None,
                                        video_path[current_iter * batch_size
                                                        + single_batch] if video_path else None,
                                        audio_path[current_iter * batch_size
                                                        + single_batch] if audio_path else None,
                                        max_output_length,
                                        self.block_size,
                                        req_idx=single_batch,
                                        adapter_id=self.adapter_id,
                                        batch_size=batch_size))  
                    for single_batch in range(batch_size)]
        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
        return MultimodalRequestOut(req_list, batch_size, image_path, video_path, audio_path, input_texts)
    
    def make_cache_manager(self):
        if not self.cache_manager:
            if self.max_prefill_tokens == -1:
                self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
            cache_block_size = self.block_size * self.model.num_kv_heads * self.model.head_size
            dtype_size = CacheManager.get_dtype_size(self.dtype)
            total_cache_size = self.model.num_layers * cache_block_size * 2 * dtype_size
            # 1 << 30正好是1G
            max_memory = ENV.memory_fraction * self.max_memory
            free_memory = max_memory - ENV.reserved_memory_gb * (1 << 30) - (
                self.warm_up_memory if self.warm_up_memory != 0 else self.init_memory)
            print_log(self.rank, logger.info,
                      f"infer max_memory(GB): {max_memory / (1024 ** 3): .2f}, "
                      f"warm_up_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}, "
                      f"free_memory(GB): {free_memory / (1024 ** 3): .2f}")
            try:
                num_blocks = int(free_memory // total_cache_size)
            except ZeroDivisionError as e:
                raise ZeroDivisionError("total_cache_size should not be 0") from e
            print_log(self.rank, logger.info, f"num_blocks: {num_blocks}, free_memory: {free_memory}")
            cache_config = CacheConfig(num_blocks, self.block_size)
            self.cache_manager = CacheManager(cache_config, self.model_config)
    
    def save_token_ids(self, req_list):
        for idx, req in enumerate(req_list):
            input_ids_save_filename = f"input_ids_{idx}.pth"
            input_ids_save_path = os.path.join(ENV.token_ids_save_folder, input_ids_save_filename)
            input_ids_save_path = file_utils.standardize_path(input_ids_save_path)
            file_utils.check_file_safety(input_ids_save_path, 'w')
            torch.save(req.input_ids.cpu(), input_ids_save_path)
                
            output_ids_save_filename = f"output_ids_{idx}.txt"
            output_ids_save_path = os.path.join(ENV.token_ids_save_folder, output_ids_save_filename)
            with file_utils.safe_open(output_ids_save_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(map(str, req.out_token_list)))

    def performance_save(self, input_len, all_token_num_list, e2e_time_all, batch):
        output_len = all_token_num_list[0][1]
        e2e_throughput = batch * output_len / (e2e_time_all + 1e-12)
        e2e_time = e2e_time_all * 1000
        e2e_throughput = e2e_throughput
        if ENV.benchmark_filepath:
            benchmark_csv = ENV.benchmark_filepath
        else:
            script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.dirname(os.path.dirname(script_path))
            benchmark_csv = os.path.join(script_path, "examples/atb_models/benchmark_result/benchmark.csv")
        if self.local_rank == 0:
            with file_utils.safe_open(benchmark_csv, newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                second_row = next(csv_reader)
                embedding_length = int(second_row[1])
                first_token_time = float(second_row[4])
                non_first_token_time = float(second_row[5])
            if not os.path.exists(self.perf_file):
                file_utils_write = file_utils.safe_open(self.perf_file, 'a')
                file_utils_write.write(PERF_COLUMNS)
                file_utils_write.write(f"{batch}, {input_len}, {output_len},{embedding_length}, \
                {first_token_time},{non_first_token_time}, {e2e_time}, \
                {e2e_throughput}\n")
                file_utils_write.close()
            else:
                file_utils_write = file_utils.safe_open(self.perf_file, 'a')
                file_utils_write.write(f"{batch}, {input_len}, {output_len},{embedding_length}, \
                {first_token_time},{non_first_token_time}, {e2e_time}, \
                {e2e_throughput}\n")
                file_utils_write.close()
        
    def performance_test(self, performance_inputs):
        all_input_texts = performance_inputs.all_input_texts
        all_token_num_list = performance_inputs.all_token_num_list
        e2e_time_all = performance_inputs.e2e_time_all
        batch = performance_inputs.batch
        input_len = self.tokenizer([str(all_input_texts[0])], return_tensors="pt")["input_ids"].flatten().shape[0]
        self.performance_save(input_len, all_token_num_list, e2e_time_all, batch)
    
    def precision_save(self, precision_inputs, **kwargs):
        answer_pairs = kwargs.get("answer_pairs", None)
        if not answer_pairs:
            raise ValueError("answer_pairs is not provided.")
        if not os.path.exists(self.pred_file):
            with file_utils.safe_open(self.pred_file, "w") as f:
                json.dump(answer_pairs, f)
        else:
            with file_utils.safe_open(self.pred_file, "r") as f:
                old_data = json.load(f)
            old_data.update(answer_pairs)
            old_data = dict(sorted(old_data.items()))
            with file_utils.safe_open(self.pred_file, "w") as f:
                json.dump(old_data, f)

    def profiling(self, inputs, batch_size, max_output_length):
        all_input_texts = []
        e2e_time_all = 0
        batch = None
        req_list = None
        print_log(self.rank, logger.debug, "enter profiling")
        profiling_path = ENV.profiling_filepath
        profiling_path = file_utils.standardize_path(profiling_path)
        file_utils.check_file_safety(profiling_path, 'w')
        if not os.path.exists(profiling_path):
            os.makedirs(profiling_path, mode=0o750, exist_ok=True)
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
            experimental_config=experimental_config):
            multimodalrequestout = self.prepare_request(inputs,
                                                        batch_size,
                                                        max_output_length,
                                                        current_iter=0)
            req_list = multimodalrequestout.req_list
            batch = multimodalrequestout.batch
            image_file_list = multimodalrequestout.image_file_list
            video_file_list = multimodalrequestout.video_file_list
            audio_file_list = multimodalrequestout.audio_file_list
            final_input_texts = multimodalrequestout.input_texts
            generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        torch.npu.synchronize()
        e2e_end = time.time()
        e2e_time = e2e_end - e2e_start
        e2e_time_all += e2e_time
        all_input_texts.extend(final_input_texts)
        return RunReturns(req_list, 
                        batch, 
                        image_file_list, 
                        video_file_list,
                        audio_file_list,
                        all_input_texts, 
                        [], 
                        [], 
                        e2e_time_all)

    def run(self, inputs, batch_size, max_output_length, ignore_eos, max_iters):
        all_generate_text_list = []
        all_token_num_list = []
        e2e_time_all = 0
        batch = None
        req_list = None
        print_log(self.rank, logger.debug, "no profiling")
        torch.npu.synchronize()
        if ignore_eos:
            self.model.postprocessor.eos_token_id = []
        for current_iter in range(max_iters):
            e2e_start = time.time()
            multimodalrequestout = self.prepare_request(inputs,
                                                        batch_size,
                                                        max_output_length,
                                                        current_iter)
            req_list = multimodalrequestout.req_list
            batch = multimodalrequestout.batch
            image_file_list = multimodalrequestout.image_file_list
            video_file_list = multimodalrequestout.video_file_list
            audio_file_list = multimodalrequestout.audio_file_list
            final_input_texts = multimodalrequestout.input_texts
            print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
            print_log(self.rank, logger.info, f'current iter: {current_iter}')
            generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)

            generate_text_list, token_num_list = decode_token(req_list, self.tokenizer, self.skip_special_tokens)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
            e2e_time_all += e2e_time
            all_generate_text_list.extend(generate_text_list)
            all_token_num_list.extend(token_num_list)
        return RunReturns(req_list, 
                          batch, 
                          image_file_list, 
                          video_file_list,
                          audio_file_list,
                          final_input_texts, 
                          all_generate_text_list, 
                          all_token_num_list, 
                          e2e_time_all)

    def warm_up(self):
        image_path = self.warm_up_inputs.image_path
        video_path = self.warm_up_inputs.video_path
        audio_path = self.warm_up_inputs.audio_path
        input_texts = self.warm_up_inputs.input_texts
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (self.max_input_length + self.max_output_length)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = math.ceil((self.max_input_length + self.max_output_length) /
                                                self.block_size) * self.max_batch_size
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        req_list = [request_from_multimodalinputs(
                        self.processor,
                        self.model,
                        MultiModalReqParams(input_texts[0] if input_texts else None,
                                                image_path[0] if image_path else None,
                                                video_path[0] if video_path else None,
                                                audio_path[0] if audio_path else None,
                                                self.max_output_length,
                                                self.block_size,
                                                req_idx=single_batch,
                                                adapter_id=self.adapter_id,
                                                batch_size=self.max_batch_size))
                        for single_batch in range(self.max_batch_size)
                    ]
        self.model.postprocessor.max_new_tokens = 2
        
        generate_req(req_list, self.model, self.max_batch_size, self.max_prefill_tokens, self.cache_manager)
        self.warm_up_memory = int(
            self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.model.soc_info.need_nz))
        print_log(self.rank, logger.info, f'warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}')
        print_log(self.rank, logger.info, "---------------end warm_up---------------")
    
    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        self.make_cache_manager()   
        self.model.postprocessor.max_new_tokens = max_output_length
        all_generate_text_list = []
        all_token_num_list = []
        e2e_time_all = 0
        if not ENV.profiling_enable:
            if not max_iters:
                raise ValueError("max_iters is not provided")
            infer_results = self.run(mm_inputs,
                        batch_size, 
                        max_output_length, 
                        ignore_eos,
                        max_iters)
        else:
            infer_results = self.profiling(mm_inputs,
                                        batch_size, 
                                        max_output_length)
        all_generate_text_list = infer_results.all_generate_text_list
        all_token_num_list = infer_results.all_token_num_list
        e2e_time_all = infer_results.e2e_time_all
        if self.predict_result:
            if self.local_rank == 0:   
                self.precision_save(infer_results)
        if self.performance:
            self.performance_test(infer_results)
        if ENV.token_ids_save_enable:
            if self.local_rank == 0:
                self.save_token_ids(infer_results.req_list)
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return all_generate_text_list, all_token_num_list, e2e_time_all
    

num_validator = NumberArgumentValidator(min_value=0)
path_validator = StringArgumentValidator(min_length=0, max_length=4096)
bool_validator = BooleanArgumentValidator()
max_position_embeddings_validator = NumberArgumentValidator(min_value=0, allow_none=True)
max_prefill_tokens_validator = NumberArgumentValidator(min_value=0, special_values=[-1])
block_size_validator = NumberArgumentValidator(min_value=0, max_value=576)

parser = ArgumentParser(description="")
parser.add_argument('--model_path', help="model and tokenizer path", default='', validator=path_validator)
parser.add_argument('--prediction_result', action=ArgumentAction.STORE_TRUE.value, validator=bool_validator)
parser.add_argument('--performance', action=ArgumentAction.STORE_TRUE.value, validator=bool_validator)
parser.add_argument('--max_position_embeddings', type=int, default=None,
                    validator=max_position_embeddings_validator)
parser.add_argument('--max_input_length', type=int, default=4096, validator=num_validator)
parser.add_argument('--max_output_length', type=int, default=256, validator=num_validator)
parser.add_argument('--max_prefill_tokens', type=int, default=-1, validator=max_prefill_tokens_validator)
parser.add_argument("--max_batch_size", type=int, default=1, validator=num_validator)
parser.add_argument("--block_size", type=int, default=128, validator=block_size_validator)

parser.add_argument('--num_beams', type=int, help="Use beam search if num_beams >1", default=1,
                    validator=num_validator)
parser.add_argument('--temperature', type=float, default=1.0, validator=num_validator)
parser.add_argument('--top_k', type=int, default=1, validator=num_validator)
parser.add_argument('--top_p', type=float, default=0.0, validator=num_validator)
parser.add_argument('--length_penalty', type=float, default=1.0, validator=num_validator)
parser.add_argument('--repetition_penalty', type=float, default=1.0, validator=num_validator)
parser.add_argument('--presence_penalty', type=float, default=0.0, validator=num_validator)
parser.add_argument('--frequency_penalty', type=float, default=0.0, validator=num_validator)
parser.add_argument('--ignore_eos', action=ArgumentAction.STORE_TRUE.value, validator=bool_validator)
parser.add_argument('--trust_remote_code', action=ArgumentAction.STORE_TRUE.value, validator=bool_validator)