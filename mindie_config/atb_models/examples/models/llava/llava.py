# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
import math
import os

import torch
import torch_npu
from transformers import AutoProcessor

from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from atb_llm.utils import argument_utils
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.multimodal_utils import is_image_path, is_video_path, MultimodalInput, MultimodalRequestOut
from atb_llm.models.base.model_utils import safe_from_pretrained
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator, num_validator
from examples.run_pa import parse_ids
from examples.server.request import request_from_multimodalinputs, MultiModalReqParams


STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/llava/llava_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/llava/predict_result.json"


def is_llava_next34b(config):
    num_hidden_layers = config.text_config.num_hidden_layers
    architectures = config.architectures
    if num_hidden_layers in [60] and architectures[0] == "LlavaNextForConditionalGeneration":
        return True
    return False


@dataclass
class LlavaReqParams(MultiModalReqParams):
    frames: int = 8
    

class LlavaRunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.processor = None
        super().__init__(**kwargs)
        self.frames = kwargs.get('frames', 8)
        
    def init_processer(self):
        if is_llava_next34b(self.model.config):
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path, use_fast=False, 
                                                  trust_remote_code=self.trust_remote_code)
        else:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path, 
                                                  trust_remote_code=self.trust_remote_code)

    def prepare_request(self, multimodalinputs, batch_size, max_output_length, current_iter):
        input_texts = multimodalinputs.input_texts
        img_path = multimodalinputs.image_path
        vid_path = multimodalinputs.video_path
        req_list = [request_from_multimodalinputs(
                    self.processor,
                    self.model,
                    LlavaReqParams(input_texts[current_iter * batch_size
                                                        + single_batch] if input_texts else None,
                                        img_path[current_iter * batch_size
                                                        + single_batch] if img_path else None,
                                        vid_path[current_iter * batch_size
                                                        + single_batch] if vid_path else None,
                                        None,
                                        max_output_length,
                                        self.block_size,
                                        req_idx=single_batch,
                                        adapter_id=self.adapter_id,
                                        batch_size=batch_size,
                                        frames=self.frames))  
                    for single_batch in range(batch_size)]
        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')
        return MultimodalRequestOut(req_list, batch_size, img_path, vid_path, None, input_texts)
            
    def precision_save(self, precision_inputs, **kwargs):
        all_input_texts = precision_inputs.all_input_texts
        all_generate_text_list = precision_inputs.all_generate_text_list
        image_file_list = precision_inputs.image_file_list
        video_file_list = precision_inputs.video_file_list
        file_list = image_file_list if image_file_list else video_file_list
        image_answer_pairs = {}
        for text_index in range(len(all_input_texts)):
            image_answer_pairs[file_list[text_index]] = all_generate_text_list[text_index]
            image_answer_pairs = dict(sorted(image_answer_pairs.items()))
        super().precision_save(precision_inputs, answer_pairs=image_answer_pairs)
            
    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        input_texts = mm_inputs.input_texts
        img_path = mm_inputs.image_path
        vid_path = mm_inputs.video_path
        path_list = img_path if img_path else vid_path
        
        if len(input_texts) != len(path_list):
            raise RuntimeError("input_text length must equal input_images length")
        if not ENV.profiling_enable:
            try:
                max_iters = math.ceil(len(path_list) / self.max_batch_size)
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)

    
def parse_arguments():
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    list_num_validator = argument_utils.ListArgumentValidator(num_validator, 
                                                              max_length=1000, 
                                                              allow_none=True)
    parser_llava = parser
    parser_llava.add_argument('--image_or_video_path',
                        help="image_or_video path",
                        default="/data/acltransformer_testdata/llava",
                        validator=path_validator,
                        )
    parser_llava.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=["USER: <image>\nDescribe this image in detail. ASSISTANT:"],
        validator=list_str_validator,
        )
    parser_llava.add_argument(
        '--input_texts_for_video',
        type=str,
        nargs='+',
        default=["USER: <video>\nDescribe this video in detail. ASSISTANT:"],
        validator=list_str_validator)
    parser_llava.add_argument(
        '--video_frames',
        type=int,
        default=8,
        validator=num_validator)
    parser_llava.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None,
        validator=list_num_validator)

    return parser_llava.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        'pred_file': PRED_FILE,
        'frames': args.video_frames,
        **vars(args)
    }
    
    image_or_video_path = standardize_path(args.image_or_video_path)
    check_file_safety(image_or_video_path, 'r')
    file_name = safe_listdir(image_or_video_path)
    file_length = len(file_name)
    file_path = [os.path.join(image_or_video_path, f) for f in file_name]
    
    if is_video_path(image_or_video_path):
        new_input_texts = args.input_texts_for_video
        video_path = file_path
        image_path = None
    if is_image_path(image_or_video_path):
        new_input_texts = args.input_texts_for_image
        image_path = file_path
        video_path = None
    new_input_texts.extend([new_input_texts[-1]] * (file_length - len(new_input_texts)))
    input_dict['input_texts'] = new_input_texts
    input_dict['image_path'] = image_path
    input_dict['video_path'] = video_path
    
    pa_runner = LlavaRunner(**input_dict)
    
    if image_path:
        image_length = len(image_path)
        remainder = image_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            image_path.extend([image_path[-1]] * num_to_add)
            new_input_texts.extend([new_input_texts[-1]] * num_to_add)
    else:
        video_length = len(video_path)
        remainder = video_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            video_path.extend([video_path[-1]] * num_to_add)
            new_input_texts.extend([new_input_texts[-1]] * num_to_add)
    

    infer_params = {
        "mm_inputs": MultimodalInput(new_input_texts,
                                image_path,
                                video_path,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    pa_runner.warm_up()
    pa_runner.infer(**infer_params)