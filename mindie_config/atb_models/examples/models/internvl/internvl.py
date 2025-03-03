# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import os
import json
import torch

from atb_llm.utils import argument_utils
from atb_llm.utils import file_utils
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.multimodal_utils import MultimodalInput
from atb_llm.utils.log import logger, print_log
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from examples.run_pa import parse_ids
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator, num_validator

MAX_FILE_SIZE = 1000 * 1024 * 1024

STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/internvl/internvl_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, FirstToken(s), ResponseTime(s), E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/internvl/predict_result.json"


def is_image(file_image_name):
    ext = os.path.splitext(file_image_name)[1]
    ext = ext.lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return True
    return False


def is_video(file_video_name):
    video_ext = os.path.splitext(file_video_name)[1]
    video_ext = video_ext.lower()
    if video_ext in [".mp4", ".wmv", ".avi"]:
        return True
    return False


def is_image_path(path):
    path = standardize_path(path)
    check_file_safety(path)
    files = safe_listdir(path)
    if is_image(files[0]):
        return True
    return False


def is_video_path(path):
    path = standardize_path(path)
    check_file_safety(path)
    files = safe_listdir(path)
    if is_video(files[0]):
        return True
    return False


class InternvlRunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pred_file = PRED_FILE
        
    def init_processor(self):
        self.processor = self.model.tokenizer

    def save_token_ids(self, req_list):
        for idx, req in enumerate(req_list):
            input_ids_save_filename = f"input_ids_{idx}.pth"
            input_ids_save_path = os.path.join(ENV.token_ids_save_folder, input_ids_save_filename)
            input_ids_save_path = file_utils.standardize_path(input_ids_save_path)
            file_utils.check_file_safety(input_ids_save_path, 'w', max_file_size=MAX_FILE_SIZE)
            torch.save(req.input_ids.cpu(), input_ids_save_path)
                
            output_ids_save_filename = f"output_ids_{idx}.txt"
            output_ids_save_path = os.path.join(ENV.token_ids_save_folder, output_ids_save_filename)
            with file_utils.safe_open(output_ids_save_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(map(str, req.out_token_list)))

    def precision_save(self, precision_inputs, **kwargs):
        all_input_texts = precision_inputs.all_input_texts
        all_generate_text_list = precision_inputs.all_generate_text_list
        image_file_list = precision_inputs.image_file_list
        video_file_list = precision_inputs.video_file_list
        file_list = image_file_list if image_file_list else video_file_list
        answer_pairs = {}
        if not file_list:
            raise ValueError("Both image_file_list and video_file_list are empty.")
        if len(all_input_texts) != len(file_list):
            raise ValueError(f"Mismatched lengths between \
                all_input_texts={all_input_texts} and file_list={file_list}")
        for text_index in range(len(all_input_texts)):
            answer_pairs[file_list[text_index]] = all_generate_text_list[text_index]
            answer_pairs = dict(sorted(answer_pairs.items()))
        super().precision_save(precision_inputs, answer_pairs=answer_pairs)

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        input_texts = mm_inputs.input_texts
        image_path_list = mm_inputs.image_path
        video_path_list = mm_inputs.video_path
        path_list = image_path_list if image_path_list else video_path_list
        if len(input_texts) != len(path_list):
            raise RuntimeError("input_text length must equal input images or video length")
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(len(path_list) / self.max_batch_size)
            else:
                raise RuntimeError("f{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)


def parse_arguments():
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    list_num_validator = argument_utils.ListArgumentValidator(num_validator, 
                                                              max_length=1000, 
                                                              allow_none=True)
    video_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=4096, allow_none=True)
    parser_internvl = parser
    parser.update_argument('--max_input_length', default=12288)
    parser_internvl.add_argument('--image_or_video_path',
                        help="image_or_video path",
                        default="/data/internvl/images/",
                        validator=path_validator,
                        )
    parser_internvl.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=[
            "用500字详细描述图片",
        ],
        validator=list_str_validator)
    parser_internvl.add_argument(
        '--input_texts_for_video',
        type=str,
        nargs='+',
        default=[
            "用500字详细描述视频",
        ],
        validator=list_str_validator)
    parser_internvl.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None,
        validator=list_num_validator)
    return parser_internvl.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    image_or_video_path = standardize_path(args.image_or_video_path)
    check_file_safety(image_or_video_path, 'r')
    file_name = safe_listdir(image_or_video_path)
    file_length = len(file_name)
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        **vars(args)
    }
    if is_image_path(image_or_video_path):
        image_path = [os.path.join(image_or_video_path, f) for f in file_name]
        video_path = None
        input_dict['image_path'] = image_path
        texts = args.input_texts_for_image
    else:
        video_path = [os.path.join(image_or_video_path, f) for f in file_name]
        image_path = None
        input_dict['video_path'] = video_path
        texts = args.input_texts_for_video
    if len(texts) > file_length:
        raise ValueError(f"The number of input texts is greater than the number of files.")
    texts.extend([texts[-1]] * (file_length - len(texts)))
    input_dict['input_texts'] = texts
    pa_runner = InternvlRunner(**input_dict)
    if image_path:
        image_length = len(image_path)
        remainder = image_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            image_path.extend([image_path[-1]] * num_to_add)
            texts.extend([texts[-1]] * num_to_add)
    elif video_path:
        video_length = len(video_path)
        remainder = video_length % args.max_batch_size
        if remainder != 0:
            num_to_add = args.max_batch_size - remainder
            video_path.extend([video_path[-1]] * num_to_add)
            texts.extend([texts[-1]] * num_to_add)

    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    infer_params = {
        "mm_inputs": MultimodalInput(texts,
                                     image_path,
                                     video_path,
                                     None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }

    pa_runner.warm_up()
    generate_texts, token_nums, latency = pa_runner.infer(**infer_params)
    for i, generate_text in enumerate(generate_texts):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
        print_log(rank, logger.info, f"Latency: {latency}")