# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
import csv
import time
import json
import math

from transformers import AutoProcessor
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.multimodal_utils import MultimodalInput
from atb_llm.models.base.model_utils import safe_from_pretrained
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.run_pa import parse_ids


STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/glm4v/glm4v_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/glm4v/predict_result.json"



class Glm4vPARunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def precision_save(self, precision_inputs, **kwargs):
        all_input_texts = precision_inputs.all_input_texts
        all_generate_text_list = precision_inputs.all_generate_text_list
        image_file_list = precision_inputs.image_file_list
        image_answer_pairs = {}
        for text_index in range(len(all_input_texts)):
            image_answer_pairs[image_file_list[text_index]] = all_generate_text_list[text_index]
            image_answer_pairs = dict(sorted(image_answer_pairs.items()))
        super().precision_save(precision_inputs, answer_pairs=image_answer_pairs)
                
    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        input_texts = mm_inputs.input_texts
        image_path_list = mm_inputs.image_path
        if len(input_texts) != len(image_path_list):
            raise RuntimeError("input_text length must equal input_images length")
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(len(mm_inputs.image_path) / self.max_batch_size)
            else:
                raise RuntimeError("f{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)
    

def parse_arguments():
    glm4v_parser = argparse.ArgumentParser()
    glm4v_parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='/home/data/glm-4v-9b/',
                        )
    glm4v_parser.add_argument('--image_path',
                        help="image path",
                        default="/home/data/images/",
                        )
    glm4v_parser.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=["描述这张图片"])
    glm4v_parser.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None)
    glm4v_parser.add_argument(
        '--prediction_result',
         action=STORE_TRUE)
    glm4v_parser.add_argument(
        '--performance',
         action=STORE_TRUE)
    glm4v_parser.add_argument('--max_position_embeddings', type=int, default=None)
    glm4v_parser.add_argument('--max_input_length', type=int, default=4096)
    glm4v_parser.add_argument('--max_output_length', type=int, default=128)
    glm4v_parser.add_argument('--max_prefill_tokens', type=int, default=-1)
    glm4v_parser.add_argument("--max_batch_size", type=int, default=1)
    glm4v_parser.add_argument("--block_size", type=int, default=128)

    glm4v_parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    glm4v_parser.add_argument('--temperature', type=float, default=1.0)
    glm4v_parser.add_argument('--top_k', type=int, default=1)
    glm4v_parser.add_argument('--top_p', type=float, default=0.0)
    glm4v_parser.add_argument('--length_penalty', type=float, default=1.0)
    glm4v_parser.add_argument('--repetition_penalty', type=float, default=1.0)
    glm4v_parser.add_argument('--presence_penalty', type=float, default=0.0)
    glm4v_parser.add_argument('--frequency_penalty', type=float, default=0.0)
    glm4v_parser.add_argument('--ignore_eos', action=STORE_TRUE)
    glm4v_parser.add_argument('--trust_remote_code', action=STORE_TRUE)

    return glm4v_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    image_path = standardize_path(args.image_path)
    check_file_safety(image_path, 'r')
    file_name = safe_listdir(image_path)
    image_path = [os.path.join(image_path, f) for f in file_name]
    texts = args.input_texts_for_image
    image_length = len(image_path)
    if len(texts) != image_length:
        texts.extend([texts[-1]] * (image_length - len(texts)))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        'pred_file': PRED_FILE,
        **vars(args)
    }
    input_dict['image_path'] = image_path
    input_dict['input_texts'] = texts

    pa_runner = Glm4vPARunner(**input_dict)
    
    remainder = image_length % args.max_batch_size
    if remainder != 0:
        num_to_add = args.max_batch_size - remainder
        image_path.extend([image_path[-1]] * num_to_add)
        texts.extend([texts[-1]] * num_to_add)

    infer_params = {
        "mm_inputs": MultimodalInput(
                                texts,
                                image_path,
                                None,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    pa_runner.warm_up()
    generate_texts, token_nums, e2e_time_gene = pa_runner.infer(**infer_params)
    
    length = len(args.input_texts_for_image)
    for i, generate_text in enumerate(generate_texts):
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {args.input_texts_for_image[i]}', need_filter=True)
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')