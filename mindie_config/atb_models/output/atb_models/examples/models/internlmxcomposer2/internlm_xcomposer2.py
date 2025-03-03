# Copyright Huawei Technologies Co., Ltd. 2024-2028. All rights reserved.
import math
import os
import json

from transformers import AutoProcessor

from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.log import logger, print_log
from atb_llm.utils import argument_utils
from atb_llm.utils.multimodal_utils import MultimodalInput
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator


PERF_FILE = os.path.join(os.path.dirname(__file__), "internlmxcomposer2_performance.csv")
PRED_FILE = os.path.join(os.path.dirname(__file__), "predict_result.json")


class InternlmXC2Runner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.processor = None
        super().__init__(**kwargs)
        self.adapter_id = kwargs.get("lora_adapter_id", None)

    def init_processor(self):
        try:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path,
                                                  trust_remote_code=self.trust_remote_code)
        except AssertionError:
            self.processor = self.model.tokenizer

    def precision_save(self, precision_inputs, **kwargs):
        image_answer_pairs = {}
        all_generate_text_list = precision_inputs.all_generate_text_list
        all_generate_text_list = [answer.split('[UNUSED_TOKEN_145]')[0] for answer in all_generate_text_list]
        image_file_list = precision_inputs.image_file_list
        file_len = len(image_file_list)
        for idx in range(file_len):
            image_answer_pairs[image_file_list[idx]] = all_generate_text_list[idx]
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

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, **kwargs):
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
    parser_internlmxc2 = parser
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    

    parser_internlmxc2.add_argument(
        '--image_or_video_path',
        help="image_or_video path",
        default="/data/internlmxcomposer2/images/",
        validator=path_validator)
    parser_internlmxc2.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=["<ImageHere>Please describe this image in detail."],
        validator=list_str_validator)
    parser_internlmxc2.add_argument(
        '--lora_adapter_id',
        help="Lora input, accepted adapter id defined in lora_adapater param",
        type=str,
        default="internlmxc2",
        validator=string_validator)

    return parser_internlmxc2.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    image_or_video_path = standardize_path(args.image_or_video_path)
    check_file_safety(image_or_video_path, 'r')
    file_name = safe_listdir(image_or_video_path)
    image_path = [os.path.join(image_or_video_path, f) for f in file_name]
    texts = args.input_texts_for_image
    image_length = len(image_path)
    if len(texts) != image_length:
        texts.extend([texts[-1]] * (image_length - len(texts)))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        'image_path': image_path,
        'lora_adapter_id': args.lora_adapter_id,
        'input_texts': texts,
        **vars(args)
    }

    pa_runner = InternlmXC2Runner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    

    infer_params = {
        "mm_inputs": MultimodalInput(texts,
                                image_path,
                                None,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
        "skip_special_tokens": True
    }
    pa_runner.warm_up()
    generate_texts, token_nums, latency = pa_runner.infer(**infer_params)
    generate_texts = [answer.split('[UNUSED_TOKEN_145]')[0] for answer in generate_texts]
    all_token_nums = 0
    for i, generate_text in enumerate(generate_texts):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
        print_log(rank, logger.info, f"Latency: {latency}")
        all_token_nums += token_nums[i][1]
    print_log(rank, logger.info, f"All Token Nums: {all_token_nums}")