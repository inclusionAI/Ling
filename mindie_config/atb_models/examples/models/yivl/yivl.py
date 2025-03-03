# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import os
import json
from dataclasses import dataclass
from transformers import AutoTokenizer
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, safe_listdir
from atb_llm.utils.log import logger, print_log
from atb_llm.models.yivl.data_processor_yivl import DataProcessorYiVl
from atb_llm.utils.multimodal_utils import MultimodalInput
from atb_llm.utils.argument_utils import ListArgumentValidator, StringArgumentValidator
from examples.multimodal_runner import MultimodalPARunner, parser


_PERF_FILE = "./examples/models/yivl/yivl_performance.csv"
_PRED_FILE = "./examples/models/yivl/predict_result.json"


@dataclass
class PreProcessor:
    tokenizer : AutoTokenizer
    image_processor : DataProcessorYiVl


def get_img_list(path):
    files = safe_listdir(path, 10000) # max file num in list
    image_files = []
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            image_files.append(os.path.join(path, file))
    return image_files


class YivlRunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_processor(self):
        image_processor = DataProcessorYiVl(self.model.config.vison_tower_path,
                                trust_remote_code=self.trust_remote_code)
        self.processor = PreProcessor(self.model.tokenizer, image_processor)

    def precision_save(self, precision_inputs, **kwargs):
        image_list = precision_inputs.image_file_list
        text_res_list = precision_inputs.all_generate_text_list
        image_answer_pairs = {}
        for text_index in range(len(precision_inputs.all_input_texts)):
            image_answer_pairs[image_list[text_index]] = text_res_list[text_index]
            image_answer_pairs = dict(sorted(image_answer_pairs.items()))
        if os.path.exists(_PRED_FILE):
            with safe_open(_PRED_FILE, "r") as f:
                history_data = json.load(f)
            history_data.update(image_answer_pairs)
            history_data = dict(sorted(history_data.items()))
            with safe_open(_PRED_FILE, "w") as f:
                json.dump(history_data, f)
        else:
            with safe_open(_PRED_FILE, "w") as f:
                json.dump(image_answer_pairs, f)

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, **kwargs):
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(len(mm_inputs.image_path) / self.max_batch_size)
            else:
                raise RuntimeError("f{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)


def parse_arguments():
    parser_yivl = parser
    string_validator = StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = ListArgumentValidator(string_validator, max_length=1000)

    parser_yivl.add_argument(
        '--image_path',               
        help="image path",
        required=True,
        validator=string_validator)
    parser_yivl.add_argument(
        '--input_texts',
        type=str,
        nargs='+',
        default=["Describe this image in detail."],
        validator=list_str_validator)
    return parser_yivl.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        'perf_file': _PERF_FILE,
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    img_list = get_img_list(args.image_path)
    if (len(img_list) < 1):
        err_msg = "No image is found in given image path, please check"
        print_log(rank, logger.error, err_msg)
        raise ValueError(err_msg)
    input_dict.update({'image_path': img_list})

    pa_runner = YivlRunner(**input_dict)
    print_log(rank, logger.info, f'pa_runner: {pa_runner}')
    
    texts = args.input_texts
    image_num = len(img_list)
    texts_num = len(texts)
    if image_num > texts_num:
        texts.extend([texts[-1]] * (image_num - texts_num))
    else:
        texts = texts[:image_num]
        
    infer_params = {
        "mm_inputs": MultimodalInput(texts, img_list, None, None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    pa_runner.warm_up()
    generate_texts, token_nums, latency = pa_runner.infer(**infer_params) 
    input_texts = args.input_texts
    for i, generate_text in enumerate(generate_texts):
        length = len(input_texts)
        if i < length:
            print_log(rank, logger.info, f'Question[{i}]: {input_texts[i]}', need_filter=True)
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
        print_log(rank, logger.info, f"Latency: {latency}")