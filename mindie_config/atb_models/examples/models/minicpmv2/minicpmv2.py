# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import math
import os

from transformers import AutoProcessor

from atb_llm.utils.env import ENV
from atb_llm.utils import argument_utils
from atb_llm.utils.log import logger, print_log
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.file_utils import safe_listdir, standardize_path, check_file_safety
from atb_llm.utils.multimodal_utils import MultimodalInput
from examples.run_pa import parse_ids
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator, num_validator

STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/minicpmv2/minicpmv2_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/minicpmv2/predict_result.json"


class MinicpmV2Runner(MultimodalPARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_processer(self):
        try:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path,
                                                    trust_remote_code=self.trust_remote_code)
        except AssertionError:
            self.processor = self.model.tokenizer

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
    parser_minicpmv2 = parser
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_num_validator = argument_utils.ListArgumentValidator(num_validator, 
                                                              max_length=1000, 
                                                              allow_none=True)
    texts_validator_dict_element = argument_utils.DictionaryArgumentValidator({'role': string_validator, 
                                                                               'content': string_validator})
    texts_validator_dict = argument_utils.ListArgumentValidator(texts_validator_dict_element)
    texts_validator_str = argument_utils.ListArgumentValidator(string_validator)
    texts_validator = {str: texts_validator_str, dict: texts_validator_dict}

    parser_minicpmv2.add_argument('--image_or_video_path',
                        help="image_or_video path",
                        default="/data/acltransformer_testdata/minicpmv2",
                        validator=path_validator,
                        )
    parser_minicpmv2.add_argument(
        '--input_texts_for_image',
        type=str,
        nargs='+',
        default=[{'role': 'user', 'content': 'Write an essay about this image, at least 256 words.'}],
        validator=texts_validator)
    parser_minicpmv2.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None,
        validator=list_num_validator)

    return parser_minicpmv2.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
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
        'pred_file': PRED_FILE,
        **vars(args)
    }
    input_dict['image_path'] = image_path
    input_dict['input_texts'] = texts
        
    pa_runner = MinicpmV2Runner(**input_dict)
    
    remainder = image_length % args.max_batch_size
    if remainder != 0:
        num_to_add = args.max_batch_size - remainder
        image_path.extend([image_path[-1]] * num_to_add)
        texts.extend([texts[-1]] * num_to_add)
        
    
    infer_params = {
        "mm_inputs": MultimodalInput(texts,
                                image_path,
                                None,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    pa_runner.warm_up()
    generate_texts, token_nums, e2e_time_gene = pa_runner.infer(**infer_params)
    
    for i, generate_text in enumerate(generate_texts):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
