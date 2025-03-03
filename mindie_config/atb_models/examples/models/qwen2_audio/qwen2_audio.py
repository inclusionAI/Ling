# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
import os
import json

from transformers import AutoProcessor
from atb_llm.utils import argument_utils
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_open, safe_listdir, standardize_path, check_file_safety
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.multimodal_utils import MultimodalInput

from examples.run_pa import parse_ids
from examples.multimodal_runner import MultimodalPARunner, parser
from examples.multimodal_runner import path_validator, num_validator


STORE_TRUE = "store_true"
PERF_FILE = "./examples/models/qwen2_audio/qwen2_audio_performance.csv"
PERF_COLUMNS = "batch, input_len, output_len, embedding_len, fisrt_token_time(ms), \
                non_first_token_time(ms), ResponseTime(ms),E2E Throughput Average(Tokens/s)\n"
PRED_FILE = "./examples/models/qwen2_audio/predict_result.json"
MAX_INPUT_LENGTH = 100000


class Qwen2AudioRunner(MultimodalPARunner):
    def __init__(self, **kwargs):
        self.processor = None
        super().__init__(**kwargs)
    
    def init_processer(self):
        try:
            self.processor = safe_from_pretrained(AutoProcessor, self.model_path)
        except AssertionError:
            self.processor = self.model.tokenizer
    
    def precision_save(self, precision_inputs, **kwargs):
        all_input_texts = precision_inputs.all_input_texts
        all_generate_text_list = precision_inputs.all_generate_text_list
        audio_file_list = precision_inputs.audio_file_list
        image_answer_pairs = {}
        if len(all_input_texts) > MAX_INPUT_LENGTH:
            raise ValueError("all_input_texts length must less than MAX_INPUT_LENGTH")

        for text_index in range(len(all_input_texts)):
            image_answer_pairs[audio_file_list[text_index]] = all_generate_text_list[text_index]
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

    def infer(self, mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=None, **kwargs):
        input_texts = mm_inputs.input_texts
        audios_path_list = mm_inputs.audio_path
        if len(input_texts) != len(audios_path_list):
            raise RuntimeError("input_text length must equal input_audios length")
        if not ENV.profiling_enable:
            if self.max_batch_size > 0:
                max_iters = math.ceil(len(mm_inputs.audio_path) / self.max_batch_size)
            else:
                raise RuntimeError("f{self.max_batch_size} max_batch_size should > 0, please check")
        return super().infer(mm_inputs, batch_size, max_output_length, ignore_eos, max_iters=max_iters)
    

def parse_arguments():
    prompt_head = "<|audio_bos|><|AUDIO|><|audio_eos|>In this audio, what kind of sound can you hear? "
    prompt_body1 = "A: Laughter, B: Sigh, C: Cough, D: Throat clearing, E: Sneeze, F: Sniff, "
    prompt_body2 = "Please select the one closest to the correct answer. ASSISTANT:"
    prompt = prompt_head + prompt_body1 + prompt_body2
    
    parser_qwen2audio = parser
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    list_num_validator = argument_utils.ListArgumentValidator(num_validator, 
                                                              max_length=1000, 
                                                              allow_none=True)
    parser_qwen2audio.add_argument('--audio_path',
                        help="image_or_video path",
                        default="/data/qwen2_audio_data/audio",
                        validator=path_validator
                        )
    parser_qwen2audio.add_argument(
        '--input_texts_for_audio',
        type=str,
        nargs='+',
        default=[prompt],
        validator=list_str_validator)
    parser_qwen2audio.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None,
        validator=list_num_validator)

    return parser_qwen2audio.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size

    audio_path = standardize_path(args.audio_path)
    check_file_safety(audio_path, 'r')
    file_name = safe_listdir(audio_path, max_file_num=MAX_INPUT_LENGTH)
    audio_path_list = [os.path.join(audio_path, f) for f in file_name]
    texts = args.input_texts_for_audio
    audio_length = len(audio_path_list)
    if len(texts) != audio_length:
        texts.extend([texts[-1]] * (audio_length - len(texts)))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'perf_file': PERF_FILE,
        **vars(args)
    }
    input_dict['audio_path'] = audio_path_list
    input_dict['input_texts'] = texts

    pa_runner = Qwen2AudioRunner(**input_dict)

    remainder = audio_length % args.max_batch_size
    if remainder != 0:
        num_to_add = args.max_batch_size - remainder
        audio_path_list.extend([audio_path_list[-1]] * num_to_add)
        texts.extend([texts[-1]] * num_to_add)

    infer_params = {
        "mm_inputs": MultimodalInput(texts,
                                None,
                                None,
                                audio_path_list),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    pa_runner.warm_up()
    generate_texts, token_nums, e2e_time_gene = pa_runner.infer(**infer_params)

    for i, generate_text in enumerate(generate_texts):
        print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
        print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
