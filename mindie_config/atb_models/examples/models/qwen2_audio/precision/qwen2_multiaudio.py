# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import csv
from atb_llm.utils import argument_utils
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.multimodal_utils import MultimodalInput
from atb_llm.utils.file_utils import safe_open

from examples.run_pa import parse_ids
from examples.multimodal_runner import parser
from examples.multimodal_runner import path_validator, num_validator
from examples.models.qwen2_audio.qwen2_audio import Qwen2AudioRunner


def parse_arguments():
    parser_qwen2multiaudio = parser
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
    list_num_validator = argument_utils.ListArgumentValidator(num_validator, 
                                                              max_length=1000, 
                                                              allow_none=True)
    parser_qwen2multiaudio.add_argument('--audio_path',
                        help="image_or_video path",
                        default="/data/dataset/npu_gpu_100",
                        validator=path_validator
                        )
    parser_qwen2multiaudio.add_argument(
        '--input_texts_for_audio',
        type=str,
        nargs='+',
        default=["prompt"],
        validator=list_str_validator)
    parser_qwen2multiaudio.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None,
        validator=list_num_validator)

    return parser_qwen2multiaudio.parse_args()


def get_audio_data(data_path, csv_file, audio_root):
    audio_dict_second = []
    with safe_open(os.path.join(data_path, csv_file), encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            audio = os.path.join(audio_root, row[0])
            if os.path.exists(audio):
                audio_dict_second.append(audio)
            else:
                raise ValueError(f"can not find file: {row[0]}, please check.")
    return audio_dict_second


if __name__ == '__main__':
    args = parse_arguments()
    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }
    data_path = "./examples/models/qwen2_audio/precision/multi_audio_data/"
    audio_root = args.audio_path
    save_path = "./examples/models/qwen2_audio/precision/qwen2_audio_npu_test_pure.csv"
    audio_dict_first = get_audio_data(data_path, "data_first.csv", audio_root)
    audio_dict_second = get_audio_data(data_path, "data_second.csv", audio_root)

    pa_runner = Qwen2AudioRunner(**input_dict)
    with safe_open(save_path, mode='w', encoding='utf-8') as file:
        filenames = ["first_audio", "second_audio", "answer"]
        writer = csv.DictWriter(file, fieldnames=filenames)
        writer.writeheader()
        for idx, video_first in enumerate(audio_dict_first):
            video_second = audio_dict_second[idx]
            prompt_text1 = "What did the speaker say in the first audio? "
            prompt_text2 = "And what did the speaker say in the second audio?"
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": video_first},
                    {"type": "audio", "audio_url": video_second},
                    {"type": "text", "text": prompt_text1 + prompt_text2},
                ]},
            ] 
            inputs = [conversation]
            infer_params = {
                "mm_inputs": MultimodalInput(inputs,
                                        None,
                                        None,
                                        ['placeholder']),
                "batch_size": args.max_batch_size,
                "max_output_length": args.max_output_length,
                "ignore_eos": args.ignore_eos,
            }
            generate_texts, token_nums, e2e_time_gene = pa_runner.infer(**infer_params)

            for i, generate_text in enumerate(generate_texts):
                print_log(rank, logger.info, f'url_pre[{i}]: {video_first}')
                print_log(rank, logger.info, f'url[{i}]: {video_second}')
                print_log(rank, logger.info, f'Answer[{i}]: {generate_text}')
                print_log(rank, logger.info, f'Generate[{i}] token num: {token_nums[i]}')
                answer_dict = {"first_audio": video_first, "second_audio": video_second, "answer": generate_text}
                writer.writerow(answer_dict)

                