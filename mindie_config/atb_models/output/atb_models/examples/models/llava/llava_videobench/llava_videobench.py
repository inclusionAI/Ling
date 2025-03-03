# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json

import torch

from atb_llm.utils import argument_utils
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open, standardize_path, check_file_safety
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.multimodal_utils import MultimodalInput
from examples.multimodal_runner import parser, path_validator
from examples.models.llava.llava import LlavaRunner
from examples.models.llava.llava_videobench.videobench_utils import process_qs, select_data

STORE_TRUE = "store_true"
PRED_FILE = "./examples/models/llava/predict_result.json"
MAX_TEXT_LENGTH = 4096
MAX_STRING_LENGTH = 4096

            
def parse_arguments():
    string_validator = argument_utils.StringArgumentValidator(min_length=0, max_length=1000)
    list_str_validator = argument_utils.ListArgumentValidator(string_validator, max_length=1000)
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
        validator=list_str_validator)
    parser_llava.add_argument(
        '--input_texts_for_video',
        type=str,
        nargs='+',
        default=["USER: <video>\nDescribe this video in detail. ASSISTANT:"],
        validator=list_str_validator)
    parser_llava.add_argument('--dataset_name', type=str, default='Driving-decision-making', validator=string_validator)
    parser_llava.add_argument('--Eval_QA_root',
                        type=str,
                        default='/usr/local/Ascend/Video-Bench/',
                        help="folder containing QA JSON files",
                        validator=path_validator,
                        )
    parser_llava.add_argument('--chat_conversation_output_folder', type=str, default='./Chat_results',
                               validator=path_validator)
    return parser_llava.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    eval_qa_root = standardize_path(args.Eval_QA_root)
    check_file_safety(eval_qa_root, 'r')
    dataset_qajson = select_data(eval_qa_root)
    image_or_video_path = standardize_path(args.image_or_video_path)
    check_file_safety(image_or_video_path, 'r')
    if image_or_video_path.endswith('/'):
        image_or_video_path = image_or_video_path[:-1]
    args.dataset_name = image_or_video_path.split('/')[-1]
    path_video = image_or_video_path.split('/')[:-2]
    eval_video_root = '/'.join(path_video) + '/'   
    eval_video_root = standardize_path(eval_video_root)
    check_file_safety(eval_video_root, 'r')


    if args.dataset_name is None:
        dataset_name_list = list(dataset_qajson.keys())
    else:
        dataset_name_list = [args.dataset_name]
        print_log(0, logger.info, f"Specifically run {dataset_name_list[0]}")
    print_log(0, logger.info, f"dataset_name_list: {dataset_name_list}")

    chat_conversation_output_folder = standardize_path(args.chat_conversation_output_folder)
    check_file_safety(chat_conversation_output_folder, 'r')
    os.makedirs(chat_conversation_output_folder, exist_ok=True)

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    pa_runner = LlavaRunner(**input_dict)

    for dataset_name in dataset_name_list:
        if dataset_name not in dataset_qajson:
            continue
        qa_json = dataset_qajson[dataset_name]
        print_log(0, logger.info, f"Dataset name:{dataset_name}, qa_json:{qa_json}!")
        with safe_open(qa_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        args.input_texts_for_image = []
        args.input_texts_for_video = []
        args.input_file_list = []
        eval_dict = {}
        for _, (q_id, item) in enumerate(data.items()):
            try:
                question, vid_path = process_qs(item, eval_video_root, "USER: <video>\n")
                question += " Please respond with only the corresponding options and do not provide any explanations \
or additional information. ASSISTANT:"
                print_log(0, logger.info, f"input questions:{question}!\n")
                args.input_texts_for_video.append(question)
                image_or_video_path = "/".join(vid_path.split('/')[:-1])
                image_or_video_name = vid_path.split('/')[-1]
                args.input_file_list.append(os.path.join(image_or_video_path, image_or_video_name))

            except Exception as e:
                raise Exception from e

    infer_params = {
        "mm_inputs": MultimodalInput(args.input_texts_for_video,
                                None,
                                args.input_file_list,
                                None),
        "batch_size": args.max_batch_size,
        "max_output_length": args.max_output_length,
        "ignore_eos": args.ignore_eos,
    }
    all_generate_text_list, all_token_num_list, e2e_time_all = pa_runner.infer(**infer_params)

    eval_dict = {}
    for idx, (q_id, item) in enumerate(data.items()):
        video_id = item['video_id']
        question = item['question']
        dataset = item['dataset']
        eval_dict[q_id] = {
            "video_id": video_id,
            "question": question,
            "output_sequence": all_generate_text_list[idx],
            'dataset': dataset
        }
        print_log(0, logger.info, f"q_id:{q_id}, question:{question}, output:{all_generate_text_list[idx]}!\n")
    eval_dataset_json = f'{chat_conversation_output_folder}/{dataset_name}_eval.json'
    if local_rank == 0:
        with safe_open(eval_dataset_json, 'w', encoding='utf-8') as f:
            json.dump(eval_dict, f, indent=2)