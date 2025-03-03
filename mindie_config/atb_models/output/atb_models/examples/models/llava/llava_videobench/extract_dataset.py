# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os
import json
import random
import shutil

from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_open
from videobench_utils import select_data

parser = argparse.ArgumentParser()
parser.add_argument("--Eval_QA_root", type=str, default="./Video-Bench/", help="folder containing QA JSON files")
parser.add_argument("--Eval_Video_root", type=str, default="./VideoBench/", help="folder containing video data")
parser.add_argument("--sampling_output_folder", type=str, default="./VideoBench/Eval_video", help="")
parser.add_argument("--json_output_path", type=str, default='./Video-Bench/Eval_QA', help="")
parser.add_argument("--correct_answer_file", type=str, default='./Video-Benc/ANSWER.json', help="")
parser.add_argument("--nums", type=int, default=1000, help="")
args = parser.parse_args()

os.makedirs(args.sampling_output_folder, exist_ok=True)

eval_qa_root = args.Eval_QA_root
eval_video_root = args.Eval_Video_root
dataset_qajson = select_data(eval_qa_root)
dataset_qajson.pop("sampling_data", None)

dataset_name_list = list(dataset_qajson.keys())

with safe_open(args.correct_answer_file, 'r', encoding='utf-8') as f:
    answer = json.load(f)

all_dict = {}
for dataset_name in dataset_name_list:
    if dataset_name not in dataset_qajson:
        continue
    qa_json = dataset_qajson[dataset_name]
    print_log(0, logger.info, f"Dataset name:{dataset_name}, qa_json:{qa_json}!")
    with safe_open(qa_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for _, (q_id, item) in enumerate(data.items()):
        all_dict[q_id] = item

if len(all_dict) >= args.nums:
    keys = list(all_dict.keys())
    selected_items = random.sample(keys, args.nums)
    eval_dict = {}
    for key in selected_items:
        if key not in all_dict:
            continue
        eval_dict[key] = all_dict[key]
else:
    print_log(0, logger.info, "Not enough data to sample items.")

sampling_dataset = f'{args.sampling_output_folder}/sampling_data'
os.makedirs(sampling_dataset, exist_ok=True)
data_dict = {}
for dataset_name, info in eval_dict.items():
    vid_path = os.path.join(args.Eval_Video_root, info.get('vid_path'))
    if vid_path and os.path.exists(vid_path):
        file_name = os.path.basename(vid_path)
        new_file_path = os.path.join(sampling_dataset, file_name)
        shutil.copy2(vid_path, new_file_path)
        print_log(0, logger.info, f"file {file_name} copied to {new_file_path}")
        write_file_path = 'Eval_video/sampling_data/' + file_name
        if dataset_name in eval_dict:
            eval_dict[dataset_name]['vid_path'] = write_file_path
        data_name = vid_path.split('/')[-2]
        if not data_dict.get(data_name):
            data_dict[data_name] = {}
        if data_name in data_dict:
            data_dict[data_name][dataset_name] = answer[data_name][dataset_name]
        if dataset_name in eval_dict:
            eval_dict[dataset_name]['dataset'] = data_name
    else:
        print_log(0, logger.info, f"file {file_name} does not exist!")

sampling_dataset_json = f'{args.json_output_path}/sampling_datasets_QA_new.json'
with safe_open(sampling_dataset_json, 'w', encoding='utf-8') as f:
    json.dump(eval_dict, f, indent=2)

answer_json = f'{args.json_output_path}/sampling_answer.json'
with safe_open(answer_json, 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2)
