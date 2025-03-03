# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import time
import os
import json
import numpy as np
import torch
import av
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_open, standardize_path, check_file_safety
from atb_llm.utils.multimodal_utils import check_video_path
from atb_llm.models.base.model_utils import safe_from_pretrained
from .videobench_utils import select_data, process_qs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_video_pyav(container_, indices_):
    frames = []
    container_.seek(0)
    start_index_ = indices_[0]
    end_index = indices_[-1]
    for i, frame in enumerate(container_.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index_ and i in indices_:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='sampling_data', help="The type of LLM")
parser.add_argument("--Eval_QA_root",
                    type=str,
                    default='/workspace/llava_gpu/Video-Bench/',
                    help="folder containing QA JSON files",
                    )
parser.add_argument("--Eval_Video_root",
                    type=str,
                    default='/workspace/llava_gpu/VideoBench/',
                    help="folder containing video data",
                    )
parser.add_argument('--model_path',
                    help="model and tokenizer path",
                    default='/workspace/llava_gpu/LLaVA-NeXT-Video-7B-hf',
                    )
parser.add_argument("--chat_conversation_output_folder", type=str, default='./Chat_results', help="")
args = parser.parse_args()

eval_qa_root = standardize_path(args.Eval_QA_root)
check_file_safety(eval_qa_root, 'r')

eval_video_root = standardize_path(args.Eval_Video_root)
check_file_safety(eval_video_root, 'r')

model_path = standardize_path(args.model_path)
check_file_safety(model_path, 'r')

chat_conversation_output_folder = standardize_path(args.chat_conversation_output_folder)
check_file_safety(chat_conversation_output_folder, 'w')

dataset_qajson = select_data(eval_qa_root)

if args.dataset_name is None:
    dataset_name_list = list(dataset_qajson.keys())
elif args.dataset_name in dataset_qajson.keys():
    dataset_name_list = [args.dataset_name]
    print_log(0, logger.info, f"Specifically run {dataset_name_list[0]}")
else:
    print_log(0, logger.info, "dataset_name must be in dataset_qajson.")
    
print_log(0, logger.info, f"dataset_name_list: {dataset_name_list}")

os.makedirs(chat_conversation_output_folder, exist_ok=True)

model = safe_from_pretrained(LlavaNextVideoForConditionalGeneration, model_path, device_map="auto")
processor = safe_from_pretrained(LlavaNextVideoProcessor, model_path)


e2e_time_all_list = []
e2e_time_all = 0

for dataset_name in dataset_name_list:
    if dataset_name not in dataset_qajson:
        continue
    qa_json = dataset_qajson[dataset_name]
    print_log(0, logger.info, f"Dataset name:{dataset_name}, qa_json:{qa_json}!")
    with safe_open(qa_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    eval_dict = {}
    for _, (q_id, item) in enumerate(data.items()):
        video_id = item['video_id']
        question, vid_path = process_qs(item, eval_video_root, '')

        vid_path = check_video_path(vid_path)
        container = av.open(vid_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        video = read_video_pyav(container, indices)
        container.close()

        question_dict = {}
        question_dict["type"] = "text"
        question_dict["text"] = question
        content = []
        content.append(question_dict)
        content.append({"type":"video"})
        conversation_dict = {}
        conversation_dict["role"] = "user"
        conversation_dict["content"] = content
        conversation = []
        conversation.append(conversation_dict)
        e2e_start = time.time()
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        prompt = prompt.split("ASSISTANT:")[0] + "Please respond with only the \
corresponding options and do not provide any explanations or additional information. ASSISTANT:"
        print_log(0, logger.info, f"prompt: {prompt}")
        inputs = processor(text=prompt, videos=video, return_tensors="pt")
        inputs = inputs.to(device)
        output = model.generate(**inputs, max_new_tokens=60)

        out = output
        out = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        e2e_end = time.time()
        e2e_time = e2e_end - e2e_start
        e2e_time_all += e2e_time
        start_index = out[0].find("ASSISTANT:") + len("ASSISTANT:")
        new_str = out[0][start_index:]
        output = []
        output.append(new_str)
        dataset = item['dataset']
        eval_dict[q_id] = {
            'video_id': video_id,
            'question': question,
            'output_sequence': output,
            'time': e2e_time,
            'dataset': dataset
        }
        print_log(0, logger.info, f"q_id:{q_id}, output:{output}!\n")
    eval_dataset_json = f'{args.chat_conversation_output_folder}/{dataset_name}_eval.json'

    with safe_open(eval_dataset_json, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=2)
