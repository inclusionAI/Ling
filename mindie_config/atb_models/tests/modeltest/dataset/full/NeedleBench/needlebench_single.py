# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2020 OpenCompass Authors. All rights reserved.
# Copyright (c) Microsoft Corporation.
import enum
import math
import random
import os
import json
import tempfile
import shutil
from dataclasses import dataclass
import tiktoken
from atb_llm.utils import file_utils


DATA = "data"
FULL = "full"
NEEDLEBENCH = "needlebench"


class Position(enum.Enum):
    START = 0
    END = 1


@dataclass
class NeedleBenchConfig:
    context_length_list: list
    depth_percent_list: list
    num_repeats_per_file: int
    file_names: list
    length_buffer: int
    tokenizer_model: str
    position: Position
    language: str


def logistic(x, length=100, x0=50, k=0.1):
    return round(length / (1 + math.exp(-k * (x - x0))), 3)


def generate_linear_space(start, end, num):
    if num == 1:
        return [start]
    elif num < 1:
        raise ValueError('num must be at least 1.')
    step = (end - start) / (num - 1)
    return [start + step * i for i in range(num)]


def generate_depth_percents(intervals, interval_type):
    if interval_type == 'linear':
        return generate_linear_space(0, 100, intervals)
    elif interval_type == 'sigmoid':
        linear_space = generate_linear_space(0, 100, intervals)
        return [logistic(x) for x in linear_space]
    else:
        raise ValueError('Unsupported interval type')


def load_datasets(config: NeedleBenchConfig):

    def get_random_line(counter, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [
                json.loads(line.strip())
                for line in file
                if json.loads(line.strip())['language'] == config.language
            ]

        if lines:
            random.seed(counter)
            random_line = random.choice(lines)
            return {
                'needle': random_line['needle'],
                'retrieval_question': random_line['retrieval_question'],
                'keyword': random_line['arg2']
            }
        else:
            return {}

    def _generate_context(tokens_context, depth_percent, tokenizer, needle):
        tokens_needle = tokenizer.encode(needle)
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_context = (tokens_context[:insertion_point] +
                        tokens_needle + tokens_context[insertion_point:])
        new_context = tokenizer.decode(tokens_context)
        return new_context
    
    def _modify_retrieval_question(retrieval_question):
        language = config.language
        if language == 'Chinese':
            parts = retrieval_question.split('请按照')
            guide_retrieval_question = (parts[0] + '在回答之前，请思考文档中与此问题'
                                        '最相关的内容是什么。请按照' + parts[1])
            return guide_retrieval_question
        elif language == 'English':
            parts = retrieval_question.split('Please answer in the format')
            guide_retrieval_question = (
            parts[0] + 'Before answering, please consider'
            ' what in the document is most relevant to this question.'
            ' Please answer in the format' + parts[1])
            return guide_retrieval_question
        else:
            raise ValueError(f"Language '{language}' is not supported.")

    def _generate_prompt(context, retrieval_question): # End: 将问句添加到末尾
        retrieval_question = _modify_retrieval_question(
            retrieval_question)
        language = config.language
        position = config.position
        if language == 'Chinese':
            if position == Position.END:
                prompt = (
                    '你是一个善于回答用户问题的智能AI助手\n'
                    '请保持你的回答简洁清楚。不要说和下面文档中的无关的话'
                    '，或重复你的回答\n'
                    f'用户现在给你的文档是{context}\n\n'
                    f'现在请问：{retrieval_question}'
                )
            elif position == Position.START:
                prompt = (
                    '你是一个善于回答用户问题的智能AI助手\n'
                    '请保持你的回答简洁清楚。不要说和下面文档中的无关的话'
                    '，或重复你的回答\n'
                    f'现在请问：{retrieval_question}',
                    f'用户现在给你的文档是{context}\n\n'
                )
            else:
                raise ValueError('Unsupported position. '
                                'Position must be POSITION_END or POSITION_START.')
        elif language == 'English':
            if position == Position.END:
                prompt = ('You are an intelligent AI assistant skilled in '
                        'answering user questions.\n'
                        'Please keep your answers concise and clear. Do '
                        'not talk about irrelevant topics or repeat '
                        'your answers.\nThe document '
                        f'given to you by the user is {context}\n\n'
                        f'Now, the question is: {retrieval_question}')
            elif position == Position.START:
                prompt = ('You are an intelligent AI assistant skilled in '
                        'answering user questions.\n'
                        'Please keep your answers concise and clear. Do '
                        'not talk about irrelevant topics or repeat '
                        'your answers.\n'
                        f'Now, the question is: {retrieval_question}'
                        'The document given to you by the user'
                        f' is {context}\n\n')
            else:
                raise ValueError(f'Unsupported position {position}. '
                                'Position must be POSITION_END or POSITION_START.')
        else:
            raise ValueError(f"Language '{language}' is not supported.")
                    
        return prompt
    
    def load_dataset(original_context_length, depth_percent):
        dataset = []
        current_dir = os.path.abspath(__file__)
        levels_up = 4
        for _ in range(levels_up):
            current_dir = os.path.dirname(current_dir)  
        needle_file_path = os.path.join(current_dir, DATA, NEEDLEBENCH, "needles.jsonl")
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
        dest_filepath = os.path.join(cache_dir, cache_file)
        max_file_size = 5 * 1024 * 1024
        if not file_utils.is_path_exists(dest_filepath):
            src_filepath = os.path.join(current_dir, "dataset", FULL, "NeedleBench", cache_file)
            if file_utils.is_path_exists(src_filepath):
                src_filepath = file_utils.standardize_path(src_filepath)
                file_utils.check_file_safety(src_filepath, 'r', max_file_size=max_file_size)
                dest_filepath = file_utils.standardize_path(dest_filepath)
                file_utils.check_file_safety(dest_filepath, 'w', max_file_size=max_file_size)
                shutil.copyfile(src_filepath, dest_filepath)
        tokenizer = tiktoken.encoding_for_model(config.tokenizer_model)
        for file_name in config.file_names:
            file_path = os.path.join(current_dir, DATA, NEEDLEBENCH, f"{file_name}.jsonl")
            with file_utils.safe_open(file_path, 'r', encoding='utf-8') as f:
                lines_bak = [json.loads(line.strip()) for line in f]
            lines = lines_bak.copy()
            for counter in range(config.num_repeats_per_file):
                random.seed(counter)
                random.shuffle(lines)
                random_needle = get_random_line(counter, needle_file_path)
                needle = '\n' + random_needle.get('needle') + '\n'
                retrieval_question = random_needle.get('retrieval_question')
                keyword = random_needle.get('keyword')
                
                context_length = original_context_length - config.length_buffer
                target_length_per_record = context_length - len(
                    tokenizer.encode(needle))
                target_length_per_record = max(target_length_per_record, 0)
                accumulated_tokens = []
                for line in lines:
                    tokens_current_line = tokenizer.encode(line.get('text'))
                    accumulated_tokens.extend(tokens_current_line)

                    if len(accumulated_tokens) >= target_length_per_record:
                        break
                data = {'prompt':'', 'answer':''}
                processed_text = _generate_context(
                accumulated_tokens[:target_length_per_record], depth_percent,
                tokenizer, needle)

                processed_prompt = _generate_prompt(processed_text,
                    retrieval_question)
                data['prompt'] = processed_prompt
                data['answer'] = needle + '*' + keyword
                dataset.append(data)

        return dataset

    datasets = []
    data = 'data'
    for file_name in config.file_names:
        for original_context_length in config.context_length_list:
            for depth_percent in config.depth_percent_list:
                dataset = {'name': f'{file_name}_{original_context_length}_{depth_percent}', data: ''}
                dataset[data] = load_dataset(original_context_length, depth_percent)
                datasets.append(dataset)
    return datasets


def test_single_4k():
    context_lengths = list(range(1000, 5000, 1000))
    document_depth_percent_intervals = 20
    document_depth_percent_interval_type = 'linear'
    file_list = ['PaulGrahamEssays']
    depths_list = generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type)
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                600, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets


def test_single_8k():
    context_lengths = list(range(5000, 9000, 1000))
    document_depth_percent_intervals = 20
    document_depth_percent_interval_type = 'linear'
    file_list = ['PaulGrahamEssays']
    depths_list = generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type)
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                600, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets


def test_single_32k():
    context_lengths = [9000, 13000, 17000, 21000, 25000, 29000, 31000, 32000]
    depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
    file_list = ['PaulGrahamEssays']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                3000, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets


def test_single_128k():
    context_lengths = [16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000]
    depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
    file_list = ['PaulGrahamEssays']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                600, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets


def test_single_200k():
    context_lengths = [16000, 48000, 80000, 112000, 128000, 144000, 176000, 200000]
    depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
    file_list = ['PaulGrahamEssays']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                600, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets


def test_single_256k():
    context_lengths = [32000, 128000, 256000]
    depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
    file_list = ['PaulGrahamEssays']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                600, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets


def test_single_1000k():
    context_lengths = [20000, 160000, 300000, 440000, 580000, 720000, 860000, 1000000]
    depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
    file_list = ['PaulGrahamEssays']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                600, 'gpt-4', Position.END, 'English')
    needlebench_en_datasets = load_datasets(config)

    file_list = ['zh_finance']
    config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                200, 'gpt-4', Position.END, 'Chinese')
    needlebench_zh_datasets = load_datasets(config)
    return needlebench_en_datasets + needlebench_zh_datasets
