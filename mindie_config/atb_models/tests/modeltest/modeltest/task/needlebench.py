# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Copyright 2020 OpenCompass Authors. All rights reserved.
# Copyright (c) Microsoft Corporation.
import json
import os
import enum
import tempfile
import random
import re
import shutil
import math
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken
from modeltest.metric.acc import AccMetric
from atb_llm.utils import file_utils
from .precision_task import PrecisionTask


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


class NeedleBenchPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.context_length = str(task_config.requested_max_input_length // 1000) + "k"
        self.needles_file = "needles.jsonl"
    
    def get_single_4k_config(self):
        context_lengths = list(range(1000, 5000, 1000))
        document_depth_percent_intervals = 20
        document_depth_percent_interval_type = 'linear'
        file_list = ['PaulGrahamEssays.jsonl']
        depths_list = generate_depth_percents(
                document_depth_percent_intervals,
                document_depth_percent_interval_type)
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    600, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def get_single_8k_config(self):
        context_lengths = list(range(5000, 9000, 1000))
        document_depth_percent_intervals = 20
        document_depth_percent_interval_type = 'linear'
        file_list = ['PaulGrahamEssays.jsonl']
        depths_list = generate_depth_percents(
                document_depth_percent_intervals,
                document_depth_percent_interval_type)
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    600, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def get_single_32k_config(self):
        context_lengths = [9000, 13000, 17000, 21000, 25000, 29000, 31000, 32000]
        depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
        file_list = ['PaulGrahamEssays.jsonl']
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    3000, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def get_single_128k_config(self):
        context_lengths = [16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000]
        depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
        file_list = ['PaulGrahamEssays.jsonl']
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    600, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def get_single_200k_config(self):
        context_lengths = [16000, 48000, 80000, 112000, 128000, 144000, 176000, 200000]
        depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
        file_list = ['PaulGrahamEssays.jsonl']
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    600, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def get_single_256k_config(self):
        context_lengths = [32000, 128000, 256000]
        depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
        file_list = ['PaulGrahamEssays.jsonl']
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    600, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def get_single_1000k_config(self):
        context_lengths = [20000, 160000, 300000, 440000, 580000, 720000, 860000, 1000000]
        depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
        file_list = ['PaulGrahamEssays.jsonl']
        en_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    600, 'gpt-4', Position.END, 'English')
        file_list = ['zh_finance.jsonl']
        cn_config = NeedleBenchConfig(context_lengths, depths_list, 10, file_list,
                                    200, 'gpt-4', Position.END, 'Chinese')
        return [en_config, cn_config]

    def prepare_data(self, metric):
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
        
        def load_dataset(config, file_name, original_context_length, depth_percent):
            dataset = []
            needle_file_path = os.path.join(self.local_dataset_folder, self.task_config.local_dataset_path,
                self.needles_file)
            cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
            dest_filepath = os.path.join(cache_dir, cache_file)
            if not file_utils.is_path_exists(dest_filepath):
                src_filepath = os.path.join(
                    self.local_dataset_folder,
                    self.task_config.local_dataset_path,
                    cache_file)
                max_file_size = 5 * 1024 * 1024
                src_filepath = file_utils.standardize_path(src_filepath)
                if file_utils.is_path_exists(src_filepath):
                    file_utils.check_file_safety(src_filepath, 'r', max_file_size=max_file_size)
                    dest_filepath = file_utils.standardize_path(dest_filepath)
                    file_utils.check_file_safety(dest_filepath, 'w', max_file_size=max_file_size)
                    shutil.copyfile(src_filepath, dest_filepath)
            tokenizer = tiktoken.encoding_for_model(config.tokenizer_model)
            with file_utils.safe_open(file_name, 'r', encoding='utf-8') as f:
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
        config_map = {
            '4k': self.get_single_4k_config,
            '8k': self.get_single_8k_config,
            '32k': self.get_single_32k_config,
            '128k': self.get_single_128k_config,
            '256k': self.get_single_256k_config,
            '200k': self.get_single_200k_config,
            '1000k': self.get_single_1000k_config
        }
        configs = config_map.get(self.context_length)()
        if self.task_config.subject_mapping is None:
            self.task_config.subject_mapping = {}
        for config in configs:
            for file_name in tqdm(config.file_names):
                entry = os.path.join(
                    self.local_dataset_folder,
                    self.task_config.local_dataset_path,
                    file_name)
                if isinstance(metric, AccMetric):
                    metric.correct_num_list.append(0)
                dataset = []
                for original_context_length in config.context_length_list:
                    for depth_percent in config.depth_percent_list:
                        dataset_name = f'{file_name}_{original_context_length}_{depth_percent}'
                        self.task_config.subject_mapping[dataset_name] = {"name": dataset_name}
                        dataset = load_dataset(config, entry, original_context_length, depth_percent)
                        datasets.append(dataset)
                        if isinstance(metric, AccMetric):
                            metric.correct_num_list.append(0)
        return datasets


    def build_queries(self, _, batched_data, model_config):
        return [item['prompt'] for item in batched_data]

    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        def trim_prediction(prediction, reference):
            l08 = int(0.8 * len(reference))
            l12 = int(1.2 * len(reference))
            trimmed_prediction = prediction[:l12]

            if len(trimmed_prediction) > l08 and \
            reference[-1] in trimmed_prediction[l08:]:
                end_pos = l08 + trimmed_prediction[l08:].index(reference[-1]) + 1
                trimmed_prediction = trimmed_prediction[:end_pos]

            return trimmed_prediction
        
        def score(prediction, gold):
            total_score = 0
            keyword = gold.split('*')[1]
            reference = gold.split('*')[0]
            raw_prediction = prediction
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)

            prediction = trim_prediction(prediction, reference)

            edit_distance = levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 1 - edit_distance / max_len if max_len != 0 else 1

            if keyword in raw_prediction:
                score = 1
            else:
                score = 0.2 * score

            detail = {
            'pred': prediction,
            'answer': reference,
            'edit_distance': edit_distance,
            'score': score
            }
            total_score += score
            result = {'score': total_score, 'detail': detail}
            return result
                
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)

            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        for idx, item in enumerate(batched_data):
            ans = item['answer']
            response = generate_token_lists[idx]
            acc = score(response, ans).get('score')
            metric.csv_debug.get("golden_result", []).append(item['answer'])
            metric.csv_debug.get("test_result", []).append(response)
            metric.csv_debug.get("pass", []).append(acc)
            metric.correct_num += acc
            metric.correct_num_list[sub_dataset_idx] += acc

