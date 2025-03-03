# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

"""Copyright (c) 2022, salesforce.com, inc.

All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
import json
from modeltest.metric.acc import AccMetric
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.multimodal_utils import MultimodalInput
from .precision_task import PrecisionTask


MAX_TARGET_LENGTH = 2048
MAX_BATCH_LEN = 2048


class TextVQAPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)

        textvqa_gt_json_path = os.path.abspath(self.task_config.local_dataset_path).split('.')[0] + "_annotations.json"
        self.vqa_gt_answer = {}
        with open(textvqa_gt_json_path) as f:
            gt_answers = json.load(f)['annotations']
            for item in gt_answers:
                question_id = item['question_id']
                answers = item['answers']
                self.vqa_gt_answer[question_id] = answers
        self.vqa_eval = VQAEvalMethod()
        self.gt_ans_id = 0

    @staticmethod
    def get_qwen_vl_queries(batched_data, prefix, suffix):
        queries = [[{"image": item["image"]},
                    {"text": prefix + item["question"] + suffix}] for item in batched_data]
        return queries
         
    @staticmethod
    def get_qwen2_vl_queries(batched_data, prefix, suffix):
        queries = [[{"text": prefix},
                    {"image": item["image"]},
                    {"text": item["question"] + suffix}] for item in batched_data]
        return queries

    @staticmethod
    def get_yivl_queries(batched_data, prefix, suffix):
        queries = MultimodalInput([item["question"] for item in batched_data], 
                                  [item["image"] for item in batched_data], 
                                  None, 
                                  None)
        return queries

    @staticmethod
    def get_internvl_queries(batched_data, prefix, suffix):
        queries = MultimodalInput(
            [prefix + item["question"] + suffix for item in batched_data], 
            [item["image"] for item in batched_data], 
            None, 
            None
            )
        return queries
    
    @staticmethod
    def get_cogvlm2_queries(batched_data, prefix, suffix):
        from examples.models.cogvlm2.run_pa import InputAttrs
        return [InputAttrs([prefix + item["question"] + suffix], item["image"]) for item in batched_data]

    def prepare_data(self, metric):
        if isinstance(metric, AccMetric):
            metric.correct_num_list.append(0)

        textvqa_datasets = []
        with safe_open(self.task_config.local_dataset_path, encoding='utf-8') as f:
            dataset = []
            for line in f:
                line_json = json.loads(line)
                dataset.append(line_json)
            textvqa_datasets.append(dataset)
        return textvqa_datasets

    def build_queries(self, _, batched_data, model_config):
        prefix = model_config.mm_model.get('prompt_prefix')
        suffix = model_config.mm_model.get('prompt_suffix')
        func_map = {
            "qwen_vl": "get_qwen_vl_queries",
            "qwen2_vl": "get_qwen2_vl_queries",
            "internvl": "get_internvl_queries",
            "cogvlm2": "get_cogvlm2_queries",
            "yivl": "get_yivl_queries",
        }
        try:
            func_name = func_map[model_config.model_name]
        except KeyError as e:
            raise KeyError(f"Unsupported! Please choose from [{func_map.keys()}].") from e
        func = getattr(self, func_name)
        return func(batched_data, prefix, suffix)

    def result_judge(self, metric, generate_token_lists, _, sub_dataset_idx, batched_data):
        if len(batched_data) > MAX_BATCH_LEN:
            raise ValueError(f"Batch size should smaller than {MAX_BATCH_LEN}, got {len(batched_data)}")

        answers = generate_token_lists
        answer_results = [answer.lstrip() if answer else "-1" for answer in answers]

        self.vqa_gt_answer
        for idx, item in enumerate(batched_data):
            question_id = item['question_id']
            gt_answers_list = self.vqa_gt_answer[question_id]
            res_data = answer_results[idx]
            res_data = res_data.split("<|endoftext|>")[0]
            res_data = res_data.split("<|im_end|>")[0]
            res_data = res_data.split("<|end_of_text|>")[0]
            res_data = res_data.split("###")[0]
            res_data = res_data.replace('\n', ' ')
            res_data = res_data.replace('\t', ' ')
            res_data = res_data.strip()
            res_data = self.vqa_eval.process_punctuation(res_data)
            res_data = self.vqa_eval.process_digit_article(res_data)
            acc_list = []
            gt_answers = [ans['answer'] for ans in gt_answers_list]

            if len(set(gt_answers)) > 1:
                for ans_dict in gt_answers_list:
                    ans_dict['answer'] = self.vqa_eval.process_punctuation(
                        ans_dict['answer'])
            for gt_ans in gt_answers_list:
                other_gt_ans = [item for item in gt_answers_list if item != gt_ans]
                matched_ans = [item for item in other_gt_ans if item['answer'] == res_data]
                acc = min(1, len(matched_ans) / 3)
                acc_list.append(acc)

            if len(acc_list) == 0:
                raise ValueError("Length of acc_list should be positive, got 0.")

            avg_acc = sum(acc_list) / len(acc_list)
            metric.correct_num += avg_acc
            metric.correct_num_list[sub_dataset_idx] += avg_acc


# The code below is based on the VQAEval class at the following URL:
# (https://github.com/bupt-cist/vqa-playground-pytorch/blob/master/official_test.py)
class VQAEvalMethod:
    def __init__(self):
        self.contractions = {
            'aint': "ain't",
            'arent': "aren't",
            'cant': "can't",
            'couldve': "could've",
            'couldnt': "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            'didnt': "didn't",
            'doesnt': "doesn't",
            'dont': "don't",
            'hadnt': "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            'hasnt': "hasn't",
            'havent': "haven't",
            'hed': "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            'hes': "he's",
            'howd': "how'd",
            'howll': "how'll",
            'hows': "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            'Im': "I'm",
            'Ive': "I've",
            'isnt': "isn't",
            'itd': "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            'itll': "it'll",
            "let's": "let's",
            'maam': "ma'am",
            'mightnt': "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            'mightve': "might've",
            'mustnt': "mustn't",
            'mustve': "must've",
            'neednt': "needn't",
            'notve': "not've",
            'oclock': "o'clock",
            'oughtnt': "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            'shant': "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            'shouldve': "should've",
            'shouldnt': "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": 'somebodyd',
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            'somebodyll': "somebody'll",
            'somebodys': "somebody's",
            'someoned': "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            'someonell': "someone'll",
            'someones': "someone's",
            'somethingd': "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            'somethingll': "something'll",
            'thats': "that's",
            'thered': "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            'therere': "there're",
            'theres': "there's",
            'theyd': "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            'theyll': "they'll",
            'theyre': "they're",
            'theyve': "they've",
            'twas': "'twas",
            'wasnt': "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            'weve': "we've",
            'werent': "weren't",
            'whatll': "what'll",
            'whatre': "what're",
            'whats': "what's",
            'whatve': "what've",
            'whens': "when's",
            'whered': "where'd",
            'wheres': "where's",
            'whereve': "where've",
            'whod': "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            'wholl': "who'll",
            'whos': "who's",
            'whove': "who've",
            'whyll': "why'll",
            'whyre': "why're",
            'whys': "why's",
            'wont': "won't",
            'wouldve': "would've",
            'wouldnt': "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            'yall': "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            'youd': "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            'youll': "you'll",
            'youre': "you're",
            'youve': "you've",
        }

        self.manual_map = {
            'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
        }

        self.articles = ['a', 'an', 'the']

        self.period_strip = re.compile('(?!<=\d)(\.)(?!\d)')
        self.comma_strip = re.compile('(\d)(,)(\d)')
        self.punct = [
            ';',
            r'/',
            '[',
            ']',
            '"',
            '{',
            '}',
            '(',
            ')',
            '=',
            '+',
            '\\',
            '_',
            '-',
            '>',
            '<',
            '@',
            '`',
            ',',
            '?',
            '!',
        ]

    def process_punctuation(self, in_text):
        if len(in_text) > MAX_TARGET_LENGTH:
            raise ValueError(
                f"Invalid in_text length, should be no more than {MAX_TARGET_LENGTH} but got {len(in_text)}"
            ) 
        out_text = in_text
        for p in self.punct:
            if (p + ' ' in in_text or ' ' + p
                    in in_text) or (re.search(self.comma_strip, in_text)):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.period_strip.sub('', out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        if len(in_text) > MAX_TARGET_LENGTH:
            raise ValueError(
                f"Invalid in_text length, should be no more than {MAX_TARGET_LENGTH} but got {len(in_text)}"
            ) 
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.manual_map.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
        for word_id, word in enumerate(out_text):
            if word in self.contractions:
                out_text[word_id] = self.contractions[word]

        return ' '.join(out_text)
