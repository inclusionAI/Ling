# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re
import os
import pandas as pd
from modeltest.metric.acc import AccMetric
from atb_llm.utils.log.logging import logger
from .precision_task import PrecisionTask


class CEvalMMLUFewShotsPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.labels = []
        self.model_config = None
    
    def prepare_data(self, metric):
        for _ in self.task_config.subject_mapping:
            if isinstance(metric, AccMetric):
                metric.correct_num_list.append(0)
        _, val_list = self.load_dataset_by_task_name()

        return val_list
    
    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        self.model_config = model_config
        dev_list, _ = self.load_dataset_by_task_name()
        q_num = len(batched_data)
        dev_data = dev_list[sub_dataset_idx]
        task_name = list(self.task_config.subject_mapping.items())[sub_dataset_idx][0]
        prompt_ends = [self.format_example(batched_data, j, include_answer=False)
                    for j in range(q_num)]
        train_prompts = [self.gen_prompt(dev_data, task_name, self.task_config.shots)] * q_num
        prompt = [train + prpt for train, prpt in zip(train_prompts, prompt_ends)]
        self.labels = [batched_data[j][len(batched_data[0]) - 1]
                    for j in range(q_num)]
        prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt] 
        return prompts

    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        for idx, generate_token_list in enumerate(generate_token_lists):
            logger.debug('Question[%d]: %s', len(batched_data) + idx, 
                self.build_queries(sub_dataset_idx, batched_data, self.model_config))
            logger.debug('Answer[%d]: %s', len(batched_data) + idx, generate_token_list)

        answer_results = [generate_token_list.lstrip()[0] if generate_token_list 
                    else "-1" for generate_token_list in generate_token_lists]
        is_correct = ["Correct" if answer_result == label else "Wrong"
                    for answer_result, label in zip(answer_results, self.labels)]
        for idx, is_pass in enumerate(is_correct):
            metric.csv_debug.get("golden_result").append(self.labels[idx])
            metric.csv_debug.get("test_result").append(answer_results[idx])
            metric.csv_debug.get("pass").append(is_pass)
            if is_pass != "Correct":
                logger.debug(">>>推理结果 is : %s", answer_results[idx])
                logger.debug(">>>真实结果 is : %s", self.labels[idx])
            else:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1

    def load_dataset_by_task_name(self):
        row_begin_idx, col_begin_idx = self.get_row_col(), self.get_row_col()
        dev_list, val_list = [], []
        test_set_name = self.get_test_set()
        test_set_path = f"_{test_set_name}.csv"
        for _, subject_name in enumerate(self.task_config.subject_mapping):
            origin_dev_df = pd.read_csv(os.path.join(self.task_config.local_dataset_path, "dev", 
                                subject_name + "_dev.csv"), header=None)
            origin_val_df = pd.read_csv(os.path.join(self.task_config.local_dataset_path, test_set_name,  
                                subject_name + test_set_path), header=None)
            dev_df = origin_dev_df.iloc[row_begin_idx:row_begin_idx + self.task_config.shots, col_begin_idx:]
            val_df = origin_val_df.iloc[row_begin_idx:, col_begin_idx:]
            dev_data = dev_df.values.tolist()
            val_data = val_df.values.tolist()
            dev_list.append(dev_data)
            val_list.append(val_data)
        return dev_list, val_list

    def format_example(self, batch_data, idx, include_answer=True):
        prompt = batch_data[idx][0]
        choices_list_len = len(self.task_config.choices)
        for i in range(choices_list_len):
            prompt += "\n{}. {}".format(self.task_config.choices[i], batch_data[idx][i + 1])
        prompt = ''.join(prompt)
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(batch_data[idx][choices_list_len + 1])
        return prompt

    def format_subject(self, subject):
        subject_parts = subject.split("_")
        formatted_subjects = ""
        for entry in subject_parts:
            formatted_subjects += " " + entry
        return formatted_subjects

    def gen_prompt(self, train_data, subject, k=-1):
        prompt = self.task_config.prompt.format(
            self.format_subject(subject)
        )
        if k == -1:
            k = train_data.shape[0]
        for i in range(k):
            prompt += self.format_example(train_data, i)
        return prompt

    def get_test_set(self):
        raise NotImplementedError("Subclasses should implement get_test_set.")

    def get_row_col(self):
        raise NotImplementedError("Subclasses should implement get_row_col.")


class CEvalMMLUZeroShotPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.labels = []
        self.model_config = None

    @staticmethod
    def _postprocess(text: str, options: str, cushion=True) -> str:
        patterns = [
            f'答案是?\s?([{options}])',
            f'答案是?\s?：([{options}])',
            f'答案是?\s?:([{options}])',
            f'答案应该?是\s?([{options}])',
            f'答案应该?选\s?([{options}])',
            f'答案为\s?([{options}])',
            f'答案选\s?([{options}])',
            f'选择?\s?([{options}])',
            f'故选?\s?([{options}])'
            f'只有选?项?\s?([{options}])\s?是?对',
            f'只有选?项?\s?([{options}])\s?是?错',
            f'只有选?项?\s?([{options}])\s?不?正确',
            f'只有选?项?\s?([{options}])\s?错误',
            f'说法不?对选?项?的?是\s?([{options}])',
            f'说法不?正确选?项?的?是\s?([{options}])',
            f'说法错误选?项?的?是\s?([{options}])',
            f'([{options}])\s?是正确的',
            f'([{options}])\s?是正确答案',
            f'选项\s?([{options}])\s?正确',
            f'所以答\s?([{options}])',
            f'所以\s?([{options}][.。$]?$)',
            f'所有\s?([{options}][.。$]?$)',
            f'[\s，：:,]([{options}])[。，,\.]?$',
            f'[\s，,：:][故即]([{options}])[。\.]?$',
            f'[\s，,：:]因此([{options}])[。\.]?$',
            f'[是为。]\s?([{options}])[。\.]?$',
            f'因此\s?([{options}])[。\.]?$',
            f'显然\s?([{options}])[。\.]?$',
            '答案是\s?(\S+)(?:。|$)',
            '答案应该是\s?(\S+)(?:。|$)',
            '答案为\s?(\S+)(?:。|$)',
            f'[Tt]he answer is \(?([{options}])\)?',
            f'[Tt]he answer is option \(?([{options}])\)?',
            f'[Tt]he correct answer is \(?([{options}])\)?',
            f'[Tt]he correct answer is option \(?([{options}])\)?',
            f'[Tt]he answer to the question is \(?([{options}])\)?',
            f'^选项\s?([{options}])',
            f'^([{options}])\s?选?项',
            f'(\s|^)[{options}][\s。，,：:\.$]',
            f'(\s|^)[{options}](\s|$)',
            '1.\s?(.*?)$',
            f'1.\s?([{options}])[.。$]?$',
        ]
        cushion_patterns = [
            f'([{options}]):',
            f'[{options}]',
        ]

        if cushion:
            patterns.extend(cushion_patterns)
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                outputs = match.group(0)
                for i in options:
                    if i in outputs:
                        return i
        return ''

    def prepare_data(self, metric):
        for _ in self.task_config.subject_mapping:
            if isinstance(metric, AccMetric):
                metric.correct_num_list.append(0)
        val_list = self.load_dataset_by_task_name()

        return val_list

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        self.model_config = model_config
        q_num = len(batched_data)
        task_name = list(self.task_config.subject_mapping.items())[sub_dataset_idx][self.get_row_col()]
        prompt = [self.format_example(task_name, batched_data, j) for j in range(q_num)]
        self.labels = [batched_data[j][len(batched_data[0]) - 1]
                    for j in range(q_num)]
        prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt] 
        return prompts

    def result_judge(self, metric, generate_token_lists, logits, sub_dataset_idx, batched_data):
        for idx, generate_token_list in enumerate(generate_token_lists):
            logger.debug('Question[%d]: %s', len(batched_data) + idx, 
                self.build_queries(sub_dataset_idx, batched_data, self.model_config))
            logger.debug('Answer[%d]: %s', len(batched_data) + idx, generate_token_list)

        answer_results = [self._postprocess(generate_token_list, "ABCD") 
                        for generate_token_list in generate_token_lists]
        is_correct = ["Correct" if answer_result == label else "Wrong"
                    for answer_result, label in zip(answer_results, self.labels)]
        for idx, is_pass in enumerate(is_correct):
            metric.csv_debug.get("golden_result").append(self.labels[idx])
            metric.csv_debug.get("test_result").append(answer_results[idx])
            metric.csv_debug.get("pass").append(is_pass)
            if is_pass != "Correct":
                logger.debug(">>>推理结果 is : %s", answer_results[idx])
                logger.debug(">>>真实结果 is : %s", self.labels[idx])
            else:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1
        
    def load_dataset_by_task_name(self):
        val_list = []
        row_begin_idx, col_begin_idx = self.get_row_col(), self.get_row_col()
        test_set_name = self.get_test_set()
        test_set_path = f"_{test_set_name}.csv"
        for subject_name in self.task_config.subject_mapping:
            origin_val_df = pd.read_csv(os.path.join(self.task_config.local_dataset_path, test_set_name, 
                                subject_name + test_set_path), header=None)
            val_df = origin_val_df.iloc[row_begin_idx:, col_begin_idx:]
            val_data = val_df.values.tolist()
            val_list.append(val_data)
        return val_list

    def format_example(self, name, batch_data, idx):
        raise NotImplementedError("Subclasses should implement get_test_set.")        

    def get_test_set(self):
        raise NotImplementedError("Subclasses should implement get_test_set.")

    def get_row_col(self):
        raise NotImplementedError("Subclasses should implement get_test_set.")