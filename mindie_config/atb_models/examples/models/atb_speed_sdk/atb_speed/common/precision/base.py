#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
precision base
"""
import json
import os
from string import ascii_letters

import pandas as pd
import torch
from atb_llm.utils.file_utils import safe_open
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher.base import BaseLauncher
from atb_speed.common.utils import torch_parallel_info
from tqdm import tqdm

HARD_TASK = (
    "advanced_mathematics", "discrete_mathematics", "probability_and_statistics", "college_chemistry",
    "college_physics", "high_school_mathematics", "high_school_chemistry", "high_school_physics"
)


class Record:
    """only keep one card result when debug is False"""

    def __init__(self, log_dir, log_flag, debug=False):
        self.debug = debug
        self.flag = log_flag if debug else ""
        self.log_name = os.path.join(log_dir, f"device{self.flag}.log")
        self.cache_name = os.path.join(log_dir, f"cache{self.flag}.txt")
        self.begin_idx = self.load_cache()

    def log(self, *msg):
        if self.debug or torch_parallel_info.is_rank_0:
            with safe_open(self.log_name, "a", encoding="utf-8") as f:
                f.write(" ".join([str(i) for i in msg]) + '\n')

    def load_cache(self):
        if not os.path.exists(self.cache_name):
            self.log("[-] No cache file, cache will be created")
            return 0
        self.log("[~] Loading cache on last abnormal exit ... (and continue with the cache)")
        with safe_open(self.cache_name, "r", encoding="utf-8") as f:
            cache = f.read().strip().split()
        if not cache:
            return 0
        cache = [row.split(",") for row in cache]
        start_idx = cache[-1][0]
        self.log(f"[+] Load cache successfully! start idx: {start_idx}")
        return int(start_idx) + 1

    def update_cache(self, task_name, question_id, truth_answer, predict_answer):
        if self.debug or torch_parallel_info.is_rank_0:
            with safe_open(self.cache_name, "a", encoding="utf-8") as f:
                f.write(f"{question_id},{task_name},{truth_answer},{predict_answer}\n")


class PrecisionTestBase:
    def __init__(self, launcher: BaseLauncher, workdir="", **kwargs):
        workdir = atb_speed_config.precision.work_dir if not workdir else workdir
        self.data_dir = os.path.join(workdir, "data")
        self.result_output_dir = os.path.join(workdir, "test_result")
        self.init_result_dir()
        self.choices = ["A", "B", "C", "D"]
        self.shot = 5
        self.batch = 1
        self.seq_len_out = 32

        self.model, self.tokenizer = launcher.model, launcher.tokenizer
        self.local_rank = launcher.local_rank
        self.launcher = launcher
        self.recorder = Record(self.result_output_dir, self.local_rank)
        self.subject_mapping_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 f"{atb_speed_config.precision.mode}_subject_mapping.json")
        # kwargs have higher priority
        if atb_speed_config.precision:
            self.update_param(atb_speed_config.precision.__dict__)
        self.update_param(kwargs)

    @staticmethod
    def format_subject(subject):
        sub_list = subject.split("_")
        final_str = ""
        for entry in sub_list:
            final_str += " " + entry
        return final_str

    def update_param(self, param_dict):
        for key, value in param_dict.items():
            setattr(self, key, value)
            self.recorder.log(f"[+] set {key} to {value}")

    def init_result_dir(self):
        if torch_parallel_info.is_rank_0:
            os.makedirs(self.result_output_dir, exist_ok=True)
        if torch_parallel_info.world_size > 1:
            torch.distributed.barrier()

    def compute_metric(self, subject_mapping):
        run_results = pd.read_csv(
            self.recorder.cache_name,
            names=['question_id', 'task_name', 'truth_answer', 'predict_answer'])
        classes_acc = dict()
        subject_acc = dict()
        hard_task = [0, 0]
        for task in subject_mapping:
            class_of_task = subject_mapping[task][2]
            this_task = run_results.loc[run_results.task_name == task]
            if not this_task.shape[0]:
                continue
            correct_num = (this_task.truth_answer == this_task.predict_answer).sum()
            if class_of_task not in classes_acc:
                classes_acc[class_of_task] = [0, 0]  # correct num, total num
            if task in HARD_TASK:
                hard_task[0] += correct_num
            hard_task[1] += this_task.shape[0]
            subject_acc[task] = correct_num / this_task.shape[0]
            classes_acc[class_of_task][0] += correct_num
            classes_acc[class_of_task][1] += this_task.shape[0]

        avg_acc = sum([i[0] for i in classes_acc.values()]) / sum([j[1] for j in classes_acc.values()])
        for c in classes_acc:
            classes_acc[c] = classes_acc[c][0] / classes_acc[c][1]
        classes_acc["Avg"] = avg_acc
        classes_acc["Avg(Hard)"] = hard_task[0] / hard_task[1]
        with safe_open(os.path.join(self.result_output_dir, f"result{self.recorder.flag}_subject_acc.json"), "w") as fp:
            json.dump(subject_acc, fp)
        with safe_open(os.path.join(self.result_output_dir, f"result{self.recorder.flag}_classes_acc.json"), "w") as fp:
            json.dump(classes_acc, fp)
        if torch_parallel_info.is_rank_0:
            self.launcher.logger.info(f"[+] Avg acc: {classes_acc['Avg']}")

    def get_subject_mapping(self):
        with safe_open(self.subject_mapping_path, "r", encoding="utf-8") as f:
            subject_mapping = json.load(f)
        return subject_mapping

    def load_csv_by_task_name(self, task_name):
        dev_df = pd.read_csv(os.path.join(self.data_dir, "dev", task_name + "_dev.csv"), header=None)[
                 :self.shot + 1]
        val_df = pd.read_csv(os.path.join(self.data_dir, "val", task_name + "_val.csv"), header=None)

        return dev_df, val_df

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = len(self.choices)
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def batch_infer(self, qr_pair, begin_idx):
        prompts = [item['prompt'] for item in qr_pair]
        truth_answers = [item['answer'] for item in qr_pair]
        task_names = [item['task_name'] for item in qr_pair]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding='longest')
        inputs = inputs.to(self.model.device)
        input_len = len(inputs.input_ids[0])
        with torch.no_grad():
            output = self.model.generate(inputs.input_ids,
                                         attention_mask=inputs.attention_mask,
                                         max_new_tokens=self.seq_len_out)
        answers = self.tokenizer.batch_decode(output.to(torch.int32)[:, input_len:])

        for prompt, truth_answer, task_name, ori_answer in zip(prompts, truth_answers, task_names, answers):
            self.recorder.log("\n========== prompt start ==========\n", prompt,
                              "\n==========  prompt end  ==========\n")
            self.recorder.log(f"[+] prompt length: {input_len}")
            self.recorder.log("\n========== answer start ==========\n", ori_answer,
                              "\n==========  answer end  ==========\n")
            answer_list = [char.upper() for char in ori_answer if char in ascii_letters]
            answer = answer_list[0] if answer_list else "-1"
            is_correct = "Correct" if answer == truth_answer else "Wrong"
            self.recorder.log(f"[{is_correct}] predict: {answer}, label: {truth_answer}")
            self.recorder.update_cache(task_name, begin_idx, truth_answer, answer)
            begin_idx += 1

    def run(self):
        subject_mapping = self.get_subject_mapping()
        subject_name_list = sorted(list(subject_mapping.keys()))
        qr_pair = []

        total_len = 0
        begin_idx = self.recorder.begin_idx
        for task_name in subject_name_list:
            dev_df, val_df = self.load_csv_by_task_name(task_name)
            total_len += len(val_df)
            if len(val_df) <= begin_idx:
                self.recorder.log(f"[~] Skip Task: {task_name}")
                begin_idx -= len(val_df)
                continue

            for i in range(val_df.shape[0]):
                if begin_idx > 0:
                    begin_idx -= 1
                    continue
                for cut_shot in range(self.shot):
                    prompt_end = self.format_example(val_df, i, include_answer=False)
                    train_prompt = self.gen_prompt(dev_df, task_name, self.shot - cut_shot)
                    prompt = train_prompt + prompt_end
                    input_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])
                    if input_len > 2000:
                        continue
                    label = val_df.iloc[i, val_df.shape[1] - 1]
                    qr_pair.append({'task_name': task_name, 'prompt': prompt, 'answer': label})
                    break
        pbar = None
        if torch_parallel_info.is_rank_0:
            pbar = tqdm(total=total_len, initial=self.recorder.begin_idx)
        for i in range(0, len(qr_pair), self.batch):
            self.batch_infer(qr_pair[i: i + self.batch], i + self.recorder.begin_idx)
            if torch_parallel_info.is_rank_0:
                pbar.update(self.batch if i + self.batch <= len(qr_pair) else len(qr_pair) - i)
        if torch_parallel_info.is_rank_0:
            pbar.close()
        self.compute_metric(subject_mapping)


class CEVALPrecisionTest(PrecisionTestBase):
    """
    CEVAL
    """

    def load_csv_by_task_name(self, task_name):
        dev_df, val_df = super().load_csv_by_task_name(task_name)

        # remove the first row "column names" and the first column "id"
        dev_df = dev_df.iloc[1:, 1:]
        val_df = val_df.iloc[1:, 1:]

        return dev_df, val_df


class MMLUPrecisionTest(PrecisionTestBase):
    """
    MMLU
    """

    def compute_metric(self, subject_mapping):
        subject_mapping_adapt = {k: [None, None, v] for k, v in subject_mapping.items()}
        return super().compute_metric(subject_mapping_adapt)
