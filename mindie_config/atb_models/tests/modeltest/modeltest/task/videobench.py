# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import json
from atb_llm.utils.file_utils import safe_open
from modeltest.metric.acc import AccMetric
from .precision_task import PrecisionTask


class VideoBenchPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
        self.eval_qa_root = self.task_config.local_dataset_path
        answer_json_path = os.path.join(task_config.local_dataset_path, "answer/ANSWER.json")
        with safe_open(answer_json_path, 'r', encoding='utf-8') as f:
            self.gt_answer = json.load(f)
    
    @staticmethod
    def get_qwen2_vl_queries(batched_data, prefix, suffix):
        queries = [[{"video": item["video"]},
                    {"text": item["question"]}] for item in batched_data]
        return queries

    @staticmethod
    def _find_choice(result):
        choice_list = ['A', 'B', 'C', 'D', 'E', 'F']
        for choice in choice_list:
            if choice in result:
                return choice
        return ""
    
    def prepare_data(self, metric):
        dataset_qajson = {
        "ActivityNet": f"{self.eval_qa_root}/ActivityNet_QA_new.json",
        "Driving-decision-making": f"{self.eval_qa_root}/Driving-decision-making_QA_new.json",
        "Driving-exam": f"{self.eval_qa_root}/Driving-exam_QA_new.json",
        "MOT": f"{self.eval_qa_root}/MOT_QA_new.json",
        "MSRVTT": f"{self.eval_qa_root}/MSRVTT_QA_new.json",
        "MSVD": f"{self.eval_qa_root}/MSVD_QA_new.json",
        "MV": f"{self.eval_qa_root}/MV_QA_new.json",
        "NBA": f"{self.eval_qa_root}/NBA_QA_new.json",
        "SQA3D": f"{self.eval_qa_root}/SQA3D_QA_new.json",
        "TGIF": f"{self.eval_qa_root}/TGIF_QA_new.json",
        "TVQA": f"{self.eval_qa_root}/TVQA_QA_new.json",
        "Ucfcrime": f"{self.eval_qa_root}/Ucfcrime_QA_new.json",
        "Youcook2": f"{self.eval_qa_root}/Youcook2_QA_new.json",
        "sampling_data":f"{self.eval_qa_root}/sampling_datasets_QA_new.json"
        }
        datasets = []
        for dataset_name in self.task_config.subject_mapping.keys():
            if dataset_name not in dataset_qajson:
                continue
            qa_json = dataset_qajson.get(dataset_name)
            # 返回一个包含字典的列表的列表
            if isinstance(metric, AccMetric):
                metric.correct_num_list.append(0)
            with safe_open(qa_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sub_dataset = []
            for _, (key, item) in enumerate(data.items()):
                try:
                    question = item['question']
                    choices_ = "choices"
                    if len(item[choices_]) == 6:
                        question += (
                f"Choices: A.{item[choices_]['A']} B.{item[choices_]['B']} C.{item[choices_]['C']}"
                + f" D.{item[choices_]['D']} E.{item[choices_]['E']} F.{item[choices_]['F']}"
                + f" \n Among the six options A, B, C, D, E, F above, the one closest to the correct answer is:"
                        )
                    elif len(item[choices_]) == 5:
                        question += (
                f" A.{item[choices_]['A']} B.{item[choices_]['B']} C.{item[choices_]['C']}"
                + f" D.{item[choices_]['D']} E.{item[choices_]['E']}"
                + f" \n Among the five options A, B, C, D, E above, the one closest to the correct answer is: "
                        )
                    elif len(item[choices_]) == 4:
                        question += (
                f" A.{item[choices_]['A']} B.{item[choices_]['B']} C.{item[choices_]['C']}"
                + f" D.{item[choices_]['D']}" 
                + f" \n Among the four options A, B, C, D above, the one closest to the correct answer is:"
                        )
                    elif len(item[choices_]) == 3:
                        question += (
                f" A.{item[choices_]['A']} B.{item[choices_]['B']} C.{item[choices_]['C']}"
                + f" \n Among the three options A, B, C above, the one closest to the correct answer is: "
                        )
                    elif len(item[choices_]) == 2:
                        question += (
                f" A.{item[choices_]['A']} B.{item[choices_]['B']}" 
                + f" \n Among the two options A, B above, the one closest to the correct answer is: "
                        )

                    question += (
                " Please respond with only the corresponding options and do not provide any explanations" 
                + " or additional information. ASSISTANT:"
                                )

                    sub_dataset.append({"question": question, "video" : item['vid_path'], "qid" : key})

                except Exception as e:
                    raise Exception from e
                
            datasets.append(sub_dataset)
        return datasets

    def build_queries(self, _, batched_data, model_config):
        prefix = model_config.mm_model.get('prompt_prefix')
        suffix = model_config.mm_model.get('prompt_suffix')
        func_map = {
            "qwen2_vl": "get_qwen2_vl_queries",
        }
        try:
            func_name = func_map[model_config.model_name]
        except KeyError as e:
            raise KeyError(f"Unsupported! Please choose from [{func_map.keys()}].") from e
        func = getattr(self, func_name)
        return func(batched_data, prefix, suffix)


    def result_judge(self, metric, generate_token_lists, _, sub_dataset_idx, batched_data):
        for idx, item in enumerate(batched_data):
            qid = item.get("qid")
            answer = generate_token_lists[idx]
            choice = self._find_choice(answer)
            sub_dataset = list(self.task_config.subject_mapping.keys())[sub_dataset_idx]
            gt_choice = self.gt_answer.get(sub_dataset).get(qid).get("answer")

            if choice == gt_choice:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1

