# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from modeltest.metric.acc import AccMetric
from atb_llm.utils.multimodal_utils import MultimodalInput
from .precision_task import PrecisionTask


class VocalSoundPrecisionTask(PrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)

    @staticmethod
    def get_qwen2_audio_queries(batched_data, input_texts):
        queries = MultimodalInput(input_texts, None, None, [[item['audio_path']] for item in batched_data]) 
        return queries
    
    @staticmethod
    def _find_choice(result):
        choose_map = {
            "A": "Laughter",
            "B": "Sigh",
            "C": "Cough",
            "D": "Throat clearing", 
            "E": "Sneeze",
            "F": "Sniff"
        }
        predict_result = None
        for choose in choose_map:
            choose_ = choose + '.'
            if (choose_map[choose].lower() in result.lower()) or (choose_ in result) or (choose in result):
                predict_result = choose
                break
        return predict_result
    
    def prepare_data(self, metric):
        # 返回一个包含字典的列表的列表
        if isinstance(metric, AccMetric):
            metric.correct_num_list.append(0)
        datasets = []
        for _, _, files in os.walk(self.task_config.local_dataset_path):
            audio_paths = []
            for file in files:
                audio_paths.append({'audio_path': os.path.join(self.task_config.local_dataset_path, file)})
        datasets.append(audio_paths)
        return datasets

    def build_queries(self, _, batched_data, model_config):
        input_texts = model_config.mm_model.get('input_texts')
        func_map = {
            "qwen2_audio": "get_qwen2_audio_queries",
        }
        try:
            func_name = func_map[model_config.model_name]
        except KeyError as e:
            raise KeyError(f"Unsupported! Please choose from [{func_map.keys()}].") from e
        func = getattr(self, func_name)
        return func(batched_data, input_texts)

    def result_judge(self, metric, generate_token_lists, _, sub_dataset_idx, batched_data):
        gt_answer_map = {
            "laughter": "A",
            "sigh": "B",
            "cough": "C",
            "throatclearing": "D", 
            "sneeze": "E",
            "sniff": "F"
        }
        for idx, _ in enumerate(batched_data):
            choice = self._find_choice(generate_token_lists[idx])
            gt_answer = batched_data[idx].get('audio_path').split('.')[0].split('_')[-1]
            gt_choice = gt_answer_map.get(gt_answer)

            if choice == gt_choice:
                metric.correct_num += 1
                metric.correct_num_list[sub_dataset_idx] += 1
