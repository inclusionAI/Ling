# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import shutil
import pandas as pd
from atb_llm.utils.log.logging import logger


class Metric():
    def __init__(
            self, 
            metric_type,
            task_config,
            model_config,
            device_type,
            runner_config,
            output_dir,
            result_file_name=None
    ):
        self.metric_type = metric_type
        self.task_config = task_config
        self.model_config = model_config
        self.runner_config = runner_config
        self.output_dir = output_dir
        result_base_path = os.path.join(
            f"{self.model_config.data_type}",
            self.model_config.model_name
        )
        self.result_dir = os.path.join(
            self.output_dir,
            "results",
            device_type, 
            f"{self.task_config.task_type}_test",
            self.task_config.task_name,
            result_base_path
        )
        self.data_dir = os.path.join(
            self.output_dir,
            "data",
            device_type, 
            f"{self.task_config.task_type}_test",
            self.task_config.task_name,
            result_base_path
        )
        self.log_dir = os.path.join(
            self.output_dir,
            "logs"
        )
        self.debug_dir = os.path.join(
            self.output_dir,
            "debug",
            device_type, 
            f"{self.task_config.task_type}_test",
            self.task_config.task_name,
            result_base_path
        )
        self.result_file_path = os.path.join(
            self.result_dir, 
            f"{self.task_config.task_name}_{self.model_config.model_type}_batch{self.runner_config.batch_size}_"
            f"tp{self.runner_config.tp}_result.csv"
        ) if result_file_name is None else result_file_name
        self.debug_info_path = os.path.join(
            self.debug_dir, 
            f"{self.task_config.task_name}_{self.model_config.model_type}_batch{self.runner_config.batch_size}_"
            f"tp{self.runner_config.tp}_debug_info.csv"
        )      
        self.case_num = 0
        self.case_num_list = []
        self.error_num = 0
        self.error_list: list[tuple] = [] # [(sub_dataset_idx, batch_idx)]
        self.csv_debug = {
            "key": [],
            "queries": [],
            "input_token_ids": [],
            "output_token_ids": [], 
            "test_result": [],
            "golden_result": [],
            "pass": []
        }
        self.post_init()

    def post_init(self):
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.__create_folder(self.data_dir)

    def print_metric(self):
        logger.info(f"{self.task_config.task_name} test result saved to: {self.result_file_path}")

    def save_debug(self):        
        df = pd.DataFrame(self.csv_debug)
        df.to_csv(self.debug_info_path, index=False, encoding='utf-8')
        logger.info(f"{self.task_config.task_name} debug info saved to: {self.debug_info_path}")

    def __create_folder(self, folder_path):
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path, ignore_errors=True)
            except Exception as e:
                self.logger.error("Error deleting folder %s: %s", folder_path, e)
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(folder_path):
            self.logger.error("folder %s create fail", folder_path)
            raise RuntimeError(f"folder {folder_path} create fail")