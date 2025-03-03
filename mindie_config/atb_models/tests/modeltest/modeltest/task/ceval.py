# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from .ceval_mmlu import CEvalMMLUFewShotsPrecisionTask, CEvalMMLUZeroShotPrecisionTask


class CEvalFewShotsPrecisionTask(CEvalMMLUFewShotsPrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)

    def get_test_set(self):
        return 'val'
    
    def get_row_col(self):
        return 1


class CEvalZeroShotPrecisionTask(CEvalMMLUZeroShotPrecisionTask):
    def __init__(self, task_config) -> None:
        super().__init__(task_config)
    
    def format_example(self, name, batch_data, idx):
        question = batch_data[idx][0]
        option_a = batch_data[idx][1]
        option_b = batch_data[idx][2]
        option_c = batch_data[idx][3]
        option_d = batch_data[idx][4]
        prompt = self.task_config.prompt.format(name["name_ch"], 
                                                question, option_a, option_b, option_c, option_d)
        return prompt
    
    def get_test_set(self):
        return 'val'
    
    def get_row_col(self):
        return 1