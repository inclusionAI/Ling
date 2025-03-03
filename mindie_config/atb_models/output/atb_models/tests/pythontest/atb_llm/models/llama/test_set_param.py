# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import unittest
import json
import torch
import torch_npu


class TestSetParam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_npu.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError(
                "env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path,
                                "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)

    def setUp(self):
        self.llama_model = torch.classes.ModelTorch.ModelTorch("llama_LlamaDecoderModel")

    def test_set_param(self):
        codel_param = {
            "normEps": 1e-5,
            "normType": None
        }
        self.llama_model.set_param(json.dumps({**codel_param}))


if __name__ == '__main__':
    unittest.main()