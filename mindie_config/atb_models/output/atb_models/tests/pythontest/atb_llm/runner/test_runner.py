# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import unittest
from unittest.mock import patch

import pandas as pd
import torch
import torch_npu

from atb_llm.runner.model_runner import ModelRunner
from examples.server.cache import CacheConfig, ModelConfig, CacheManager


dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8
}


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 6
        self.bos_token_id = 7
        self.eos_token_id = 8

    @staticmethod
    def decode(self, token_id):
        return "A test string"

    def add_special_tokens(self, tokens):
        pass


class TestModelRunner(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.config = None
        self.tensor_names = {}

    @patch('atb_llm.utils.weights.Weights.get_tensor')
    @patch('atb_llm.utils.weights.Weights._get_slice')
    @patch('transformers.configuration_utils.PretrainedConfig.get_config_dict')
    def test_model_runner(self, mock_get_config_dict, mock_get_slice, mock_get_tensor):
        mock_get_config_dict.side_effect = self._get_config_dict_side_effect
        mock_get_slice.side_effect = self._get_slice_side_effect
        mock_get_tensor.side_effect = self._get_tensor_side_effect
        
        test_config = None
        with open("tests/pythontest/atb_llm/runner/test_config.json", "r") as f:
            test_config = json.load(f)
        model_names = test_config.keys()        

        with patch("atb_llm.utils.weights.weight_files", return_value=[]) as _, \
            patch('transformers.AutoTokenizer.from_pretrained', return_value=FakeTokenizer()) as _:
            for name in model_names:
                self.config = test_config[name]["config"]
                self.tensor_names = {}
                self.parse_tensor_info(test_config[name]["weights"])

                # init
                model_runner = ModelRunner("", 0, 1)
                
                # load weights
                model_runner.load_weights()

                for batch_size in [1, 16]:
                    block_size = 128
                    # prepare inputs
                    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64).repeat(batch_size).npu()
                    length = int(input_ids.shape[0] / batch_size)
                    position_ids = torch.arange(length, dtype=torch.int32).repeat(batch_size).npu()
                    block_tables_tensor = torch.arange(batch_size, dtype=torch.int32).view(batch_size, -1).npu()
                    slots = torch.cat([torch.arange(length, dtype=torch.int32) + b * block_size \
                        for b in range(batch_size)], dim=0).npu()
                    input_lengths_tensor = torch.tensor([length] * batch_size, dtype=torch.int64).npu()
                    prefill_head_indices = torch.cat([torch.tensor([length - 1 + b * length], dtype=torch.int64) \
                                                    for b in range(batch_size)], dim=0).npu()

                    model_config = ModelConfig(
                        model_runner.num_heads,
                        model_runner.num_kv_heads,
                        model_runner.config.num_key_value_heads \
                                                if hasattr(model_runner.config, 'num_key_value_heads') \
                                                else model_runner.config.num_kv_heads,
                        model_runner.k_head_size,
                        model_runner.v_head_size,
                        model_runner.num_layers,
                        model_runner.device,
                        model_runner.dtype,
                        model_runner.soc_info,
                        model_runner.kv_quant_type,
                        model_runner.fa_quant_type)

                    cache_config = CacheConfig(9, block_size)
                    cache_manager = CacheManager(cache_config, model_config)
                    
                    # forward
                    model_runner.forward(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        is_prefill=True,
                        block_tables=block_tables_tensor,
                        kv_cache=cache_manager.kv_cache,
                        slots=slots,
                        input_lengths=input_lengths_tensor,
                        max_seq_len=length,
                        lm_head_indices=prefill_head_indices
                    )

    def parse_tensor_info(self, csv_path):
        data = pd.read_csv(csv_path, sep='|')
        for i in range(len(data)):
            name = data.iloc[i, 1].strip()
            self.tensor_names[name] = {}
            self.tensor_names[name]["shape"] = list(map(int, data.iloc[i, 2].split(',')))
            self.tensor_names[name]["dtype"] = dtype_map.get(data.iloc[i, 3])

    def _get_config_dict_side_effect(self, model_path, **kwargs):
        kwargs.pop("cache_dir", None)
        kwargs.pop("force_download", False)
        kwargs.pop("resume_download", False)
        kwargs.pop("proxies", None)
        kwargs.pop("local_files_only", False)
        kwargs.pop("revision", None)
        
        return self.config, kwargs
    
    def _get_tensor_side_effect(self, tensor_name):
        tensor_shape = self.tensor_names.get(tensor_name).get("shape")
        tensor_dtype = self.tensor_names.get(tensor_name).get("dtype")
        return torch.randn(tensor_shape, dtype=tensor_dtype)

    def _get_slice_side_effect(self, tensor_name):
        tensor = self._get_tensor_side_effect(tensor_name)
        tensor.get_shape = tensor.size
        return tensor

if __name__ == "__main__":
    unittest.main()