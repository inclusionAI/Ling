# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import unittest
from unittest.mock import Mock
import random

import torch

from atb_llm.utils.dist import initialize_distributed
from atb_llm.utils.adapter_manager import AdapterWeightLoader
import atb_llm.utils.weights as weights_module


class FakeSafeTensor:
    def __init__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        self.tensor = tensor
        self.dtype = self.tensor.dtype

    def __getitem__(self, index):
        return self.tensor[index]

    def shape(self):
        return self.tensor.shape

    def get_shape(self):
        return self.tensor.shape

    def split(self, split_size, dim=0):
        return self.tensor.split(split_size, dim)


class TestAdapterWeightLoader(unittest.TestCase):

    def setUp(self):
        world_size = 2 * random.randint(1, 4)
        rank = random.randint(0, world_size - 1)
        npu_id = 0
        self.process_group, self.device = initialize_distributed(rank, npu_id, world_size)
        self.model_name_or_path = "test_model_path"
        self.dtype = torch.float16

        self.n = world_size * random.randint(1, 1024)
        self.k = world_size * random.randint(1, 1024)
        self.r = 2 ** random.randint(1, 6)

        mock_weight_files = Mock(return_value=[])
        weights_module.weight_files = mock_weight_files

        self.adapter_weight_loader = AdapterWeightLoader(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
        )

    def test_load_lora_a_tensor_parallel_dim_0(self):
        fake_lora_a_tensor = torch.randn(self.r, self.k, dtype=self.dtype)
        self.adapter_weight_loader.get_tensor = Mock(return_value=fake_lora_a_tensor)

        lora_tensor = self.adapter_weight_loader.get_lora_tensor(
            "test_lora_tensor", is_lora_a=True, padding_shape=(1, self.k),
            tensor_parallel_dim=0, align_size=128
        )
        self.assertTrue(torch.equal(lora_tensor, fake_lora_a_tensor))

    def test_load_lora_a_tensor_parallel_dim_1(self):
        fake_lora_a_tensor = torch.randn(self.r, self.k, dtype=self.dtype)
        self.adapter_weight_loader._get_slice = Mock(return_value=FakeSafeTensor(fake_lora_a_tensor))

        lora_tensor = self.adapter_weight_loader.get_lora_tensor(
            "test_lora_tensor", is_lora_a=True, padding_shape=(1, self.k),
            tensor_parallel_dim=1, align_size=1
        )
        chunk_size = self.k // self.process_group.size()
        start = chunk_size * self.process_group.rank()
        end = chunk_size * (self.process_group.rank() + 1)
        self.assertTrue(torch.equal(lora_tensor, fake_lora_a_tensor[:, start:end]))

    def test_load_lora_a_dummy(self):
        lora_tensor = self.adapter_weight_loader.get_lora_tensor(
            "test_lora_tensor", is_lora_a=True, padding_shape=(1, self.k),
            tensor_parallel_dim=1, align_size=1
        )
        self.assertTrue(torch.equal(lora_tensor, torch.zeros((1, self.k), dtype=self.dtype)))

    def test_load_lora_b_tensor_parallel_dim_0(self):
        fake_lora_b_tensor = torch.randn(self.n, self.r, dtype=self.dtype)
        self.adapter_weight_loader._get_slice = Mock(return_value=FakeSafeTensor(fake_lora_b_tensor))

        lora_tensor = self.adapter_weight_loader.get_lora_tensor(
            "test_lora_tensor", is_lora_a=False, padding_shape=(self.n, 1),
            tensor_parallel_dim=0, align_size=1
        )
        chunk_size = self.n // self.process_group.size()
        start = chunk_size * self.process_group.rank()
        end = chunk_size * (self.process_group.rank() + 1)
        self.assertTrue(torch.equal(lora_tensor, fake_lora_b_tensor[start:end, :]))

    def test_load_lora_b_tensor_parallel_dim_1(self):
        fake_lora_b_tensor = torch.randn(self.n, self.r, dtype=self.dtype)
        self.adapter_weight_loader.get_tensor = Mock(return_value=fake_lora_b_tensor)

        lora_tensor = self.adapter_weight_loader.get_lora_tensor(
            "test_lora_tensor", is_lora_a=False, padding_shape=(self.n, 1),
            tensor_parallel_dim=1, align_size=1
        )
        self.assertTrue(torch.equal(lora_tensor, fake_lora_b_tensor))
    
    def test_load_lora_b_dummy(self):
        lora_tensor = self.adapter_weight_loader.get_lora_tensor(
            "test_lora_tensor", is_lora_a=False, padding_shape=(self.n, 1),
            tensor_parallel_dim=1, align_size=1
        )
        self.assertTrue(torch.equal(lora_tensor, torch.zeros((self.n, 1), dtype=self.dtype)))
