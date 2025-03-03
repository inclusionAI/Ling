# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import unittest
from unittest.mock import Mock, patch, mock_open
import random


import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from ddt import ddt, data, unpack

from atb_llm.utils.dist import initialize_distributed
from atb_llm.utils.adapter_manager import AdapterWeightLoader, \
    AdapterManager, AdapterInfo, AdapterIdsType, AdapterIdsRecord
import atb_llm.utils.weights as weights_module
from atb_llm.utils.layers.linear.fast_linear import FastLinear
from atb_llm.utils.layers.linear.lora import Lora
from atb_llm.models.base.config import LoraConfig
import atb_llm.utils.adapter_manager as adapter_manager_module
from atb_llm.utils.log import print_log, logger
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.file_utils import MAX_PATH_LENGTH


BASE_ADAPTER_NAME = "base"
SORTED_ADAPTER_NAME = "sorted"
ADATPER1_STR = "adapter1"
ADATPER2_STR = "adapter2"
ADATPER3_STR = "adapter3"
ADATPER4_STR = "adapter4"
ADATPER5_STR = "adapter5"


class FakeModel(nn.Module):
    def __init__(self, linear_module_prefixes):
        super().__init__()
        linear_module = FastLinear(torch.rand((4096, 2048), dtype=torch.float16), None)
        linear_module.prefixes = linear_module_prefixes
        self.linear_module = linear_module


class FakeModelWithLora(nn.Module):
    def __init__(self, lora_module_dict):
        super().__init__()
        self.soc_info = NPUSocInfo()
        linear_module = FastLinear(torch.rand((4096, 2048), dtype=torch.float16), None)
        linear_module.prefixes = ["linear_module_1"]
        linear_module.lora = lora_module_dict
        self.linear_module = linear_module


@ddt
class TestAdapterManger(unittest.TestCase):
    def setUp(self):
        self.lora_adapter = {ADATPER1_STR: "fake_adapter_1_path", ADATPER2_STR: "fake_adapter_2_path"}

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
        adapter_manager_module.weight_files = mock_weight_files

        self.adapter_weight_loader = AdapterWeightLoader(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
        )
        self.adapter_manager = AdapterManager(self.adapter_weight_loader)

    @data((None, [BASE_ADAPTER_NAME]), ([None, None], [BASE_ADAPTER_NAME]),
          ([ADATPER1_STR, ADATPER1_STR], [ADATPER1_STR]),
          ([ADATPER3_STR], [BASE_ADAPTER_NAME]), ([ADATPER1_STR, ADATPER2_STR], [ADATPER1_STR, ADATPER2_STR]))
    @unpack
    def test_preprocess_adatper_ids(self, adapter_ids, expected_adapter_ids):
        self._update_adapter_ids_registry()
        effective_adapter_ids = self.adapter_manager.preprocess_adatper_ids(adapter_ids)
        self.assertEqual(effective_adapter_ids, expected_adapter_ids)

    @data(([BASE_ADAPTER_NAME], AdapterIdsType.SINGLE), ([BASE_ADAPTER_NAME, ADATPER1_STR], AdapterIdsType.MIXED),
          ([ADATPER1_STR, ADATPER2_STR], AdapterIdsType.SORTED))
    @unpack
    def test_update_adapter_check_modification(self, adapter_ids, adapter_ids_type):
        self._update_adapter_ids_registry()
        need_update = self.adapter_manager.update_adapter(adapter_ids)
        self.assertTrue(need_update)
        self.assertEqual(self.adapter_manager.previous_adapter_ids, AdapterIdsRecord(adapter_ids_type, adapter_ids))

    @data(([BASE_ADAPTER_NAME], [ADATPER1_STR], True), ([BASE_ADAPTER_NAME], [BASE_ADAPTER_NAME], False),
          ([ADATPER1_STR, BASE_ADAPTER_NAME], [ADATPER1_STR, ADATPER2_STR], False),
          ([ADATPER1_STR, BASE_ADAPTER_NAME], [ADATPER2_STR, ADATPER1_STR], True))
    @unpack
    def test_update_adapter_check_return_value(self, previous_adapter_ids, adapter_ids, expected_result):
        self._update_adapter_ids_registry()
        self.adapter_manager.update_adapter(previous_adapter_ids)
        need_update = self.adapter_manager.update_adapter(adapter_ids)
        self.assertEqual(need_update, expected_result)

    @data((None, True), ([None, None], True), ([ADATPER1_STR, ADATPER2_STR], True),
         ([ADATPER2_STR, ADATPER1_STR], False), ([BASE_ADAPTER_NAME, ADATPER1_STR], False),
         ([ADATPER1_STR, ADATPER3_STR], True))
    @unpack
    def test_check_adapter_ids_is_sorted(self, adapter_ids, expected_result):
        self._update_adapter_ids_registry()
        actual_result = self.adapter_manager.check_adapter_ids_is_sorted(adapter_ids)
        self.assertEqual(actual_result, expected_result)

    @data((256, 1), (256, 256))
    @unpack
    def test_get_base_weight_shape(self, expected_n, expected_k):
        linear_module = FastLinear(torch.rand((expected_n, expected_k), dtype=self.dtype), None)

        n, k = self.adapter_manager.get_base_weight_shape(linear_module)
        self.assertEqual(n, expected_n)
        self.assertEqual(k, expected_k)

    def test_get_scaling(self):
        scaling = self.adapter_manager.get_scaling(LoraConfig(r=4, lora_alpha=64), prefix="")
        self.assertEqual(scaling, 16)
        pattern_key = "model.layer.0.attention.qkv"
        scaling = self.adapter_manager.get_scaling(LoraConfig(
            r=4,
            lora_alpha=64,
            rank_pattern={"attention.qkv": 8},
            alpha_pattern={"attention.qkv": 16}),
            prefix=pattern_key)
        self.assertEqual(scaling, 2)
        scaling = self.adapter_manager.get_scaling(LoraConfig(r=4, lora_alpha=64, use_rslora=True), prefix="")
        self.assertEqual(scaling, 32)

    def test_load_adapter_duplicate(self):
        self.adapter_manager.adapter_info_registry = {
            ADATPER1_STR: AdapterInfo(
                idx=len(self.adapter_manager.adapter_info_registry), adapter_path="fake_adapter_1_path")
        }
        self.adapter_manager.load_adapter(ADATPER1_STR, "fake_adapter_1_path", self.device)
        self.adapter_manager.load_lora_weight = Mock()
        self.adapter_manager.load_lora_weight.add_operation.assert_not_called()

    @patch("os.fdopen", new_callable=mock_open, read_data='{"r": 2, "lora_alpha": 8}')
    @patch("os.open", return_value=3)
    def test_load_adapter_single_prefix(self, mock_open_func, mock_fdopen):
        adapter_manager_module.standardize_path = Mock(side_effect=self._mock_standardize_path)
        adapter_manager_module.check_file_safety = Mock(side_effect=self._mock_check_file_safety)
        adapter_manager_module.file_utils_safe_open = Mock(side_effect=self._mock_safe_open)
        adapter_manager_module.check_path_permission = Mock(return_value=None)
        self.adapter_manager.get_base_weight_shape = Mock(return_value=(self.n, self.k))
        fake_lora_a_tensor = torch.rand((self.r, self.k), dtype=self.dtype)
        fake_lora_b_tensor = torch.rand((self.n, self.r), dtype=self.dtype)
        self.adapter_manager.base_model = FakeModel(["linear_module"])
        self.adapter_manager.lora_weights_loader = self.adapter_weight_loader
        self.adapter_manager.lora_weights_loader.get_lora_tensor = Mock(
            side_effect=[fake_lora_a_tensor, fake_lora_b_tensor]
        )
        self.adapter_manager.load_adapter(ADATPER1_STR, "fake_adapter_1_path", self.device)
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[ADATPER1_STR],
            AdapterInfo(idx=0, adapter_path="fake_adapter_1_path", config=LoraConfig(r=2, lora_alpha=8))
        )
        self.assertIsInstance(self.adapter_manager.base_model.linear_module.lora[ADATPER1_STR], Lora)
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[ADATPER1_STR].lora_a, fake_lora_a_tensor))
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[ADATPER1_STR].lora_b,
            (fake_lora_b_tensor * 4).T.contiguous()  # lora_alpha / r
        ))

    @patch("os.fdopen", new_callable=mock_open, read_data='{"r": 2, "lora_alpha": 8}')
    @patch("os.open", return_value=3)
    def test_load_adapter_multiple_prefix(self, mock_open_func, mock_fdopen):
        adapter_manager_module.standardize_path = Mock(side_effect=self._mock_standardize_path)
        adapter_manager_module.check_file_safety = Mock(side_effect=self._mock_check_file_safety)
        adapter_manager_module.file_utils_safe_open = Mock(side_effect=self._mock_safe_open)
        adapter_manager_module.check_path_permission = Mock(return_value=None)
        self.adapter_manager.get_base_weight_shape = Mock(return_value=(self.n, self.k))
        fake_lora_a_tensor_1 = torch.rand((self.r, self.k), dtype=self.dtype)
        fake_lora_a_tensor_2 = torch.rand((self.r * 2, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, self.r), dtype=self.dtype)
        fake_lora_b_tensor_2 = torch.rand((self.n, self.r * 2), dtype=self.dtype)
        self.adapter_manager.base_model = FakeModel(["linear_module1", "linear_module2"])
        self.adapter_manager.lora_weights_loader = self.adapter_weight_loader
        self.adapter_manager.lora_weights_loader.get_lora_tensor = Mock(
            side_effect=[fake_lora_a_tensor_1, fake_lora_a_tensor_2,
                         fake_lora_b_tensor_1, fake_lora_b_tensor_2]
        )
        self.adapter_manager.load_adapter(ADATPER1_STR, "fake_adapter_1_path", self.device)
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[ADATPER1_STR],
            AdapterInfo(idx=0, adapter_path="fake_adapter_1_path", config=LoraConfig(r=2, lora_alpha=8))
        )
        self.assertIsInstance(self.adapter_manager.base_model.linear_module.lora[ADATPER1_STR], Lora)
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[ADATPER1_STR].lora_a,
            torch.cat([fake_lora_a_tensor_1, fake_lora_a_tensor_2], 0)))
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[ADATPER1_STR].lora_b,
            (torch.block_diag(fake_lora_b_tensor_1, fake_lora_b_tensor_2) * 4).T.contiguous()
        ))

    def test_load_dummy_adapter(self):
        self.adapter_manager.base_model = FakeModel(["linear_module1", "linear_module2"])
        self.adapter_manager.load_dummy_adapter()
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME],
            AdapterInfo(idx=0, adapter_path="",
                        config=LoraConfig(r=1, lora_alpha=1, use_rslora=False))
        )
        self.assertIsInstance(self.adapter_manager.base_model.linear_module.lora[BASE_ADAPTER_NAME], Lora)
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[BASE_ADAPTER_NAME].lora_a,
            torch.zeros([1, 2048], dtype=self.dtype)))
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[BASE_ADAPTER_NAME].lora_b,
            torch.zeros([1, 4096], dtype=self.dtype)))

    def test_concate_sorted_adapter(self):
        self.adapter_manager.adapter_info_registry = {
            ADATPER2_STR: AdapterInfo(idx=0, adapter_path="fake_adapter_2_path"),
            ADATPER1_STR: AdapterInfo(idx=1, adapter_path="fake_adapter_1_path")
        }
        lora_module_dict = nn.ModuleDict()
        max_r = 64
        fake_lora_a_tensor_1 = torch.rand((max_r, self.k), dtype=self.dtype)
        fake_lora_a_tensor_2 = torch.rand((max_r, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, max_r), dtype=self.dtype).T.contiguous()
        fake_lora_b_tensor_2 = torch.rand((self.n, max_r), dtype=self.dtype).T.contiguous()
        lora_module_dict[ADATPER1_STR] = Lora(fake_lora_a_tensor_1, fake_lora_b_tensor_1, r=1, alpha=1)
        lora_module_dict[ADATPER2_STR] = Lora(fake_lora_a_tensor_2, fake_lora_b_tensor_2, r=1, alpha=1)
        self.adapter_manager.base_model = FakeModelWithLora(lora_module_dict)
        self.adapter_manager.concate_sorted_adapter()
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[SORTED_ADAPTER_NAME],
            AdapterInfo(idx=2, adapter_path="",
                        config=LoraConfig(r=1, lora_alpha=1, use_rslora=False))
        )
        self.assertIsInstance(self.adapter_manager.base_model.linear_module.lora[SORTED_ADAPTER_NAME], Lora)
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[SORTED_ADAPTER_NAME].lora_a,
            torch.cat([fake_lora_a_tensor_2.unsqueeze(0), fake_lora_a_tensor_1.unsqueeze(0)], dim=0))
        )
        self.assertTrue(torch.equal(
            self.adapter_manager.base_model.linear_module.lora[SORTED_ADAPTER_NAME].lora_b,
            torch.cat([fake_lora_b_tensor_2.unsqueeze(0), fake_lora_b_tensor_1.unsqueeze(0)], dim=0))
        )

    def test_sort_adapter_ids(self):
        candidates = [ADATPER1_STR, ADATPER2_STR, ADATPER3_STR, ADATPER4_STR, ADATPER5_STR]
        for i, item in enumerate(candidates):
            self.adapter_manager.adapter_info_registry[item] = AdapterInfo(idx=i, adapter_path="")
        adapter_ids = [ADATPER2_STR, ADATPER4_STR, ADATPER5_STR, ADATPER5_STR, ADATPER1_STR]
        sorted_adapter_idx, revert_adapter_idx = self.adapter_manager.sort_adapter_ids(adapter_ids)
        self.assertTrue(sorted_adapter_idx, [4, 0, 1, 2, 3])
        self.assertTrue(revert_adapter_idx, [1, 2, 3, 4, 0])

    def test_get_adapters_single_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.update_adapter([ADATPER1_STR])
        lora_module_dict = nn.ModuleDict()
        fake_lora_a_tensor_1 = torch.rand((self.r, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, self.r), dtype=self.dtype)
        lora_module_dict[ADATPER1_STR] = Lora(fake_lora_a_tensor_1, fake_lora_b_tensor_1, r=1, alpha=1)
        self.adapter_manager.base_model = FakeModelWithLora(lora_module_dict)
        adapter_weights = self.adapter_manager.get_adapters([ADATPER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.equal(adapter_weights[0], fake_lora_a_tensor_1))
        self.assertTrue(torch.equal(adapter_weights[1], fake_lora_b_tensor_1))

    def test_get_adapters_mixed_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.update_adapter([ADATPER2_STR, ADATPER1_STR])
        lora_module_dict = nn.ModuleDict()
        max_r = 64
        fake_lora_a_tensor_1 = torch.rand((max_r, self.k), dtype=self.dtype)
        fake_lora_a_tensor_2 = torch.rand((max_r, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, max_r), dtype=self.dtype).T.contiguous()
        fake_lora_b_tensor_2 = torch.rand((self.n, max_r), dtype=self.dtype).T.contiguous()
        lora_module_dict[ADATPER2_STR] = Lora(fake_lora_a_tensor_2, fake_lora_b_tensor_2, r=1, alpha=1)
        lora_module_dict[ADATPER1_STR] = Lora(fake_lora_a_tensor_1, fake_lora_b_tensor_1, r=1, alpha=1)
        self.adapter_manager.base_model = FakeModelWithLora(lora_module_dict)
        adapter_weights = self.adapter_manager.get_adapters([ADATPER2_STR, ADATPER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.equal(
            adapter_weights[0],
            pad_sequence([fake_lora_a_tensor_2, fake_lora_a_tensor_1], batch_first=True)
        ))
        self.assertTrue(torch.equal(
            adapter_weights[1],
            pad_sequence([fake_lora_b_tensor_2, fake_lora_b_tensor_1], batch_first=True)
        ))

    def test_get_adapters_sorted_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.update_adapter([ADATPER2_STR, ADATPER1_STR])
        lora_module_dict = nn.ModuleDict()
        max_r = 64
        fake_lora_a_tensor_1 = torch.rand((max_r, self.k), dtype=self.dtype)
        fake_lora_a_tensor_2 = torch.rand((max_r, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, max_r), dtype=self.dtype).T.contiguous()
        fake_lora_b_tensor_2 = torch.rand((self.n, max_r), dtype=self.dtype).T.contiguous()
        lora_module_dict[ADATPER2_STR] = Lora(fake_lora_a_tensor_2, fake_lora_b_tensor_2, r=1, alpha=1)
        lora_module_dict[ADATPER1_STR] = Lora(fake_lora_a_tensor_1, fake_lora_b_tensor_1, r=1, alpha=1)
        self.adapter_manager.base_model = FakeModelWithLora(lora_module_dict)
        adapter_weights = self.adapter_manager.get_adapters([ADATPER2_STR, ADATPER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.equal(
            adapter_weights[0],
            pad_sequence([fake_lora_a_tensor_2, fake_lora_a_tensor_1], batch_first=True)
        ))
        self.assertTrue(torch.equal(
            adapter_weights[1],
            pad_sequence([fake_lora_b_tensor_2, fake_lora_b_tensor_1], batch_first=True)
        ))

    def _update_adapter_ids_registry(self):
        for adapter_id, adapter_path in self.lora_adapter.items():
            self.adapter_manager.adapter_info_registry[adapter_id] = AdapterInfo(
                idx=len(self.adapter_manager.adapter_info_registry), adapter_path=adapter_path)
        self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.adapter_manager.adapter_info_registry), adapter_path="")

    def _mock_standardize_path(self, path, max_path_length=MAX_PATH_LENGTH, check_link=True):
        return path

    def _mock_check_file_safety(self, path):
        print_log(self.process_group.rank(), logger.info, f"path {path} is not used in _mock_check_file_safety")
        pass

    def _mock_from_dict(self, path):
        print_log(self.process_group.rank(), logger.info, f"path {path} is not used in _mock_check_file_safety")
        return {"r": 2, "lora_alpha": 8}

    def _mock_safe_open(self, file_path: str, mode='r',
        encoding=None, permission_mode=0o600, is_exist_ok=True, **kwargs):
        return os.fdopen(os.open(file_path, os.O_RDONLY), "r", encoding=encoding)
