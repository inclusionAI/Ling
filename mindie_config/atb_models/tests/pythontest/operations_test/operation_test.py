# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import sys
import json
import re
import unittest
import logging

import torch

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))
import _libatb_torch as atb

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")    

DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
if DEVICE_ID is not None:
    torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))

    
class OperationTest(unittest.TestCase):
    def run_compare(self, op_set, inputs, outputs, bind=None):
        op_type, op_param, op_name = op_set
        operation = atb._BaseOperation(op_type=op_type,
                                       op_param=json.dumps(op_param),
                                       op_name=op_name)

        if bind is None:
            operation.forward(inputs, outputs)
        else:
            operation.forward(inputs, outputs, bind)
        outputs_tensors = list(outputs.values())
        golden_out_tensors = self.get_golden(inputs)
        self.__compare_all_golden(outputs_tensors, golden_out_tensors)

    def __compare_all_golden(self, out_tensors, golden_out_tensors):
        self.assertEqual(len(out_tensors), len(golden_out_tensors))
        tensor_count = len(out_tensors)
        for i in range(tensor_count):
            self.assertTrue(self._compare_golden(
                out_tensors[i], golden_out_tensors[i]))

    def _compare_golden(self, out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
        logger = logging.getLogger()
        logger.info('Current rtol and atol are %s, %s\n', rtol, atol)
        result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
        if not result:
            logger.info("out_tensor.shape: %s, golden_out_tensor.shape: %s", out_tensor.shape, golden_out_tensor.shape)
            logger.info("out_tensor:\n %s", out_tensor)
            logger.info("golden_out_tensor:\n %s", golden_out_tensor)
        logger.info('\nCompare_Result: %s\n', result)
        return result
