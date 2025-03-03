/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#ifdef USE_TORCH_RUNNER
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#endif
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/operation_call.h"
#include "acltransformer/utils/tensor_util.h"

using namespace AclTransformer;

static void SetCurrentDevice()
{
    const int deviceId = 0;
    ASD_LOG(DEBUG) << "AsdRtDeviceSetCurrent " << deviceId;
    int ret = AsdRtDeviceSetCurrent(deviceId);
    if (ret != ASDRT_SUCCESS) {
        ASD_LOG(ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;
        return;
    }
    ASD_LOG(DEBUG) << "AsdRtDeviceSetCurrent success";
}

static void CreateTensor(AsdOps::Tensor &tensor)
{
    tensor.dataSize = TensorUtil::CalcTensorDataSize(tensor);
    int ret = AsdRtMemMallocDevice(&tensor.data, tensor.dataSize, ASDRT_MEM_DEFAULT);
    ASD_LOG_IF(ret != ASDRT_SUCCESS, ERROR) << "AsdRtMemMallocDevice fail, ret:" << ret;
}

static void CreateTensors(AsdOps::SVector<AsdOps::Tensor> &inTensors, AsdOps::SVector<AsdOps::Tensor> &outTensors)
{
    AsdOps::TensorDesc desc = {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_NCHW, {2, 3}};

    const int inTensorCount = 2;
    inTensors.resize(inTensorCount);

    AsdOps::Tensor &inTensor0 = inTensors.at(0);
    inTensor0.desc = desc;
    CreateTensor(inTensor0);

    AsdOps::Tensor &inTensor1 = inTensors.at(1);
    inTensor1.desc = desc;
    CreateTensor(inTensor1);

    outTensors.resize(1);
    AsdOps::Tensor &outTensor0 = outTensors.at(0);
    outTensor0.desc = desc;
    CreateTensor(outTensor0);
}

static void FreeTensor(AsdOps::Tensor &tensor)
{
    if (tensor.data) {
        int ret = AsdRtMemFreeDevice(tensor.data);
        ASD_LOG_IF(ret != ASDRT_SUCCESS, ERROR) << "AsdRtMemFreeDevice fail";
        tensor.data = nullptr;
        tensor.dataSize = 0;
    }
}

static void FreeTensors(AsdOps::SVector<AsdOps::Tensor> &inTensors, AsdOps::SVector<AsdOps::Tensor> &outTensors)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        FreeTensor(inTensors.at(i));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        FreeTensor(outTensors.at(i));
    }
}

static void InitTorch()
{
#ifdef USE_TORCH_RUNNER
    ASD_LOG(DEBUG) << "c10_npu::NPUCachingAllocator::init";
    c10_npu::NPUCachingAllocator::init();
#endif
}

int main(int argc, const char *argv[])
{
    SetCurrentDevice();

    AclTransformer::OperationCall opCall("AddOperation", AclTransformer::AddParam());

    void *stream = nullptr;
    int ret = AsdRtStreamCreate(&stream);
    if (ret != 0) {
        ASD_LOG(ERROR) << "AsdRtStreamCreate fail, ret:" << ret;
        return 1;
    }

    InitTorch();

    AsdOps::SVector<AsdOps::Tensor> inTensors;
    AsdOps::SVector<AsdOps::Tensor> outTensors;
    CreateTensors(inTensors, outTensors);

    ret = opCall.ExecuteSync(inTensors, outTensors, stream);
    if (ret != 0) {
        ASD_LOG(ERROR) << "ExecuteSync fail, ret:" << ret;
    } else {
        ASD_LOG(DEBUG) << "ExecuteSync success";
    }

    AsdRtStreamDestroy(stream);

    FreeTensors(inTensors, outTensors);

    return 0;
}