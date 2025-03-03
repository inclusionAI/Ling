
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
#include "utils.h"
#include <iostream>
#pragma GCC diagnostic push
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include <acl/acl.h>
#include <atb/utils.h>
#include <atb_speed/utils/file_system.h>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif

#include "atb_speed/log.h"
#include "atb_speed/utils/tensor_util.h"

namespace atb_speed {
void *Utils::GetCurrentStream()
{
    int32_t devId = 0;
    aclrtGetDevice(&devId);
    void *stream = c10_npu::getCurrentNPUStream(devId).stream();
    if (stream == nullptr) {
        ATB_SPEED_LOG_ERROR("get current stream fail");
    }
    return stream;
}

int64_t Utils::GetTensorNpuFormat(const at::Tensor &tensor)
{
#ifdef TORCH_HIGHER_THAN_PTA6
    return at_npu::native::get_npu_format(tensor);
#else
    return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

at::Tensor Utils::NpuFormatCast(const at::Tensor &tensor)
{
#ifdef TORCH_HIGHER_THAN_PTA6
    return at_npu::native::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
#else
    return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
#endif
}

void Utils::BuildVariantPack(const std::vector<torch::Tensor> &inTensors, const std::vector<torch::Tensor> &outTensors,
                             atb::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2Tensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2Tensor(outTensors.at(i)));
    }
}

bool Utils::AtTensorShapeEqualToTensor(const at::Tensor &atTensor, const atb::TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == atTensor.sizes().size()) {
        return false;
    }
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        if (tensorDesc.shape.dims[i] != atTensor.sizes()[i]) {
            return false;
        }
    }
    return true;
}

atb::Tensor Utils::AtTensor2Tensor(const at::Tensor &atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},    {at::ScalarType::Byte, ACL_UINT8},  {at::ScalarType::Char, ACL_INT8},
        {at::ScalarType::Half, ACL_FLOAT16}, {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},   {at::ScalarType::BFloat16, ACL_BF16},
    };

    if (!atTensor.is_contiguous()) {
        ATB_SPEED_LOG_ERROR("atTensor is not contiguous");
    }
    
    atb::Tensor tensor;
    tensor.desc.format = static_cast<aclFormat>(GetTensorNpuFormat(atTensor));
    tensor.deviceData = atTensor.data_ptr();

    tensor.desc.shape.dimNum = atTensor.sizes().size();
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        tensor.desc.shape.dims[i] = atTensor.sizes()[i];
    }

    if (tensor.desc.shape.dimNum == 1 && tensor.desc.shape.dims[0] == 0) {
        tensor.desc.shape.dimNum = 0;
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        tensor.desc.dtype = it->second;
    } else {
        throw std::runtime_error("AtTensor2Tensor: not support dtype");
    }

    tensor.dataSize = atb::Utils::GetTensorSize(tensor);

    return tensor;
}

at::Tensor Utils::CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc)
{
    static std::map<aclDataType, at::ScalarType> dtypeMap = {
        {ACL_BOOL, at::ScalarType::Bool},    {ACL_UINT8, at::ScalarType::Byte},  {ACL_INT8, at::ScalarType::Char},
        {ACL_FLOAT16, at::ScalarType::Half}, {ACL_FLOAT, at::ScalarType::Float}, {ACL_INT32, at::ScalarType::Int},
        {ACL_INT64, at::ScalarType::Long},   {ACL_BF16, at::ScalarType::BFloat16},
    };
    at::TensorOptions options = at::TensorOptions();
    auto it = dtypeMap.find(tensorDesc.dtype);
    if (it != dtypeMap.end()) {
        options = options.dtype(it->second);
    } else {
        throw std::runtime_error("CreateAtTensorFromTensorDesc: not support dtype");
    }

    options = options.layout(torch::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());

    ATB_SPEED_LOG_DEBUG("tensor_with_format stat, " << atb_speed::TensorUtil::TensorDescToString(tensorDesc));

#ifdef TORCH_HIGHER_THAN_PTA6
    at::Tensor newTensor = at_npu::native::empty_with_format(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);
#else
    at::Tensor newTensor = at_npu::native::NPUNativeFunctions::tensor_with_format(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);
#endif

    ATB_SPEED_LOG_DEBUG("tensor_with_format end, newTensor.format:" << GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous());
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        ATB_SPEED_LOG_WARN("tensor_with_format newTensor.format:" << GetTensorNpuFormat(newTensor)
                      << " != " << tensorDesc.format);
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    ATB_SPEED_LOG_DEBUG("tensor_with_format success, newTensor.options:" << newTensor.options()
                  << ", format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous());

    return newTensor;
}

void Utils::ContiguousAtTensor(std::vector<torch::Tensor> &atTensors)
{
    for (size_t i = 0; i < atTensors.size(); ++i) {
        if (!atTensors.at(i).is_contiguous()) {
            atTensors.at(i) = atTensors.at(i).contiguous();
        }
    }
}

void Utils::ContiguousAtTensor(torch::Tensor &atTensor)
{
    if (!atTensor.is_contiguous()) {
        atTensor = atTensor.contiguous();
    }
}
}
