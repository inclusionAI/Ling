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
#include "buffer_device.h"
#include <acl/acl.h>
#include <atb_speed/utils/timer.h>
#include <atb/types.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/statistic.h"
#include "pytorch/adapter/utils/utils.h"

namespace atb_speed {
constexpr uint64_t KB_1 = 1024;
constexpr uint64_t MB_1 = 1024 * 1024;
constexpr uint64_t GB_1 = 1024 * 1024 * 1024;
constexpr uint64_t DIM_NUM_2 = 2;
BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    ATB_SPEED_LOG_DEBUG("BufferDevice::BufferDevice called, bufferSize:" << bufferSize);
    bufferSize_ = bufferSize;
    if (bufferSize_ > 0) {
        ATB_SPEED_LOG_DEBUG("BufferDevice::GetBuffer bufferSize:" << bufferSize_);
        atTensor_ = CreateAtTensor(bufferSize_);
        buffer_ = atTensor_.data_ptr();
    }
}

BufferDevice::~BufferDevice() {}

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ATB_SPEED_LOG_DEBUG("BufferDevice::GetBuffer bufferSize:" << bufferSize << "<= bufferSize_:" << bufferSize_
                        << ", not new device mem.");
        return atTensor_.data_ptr();
    }
   
    if (aclrtSynchronizeDevice() != 0) {
        return nullptr;
    }

    atTensor_.reset();
    atTensor_ = CreateAtTensor(bufferSize);
    bufferSize_ = uint64_t(atTensor_.numel());
    ATB_SPEED_LOG_DEBUG("BufferDevice::GetBuffer new bufferSize:" << bufferSize);
    buffer_ = atTensor_.data_ptr();
    return atTensor_.data_ptr();
}

torch::Tensor BufferDevice::CreateAtTensor(const uint64_t bufferSize) const
{
    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;

    tensorDesc.shape.dimNum = DIM_NUM_2;
    tensorDesc.shape.dims[0] = KB_1;
    tensorDesc.shape.dims[1] = bufferSize / KB_1 + int(1);

    return Utils::CreateAtTensorFromTensorDesc(tensorDesc);
}
} // namespace atb_speed
