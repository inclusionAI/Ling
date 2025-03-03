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
#include "atb_speed/utils/tensor_util.h"
#include <sstream>
#include <fstream>
#include <string>
#include <acl/acl.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"

namespace atb_speed {
std::string TensorUtil::TensorToString(const atb::Tensor &tensor)
{
    std::stringstream ss;
    ss << TensorDescToString(tensor.desc) << ", deviceData:" << tensor.deviceData << ", hostData:" << tensor.hostData
       << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

std::string TensorUtil::TensorDescToString(const atb::TensorDesc &tensorDesc)
{
    std::stringstream ss;
    ss << "dtype: " << tensorDesc.dtype << ", format: " << tensorDesc.format << ", shape:[";
    for (size_t i = 0; i < tensorDesc.shape.dimNum; ++i) {
        if (i == 0) {
            ss << tensorDesc.shape.dims[i];
        } else {
            ss << ", " << tensorDesc.shape.dims[i];
        }
    }
    ss << "]";

    return ss.str();
}

uint64_t TensorUtil::GetTensorNumel(const atb::Tensor &tensor) { return GetTensorNumel(tensor.desc); }

uint64_t TensorUtil::GetTensorNumel(const atb::TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == 0) {
        return 0;
    }

    int64_t elementCount = 1;
    for (size_t i = 0; i < tensorDesc.shape.dimNum; i++) {
        elementCount = CheckIntMulOverFlow(elementCount, tensorDesc.shape.dims[i]);
    }

    return elementCount;
}

bool TensorUtil::TensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB)
{
    if (tensorDescA.dtype == tensorDescB.dtype && tensorDescA.format == tensorDescB.format &&
        tensorDescA.shape.dimNum == tensorDescB.shape.dimNum) {
        for (size_t i = 0; i < tensorDescA.shape.dimNum; i++) {
            if (tensorDescA.shape.dims[i] != tensorDescB.shape.dims[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}
} // namespace atb_speed