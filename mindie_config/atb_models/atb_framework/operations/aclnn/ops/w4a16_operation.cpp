/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include <sstream>
#include <memory>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/utils/utils.h"
#include "w4a16_operation.h"

namespace atb_speed {
namespace common {

W4A16Operation::W4A16Operation(
    const std::string &name,
    AclNNWeightQuantBatchMatmulParam param) : QuantBatchMatmulOperation(name, param), param_(param) {}

atb::Status W4A16Operation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = (i == 4) ? (i + 2) : i;  // 4, 2: bias在aclExecutor中的idx为6
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);

        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.inTensors.at(i));
        if (i == 1) {  // 1: weight
            squeezedAtbTensor.desc.dtype = ACL_INT4;
            int nDim = param_.transposeB ? DIM0 : DIM1;
            squeezedAtbTensor.desc.shape.dims[nDim] = CheckIntMulOverFlow(
                squeezedAtbTensor.desc.shape.dims[nDim], 2);  // 2: n维shape * 2
        }

        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape,
            squeezedAtbTensor.desc.shape, squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed