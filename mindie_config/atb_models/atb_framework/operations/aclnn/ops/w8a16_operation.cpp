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
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"
#include "w8a16_operation.h"

namespace atb_speed {
namespace common {

W8A16Operation::W8A16Operation(
    const std::string &name,
    AclNNWeightQuantBatchMatmulParam param) : QuantBatchMatmulOperation(name, param), param_(param) {}

atb::Status W8A16Operation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = (i == 4) ? (i + 2) : i;  // 4, 2: bias在aclExecutor中的idx为6
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.inTensors.at(i));
        if ((i == 1) || (i == 2) || (i == 3)) {  // 1, 2, 3: weight, weight_scale, weight_offset
            if (squeezedAtbTensor.desc.shape.dimNum != NUM2) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " weight tensor dimNum after combine batch size "
                               << "and seq len axis should be 2, but got " << squeezedAtbTensor.desc.shape.dimNum);
                return atb::ERROR_INTERNAL_ERROR;
            }
            // StorageShape
            atb::Dims storageDims = squeezedAtbTensor.desc.shape;
            if (i == 1) {  // weight的storageShape会根据NZ和ND格式而有所不同
                storageDims = GetWeightStorageShape(squeezedAtbTensor.desc);
            }
            // ViewShape and Stride
            atb::Dims viewDims = squeezedAtbTensor.desc.shape;
            if (Is910B() && this->param_.transposeB) {
                aclnnTensor->strides = GetTransposeTensorStride(viewDims);
                viewDims.dims[0] = squeezedAtbTensor.desc.shape.dims[1];
                viewDims.dims[1] = squeezedAtbTensor.desc.shape.dims[0];
            } else {
                aclnnTensor->strides = GetCopyTensorStride(viewDims);
            }
            CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(viewDims, storageDims, squeezedAtbTensor, aclnnTensor));
        } else {
            aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
            CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape,
                squeezedAtbTensor.desc.shape, squeezedAtbTensor, aclnnTensor));
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}
} // namespace common
} // namespace atb_speed