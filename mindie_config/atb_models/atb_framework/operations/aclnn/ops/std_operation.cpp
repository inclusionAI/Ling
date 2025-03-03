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
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <unistd.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_std.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "std_operation.h"

namespace atb_speed {
namespace common {


StdOperation::StdOperation(const std::string &name) : AclNNOperation(name)
{
    this->opName_ = name;
}


StdOperation::~StdOperation()
{
    if (dim != nullptr) {
        aclDestroyIntArray(dim);
        dim = nullptr;
    }
}

atb::Status StdOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "StdOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    outTensorDescs.at(0).shape.dims[dimData.at(0)] = 1;

    ATB_SPEED_LOG_DEBUG(opName_ << "StdOperation infer shape end"
                  << " format: " << inTensorDescs.at(0).format << " dimNum: " << inTensorDescs.at(0).shape.dimNum
                  << " dims: " << inTensorDescs.at(0).shape.dims[0]);
    return 0;
}

uint32_t StdOperation::GetInputNum() const
{
    return NUM1;
}


uint32_t StdOperation::GetOutputNum() const
{
    return NUM1;
}

atb::Status StdOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape,
            atbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status StdOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor atbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape,
            atbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}


int StdOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    if (dim == nullptr) {
        dim = aclCreateIntArray(dimData.data(), 1);
    }
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnStdGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,
        dim,
        0,
        true,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}


int StdOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " StdOperation start");
    int ret = aclnnStd(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " StdOperation end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed
