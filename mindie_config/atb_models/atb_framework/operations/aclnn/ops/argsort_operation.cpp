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
#include "argsort_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_argsort.h"
#include "operations/aclnn/utils/utils.h"
namespace atb_speed {
namespace common {

ArgSortOperation::ArgSortOperation(const std::string &name) : AclNNOperation(name) {
}

ArgSortOperation::~ArgSortOperation() {
}

// 输入输出都只有一个，并且shape,format和dimnum都一致
atb::Status ArgSortOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "ArgSortOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = ACL_INT64;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    ATB_SPEED_LOG_DEBUG(opName_ << "ArgSortOperation infer shape end"
                  << " format: " << inTensorDescs.at(0).format << " dimNum: " << inTensorDescs.at(0).shape.dimNum
                  << " dims: " << inTensorDescs.at(0).shape.dims[0]);
    return 0;
}

uint32_t ArgSortOperation::GetInputNum() const
{
    return NUM1;
}

uint32_t ArgSortOperation::GetOutputNum() const
{
    return NUM1;
}

atb::Status ArgSortOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.inTensors.at(i));

        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
            squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status ArgSortOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.outTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
            squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int ArgSortOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnArgsortGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,
        0,
        false,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int ArgSortOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnArgsort(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
}  // namespace common
}  // namespace atb_speed