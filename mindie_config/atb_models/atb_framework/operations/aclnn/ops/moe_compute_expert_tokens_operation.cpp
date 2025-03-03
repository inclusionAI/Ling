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
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_compute_expert_tokens.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"
#include "moe_compute_expert_tokens_operation.h"

namespace atb_speed {
namespace common {

MoeComputeExpertTokensOperation::MoeComputeExpertTokensOperation(
    const std::string &name, MoeComputeExpertTokensParam param) : AclNNOperation(name), param_(param) {}
MoeComputeExpertTokensOperation::~MoeComputeExpertTokensOperation() {}

atb::Status MoeComputeExpertTokensOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeComputeExpertTokensOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = DIM1;

    ATB_SPEED_LOG_DEBUG(opName_
                  << "MoeComputeExpertTokensOperation infer shape origin "
                  << "inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);
    outTensorDescs.at(DIM0).shape.dims[DIM0] = param_.expertNum;

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeComputeExpertTokensOperation infer shape end");
    return 0;
}
uint32_t MoeComputeExpertTokensOperation::GetInputNum() const
{
    return DIM1;
}

uint32_t MoeComputeExpertTokensOperation::GetOutputNum() const
{
    return DIM1;
}

int MoeComputeExpertTokensOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeComputeExpertTokensOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMoeComputeExpertTokensGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        param_.expertNum,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeComputeExpertTokensOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeComputeExpertTokensOperation start");

    int ret = aclnnMoeComputeExpertTokens(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeComputeExpertTokensOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed