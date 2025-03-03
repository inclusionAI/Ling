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
#include "moe_topk_softmax_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_softmax.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MoeTopkSoftmaxOperation::MoeTopkSoftmaxOperation(
    const std::string &name, MoeTopkSoftmaxParam param) : AclNNOperation(name), param_(param) {}
MoeTopkSoftmaxOperation::~MoeTopkSoftmaxOperation() {}

atb::Status MoeTopkSoftmaxOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTopkSoftmaxOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    outTensorDescs.at(DIM2).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM2).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM2).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTopkSoftmaxOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = param_.topkNum;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM1).shape.dims[DIM1] = param_.topkNum;
    outTensorDescs.at(DIM2).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM2).shape.dims[DIM1] = param_.topkNum;

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTopkSoftmaxOperation infer shape end");
    return 0;
}
uint32_t MoeTopkSoftmaxOperation::GetInputNum() const
{
    return DIM1;
}

uint32_t MoeTopkSoftmaxOperation::GetOutputNum() const
{
    return DIM3;
}

int MoeTopkSoftmaxOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeTopkSoftmaxOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        nullptr,
        param_.topkNum,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM2)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeTopkSoftmaxOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeTopkSoftmaxOperation start");

    int ret = aclnnMoeGatingTopKSoftmax(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeTopkSoftmaxOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed