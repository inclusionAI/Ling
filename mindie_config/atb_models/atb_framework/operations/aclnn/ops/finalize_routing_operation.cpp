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
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"
#include "aclnnop/aclnn_moe_finalize_routing.h"
#include "finalize_routing_operation.h"

namespace atb_speed {
namespace common {

FinalizeRoutingOperation::FinalizeRoutingOperation(const std::string &name) : AclNNOperation(name) {}
FinalizeRoutingOperation::~FinalizeRoutingOperation() {}

atb::Status FinalizeRoutingOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "FinalizeRoutingOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM1).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM1).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM1).shape.dimNum;

    ATB_SPEED_LOG_DEBUG(opName_ << "FinalizeRoutingOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM1).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[DIM1];

    ATB_SPEED_LOG_DEBUG(opName_ << "FinalizeRoutingOperation infer shape end");
    return 0;
}
uint32_t FinalizeRoutingOperation::GetInputNum() const
{
    return NUM7;
}

uint32_t FinalizeRoutingOperation::GetOutputNum() const
{
    return DIM1;
}

int FinalizeRoutingOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " FinalizeRoutingOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMoeFinalizeRoutingGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM2)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM3)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM4)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM5)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM6)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int FinalizeRoutingOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " FinalizeRoutingOperation start");

    int ret = aclnnMoeFinalizeRouting(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " FinalizeRoutingOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed