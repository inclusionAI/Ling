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

#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_moe_token_unpermute.h"
#include "moetoken_unpermute_operation.h"

namespace atb_speed::common {

MoeTokenUnpermuteOperation::MoeTokenUnpermuteOperation(const std::string &name) : AclNNOperation(name) {}

MoeTokenUnpermuteOperation::~MoeTokenUnpermuteOperation()
{
    ATB_SPEED_LOG_DEBUG("MoeTokenPermuteOperation deconstruct");
    this->DestroyOperation();
}

/**
    *
    * @param[in] inTensorDesc: dimNum <= 8,
    * @param[in] outTensorDesc: dimNum <= 8
    * @return atb::Status
    */
atb::Status MoeTokenUnpermuteOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTokenUnpermuteOperation infer shape start");
    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM2).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTokenUnpermuteOperation infer shape end, inTensor0:"
                << " format: " << inTensorDescs.at(DIM0).format << " dimNum: " << inTensorDescs.at(DIM0).shape.dimNum
                << " dims: " << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                << inTensorDescs.at(DIM0).shape.dims[DIM1]);
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTokenUnpermuteOperation infer shape end, inTensor1:"
                << " format: " << inTensorDescs.at(DIM1).format << " dimNum: " << inTensorDescs.at(DIM1).shape.dimNum
                << " dims: " << inTensorDescs.at(DIM1).shape.dims[DIM0]);
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTokenUnpermuteOperation infer shape end, inTensor2:"
                << " format: " << inTensorDescs.at(DIM2).format << " dimNum: " << inTensorDescs.at(DIM2).shape.dimNum
                << " dims: " << inTensorDescs.at(DIM2).shape.dims[DIM0] << ", "
                << inTensorDescs.at(DIM2).shape.dims[DIM1]);
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeTokenUnpermuteOperation infer shape end, outTensor0:"
                << " format: " << outTensorDescs.at(DIM0).format << " dimNum: " << outTensorDescs.at(DIM0).shape.dimNum
                << " dims: " << outTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                << outTensorDescs.at(DIM0).shape.dims[DIM1]);

    return atb::NO_ERROR;
}


uint32_t MoeTokenUnpermuteOperation::GetInputNum() const
{
    return NUM3;
}

uint32_t MoeTokenUnpermuteOperation::GetOutputNum() const
{
    return NUM1;
}

int MoeTokenUnpermuteOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMoeTokenUnpermuteGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,     // permutedTokens
        aclnnVariantPack.aclInTensors.at(1)->tensor,     // sortedIndices
        aclnnVariantPack.aclInTensors.at(2)->tensor,     // probsOptional
        false,                                           // paddedMode
        nullptr,                                         // restoreShape
        aclnnVariantPack.aclOutTensors.at(0)->tensor,    // out
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                    << ", ret: " << ret
                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeTokenUnpermuteOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnMoeTokenUnpermute(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                    << ", ret: " << ret);
    return ret;
}
}  // namespace atb_speed::common