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
#include "aclnnop/aclnn_repeat.h"
#include "repeat_operation.h"

namespace atb_speed::common {

RepeatOperation::RepeatOperation(
    const std::string &name,
    atb_speed::common::AclNNRepeatParam param
) : AclNNOperation(name), param_(param)
{
    this->opName_ = name;
    this->param_ = param;
}

RepeatOperation::~RepeatOperation()
{
    ATB_SPEED_LOG_DEBUG("RepeatOperation deconstruct");
    this->DestroyOperation();
}

/**
    *
    * @param[in] inTensorDesc: dimNum <= 8,
    * @param[in] outTensorDesc: dimNum <= 8
    * @return atb::Status
    */
atb::Status RepeatOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "RepeatOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i] * param_.repeatsArray[i];
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "RepeatOperation infer shape end"
                << " format: " << inTensorDescs.at(0).format << " dimNum: " << inTensorDescs.at(0).shape.dimNum
                << " dims: " << inTensorDescs.at(0).shape.dims[0]);
    return 0;
}


uint32_t RepeatOperation::GetInputNum() const
{
    return DIM1;
}

uint32_t RepeatOperation::GetOutputNum() const
{
    return DIM1;
}

int RepeatOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    // 创建normalizedShape aclIntArray
    aclIntArray *repeats =  aclCreateIntArray(param_.repeatsArray.data(), param_.repeatsArray.size());
    int ret = aclnnRepeatGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,     // input
        repeats,                                         // repeatShape
        aclnnVariantPack.aclOutTensors.at(0)->tensor,    // out
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                    << ", ret: " << ret
                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int RepeatOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnRepeat(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                    << ", ret: " << ret);
    return ret;
}
}  // namespace atb_speed::common