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

#include "aclnnop/aclnn_index_put_impl.h"

#include "operations/aclnn/utils/utils.h"
#include "indexput_operation.h"

namespace atb_speed {
namespace common {
IndexputOperation::IndexputOperation(const std::string &name, AclNNIndexputParam param)
    : AclNNOperation(name), param_(param)
{
    ATB_SPEED_LOG_DEBUG("IndexputOperation, param: " << param_.ToString());
}

IndexputOperation::~IndexputOperation()
{
    ATB_SPEED_LOG_DEBUG("~IndexputOperation");
    this->DestroyOperation();
}

uint32_t IndexputOperation::GetInputNum() const { return NUM3; }

uint32_t IndexputOperation::GetOutputNum() const { return NUM1; }

atb::Status IndexputOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                          atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0) = inTensorDescs.at(0);
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return atb::NO_ERROR;
}

atb::Status IndexputOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputNum = this->GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);
    for (uint32_t i = 0; i < inputNum; ++i) {
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), static_cast<int>(i));
        if (i == 1) {
            aclnnVariantPack.aclInTensors.at(i)->tensorListidx = 0;
            aclnnVariantPack.aclInTensors.at(i)->tensorIdx = 0;
        }
    }

    vectorList.clear();
    vectorList.push_back(aclnnVariantPack.aclInTensors.at(1)->tensor);
    aclnnVariantPack.aclInTensorList.clear();
    aclnnVariantPack.aclInTensorList.push_back(aclCreateTensorList(vectorList.data(), vectorList.size()));

    aclnnVariantPack.aclOutTensors.clear();
    aclnnVariantPack.aclOutTensors.push_back(CreateTensor(variantPack.outTensors.at(0), 0));

    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return 0;
}

int IndexputOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int ret = aclnnIndexPutImplGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor, aclnnVariantPack.aclInTensorList.at(0),
        aclnnVariantPack.aclInTensors.at(2)->tensor, param_.accumulate, param_.unsafe,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:" << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end");
    return ret;
}

int IndexputOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret =
        aclnnIndexPutImpl(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed