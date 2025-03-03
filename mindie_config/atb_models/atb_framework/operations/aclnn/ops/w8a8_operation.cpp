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
#include "operations/aclnn/core/acl_nn_operation.h"
#include "aclnnop/aclnn_quant_matmul_v4.h"
#include "w8a8_operation.h"

namespace atb_speed {
namespace common {

W8A8Operation::W8A8Operation(
    const std::string &name,
    AclNNQuantMatmulParam param) : AclNNOperation(name), param_(param) {}

W8A8Operation::~W8A8Operation()
{
    ATB_SPEED_LOG_DEBUG("W8A8Operation deconstructor");
    this->DestroyOperation();
}

atb::Status W8A8Operation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                      atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

    const auto &scaleTensorDesc = inTensorDescs.at(NUM2);
    if (scaleTensorDesc.dtype == aclDataType::ACL_FLOAT) {
        outTensorDescs.at(0).dtype = aclDataType::ACL_FLOAT16;
    } else if (scaleTensorDesc.dtype == aclDataType::ACL_BF16) {
        outTensorDescs.at(0).dtype = aclDataType::ACL_BF16;
    }

    int nDim = param_.transposeB ? DIM0 : DIM1;
    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                       << inTensorDescs.at(DIM0).shape.dims[DIM1] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM2]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM1).shape.dims[nDim];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM1]);
        ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1]);
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[nDim];
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t W8A8Operation::GetInputNum() const
{
    uint32_t inputNum = 4;
    if (param_.hasBias) {
        ++inputNum;
    }
    return inputNum;
}

uint32_t W8A8Operation::GetOutputNum() const { return NUM1; }

int W8A8Operation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputIdx = 3;
    aclTensor* perTokenScaleTensor = aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor;
    aclTensor* biasTensor = param_.hasBias ? aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    int ret = aclnnQuantMatmulV4GetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // 0: input
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // 1: weight
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // 2: scale
        nullptr,  // offset
        perTokenScaleTensor,  // per token scale
        biasTensor,  // bias
        false, // transposeX1
        param_.transposeB, // transposeX2
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " end, ret:"
                  << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int W8A8Operation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    int ret = aclnnQuantMatmulV4(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("ExecuteAclNNOp failed, ret: " << ret);
    }
    return ret;
}

} // namespace common
} // namespace atb_speed