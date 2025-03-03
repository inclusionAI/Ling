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
#include "grouped_matmul_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

GroupedMatmulOperation::GroupedMatmulOperation(
    const std::string &name,
    AclNNGroupedMatmulParam param) : AclNNOperation(name), param_(param) {
}

GroupedMatmulOperation::~GroupedMatmulOperation() {
}

atb::Status GroupedMatmulOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "GroupedMatmulOperation infer shape start");
    ATB_SPEED_LOG_DEBUG(opName_ << "exports" << inTensorDescs.at(DIM2).shape.dims[DIM0]);

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = param_.outDataType == ACL_BF16 || inTensorDescs.at(DIM0).dtype == ACL_BF16 ? \
        ACL_BF16 : ACL_FLOAT16;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    int nDim = param_.transposeB ? DIM1 : DIM2;
    ATB_SPEED_LOG_DEBUG(opName_ << "GroupedMatmulOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);
    ATB_SPEED_LOG_DEBUG(opName_ << "GroupedMatmulOperation infer shape origin inTensorDescs.at(DIM1).shape.dims[nDim]"
                  << inTensorDescs.at(DIM1).shape.dims[nDim]);

    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[nDim];

    ATB_SPEED_LOG_DEBUG(opName_ << "GroupedMatmulOperation infer shape end");
    return 0;
}

uint32_t GroupedMatmulOperation::GetInputNum() const
{
    uint32_t inputNum = DIM3;
    if (param_.hasBias) {
        inputNum += DIM1;
    }
    if (param_.quantType != NONE) {
        inputNum += DIM2;
    }
    return inputNum;
}

uint32_t GroupedMatmulOperation::GetOutputNum() const
{
    return DIM1;
}

atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc)
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // ND格式下，storageShape和originalShape一致
    if (atbTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        // nz格式
        storageTensorDims.dimNum = 5;  // 5: 5维
        // (group_size, k, n) => (group_size, k / 16, n / 16, 16, 16)
        // (group_size, n, k) => (group_size, n / 16, k / 16, 16, 16)
        storageTensorDims.dims[0] = atbTensorDesc.shape.dims[0];
        storageTensorDims.dims[1] = 1 + ((atbTensorDesc.shape.dims[1] - 1) / 16);  // 1, 16：1: 维度, 16: padding大小
        storageTensorDims.dims[2] = 1 + ((atbTensorDesc.shape.dims[2] - 1) / 16);  // 2, 16：1: 维度, 16: padding大小
        storageTensorDims.dims[3] = 16;  // 3, 16：NZ格式要求
        storageTensorDims.dims[4] = 16;  // 4, 16：NZ格式要求
    }
    return storageTensorDims;
}

atb::Status GroupedMatmulOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    inputVectorOfTensor.resize(GetInputNum());
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    uint32_t inTensorCount = aclnnVariantPack.aclInTensors.size();
    for (size_t i = 0; i < inTensorCount; i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        if (i == inTensorCount - 1) {
            aclnnTensor->tensorIdx = 7; // 7 : for the last tensor
        } else {
            aclnnTensor->tensorListidx = i;
            aclnnTensor->tensorIdx = 0;
        }
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.inTensors.at(i);

        // StorageShape
        atb::Dims storageTensorDims = GetWeightStorageShape(squeezedAtbTensor.desc);

        // ViewShape and Stride
        atb::Dims viewDims = squeezedAtbTensor.desc.shape;
        if (squeezedAtbTensor.desc.shape.dimNum >= 3 && this->param_.transposeB) {  // 3: 维度
            aclnnTensor->strides = GetTransposeTensorStride(viewDims);
            viewDims.dims[0] = squeezedAtbTensor.desc.shape.dims[0];
            viewDims.dims[1] = squeezedAtbTensor.desc.shape.dims[2];  // 1, 2: 后两维转置
            viewDims.dims[2] = squeezedAtbTensor.desc.shape.dims[1];  // 1, 2: 后两维转置
        } else {
            aclnnTensor->strides = GetCopyTensorStride(viewDims);
        }

        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(viewDims, storageTensorDims, squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }

    aclnnVariantPack.aclInTensorList.clear();

    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size() - 1; i++) {
        inputVectorOfTensor.at(i).clear();
        inputVectorOfTensor.at(i).push_back(aclnnVariantPack.aclInTensors.at(i)->tensor);
        aclnnVariantPack.aclInTensorList.push_back(aclCreateTensorList(
            inputVectorOfTensor.at(i).data(), inputVectorOfTensor.at(i).size()));
    }
    return atb::NO_ERROR;
}

atb::Status GroupedMatmulOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorListidx = i;
        aclnnTensor->tensorIdx = 0;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
            squeezedAtbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    yTensorVector.clear();
    yTensorVector.push_back(aclnnVariantPack.aclOutTensors.at(DIM0)->tensor);
    aclnnVariantPack.aclOutTensorList.clear();
    aclnnVariantPack.aclOutTensorList.push_back(aclCreateTensorList(yTensorVector.data(), yTensorVector.size()));
    return atb::NO_ERROR;
}

int GroupedMatmulOperation::CreateW8A8(AclNNVariantPack &aclnnVariantPack)
{
    int ret = aclnnGroupedMatmulV4GetWorkspaceSize(aclnnVariantPack.aclInTensorList.at(DIM0),
        aclnnVariantPack.aclInTensorList.at(DIM1),
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM2) : nullptr,
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM3) :
                            aclnnVariantPack.aclInTensorList.at(DIM2),
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(4) :  // 4 : index of input tensor
                            aclnnVariantPack.aclInTensorList.at(DIM3),
        nullptr, nullptr, nullptr,
        param_.hasBias ? aclnnVariantPack.aclInTensors.at(5)->tensor :  // 5 : index of input tensor
                                aclnnVariantPack.aclInTensors.at(4)->tensor,  // 4 : index of input tensor
        nullptr, nullptr, nullptr,
        splitItem, groupType, groupListType, actType,
        aclnnVariantPack.aclOutTensorList.at(DIM0),
        nullptr, nullptr,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    return ret;
}

int GroupedMatmulOperation::CreateW8A16(AclNNVariantPack &aclnnVariantPack)
{
    int ret = aclnnGroupedMatmulV4GetWorkspaceSize(aclnnVariantPack.aclInTensorList.at(DIM0),
        aclnnVariantPack.aclInTensorList.at(DIM1),
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM2) : nullptr,
        nullptr, nullptr,
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM3) :
                            aclnnVariantPack.aclInTensorList.at(DIM2),
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(4) :  // 4 : index of input tensor
                            aclnnVariantPack.aclInTensorList.at(DIM3),
        nullptr,
        param_.hasBias ? aclnnVariantPack.aclInTensors.at(5)->tensor :  // 5 : index of input tensor
                                aclnnVariantPack.aclInTensors.at(4)->tensor,  // 4 : index of input tensor
        nullptr, nullptr, nullptr,
        splitItem, groupType, groupListType, actType,
        aclnnVariantPack.aclOutTensorList.at(DIM0),
        nullptr, nullptr,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    return ret;
}

int GroupedMatmulOperation::CreateW8A8Token(AclNNVariantPack &aclnnVariantPack)
{
    int ret = aclnnGroupedMatmulV4GetWorkspaceSize(aclnnVariantPack.aclInTensorList.at(DIM0),
        aclnnVariantPack.aclInTensorList.at(DIM1),
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM2) : nullptr,
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM3) :
                            aclnnVariantPack.aclInTensorList.at(DIM2),
        nullptr, nullptr, nullptr,
        param_.hasBias ? aclnnVariantPack.aclInTensorList.at(4) :  // 5 : index of input tensor
                            aclnnVariantPack.aclInTensorList.at(3),  // 4 : index of input tensor
        param_.hasBias ? aclnnVariantPack.aclInTensors.at(5)->tensor :  // 6 : index of input tensor
                                aclnnVariantPack.aclInTensors.at(4)->tensor,  // 5 : index of input tensor
        nullptr, nullptr, nullptr,
        splitItem, groupType, groupListType, actType,
        aclnnVariantPack.aclOutTensorList.at(DIM0),
        nullptr, nullptr,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    return ret;
}

int GroupedMatmulOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = 0;
    if (param_.quantType == GmmQuantType::NONE) {
        ret = aclnnGroupedMatmulV4GetWorkspaceSize(aclnnVariantPack.aclInTensorList.at(DIM0),
            aclnnVariantPack.aclInTensorList.at(DIM1),
            param_.hasBias ? aclnnVariantPack.aclInTensorList.at(DIM2) : nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            param_.hasBias ? aclnnVariantPack.aclInTensors.at(DIM3)->tensor :
                                aclnnVariantPack.aclInTensors.at(DIM2)->tensor,
            nullptr, nullptr, nullptr,
            splitItem, groupType, groupListType, actType,
            aclnnVariantPack.aclOutTensorList.at(DIM0),
            nullptr, nullptr,
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
    } else if (param_.quantType == GmmQuantType::W8A8_CHANNEL) {
        ret = CreateW8A8(aclnnVariantPack);
    } else if (param_.quantType == GmmQuantType::W8A16_CHANNEL) {
        ret = CreateW8A16(aclnnVariantPack);
    } else {
        ret = CreateW8A8Token(aclnnVariantPack);
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int GroupedMatmulOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnGroupedMatmul start");
    int ret = aclnnGroupedMatmulV4(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnGroupedMatmul end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed