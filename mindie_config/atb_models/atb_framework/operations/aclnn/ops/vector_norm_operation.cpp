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
#include "aclnnop/aclnn_linalg_vector_norm.h"
#include "vector_norm_operation.h"


namespace atb_speed::common {

    VectorNormOperation::VectorNormOperation(
        const std::string &name,
        atb_speed::common::AclNNVectorNormParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    VectorNormOperation::~VectorNormOperation()
    {
        ATB_SPEED_LOG_DEBUG("VectorNormOperation deconstruct");
        if (dims != nullptr) {
            aclDestroyIntArray(dims);
        }
        if (param_.ord != nullptr) {
            aclDestroyScalar(param_.ord);
        }

        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: dimNum = 3, [batch_size, seq_len, hidden_size]
     * @param[in] outTensorDesc: dimNum = 3, [batch_size, seq_len, hidden_size]
     * @return atb::Status
     */
    atb::Status VectorNormOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape start");
        outTensorDesc.at(0).format = inTensorDesc.at(0).format;
        outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
        outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;

        if (inTensorDesc.at(0).shape.dimNum == DIM3) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " input shape: [input0] "
                          << inTensorDesc.at(0).shape.dims[DIM0] << ", "
                          << inTensorDesc.at(0).shape.dims[DIM1] << ", "
                          << inTensorDesc.at(0).shape.dims[DIM2]);
            outTensorDesc.at(0).shape.dims[DIM0] = inTensorDesc.at(0).shape.dims[DIM0];
            outTensorDesc.at(0).shape.dims[DIM1] = inTensorDesc.at(0).shape.dims[DIM1];
            outTensorDesc.at(0).shape.dims[DIM2] = inTensorDesc.at(0).shape.dims[DIM2];
        } else if (inTensorDesc.at(0).shape.dimNum == DIM2) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " input shape: [input0] "
                          << inTensorDesc.at(0).shape.dims[DIM0] << ", "
                          << inTensorDesc.at(0).shape.dims[DIM1]);
            outTensorDesc.at(0).shape.dims[DIM0] = inTensorDesc.at(0).shape.dims[DIM0];
            outTensorDesc.at(0).shape.dims[DIM1] = 1;
        }  else {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid dimNum = " << inTensorDesc.at(0).shape.dimNum);
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");
        return atb::NO_ERROR;
    }

    uint32_t VectorNormOperation::GetInputNum() const
    {
        return NUM1;  // inputTensorNum = 1
    }

    uint32_t VectorNormOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status VectorNormOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor start");
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " Create aclInTensor end");

        aclnnVariantPack.aclOutTensors.resize(variantPack.outTensors.size());
        for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
            aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " Create aclOutTensor end");
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor end");
        return 0;
    }

    atb::Status VectorNormOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
            ATB_SPEED_LOG_DEBUG(opName_ << " aclnnTensor = " << aclnnTensor);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    atb::Status VectorNormOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclNnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclNnVariantPack.aclOutTensors.resize(GetOutputNum());
        for (size_t i = 0; i < aclNnVariantPack.aclOutTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " outTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclNnVariantPack.aclOutTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    std::shared_ptr<AclNNTensor> VectorNormOperation::CreateTensor(atb::Tensor atbTensor, size_t tensorIdx)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor start");
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = atbTensor;
        ATB_SPEED_LOG_DEBUG(opName_ << "  atbTensor.shape0 = " << atbTensor.desc.shape.dims[0]);
        ATB_SPEED_LOG_DEBUG(opName_ << "  atbTensor.shape1 = " << atbTensor.desc.shape.dims[1]);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(atbTensor);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.shape0 = " << squeezedAtbTensor.desc.shape.dims[0]);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.shape1 = " << squeezedAtbTensor.desc.shape.dims[1]);
        ATB_SPEED_LOG_DEBUG(opName_ << " tensor dtype: " << squeezedAtbTensor.desc.dtype);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.desc.shape.dims0 = " <<
            squeezedAtbTensor.desc.shape.dims[0]);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.desc.shape.dims1 = " <<
            squeezedAtbTensor.desc.shape.dims[1]);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.desc.shape.dimNum = " <<
            squeezedAtbTensor.desc.shape.dimNum);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.desc.dtype = " <<
            squeezedAtbTensor.desc.dtype);
        ATB_SPEED_LOG_DEBUG(opName_ << "  squeezedAtbTensor.deviceData = " <<
            squeezedAtbTensor.deviceData);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype,
            aclnnTensor->strides.data(),
            0,
            squeezedAtbTensor.desc.format,
            squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.deviceData);
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor end");
        return aclnnTensor;
    }

    int VectorNormOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        float ord = 1.0;
        param_.ord = aclCreateScalar(&ord, aclDataType::ACL_FLOAT);
        std::vector<int64_t> dimData = { -1 };
        if (dims == nullptr) {
            dims = aclCreateIntArray(dimData.data(), 1);
        }

        int ret = aclnnLinalgVectorNormGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,
            param_.ord,
            dims,
            true,
            aclDataType::ACL_FLOAT16,
            aclnnVariantPack.aclOutTensors.at(0)->tensor,
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                      << ", ret: " << ret
                      << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                      << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
        return ret;
    }

    int VectorNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnLinalgVectorNorm(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                      << ", ret: " << ret);
        return ret;
    }

}  // namespace atb_speed::common
