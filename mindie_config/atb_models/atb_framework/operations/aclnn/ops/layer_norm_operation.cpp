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
#include "aclnnop/aclnn_layer_norm.h"
#include "layer_norm_operation.h"


namespace atb_speed::common {

    LayerNormOperation::LayerNormOperation(
        const std::string &name,
        atb_speed::common::AclNNLayerNormParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    LayerNormOperation::~LayerNormOperation()
    {
        ATB_SPEED_LOG_DEBUG("LayerNormOperation deconstruct");
        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: dimNum = 3, [batch_size, seq_len, hidden_size]
     * @param[in] outTensorDesc: dimNum = 3, [batch_size, seq_len, hidden_size]
     * @return atb::Status
     */
    atb::Status LayerNormOperation::InferShape(
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
        } else {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid dimNum = " << inTensorDesc.at(0).shape.dimNum);
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");
        return atb::NO_ERROR;
    }

    uint32_t LayerNormOperation::GetInputNum() const
    {
        return NUM3;  // inputTensorNum = 3
    }

    uint32_t LayerNormOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status LayerNormOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
            if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
        }
        return atb::NO_ERROR;
    }

    atb::Status LayerNormOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

    std::shared_ptr<AclNNTensor> LayerNormOperation::CreateTensor(atb::Tensor atbTensor, size_t tensorIdx)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor start");
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = atbTensor;
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(atbTensor);
        ATB_SPEED_LOG_DEBUG(opName_ << " tensor dtype: " << squeezedAtbTensor.desc.dtype);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
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

    int LayerNormOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        int64_t normalizedShapeValue[] = { aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dims[DIM2] };
        uint64_t normalizedShapeDimNum = 1;
        aclIntArray *normalizedShape = aclCreateIntArray(normalizedShapeValue, normalizedShapeDimNum);
        int ret = aclnnLayerNormWithImplModeGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,     // input
            normalizedShape,                                 // normalizedShape
            aclnnVariantPack.aclInTensors.at(1)->tensor,     // weight
            aclnnVariantPack.aclInTensors.at(2)->tensor,     // bias
            static_cast<double>(param_.layerNormEps),        // eps
            aclnnVariantPack.aclOutTensors.at(0)->tensor,    // out
            nullptr,                                         // meanOut
            nullptr,                                         // rstdOut
            static_cast<int32_t>(param_.layerNormImplMode),  // implMode
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                      << ", ret: " << ret
                      << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                      << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
        return ret;
    }

    int LayerNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnLayerNormWithImplMode(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                      << ", ret: " << ret);
        return ret;
    }

}  // namespace atb_speed::common
