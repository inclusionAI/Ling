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
#include "aclnnop/aclnn_index_select.h"
#include "index_select_operation.h"


namespace atb_speed::common {
    IndexSelectOperation::IndexSelectOperation(
        const std::string &name,
        atb_speed::common::IndexSelectParam param
    ) : AclNNOperation(name), param_(param)
    {
        ATB_SPEED_LOG_DEBUG("IndexSelectOperation construct");
        this->opName_ = name;
    }

    IndexSelectOperation::~IndexSelectOperation()
    {
        ATB_SPEED_LOG_DEBUG("IndexSelectOperation deconstruct");
        this->DestroyOperation();
    }

    uint32_t IndexSelectOperation::GetInputNum() const
    {
        return NUM2;  // inputTensorNum = 2
    }

    uint32_t IndexSelectOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    /**
     *
     * @param[in] inTensorDescs: [self, indices]
     * @param[in] outTensorDescs: out
     * @return atb::Status
     */
    atb::Status IndexSelectOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape start");
        outTensorDescs.at(0) = inTensorDescs.at(0);

        if (inTensorDescs.at(0).shape.dimNum > 8) {  // 8: tensor max dim num
            ATB_SPEED_LOG_ERROR(opName_ << " [inTensorDescs[0].dimNum should <= 8] CHECK inTensorDescs[0] = "
                            << inTensorDescs.at(0).shape.dimNum);
        }

        int64_t selfDimNum = static_cast<int64_t>(inTensorDescs.at(0).shape.dimNum);
        if ((param_.dim >= selfDimNum) || (param_.dim < -selfDimNum)) {
            ATB_SPEED_LOG_ERROR(opName_ << " [param.dim should in [-inTensorDescs[0].dimNum, inTensorDescs[0].dimNum)) "
                            << "CHECK param.dim = " << param_.dim << " inTensorDescs[0].dimNum = " << selfDimNum);
        }

        if (inTensorDescs.at(1).shape.dimNum != DIM1) {
            ATB_SPEED_LOG_ERROR(opName_ << " [inTensorDescs[1].dimNum should == 1] CHECK inTensorDescs[1] = "
                            << inTensorDescs.at(0).shape.dimNum);
        }

        outTensorDescs.at(0).shape.dims[param_.dim] = inTensorDescs.at(1).shape.dims[DIM0];

        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");
        return atb::NO_ERROR;
    }

    int IndexSelectOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        int ret = aclnnIndexSelectGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,     // self
            param_.dim,                                      // dim
            aclnnVariantPack.aclInTensors.at(1)->tensor,     // index
            aclnnVariantPack.aclOutTensors.at(0)->tensor,    // out
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                      << ", ret: " << ret
                      << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                      << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
        return ret;
    }

    int IndexSelectOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnIndexSelect(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                      << ", ret: " << ret);
        return ret;
    }

}  // namespace atb_speed::common
