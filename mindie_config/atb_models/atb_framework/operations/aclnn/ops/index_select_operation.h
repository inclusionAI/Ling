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

#ifndef ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


namespace atb_speed::common {

struct IndexSelectParam {
    int64_t dim = 0;  // select的dim
};

class IndexSelectOperation : public AclNNOperation {
public:
    explicit IndexSelectOperation(const std::string &name, IndexSelectParam param);
    ~IndexSelectOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    IndexSelectParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H
