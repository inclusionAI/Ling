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

#ifndef ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed::common {
struct AclNNRepeatParam {
    std::vector<int64_t> repeatsArray;
};

class RepeatOperation : public AclNNOperation {
public:
    explicit RepeatOperation(const std::string &name, AclNNRepeatParam param);
    ~RepeatOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

private:
    AclNNRepeatParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H