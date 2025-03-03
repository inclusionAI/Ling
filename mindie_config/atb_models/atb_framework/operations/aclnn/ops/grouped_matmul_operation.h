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
#ifndef ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
enum GmmQuantType : int {
    NONE = 0,
    W8A8_CHANNEL,
    W8A16_CHANNEL,
    W8A8_TOKEN
};

struct AclNNGroupedMatmulParam {
    bool transposeB = false;
    int quantType = NONE;
    bool hasBias = false;
    aclDataType outDataType = ACL_FLOAT16;
};

class GroupedMatmulOperation : public AclNNOperation {
public:
   explicit GroupedMatmulOperation(const std::string &name, AclNNGroupedMatmulParam param);
    ~GroupedMatmulOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

    int CreateW8A8(AclNNVariantPack &aclnnVariantPack);
    int CreateW8A16(AclNNVariantPack &aclnnVariantPack);
    int CreateW8A8Token(AclNNVariantPack &aclnnVariantPack);

    std::vector<aclTensor *> yTensorVector;
    std::vector<std::vector<aclTensor *>> inputVectorOfTensor;
    std::vector<aclTensor *> weightTensorVector;
    int64_t splitItem = 2;
    int64_t groupType = 0;
    int64_t groupListType = 0; // 0 : GMMActType::GMM_ACT_TYPE_NONE
    int64_t actType = 0;
    AclNNGroupedMatmulParam param_;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H