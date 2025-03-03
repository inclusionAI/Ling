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

#ifndef ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {
    struct AclNNLayerNormParam {
        float layerNormEps = 0;
        int64_t layerNormImplMode = 0;  // 精度模式，高精度模式(0), 高性能模式(1), 保持FLOAT16计算模式(2)
    };

    class LayerNormOperation : public AclNNOperation {
    public:
        explicit LayerNormOperation(const std::string &name, AclNNLayerNormParam param);
        ~LayerNormOperation() override;
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDesc,
            atb::SVector<atb::TensorDesc> &outTensorDesc
        ) const override;
        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        int SetAclNNWorkspaceExecutor() override;
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
        atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
        atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
        virtual std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, size_t tensorIdx);

    private:
        AclNNLayerNormParam param_;
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H
