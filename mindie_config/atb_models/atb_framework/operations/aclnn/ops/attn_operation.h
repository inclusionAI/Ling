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
#ifndef ATTN_OPERATION_H
#define ATTN_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"
#include "cstring"
namespace atb_speed {
namespace common {
struct AclNNAttnParam {
    bool hasMask = false;
    bool isFA = false;
    bool isPrefill = false;
    bool hasKVQuant = false;
    bool hasQuantOffset = false;
    int64_t headNum = 0;
    int64_t kvHeadNum = 0;
    int64_t headDim = 0;
    int64_t innerPrecise = 1; // 代表高精度/高性能选择，默认值为1（高性能）
    int64_t blockSize = 128; // page attention中KV存储每个block中最大的token个数
};

class AttnOperation : public AclNNOperation {
public:
    explicit AttnOperation(const std::string& name, AclNNAttnParam param);
    ~AttnOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
private:
    int ProcessSeqLengthTensor(atb::Tensor& tensor);
private:
    aclTensor* tensorsOfValue[1]{nullptr};
    aclTensor* tensorsOfKey[1]{nullptr};
    aclIntArray *actualSeqLengths = nullptr;
    AclNNAttnParam param_;
    std::string opName_;
    static std::vector<int> seqLencache_;
    static std::vector<int64_t> seqLen_;
};
} // namespace common
} // namespace atb_speed
#endif