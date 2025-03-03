/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_PA_MODEL_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_PA_MODEL_H

#include "atb_speed/base/model.h"

namespace atb_speed {
namespace gptneox_20b {

enum InTensorId : uint32_t {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_INPUT_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_TENSOR_MAX
};

enum InternalTensorId : uint32_t {
    INTERNAL_TENSOR_ID = 0,
    INTERNAL_TENSOR_MAX
};

enum OutTensorId : uint32_t {
    OUT_TENSOR_ID = 0,
    OUT_TENSOR_MAX,
};

class PAModel : public Model {
public:
    struct Param {
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        float layerNormEps = 0;
        float rotaryPct = 0.0;
        int rank = 0;
        int rankSize = 1;
        bool isPrefill = false;
        float qScale = 1.0;
        float qkScale = 1.0;
        std::string backend = "hccl";

        void FromString(const std::string &param);
    };

    explicit PAModel(const std::string &param);

    ~PAModel() override;

    uint32_t GetInputNum() const override;

    uint32_t GetOutputNum() const override;

    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
        std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmhead();
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;

private:
    Param param_;
    std::vector<int32_t> seqLen_;
};

} // namespace gptneox_20b
} // namespace atb_speed
#endif // ATB_SPEED_MODELS_GPTNEOX_20B_PA_MODEL_H
