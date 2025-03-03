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
#ifndef ATB_SPEED_MODELS_QWEN_MOE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN_MOE_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/qwen/layer/moe_decoder_layer.h"

namespace atb_speed {
namespace qwen {
class MoeDecoderModel : public Model {
public:
    struct Param {
        bool isFA = false;
        bool isPrefill = false;
        bool isBF16 = false;
        bool isEmbeddingParallel = false;
        bool isLmHeadParallel = true;
        bool supportSwiGLU = false;
        bool supportLcoc = false;

        int lmHeadTransposeType = -1;
        float rmsNormEps = 0;
        int numAttentionHeadsPerRank = 0;
        int hiddenSizePerAttentionHead = 0;
        int numHiddenLayers = 0;
        int numKeyValueHeadsPerRank = 0;
        int rank = 0;
        int worldSize = 1;
        int numOfExperts = 64;            // qwen1.5 equal to 60, qwen2 equal to 64
        int expertParallelDegree = 1;
        int maskStartIdx = 0;
        std::string routingMethod = "softMaxTopK";
        std::string backend = "hccl";
        std::string rankTableFile = "";
        std::vector<int> tokenOffset = {};
        std::vector<int> seqLen = {};
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> attnLinearQuantType = {};
        std::vector<std::vector<int>> mlpLinearQuantType = {};
        std::vector<std::vector<int>> moeLinearQuantType = {};
        std::vector<std::vector<int>> attnLinearTransposeType = {};
        std::vector<std::vector<int>> mlpLinearTransposeType = {};
        std::vector<std::vector<int>> moeLinearTransposeType = {};
        atb::SVector<int32_t> numOfSelectedExperts = {}; // num of selected experts, default for qwen2
        void AddParamJson(const std::string &param);
        void AddParamJsonMoE(const std::string &param);
        void AddLogInfo();
        void FromString(const std::string &param);
        void ParseBasicParams(const nlohmann::json &paramJson);
    };

    explicit MoeDecoderModel(const std::string &param);
    ~MoeDecoderModel() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddPositionalEmbedding();
    atb::Status AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmhead();
    atb::Status SetLayerParam(atb_speed::qwen::MoeDecoderLayerParam &layerParam, const int layerId);
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    Param param_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};

REGISTER_MODEL(qwen, MoeDecoderModel);
}  // namespace qwen
}  // namespace atb_speed
#endif
