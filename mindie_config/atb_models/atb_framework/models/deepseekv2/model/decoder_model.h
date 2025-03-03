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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_MODEL_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/deepseekv2/layer/decoder_layer.h"
#include "models/moe/model/decoder_model.h"

namespace atb_speed {
namespace deepseekV2 {
class DeepseekV2ModelParam : public atb_speed::moe::MoeModelParam {
public:
    // MLA参数
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    float softmaxScale = 0;
    // Grouped topk参数
    int numOfGroups = 1;
    float routedScalingFactor = 1;
    std::string routingMethod = "deviceLimited";
    std::string processLogits = "scaling";
    atb::SVector<int32_t> topkGroups = {};
    // ==== 并行策略参数 ====
    // 不配置并行策略时，默认张量并行
    // 1. 张量并行 tensor parallel
    int tpRank = 0;
    int tpSize = 1;
    std::string tpDomain = "";
    std::string tpRankTableFile = "";
    // 2. 数据并行 data parallel
    bool hasDP = false;
    int dpRank = 0;
    int dpSize = 1;
    std::string dpDomain = "";
    std::string dpRankTableFile = "";
    // 3. 专家并行 expert parallel
    // 当前暂不支持专家并行，MoE使用张量并行
    bool hasEP = false;
    int epRank = 0;
    int epSize = 1;
    std::string epDomain = "";
    std::string epRankTableFile = "";
    // MoE张量并行参数，不设置时使用TP参数
    bool hasES = false;
    int esRank = 0;
    int esSize = 1;
    std::string esDomain = "";
    std::string esRankTableFile = "";

    std::vector<std::vector<int>> attnLinearQuantType = {};
    std::vector<std::vector<int>> attnLinearTransposeType = {};

    void FromString(const std::string &param);
    void AddParamJsonMLA(const std::string &param);
    void AddParamJsonMoE(const std::string &param);
    void AddLogInfo();
};

class DecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit DecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    atb::Status AddWordEmbedding() override;
    atb::Status AddPositionalEmbedding() override;
    void SetLayerParam(DecoderLayerParam &layerParam, int64_t layerId);
    atb::Status AddLayer() override;
    atb::Status AddLayerHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId);
    atb::Status AddFinalNorm() override;
    atb::Status AddLmhead() override;
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    DeepseekV2ModelParam param_;
    std::map<std::string, uint32_t> inTensorMap_;
    uint32_t inTensorCount_ = 0;
    std::map<std::string, uint32_t> internalTensorMap_;
    uint32_t internalTensorCount_ = 0;
    int32_t layerId_ = 0;
};

REGISTER_MODEL(deepseekV2, DecoderModel);

}  // namespace deepseekV2
}  // namespace atb_speed
#endif