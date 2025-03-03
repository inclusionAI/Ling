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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_LAYER_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseekV2 {
class DecoderLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    bool enableFusedRouting = true;
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = false;
    int maskStartIdx = 0;
    int layerId = 0;
    int firstKDenseReplace = 1;
    int numOfSharedExperts = 2;       // 2:Defaulting the number of shared experts to 2
    int rank = 0;
    int worldSize = 1;
    // quant 参数
    int mlpNormQuantType = atb::infer::QUANT_UNDEFINED;
    bool isAntiOutlier = false;
    // Grouped topk参数
    int numOfGroups = 1;
    float routedScalingFactor = 1;
    // MLA参数
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    float softmaxScale = 0;
    std::string routingMethod = "deviceLimited";
    std::string processLogits = "scaling";
    std::string backend = "hccl";
    std::string rankTableFile = "";
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

    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    atb::SVector<int32_t> topkGroups = {1}; // num of selected groups
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

class DecoderLayer {
public:
    explicit DecoderLayer();
    ~DecoderLayer();

private:
    int32_t layerId_ = 0;
};

}  // namespace deepseekV2
}  // namespace atb_speed
#endif
