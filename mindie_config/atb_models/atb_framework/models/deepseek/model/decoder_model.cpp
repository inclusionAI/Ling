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
#include "models/deepseek/layer/decoder_layer.h"
#include "models/deepseek/model/decoder_model.h"

namespace atb_speed {
namespace deepseek {

constexpr int DEEPSEEK_WEIGHT_COUNT_PER_LAYER = 68;

void DeepseekModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::moe::MoeModelParam::ParseParam(paramJson);
    if (paramJson.contains("hasSharedExpert")) {
        this->hasSharedExpert = paramJson["hasSharedExpert"].get<bool>();
    }
    if (paramJson.contains("hasSharedExpertGate")) {
        this->hasSharedExpertGate = paramJson["hasSharedExpertGate"].get<bool>();
    }
    if (paramJson.contains("numOfSharedExperts")) {
        this->numOfSharedExperts = paramJson["numOfSharedExperts"].get<int>();
    }
    if (paramJson.contains("firstKDenseReplace")) {
        this->firstKDenseReplace = paramJson["firstKDenseReplace"].get<int>();
    }
    if (paramJson.contains("numOfGroups")) {
        this->numOfGroups = paramJson["numOfGroups"].get<int>();
    }
    if (paramJson.contains("numOfSelectedGroups")) {
        for (auto item : paramJson["numOfSelectedGroups"]) {
            this->topkGroups.push_back(item.get<int>());
    }
    }
}

void DeepseekModelParam::PrintParam()
{
    atb_speed::moe::MoeModelParam::PrintParam();
    std::stringstream ss;
    ss << ", hasSharedExpert: " << this->hasSharedExpert
       << ", hasSharedExpertGate: " << this->hasSharedExpertGate
       << ", numOfSharedExperts:" << this->numOfSharedExperts
       << ", firstKDenseReplace: " << this->firstKDenseReplace
       << ", numOfGroups: " << this->numOfGroups
       << ", numOfSelectedGroups: " << this->topkGroups;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

DecoderModel::DecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->param.FromString(param);
    this->weightCountPerLayer = DEEPSEEK_WEIGHT_COUNT_PER_LAYER;
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    DeepseekLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    DeepseekDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void DecoderModel::SetLayerParam(DeepseekLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::moe::MoeDecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.hasSharedExpert = this->param.hasSharedExpert;
    layerParam.hasSharedExpertGate = this->param.hasSharedExpertGate;
    layerParam.numOfSharedExperts = this->param.numOfSharedExperts;
    layerParam.firstKDenseReplace = this->param.firstKDenseReplace;
    layerParam.numOfGroups = this->param.numOfGroups;
    layerParam.topkGroups = this->param.topkGroups;
    layerParam.layerId = layerId;
}

void DecoderModel::ConstructInTensorMap()
{
    atb_speed::moe::MoeDecoderModel::ConstructInTensorMap();
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "q_len", this->inTensorMap);
    }
}

void DecoderModel::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    atb_speed::base::DecoderModel::SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "zero_hot_model"));
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len"));
    }
}

} // namespace deepseek
} // namespace atb_speed