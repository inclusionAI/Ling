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
#include <set>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/moe/model/decoder_model.h"

#include <atb/types.h>

namespace atb_speed {
namespace moe {

constexpr size_t MOE_LINEAR_TYPE_LENGTH = 4;

void MoeModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("numOfExperts")) {
        this->numOfExperts = paramJson["numOfExperts"].get<int>();
    }
    if (paramJson.contains("expertParallelDegree")) {
        this->expertParallelDegree = paramJson["expertParallelDegree"].get<int>();
    }
    if (paramJson.contains("routingMethod")) {
        this->routingMethod = paramJson["routingMethod"].get<std::string>();
    }
    if (paramJson.contains("processLogits")) {
        this->processLogits = paramJson["processLogits"].get<std::string>();
    }
    if (paramJson.contains("normHasBias")) {
        this->normHasBias = paramJson["normHasBias"].get<bool>();
    }
    if (paramJson.contains("enableFusedRouting")) {
        this->enableFusedRouting = paramJson["enableFusedRouting"].get<bool>();
    }
    if (paramJson.contains("firstKDenseReplace")) {
        this->firstKDenseReplace = atb_speed::base::VerifyParam<int>(paramJson, "firstKDenseReplace");
    }
    if (paramJson.contains("numOfSharedExperts")) {
        this->numOfSharedExperts = atb_speed::base::VerifyParam<int>(paramJson, "numOfSharedExperts");
    }
    if (paramJson.contains("hasSharedExpert")) {
        this->hasSharedExpert = atb_speed::base::VerifyParam<bool>(paramJson, "hasSharedExpert");
    }
    if (paramJson.contains("hasSharedExpertGate")) {
        this->hasSharedExpertGate = atb_speed::base::VerifyParam<bool>(paramJson, "hasSharedExpertGate");
    }
    if (paramJson.contains("maskStartIdx")) {
        maskStartIdx = atb_speed::base::VerifyParam<int>(paramJson, "maskStartIdx");
    }
    for (auto item : paramJson["numOfSelectedExperts"]) {
        this->numOfSelectedExperts.push_back(item.get<int>());
    }
    for (auto item : paramJson["moeLinearQuantType"]) {
        this->moeLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["mlpLinearQuantType"]) {
        this->mlpLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["moeLinearTransposeType"]) {
        this->moeLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["mlpLinearTransposeType"]) {
        this->mlpLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
}

void MoeModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG(", numOfExperts: " << this->numOfExperts
                  << ", expertParallelDegree: " << this->expertParallelDegree
                  << ", numOfSelectedExperts:" << this->numOfSelectedExperts
                  << ", routingMethod: " << this->routingMethod
                  << ", processLogits" << this->processLogits
                  << ", normHasBias: " << this->normHasBias
                  << ", enableFusedRouting: " << this->enableFusedRouting
                  << ", moeLinearQuantType: " << this->moeLinearQuantType
                  << ", mlpLinearQuantType: " << this->mlpLinearQuantType
                  << ", moeLinearTransposeType: " << this->moeLinearTransposeType
                  << ", mlpLinearTransposeType: " << this->mlpLinearTransposeType);
}

void MoeModelParam::CheckRoutingMethodValid()
{
    std::set<std::string> supportRoutingMethods = {"softMaxTopK", "integratedSoftmaxTopK", "deviceLimited"};
    if (supportRoutingMethods.find(this->routingMethod) == supportRoutingMethods.end()) {
        std::stringstream ss;
        ss << "The routing method " << this->routingMethod << " is not valid." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

void MoeModelParam::CheckProcessLogitsValid()
{
    std::set<std::string> supportProcessLogits = {"none", "normalization", "scaling"};
    if (supportProcessLogits.find(this->processLogits) == supportProcessLogits.end()) {
        std::stringstream ss;
        ss << "The process logits method" << this->processLogits << " is not valid." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

void MoeModelParam::CheckParam()
{
    CheckLinearParamsSufficient(this->moeLinearQuantType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->mlpLinearQuantType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->moeLinearTransposeType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->mlpLinearTransposeType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckRoutingMethodValid();
}

MoeDecoderModel::MoeDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["default_moe"] = {
        "expert_array_model", "expert_group_model", "one_hot_model", "zero_hot_model"};
    this->inTensorCandidates["fused_routing"] = {
        "in_final_hidden_state", "in_final_hidden_state_two", "in_final_bias"};
}

void MoeDecoderModel::ConstructInTensorMap()
{
    DecoderModel::ConstructInTensorMap();
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default_moe", this->inTensorMap);
}

atb::Status MoeDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    MoeLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    if (this->param.normType == atb_speed::base::RMS_NORM) {
        MoeDecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    } else {
        MoeDecoderLayer<atb::infer::LayerNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    }
    return atb::NO_ERROR;
}

void MoeDecoderModel::SetLayerParam(MoeLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.numOfExperts = this->param.numOfExperts;
    layerParam.expertParallelDegree = this->param.expertParallelDegree;
    layerParam.routingMethod = this->param.routingMethod;
    layerParam.numOfSelectedExperts = this->param.numOfSelectedExperts;
    layerParam.normHasBias = this->param.normHasBias;
    layerParam.enableFusedRouting = this->param.enableFusedRouting;
    layerParam.processLogits = this->param.processLogits;
    layerParam.moeLinearQuantType = this->param.moeLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = this->param.mlpLinearQuantType[layerId];
    layerParam.moeLinearTransposeType = this->param.moeLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = this->param.mlpLinearTransposeType[layerId];
}

void MoeDecoderModel::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    DecoderModel::SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "zero_hot_model"));
}

} // namespace moe
} // namespace atb_speed