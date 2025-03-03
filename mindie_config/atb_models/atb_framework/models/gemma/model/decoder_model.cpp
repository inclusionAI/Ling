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
#include <atb/types.h>
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include <atb_speed/log.h>

#include <vector>

#include "models/gemma/layer/decoder_layer.h"
#include "models/gemma/model/decoder_model.h"

namespace atb_speed {
namespace gemma {
GemmaDecoderModel::GemmaDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
}
void GemmaModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("hiddenSize")) {
        this->hiddenSize = paramJson["hiddenSize"].get<uint32_t>();
    }
}

void GemmaDecoderModel::SetLayerParam(GemmaLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.hiddenSize = this->param.hiddenSize;
}

atb::Status GemmaDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    GemmaLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    GemmaDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}
atb::Status GemmaDecoderModel::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(this->AddMuls());
    if (this->param.positionEmbeddingType == atb_speed::base::PositionEmbeddingType::ROPE) {
        CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}
atb::Status GemmaDecoderModel::AddMuls()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node mulsNode;
    float magnifyAttr = sqrt(this->param.hiddenSize);
    ATB_SPEED_LOG_INFO("magnify hidden_states with magnify factor " << magnifyAttr);
    atb::infer::ElewiseParam magnifyElewiseMulsParam;
    magnifyElewiseMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    magnifyElewiseMulsParam.mulsParam.varAttr = magnifyAttr;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(magnifyElewiseMulsParam, &op));
    mulsNode.operation.reset(op);
    uint32_t hiddenStatesIdx = atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states");
    mulsNode.inTensors = {&graph_.internalTensors.at(hiddenStatesIdx)};
    mulsNode.outTensors = {&graph_.internalTensors.at(hiddenStatesIdx)};
    graph_.nodes.push_back(mulsNode);
    return atb::NO_ERROR;
}
} // namespace gemma
} // namespace atb_speed