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
#include "models/starcoder/model/decoder_model.h"
#include "models/starcoder/layer/decoder_layer.h"
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include <atb/types.h>

namespace atb_speed {
namespace starcoder {

DecoderModel::DecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->weightCountWordEmbedding = 2; // 2: word embedding weight, postion embedding weight
    this->weightCountFinalNorm = 2; // 2: final norm weight, final norm bias
    this->internalTensorCandidates = {
        {"default", {"hidden_states", "positional_embedding"}}
    };
}

atb::Status DecoderModel::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(this->AddEmbeddingAdd());
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node positionalEmbeddingNode;
    atb_speed::common::WordEmbeddingParam positionalEmbeddingParam;
    this->SetWordEmbeddingParam(positionalEmbeddingParam);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(positionalEmbeddingParam, &op));
    positionalEmbeddingNode.operation.reset(op);
    positionalEmbeddingNode.inTensors = {
        &graph_.weightTensors.at(1),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids"))
    };
    positionalEmbeddingNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "positional_embedding"))
    };
    graph_.nodes.push_back(positionalEmbeddingNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddEmbeddingAdd()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node embeddingAddNode;
    atb::infer::ElewiseParam embeddingAddParam;
    embeddingAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(embeddingAddParam, &op));
    embeddingAddNode.operation.reset(op);
    embeddingAddNode.inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "positional_embedding"))
    };
    embeddingAddNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"))
    };
    graph_.nodes.push_back(embeddingAddNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    atb_speed::starcoder::DecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace starcoder
} // namespace atb_speed
