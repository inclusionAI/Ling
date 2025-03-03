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
#include "nlohmann/json.hpp"
#include "vector"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/qwen/layer/decoder_layer.h"
#include "models/qwen/model/gte_decoder_model.h"
namespace atb_speed {
namespace qwen {

// Weight count
const int WEIGHT_COUNT_WORD_EMBEDDING = 1;
const int WEIGHT_COUNT_POST_NORM = 1;

void GteDecoderModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    this->isEmbedding = paramJson["isEmbedding"].get<bool>();
    if (paramJson.contains("withEmbedding")) {
        this->withEmbedding = paramJson["withEmbedding"].get<bool>();
    }
    if (paramJson.contains("enableLogN")) {
        this->enableLogN = paramJson["enableLogN"].get<bool>();
    }
}

void GteDecoderModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
}

GteDecoderModel::GteDecoderModel(const std::string &param):atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
}

GteDecoderModel::~GteDecoderModel() {}

void GteDecoderModel::ConstructInTensorMap()
{
    this->inTensorMap.clear();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default", this->inTensorMap);

    // 添加并行解码特性或SplitFuse的Tensor
    if (this->param.enableSpeculate || this->param.enableSplitFuse) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "q_len", this->inTensorMap);
    }

    // 添加lora特性的Tensor
    if (this->param.enableLora) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "lora_common", this->inTensorMap);
        uint32_t currentTensorIdx = this->inTensorMap.size();
        for (uint32_t i = 0; i < this->param.numHiddenLayers; i++) {
            for (std::string loraWeightName : this->inTensorCandidates.at("lora_per_layer")) {
                this->inTensorMap["layer_" + std::to_string(i) + loraWeightName] = currentTensorIdx;
                currentTensorIdx++;
            }
        }
    }
}

void GteDecoderModel::ConstructInternalTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInternalTensorMap();
}

atb::Status GteDecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);

    return atb::NO_ERROR;
}


void GteDecoderModel::SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.enableLogN = param.enableLogN;
    layerParam.isEmbedding = param.isEmbedding;
}

void GteDecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    DecoderModel::SetLayerNodeInput(layerNode, layerId);

    if (param.enableLogN) {
        layerNode.inTensors.at(layerNode.inTensors.size() - 1) =
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    }
}

int64_t GteDecoderModel::BuildGraph()
{
    // 准备inTensor
    ConstructInTensorMap();
    this->graph_.inTensors.resize(this->inTensorMap.size());

    // 准备internalTensor
    ConstructInternalTensorMap();
    this->graph_.internalTensors.resize(this->internalTensorMap.size());
    // 准备outTensor
    graph_.outTensors.resize(1);  // 1: 模型输出1个out tensor

    // 准备weightTensor
    if (param.enableKvQuant) {
        weightCountPerLayer += 8;  // 8: kv cache int8 多8个inTensor
    }
    const uint64_t weightTensorSize =
        WEIGHT_COUNT_WORD_EMBEDDING +
        CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
        WEIGHT_COUNT_POST_NORM;
    graph_.weightTensors.resize(weightTensorSize);

    // 准备kv cache
    graph_.kCacheTensors.resize(param.numHiddenLayers);
    graph_.vCacheTensors.resize(param.numHiddenLayers);
    return this->AddOperationToGraph();
}

atb::Status GteDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    QwenLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    QwenDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

atb::Status GteDecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node finalNormNode;
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode.operation.reset(op);
    const uint32_t finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM;
    finalNormNode.inTensors = {
        param.withEmbedding ? \
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")) : \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_ids")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)
    };
    finalNormNode.outTensors = {&graph_.outTensors.at(0)};
 
    graph_.nodes.push_back(finalNormNode);

    return atb::NO_ERROR;
}

atb::Status GteDecoderModel::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    return atb::NO_ERROR;
}

} // namespace qwen
} // namespace atb_speed
