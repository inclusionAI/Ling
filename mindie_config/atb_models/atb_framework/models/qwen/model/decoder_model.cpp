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
#include "models/qwen/layer/decoder_layer.h"
#include "models/qwen/model/decoder_model.h"

namespace atb_speed {
namespace qwen {

void QwenModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("withEmbedding")) {
        this->withEmbedding = paramJson["withEmbedding"].get<bool>();
    }
    if (paramJson.contains("enableLogN")) {
        this->enableLogN = paramJson["enableLogN"].get<bool>();
    }
    if (paramJson.contains("isLongSeq")) {
        this->isLongSeq = paramJson["isLongSeq"].get<bool>();
    }
    if (paramJson.contains("isYarn")) {
        isYarn = paramJson["isYarn"].get<bool>();
    }
    if (paramJson.contains("mscale")) {
        this->mscale = paramJson["mscale"].get<float>();
    }
    if (paramJson.contains("enableQScale")) {
        this->enableQScale = paramJson["enableQScale"].get<bool>();
    }
}

void QwenModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("QwenModelParam: withEmbedding: " << this->withEmbedding << ", enableLogN: " << this->enableLogN
                                                          << ", isLongSeq: " << this->isLongSeq
                                                          << ", isYarn:" << this->isYarn << ", mscale:" << this->mscale
                                                          << ", enableQScale: " << this->enableQScale);
}

QwenDecoderModel::QwenDecoderModel(const std::string &param) : DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["long_seq"] = {"inv_freq", "pos_lens", "positional_ids_gather"};
    this->internalTensorCandidates["long_seq"] = {"cosine_embed_table", "sine_embed_table"};
}

void QwenDecoderModel::ConstructInTensorMap()
{
    this->inTensorMap.clear();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default", this->inTensorMap);

    // 添加长序列所需Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "long_seq", this->inTensorMap);
    }

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

void QwenDecoderModel::ConstructInternalTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInternalTensorMap();
    // 添加长序列的中间Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->internalTensorCandidates, "long_seq", this->internalTensorMap);
    }
}

atb::Status QwenDecoderModel::AddDynamicNTK()
{
    atb::Operation *op = nullptr;
    if (param.isLongSeq) {
        atb_speed::Model::Node dynamicNTKNode;
        atb::infer::DynamicNTKParam dynamicNTKParam;
        dynamicNTKParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;

        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(dynamicNTKParam, &op));
        dynamicNTKNode.operation.reset(op);

        dynamicNTKNode.inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "inv_freq")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "pos_lens"))};
        dynamicNTKNode.outTensors = {
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table"))};
        ATB_SPEED_LOG_INFO("[+] dynamicNTKNode");
        graph_.nodes.push_back(dynamicNTKNode);
    }
    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddMuls()
{
    atb::Operation *op = nullptr;

    if (param.isLongSeq && param.isYarn) {
        atb_speed::Model::Node mulsCosNode;
        atb::infer::ElewiseParam magnifyElewiseMulsParam;
        magnifyElewiseMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
        magnifyElewiseMulsParam.mulsParam.varAttr = param.mscale;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(magnifyElewiseMulsParam, &op));
        mulsCosNode.operation.reset(op);
        uint32_t cosEmbedTableIdx = atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table");
        mulsCosNode.inTensors = {&graph_.internalTensors.at(cosEmbedTableIdx)};
        mulsCosNode.outTensors = {&graph_.internalTensors.at(cosEmbedTableIdx)};
        graph_.nodes.push_back(mulsCosNode);

        atb_speed::Model::Node mulsSinNode;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(magnifyElewiseMulsParam, &op));
        mulsSinNode.operation.reset(op);
        uint32_t sinEmbedTableIdx = atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table");
        mulsSinNode.inTensors = {&graph_.internalTensors.at(sinEmbedTableIdx)};
        mulsSinNode.outTensors = {&graph_.internalTensors.at(sinEmbedTableIdx)};
        graph_.nodes.push_back(mulsSinNode);
    }

    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node positionalEmbeddingGatherNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    positionalEmbeddingGatherNode.operation.reset(op);
    if (param.isLongSeq) {
        positionalEmbeddingGatherNode.inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids_gather")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table")),
        };
    } else {
        positionalEmbeddingGatherNode.inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "cosine_table")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sine_table")),
        };
    }

    positionalEmbeddingGatherNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"))};
    ATB_SPEED_LOG_INFO("[+] positionalEmbeddingGatherNode");
    graph_.nodes.push_back(positionalEmbeddingGatherNode);

    return atb::NO_ERROR;
}

void QwenDecoderModel::SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.enableLogN = param.enableLogN;
    layerParam.enableQScale = param.enableQScale;
}

void QwenDecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    DecoderModel::SetLayerNodeInput(layerNode, layerId);

    if (param.enableLogN) {
        layerNode.inTensors.at(layerNode.inTensors.size() - 1) =
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    }
}

atb::Status QwenDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    QwenLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    QwenDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddOperationToGraph()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    CHECK_OPERATION_STATUS_RETURN(AddDynamicNTK());
    CHECK_OPERATION_STATUS_RETURN(AddMuls());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}
} // namespace qwen
} // namespace atb_speed