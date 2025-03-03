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
#include "models/llama/model/decoder_model.h"
#include "models/llama/layer/decoder_layer.h"
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include <atb/types.h>

namespace atb_speed {
namespace llama {

// Weight count
const uint64_t BLOCK_TABLES_LAYER_ID = 59;
const uint64_t SLOTS_LAYER_ID = 60;
const uint64_t SEQ_LEN_LAYER_ID = 62;
const uint64_t PFFSET_INDEX_LAYER_ID = 63;
const uint64_t RAZOR_OFFSET_LAYER_ID = 64;

void SliceTensor(atb::Tensor &tensor, int layerNum, int layerId)
{
    if (layerNum < 1) {
        ATB_SPEED_LOG_ERROR("layerNum which is smaller than 1!");
        return;
    }
    uint64_t offset = tensor.dataSize / layerNum;

    auto p = static_cast<char*>(tensor.deviceData);
    p += offset * layerId;
    tensor.deviceData = static_cast<void*>(p);
    tensor.dataSize = offset;
    tensor.desc.shape.dims[0] = tensor.desc.shape.dims[0] / layerNum;
}

static void SliceRaOffsetTensor(atb::Tensor &tensor, std::vector<int> blockNumsList, int layerNum, int layerId)
{
    if (layerNum < 1) {
        ATB_SPEED_LOG_ERROR("layerNum which is smaller than 1!");
        return;
    }
    uint64_t offset = tensor.dataSize / layerNum;

    auto p = static_cast<char*>(tensor.deviceData);
    p += offset * layerId;
    tensor.deviceData = static_cast<void*>(p);
    tensor.dataSize = blockNumsList[layerId] * sizeof(int);
    tensor.desc.shape.dims[0] = blockNumsList[layerId] / tensor.desc.shape.dims[1];
}

void LlamaModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("splitWithStride")) {
        this->splitWithStride = atb_speed::base::VerifyParam<bool>(paramJson, "splitWithStride");
    }
    if (paramJson.contains("isLongSeq")) { isLongSeq = atb_speed::base::VerifyParam<bool>(paramJson, "isLongSeq"); }
}

void LlamaModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_INFO("LlamaModelParam:splitWithStride: " << this->splitWithStride
                  << ", isLongSeq:" << isLongSeq);
}

LlamaDecoderModel::LlamaDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["default"] = {
        "input_ids", "input_embedding", "positional_ids", "cosine_table", "sine_table", "attention_mask",
        "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"};
    this->inTensorCandidates["long_seq"] = {"pos_ids_expanded", "inv_freq", "pos_lens"};
    this->internalTensorCandidates["long_seq"] = {"cosine_embed_table", "sine_embed_table"};
}

void LlamaDecoderModel::ConstructInTensorMap()
{
    DecoderModel::ConstructInTensorMap();
    // 添加DynamicNTK特性的输入Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "long_seq", this->inTensorMap);
    }
}

void LlamaDecoderModel::ConstructInternalTensorMap()
{
    DecoderModel::ConstructInternalTensorMap();
    // 添加DynamicNTK特性的中间Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->internalTensorCandidates, "long_seq", this->internalTensorMap);
    }
}

atb::Status LlamaDecoderModel::AddOperationToGraph()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    if (this->param.isLongSeq) { CHECK_OPERATION_STATUS_RETURN(this->AddDynamicNTK()); }
    if (this->param.positionEmbeddingType == atb_speed::base::PositionEmbeddingType::ROPE) {
        CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::AddDynamicNTK()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node dynamicNTKNode;
    atb::infer::DynamicNTKParam dynamicNTKParam;
    dynamicNTKParam.outDataType = this->param.isBF16 ? ACL_BF16 : ACL_FLOAT16;

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(dynamicNTKParam, &op));
    dynamicNTKNode.operation.reset(op);

    dynamicNTKNode.inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "pos_ids_expanded")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "inv_freq")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "pos_lens"))
    };
    dynamicNTKNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table"))
    };
    ATB_SPEED_LOG_DEBUG("[+] dynamicNTKNode");
    graph_.nodes.push_back(dynamicNTKNode);
    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::AddPositionalEmbedding()
{
    CHECK_OPERATION_STATUS_RETURN(atb_speed::base::DecoderModel::AddPositionalEmbedding());
    if (this->param.isLongSeq) {
        graph_.nodes.at(graph_.nodes.size() - 1).inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table"))
        };
    }

    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    LlamaLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    LlamaDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void LlamaDecoderModel::SetLayerParam(LlamaLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.splitWithStride = this->param.splitWithStride;
}

atb::Status LlamaDecoderModel::ParseParam(const std::string &paramString)
{
    CHECK_OPERATION_STATUS_RETURN(atb_speed::base::DecoderModel::ParseParam(paramString));
    nlohmann::json paramJson = atb_speed::base::StringToJson(paramString);

    this->blockNumsList_.clear();
    for (auto item : paramJson["blockNumsList"]) {
        this->blockNumsList_.push_back(atb_speed::base::VerifyParam<int>(item, "blockNumsList", true));
        ATB_SPEED_LOG_DEBUG("blockNumsList value: " << item);
    }

    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = tokenOffset.data();
    }
    auto &node = graph_.nodes.at(nodeId);
    if (tensorIdx != UINT32_MAX) {
        if (!this->param.isPrefill && this->param.enableCompressHead) {
            // OPERATION_COUNT_BEFORE_LAYER_SKIP_EMBED = 1, OPERATION_COUNT_BEFORE_LAYER_SKIP_EMBED = 2
            int operationCountBeforeLayers = this->param.skipWordEmbedding ? 1 : 2;
            auto upperBound = operationCountBeforeLayers;
            auto lowerBound = upperBound + this->param.numHiddenLayers;
            if (nodeId < static_cast<uint32_t>(upperBound) || nodeId >= static_cast<uint32_t>(lowerBound)) {
                return atb::NO_ERROR;
            }
            auto layerNum = this->param.numHiddenLayers;
            auto layerId = nodeId - upperBound;
            tensorIdx = SEQ_LEN_LAYER_ID;
            node.variantPack.inTensors.at(tensorIdx).hostData = seqLen.data() + seqLen.size() / layerNum * layerId;
        } else {
            tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len");
            graph_.inTensors.at(tensorIdx).hostData = seqLen.data();
        }
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = qLen.data();
    }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}

void LlamaDecoderModel::BuildNodeOutTensors(
    int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs)
{
    BuildNodeOutTensorImpl(nodeId, node, inTensorDescs);
}

void LlamaDecoderModel::BuildNodeVariantPack(int nodeId)
{
    int operationCountBeforeLayers =
        this->param.skipWordEmbedding ? 1 : 2;
    int upperBound = operationCountBeforeLayers;
    int lowerBound = upperBound + this->param.numHiddenLayers;
    int layerNum = this->param.numHiddenLayers;
    int layerId = nodeId - upperBound;
    if (nodeId < upperBound || nodeId >= lowerBound) {
        Model::BuildNodeVariantPack(nodeId);
    } else {
        auto &node = graph_.nodes.at(nodeId);
        atb::SVector<atb::TensorDesc> inTensorDescs;
        inTensorDescs.reserve(node.variantPack.inTensors.size());
        inTensorDescs.resize(node.variantPack.inTensors.size());

        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            CHECK_THROW(node.inTensors.at(i) == nullptr,
                modelName_ << " nodes[" << nodeId << "] "
                           << "inTensor " << i << "is NULL");

            if ((i == BLOCK_TABLES_LAYER_ID ||
                 i == SLOTS_LAYER_ID ||
                 i == SEQ_LEN_LAYER_ID ||
                 i == PFFSET_INDEX_LAYER_ID) && this->param.enableCompressHead) {
                auto curTensor = *node.inTensors.at(i);
                inTensorDescs.at(i) = curTensor.desc;
                SliceTensor(curTensor, layerNum, layerId);
                node.variantPack.inTensors.at(i) = curTensor;
                inTensorDescs.at(i) = curTensor.desc;
            } else if (i == RAZOR_OFFSET_LAYER_ID && this->param.enableCompressHead) {
                auto raOffsetTensor = *node.inTensors.at(i);
                SliceRaOffsetTensor(raOffsetTensor, blockNumsList_, layerNum, layerId);
                node.variantPack.inTensors.at(i) = raOffsetTensor;
                inTensorDescs.at(i) = raOffsetTensor.desc;
            } else {
                node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
                inTensorDescs.at(i) = node.inTensors.at(i)->desc;
            }

            ATB_SPEED_LOG_DEBUG(modelName_ << " nodes[" << nodeId << "] inTensors[" << i
                          << "]:" << TensorUtil::TensorToString(node.variantPack.inTensors.at(i)));
        }

        LlamaDecoderModel::BuildNodeOutTensors(nodeId, node, inTensorDescs);

        auto it = graph_.maxNodeIdTensorMap.find(nodeId);
        if (it != graph_.maxNodeIdTensorMap.end()) {
            for (auto tensorIt : it->second) {
                Model::FreeInternalTensor(tensorIt->deviceData);
            }
        }
    }
}
} // namespace llama
} // namespace atb_speed