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
#include "models/base/model/decoder_model.h"
#include "models/base/layer/decoder_layer.h"
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include <atb/types.h>

namespace atb_speed {
namespace base {

DecoderModel::DecoderModel(const std::string &param) : Model("DecoderModel", param)
{
    this->param.FromString(param);
    this->modelName_ += this->param.isPrefill ? "_Prefill" : "_Decoder";
    this->inTensorCandidates = {
        {"default", {
            "input_ids", "positional_ids", "cosine_table", "sine_table", "attention_mask",
            "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"}
        },
        {"compress_head_alibi", {"wins_global", "in_ra_seqlens"}},
        {"compress_head_rope", {"wins_global", "in_ra_seqlens", "pffset_index", "razor_offset", "in_reshape_seqlen"}},
        {"q_len", {"q_len"}},
        {"lora_common", {"seq_len_cum_sum"}},
        {"lora_per_layer", {
            "qkv_lora_a_0", "qkv_lora_b_0", "qkv_lora_a_1", "qkv_lora_b_1",
            "qkv_lora_a_2", "qkv_lora_b_2", "qkv_dense_lora_a", "qkv_dense_lora_b",
            "mlp_lora_a_0", "mlp_lora_b_0", "mlp_lora_a_1", "mlp_lora_b_1",
            "mlp_down_lora_a", "mlp_down_lora_b"}
        },
    };
    if (this->param.skipWordEmbedding) {
        this->inTensorCandidates["default"].at(0) = "input_embedding";
    }

    this->internalTensorCandidates = {
        {"default", {"hidden_states"}},
        {"rope", {"cosine_embedding", "sine_embedding"}},
    };
}

DecoderModel::~DecoderModel() {}

void DecoderModel::ConstructInTensorMap()
{
    this->inTensorMap.clear();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default", this->inTensorMap);

    // 添加头压缩特性的Tensor
    if (this->param.enableCompressHead) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates,
            param.positionEmbeddingType == PositionEmbeddingType::ALIBI ? "compress_head_alibi" : "compress_head_rope",
            this->inTensorMap);
    }

    // 添加并行解码特性或SplitFuse的Tensor
    if (this->param.enableSpeculate || this->param.enableSplitFuse) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "q_len", this->inTensorMap);
    }

    // 添加lora特性的Tensor
    if (this->param.enableLora) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "lora_common", this->inTensorMap);
        uint32_t currentTensorIdx = this->inTensorMap.size();
        for (uint32_t i = 0; i < this->param.numHiddenLayers; i++) {
            for (std::string loraWeightName : this->inTensorCandidates.at("lora_per_layer")) {
                this->inTensorMap["layer_" + std::to_string(i) + loraWeightName] = currentTensorIdx;
                currentTensorIdx++;
            }
        }
    }
}

void DecoderModel::ConstructInternalTensorMap()
{
    this->internalTensorMap.clear();
    // 添加默认的Tensor
    if (!this->param.skipWordEmbedding) {
        atb_speed::common::AssignTensorIdx(
            this->internalTensorCandidates, "default", this->internalTensorMap);
    }

    // 添加rope的Tensor
    if (this->param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
        atb_speed::common::AssignTensorIdx(
            this->internalTensorCandidates, "rope", this->internalTensorMap);
    }
}

uint32_t DecoderModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DecoderModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter DecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    uint32_t logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "logits_indices");
    CHECK_TENSORDESC_DIMNUM_VALID(graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // unpadInputs: [batchSize, seqLen, vocabSize] padInputs: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = this->param.isUnpadInputs ? 2 : 3;  // 2, 3: dimNum
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    int64_t seqLenAxis = this->param.isUnpadInputs ? 0 : 1;  // 2, 3: Axis
    if (this->param.isPrefill || this->param.enablePrefixCache) {
        outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
    } else {
        outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(0).shape.dims[seqLenAxis];
    }
    outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = this->param.isLmHeadParallel ? \
        CheckIntMulOverFlow(vocabSizePerRank, this->param.worldSize) : vocabSizePerRank;
    return atb::NO_ERROR;
}

int64_t DecoderModel::BuildGraph()
{
    // 准备inTensor
    this->ConstructInTensorMap();
    std::stringstream ss;
    for (auto tensor = this->inTensorMap.cbegin(); tensor != this->inTensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("inTensorMap " << ss.str());

    this->graph_.inTensors.resize(this->inTensorMap.size());
    ATB_SPEED_LOG_DEBUG("graph_.inTensorCount_ " << this->graph_.inTensors.size());

    // 准备internalTensor
    this->ConstructInternalTensorMap();
    ss.str("");
    ss.clear();
    for (auto tensor = this->internalTensorMap.cbegin(); tensor != this->internalTensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("internalTensorMap " << ss.str());
    this->graph_.internalTensors.resize(this->internalTensorMap.size());
    ATB_SPEED_LOG_DEBUG("graph_.internalTensorCount_ " << this->graph_.internalTensors.size());

    // 准备outTensor
    graph_.outTensors.resize(1);  // 1: 模型输出1个out tensor

    // 准备weightTensor
    if (this->param.enableKvQuant) {
        this->weightCountPerLayer += 8;  // 8: kv cache int8 多8个inTensor
    }
    if (this->param.enableFA3) {
        this->weightCountPerLayer += 8; // 8: FA3 多8个inTensorensor
    }
    if (this->param.enableReduceQuant) {
        this->weightCountPerLayer += 8;  // 8: lccl reduce int8 多8个inTensor
    }
    if (this->param.normType == LAYER_NORM) {
        this->weightCountFinalNorm = 2;  // 2: LayerNorm 权重数量
    }
    const uint64_t weightTensorSize =
        this->weightCountWordEmbedding +
        CheckIntMulOverFlow(this->weightCountPerLayer, this->param.numHiddenLayers) +
        this->weightCountFinalNorm + this->weightCountLmHead;
    graph_.weightTensors.resize(weightTensorSize);

    // 准备kv cache
    graph_.kCacheTensors.resize(this->param.numHiddenLayers);
    graph_.vCacheTensors.resize(this->param.numHiddenLayers);

    return this->AddOperationToGraph();
}

atb::Status DecoderModel::AddOperationToGraph()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    if (this->param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
        CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}

void DecoderModel::SetWordEmbeddingParam(atb_speed::common::WordEmbeddingParam &wordEmbeddingParam)
{
    wordEmbeddingParam.unpadInputs = this->param.isUnpadInputs;
    if (this->param.isEmbeddingParallel) {
        wordEmbeddingParam.tensorParallelInfo = {
            this->param.rank, this->param.worldSize, this->param.backend, this->param.rankTableFile
        };
    };
}

atb::Status DecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node wordEmbeddingNode;
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    this->SetWordEmbeddingParam(wordEmbeddingParam);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {
        &graph_.weightTensors.at(0),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_ids"))
    };
    wordEmbeddingNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"))
    };
    graph_.nodes.push_back(wordEmbeddingNode);
    ATB_SPEED_LOG_DEBUG("[+] base wordEmbeddingNode");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node positionalEmbeddingGatherNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    positionalEmbeddingGatherNode.operation.reset(op);
    positionalEmbeddingGatherNode.inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "cosine_table")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sine_table")),
    };
    positionalEmbeddingGatherNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"))
    };
    graph_.nodes.push_back(positionalEmbeddingGatherNode);
    ATB_SPEED_LOG_DEBUG("[+] base positionalEmbeddingGatherNode");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    if (this->param.normType == RMS_NORM) {
        DecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    } else {
        DecoderLayer<atb::infer::LayerNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddLayer()
{
    atb::Operation *op = nullptr;
    for (uint32_t layerId = 0; layerId < this->param.numHiddenLayers; ++layerId) {
        atb_speed::Model::Node layerNode;
        CHECK_OPERATION_STATUS_RETURN(this->CreateLayerOperation(&op, layerId));
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        layerNode.inTensorReshapeFuncs.resize(layerNode.operation->GetInputNum());
        SetLayerNodeInput(layerNode, layerId);
        layerNode.outTensors = {layerNode.inTensors.at(weightCountPerLayer)};  // 输出原地写在输入上
        graph_.nodes.push_back(layerNode);
    }
    ATB_SPEED_LOG_DEBUG("[+] add base layerNode num" << this->param.numHiddenLayers);
    return atb::NO_ERROR;
}

void DecoderModel::SetLayerParam(LayerParam &layerParam, uint32_t layerId)
{
    layerParam.isFA = this->param.isFA;
    layerParam.isUnpadInputs = this->param.isUnpadInputs;
    layerParam.isPrefill = this->param.isPrefill;
    layerParam.isBF16 = this->param.isBF16;
    layerParam.isLite = this->param.isLite;
    layerParam.enableSwiGLU = this->param.enableSwiGLU;
    layerParam.enableLcoc = this->param.enableLcoc;
    layerParam.enableSpeculate = this->param.enableSpeculate;
    layerParam.enableCompressHead = this->param.enableCompressHead;
    layerParam.enableSplitFuse = this->param.enableSplitFuse;
    layerParam.enableLora = this->param.enableLora;
    layerParam.loraEnableGMM = this->param.loraEnableGMM;
    layerParam.enableKvQuant = this->param.enableKvQuant;
    layerParam.enableFA3 = this->param.enableFA3;
    layerParam.kvQuantHasOffset = this->param.kvQuantHasOffset;
    layerParam.enableReduceQuant = this->param.enableReduceQuant;
    layerParam.enableAddNorm = this->param.enableAddNorm;
    layerParam.enablePrefixCache = this->param.enablePrefixCache;
    layerParam.attnBackend = this->param.attnBackend;
    layerParam.positionEmbeddingType = this->param.positionEmbeddingType;
    layerParam.normEps = this->param.normEps;
    layerParam.normType = this->param.normType;
    layerParam.quantGroupSize = this->param.quantGroupSize;
    layerParam.numAttentionHeadsPerRank = this->param.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = this->param.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = this->param.numKeyValueHeadsPerRank;
    layerParam.tensorParallelInfo = {
        this->param.rank, this->param.worldSize, this->param.backend, this->param.rankTableFile};
    layerParam.packQuantType = this->param.packQuantType[layerId];
    layerParam.linearQuantType = this->param.linearQuantType[layerId];
    layerParam.linearTransposeType = this->param.linearTransposeType[layerId];
    if (!this->param.linearHasBias.empty()) {
        layerParam.linearHasBias = this->param.linearHasBias[layerId];
    }
    layerParam.weightQuantType = this->param.weightQuantType;
}

void DecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    uint32_t inTensorId = 0;
    this->SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    if (this->param.enableCompressHead) {
        this->SetLayerNodeRaInput(layerNode, inTensorId);
    }
    if (this->param.enableSpeculate || this->param.enableSplitFuse) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len"));
    }
    if (this->param.enableLora) {
        this->SetLayerNodeLoraInput(layerNode, layerId, inTensorId);
    }
}

void DecoderModel::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    for (uint32_t weightTensorId = 0; weightTensorId < this->weightCountPerLayer; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId, this->weightCountPerLayer) + weightTensorId + this->weightCountWordEmbedding);
    }
    layerNode.inTensors.at(inTensorId++) = this->param.skipWordEmbedding ? \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"));
    if (this->param.positionEmbeddingType == atb_speed::base::PositionEmbeddingType::ROPE) {
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding"));
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"));
    } else {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "place_holder"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "place_holder"));
    }
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "block_tables"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "slots"));
}

void DecoderModel::SetLayerNodeRaInput(atb_speed::Model::Node &layerNode, uint32_t &inTensorId)
{
    std::string candidateKey = param.positionEmbeddingType == PositionEmbeddingType::ALIBI ? \
        "compress_head_alibi" : "compress_head_rope";
    for (std::string raInputName : this->inTensorCandidates.at(candidateKey)) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, raInputName)
        );
    }
}

void DecoderModel::SetLayerNodeLoraInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len_cum_sum"));
    for (std::string loraWeightName : this->inTensorCandidates.at("lora_per_layer")) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(
                this->inTensorMap, "layer_" + std::to_string(layerId) + loraWeightName)
        );
    }
}

void DecoderModel::SetFinalNormParam(atb::infer::RmsNormParam &normParam)
{
    normParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    normParam.normParam.epsilon = this->param.normEps;
}

void DecoderModel::SetFinalNormParam(atb::infer::LayerNormParam &normParam)
{
    int32_t beginParamsAxis = this->param.isFA ? 2 : 1;
    normParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    normParam.normParam.epsilon = this->param.normEps;
    normParam.normParam.beginNormAxis = beginParamsAxis;
    normParam.normParam.beginParamsAxis = 1;
}

atb::Status DecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node finalNormNode;
    if (this->param.normType == NormType::RMS_NORM) {
        atb::infer::RmsNormParam finalNormParam;
        this->SetFinalNormParam(finalNormParam);
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    } else {
        atb::infer::LayerNormParam finalNormParam;
        this->SetFinalNormParam(finalNormParam);
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    }
    finalNormNode.operation.reset(op);
    const uint32_t finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - this->weightCountFinalNorm - this->weightCountLmHead;
    finalNormNode.inTensors = {
        this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)
    };
    if (this->param.normType == NormType::LAYER_NORM) {
        finalNormNode.inTensors.push_back(&graph_.weightTensors.at(finalLayerNormWeightTensorId + 1));
    }
    finalNormNode.outTensors = {finalNormNode.inTensors.at(0)};  // 输出原地写在输入上
    graph_.nodes.push_back(finalNormNode);
    ATB_SPEED_LOG_DEBUG("[+] base finalNormNode");
    return atb::NO_ERROR;
}

void DecoderModel::SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam)
{
    lmHeadParam.unpadInputs = this->param.isUnpadInputs;
    lmHeadParam.gatherAhead = this->param.isPrefill || this->param.enablePrefixCache;
    lmHeadParam.hiddenSizePerAttentionHead = this->param.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = this->param.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = this->param.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.unpadInputs = !this->param.isFA;
    if (this->param.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo = {
            this->param.rank, this->param.worldSize, this->param.backend, this->param.rankTableFile
        };
    }
}

atb::Status DecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node lmHeadNode;
    atb_speed::common::LmHeadParam lmHeadParam;
    this->SetLmHeadParam(lmHeadParam);
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    lmHeadNode.operation.reset(op);
    const uint64_t finalLinearWeightTensorId = graph_.weightTensors.size() - this->weightCountLmHead;
    uint32_t placeHolderIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "place_holder");
    lmHeadNode.inTensors = {
        this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "logits_indices"))
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
    graph_.nodes.push_back(lmHeadNode);
    ATB_SPEED_LOG_DEBUG("[+] base lmHeadNode");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::ParseParam(const std::string &paramString)
{
    CHECK_PARAM_LT(paramString.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson = StringToJson(paramString);

    this->tokenOffset.clear();
    for (auto item : paramJson["tokenOffset"]) {
        this->tokenOffset.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("token offset value: " << item);
    }

    this->seqLen.clear();
    for (auto item : paramJson["seqLen"]) {
        this->seqLen.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("seqLen value: " << item);
    }

    this->qLen.clear();
    for (auto item : paramJson["qLen"]) {
        this->qLen.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("qLen value: " << item);
    }

    return atb::NO_ERROR;
}

atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        // 仅需在graph的intensor中bind一次
        return atb::NO_ERROR;
    }

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->tokenOffset.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->seqLen.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->qLen.data();
    }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}

} // namespace base
} // namespace atb_speed