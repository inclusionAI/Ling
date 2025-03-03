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
#include "models/deepseekv2/model/decoder_model.h"
#include <vector>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"

namespace atb_speed {
namespace deepseekV2 {

// Weight count
constexpr uint32_t WEIGHT_COUNT_PER_LAYER = 84;
constexpr uint32_t WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
constexpr uint32_t WEIGHT_COUNT_POST_NORM = 1;
constexpr uint32_t WEIGHT_COUNT_LM_HEAD = 1;
// quant linear count
constexpr uint32_t DEEPSEEKV2_LINEAR_TYPE_LENGTH = 9;

// Operation count
constexpr uint32_t OPERATION_COUNT_BEFORE_LAYER = 2;  // Word Embedding + Positional Embedding
constexpr uint32_t OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

constexpr uint32_t ATTN_LINEAR_TYPE_LENGTH = 6;

void DeepseekV2ModelParam::AddParamJsonMLA(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    if (paramJson.contains("qLoraRank")) {
        qLoraRank = paramJson["qLoraRank"].get<int>();
    }
    if (paramJson.contains("kvLoraRank")) {
        kvLoraRank = paramJson["kvLoraRank"].get<int>();
    }
    if (paramJson.contains("qkNopeHeadDim")) {
        qkNopeHeadDim = paramJson["qkNopeHeadDim"].get<int>();
    }
    if (paramJson.contains("qkRopeHeadDim")) {
        qkRopeHeadDim = paramJson["qkRopeHeadDim"].get<int>();
    }
    if (paramJson.contains("softmaxScale")) {
        softmaxScale = paramJson["softmaxScale"].get<float>();
    }
}

void DeepseekV2ModelParam::AddParamJsonMoE(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    tpRank = paramJson["rank"].get<uint32_t>();
    tpSize = CheckPositive(paramJson["worldSize"].get<int>());
    if (paramJson.contains("numOfGroups")) {
        numOfGroups = paramJson["numOfGroups"].get<int>();
    }
    if (paramJson.contains("routedScalingFactor")) {
        routedScalingFactor = paramJson["routedScalingFactor"].get<float>();
    }
    if (paramJson.contains("routingMethod")) {
        routingMethod = paramJson["routingMethod"].get<std::string>();
    }
    if (paramJson.contains("processLogits")) {
        processLogits = paramJson["processLogits"].get<std::string>();
    }
    for (auto item : paramJson["topkGroups"]) {
        topkGroups.push_back(item.get<int>());
    }
}


void DeepseekV2ModelParam::AddLogInfo()
{
    ATB_SPEED_LOG_DEBUG("DecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                << ", isBF16:" << isBF16
                << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                << isLmHeadParallel << ", enableSwiGLU: " << enableSwiGLU << ", enableLcoc:" << enableLcoc
                << ", lmHeadTransposeType: " << lmHeadTransposeType
                << ", normEps:" << normEps << ", numAttentionHeadsPerRank:"
                << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                << ", numHiddenLayers:" << numHiddenLayers
                << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
                << ", rankTableFile:" << rankTableFile
                << ", numOfExperts:" << numOfExperts << ", expertParallelDegree:" << expertParallelDegree
                << ", maskStartIdx:" << maskStartIdx << ", numOfSelectedExperts:" << numOfSelectedExperts
                << ", topkGroups:" << topkGroups << ", processLogits:" << processLogits
                << ", routedScalingFactor" << routedScalingFactor
                << "firstKDenseReplace: " << firstKDenseReplace << ", numOfSharedExperts" << numOfSharedExperts
                << "routingMethod: " << routingMethod << "tpRankTableFile: " << tpRankTableFile
                << "dpRankTableFile: " << dpRankTableFile << "epRankTableFile: " << epRankTableFile
                << "esRankTableFile: " << esRankTableFile);
}


void DeepseekV2ModelParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    ParseParam(paramJson);
    if (rank > worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater or equal to 0, please check." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    backend = paramJson["backend"].get<std::string>();
    AddParamJsonMLA(param);
    AddParamJsonMoE(param);

    for (auto item : paramJson["attnLinearQuantType"]) {
        attnLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(attnLinearQuantType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);

    for (auto item : paramJson["attnLinearTransposeType"]) {
        attnLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(attnLinearTransposeType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);

    AddLogInfo();
}

DecoderModel::DecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelInTensorCandidates = {
        {"default", {
            "in_tensor_input_ids", "in_tensor_position_ids", "in_tensor_cos_table", "in_tensor_sin_table",
            "in_tensor_attention_mask", "in_tensor_block_tables", "in_tensor_slots", "in_tensor_kvcache_idx",
            "in_final_state_model",
            "in_tensor_token_offset", "in_tensor_place_holder", "in_tensor_seq_len", "in_tensor_logits_indices",
            "in_expert_array_model", "in_expert_group_model", "in_one_hot_model", "in_zero_hot_model"}},
        {"dp", {"in_dp_input_indices_model", "in_dp_gather_indices0_model", "in_dp_gather_indices1_model"}},
    };
    return deepseekV2ModelInTensorCandidates;
}

void DecoderModel::ConstructInTensorMap()
{
    auto deepseekV2ModelInTensorCandidates = GetDeepseekV2ModelInTensorCandidates();
    uint32_t tensorIdx = 0;
    atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "default", tensorIdx, this->inTensorMap_);
    if (param_.hasDP) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "dp", tensorIdx, this->inTensorMap_);
    }
    this->inTensorCount_ = tensorIdx;
    std::stringstream ss;
    for (auto tensor = this->inTensorMap_.cbegin(); tensor != this->inTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map: " << ss.str());
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelInternalTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelInternalTensorCandidates = {
        {"default", {
            "internal_tensor_hidden_states", "internal_tensor_cos_emb", "internal_tensor_sin_emb"}},
    };
    return deepseekV2ModelInternalTensorCandidates;
}

void DecoderModel::ConstructInternalTensorMap()
{
    auto deepseekV2InternalTensorCandidates = GetDeepseekV2ModelInternalTensorCandidates();
    uint32_t tensorIdx = 0;
    atb_speed::common::AssignTensorIdx(
        deepseekV2InternalTensorCandidates, "default", tensorIdx, this->internalTensorMap_);
    this->internalTensorCount_ = tensorIdx;
    std::stringstream ss;
    for (auto tensor = this->internalTensorMap_.cbegin(); tensor != this->internalTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map: " << ss.str());
}

atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter DecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    uint32_t logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_logits_indices");
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);

    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    if (param_.isFA) {  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] =
        param_.isPrefill ? inTensorDescs.at(logitsIndicesIdx).shape.dims[0] : 1;
    } else {  // unpadInputs = true
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] =
             inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
        }
    }

    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = \
        CheckIntMulOverFlow(vocabSizePerRank, param_.tpSize);
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    return atb::NO_ERROR;
}

int64_t DecoderModel::BuildGraph()
{
    ConstructInTensorMap();
    this->graph_.inTensors.resize(this->inTensorCount_);
    ATB_SPEED_LOG_DEBUG("graph_.inTensorCount_ " << this->inTensorCount_);

    ConstructInternalTensorMap();
    this->graph_.internalTensors.resize(this->internalTensorCount_);
    ATB_SPEED_LOG_DEBUG("graph_.internalTensorCount_ " << this->internalTensorCount_);

    graph_.outTensors.resize(1);

    // set size
    const int weightTensorSize =
        WEIGHT_COUNT_WORD_EMBEDDINGNODE + CheckIntMulOverFlow(WEIGHT_COUNT_PER_LAYER, param_.numHiddenLayers) +
         WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph begin");

    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    auto wordEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    wordEmbeddingParam.unpadInputs = !param_.isFA;
    if (param_.isEmbeddingParallel) {
        wordEmbeddingParam.tensorParallelInfo.rank = param_.tpRank;
        wordEmbeddingParam.tensorParallelInfo.worldSize = param_.tpSize;
        wordEmbeddingParam.tensorParallelInfo.backend = param_.backend;
        wordEmbeddingParam.tensorParallelInfo.rankTableFile = param_.rankTableFile;
        wordEmbeddingParam.tensorParallelInfo.commDomain = param_.tpDomain;
    };
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
    wordEmbeddingNode->operation.reset(op);
    wordEmbeddingNode->inTensors = {
        &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_input_ids"))
    };
    wordEmbeddingNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                                   "internal_tensor_hidden_states"))
    };
    graph_.nodes.push_back(*wordEmbeddingNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    auto posEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    posEmbeddingNode->operation.reset(op);
    posEmbeddingNode->inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_position_ids")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_cos_table")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_sin_table")),
    };
    posEmbeddingNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                                   "internal_tensor_cos_emb")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                                   "internal_tensor_sin_emb")),
    };
    graph_.nodes.push_back(*posEmbeddingNode);
    return atb::NO_ERROR;
}

void SetMlaParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param)
{
    layerParam.qLoraRank = param.qLoraRank;
    layerParam.headNum = param.headNum;
    layerParam.qkNopeHeadDim = param.qkNopeHeadDim;
    layerParam.qkRopeHeadDim = param.qkRopeHeadDim;
    layerParam.kvLoraRank = param.kvLoraRank;
    layerParam.softmaxScale = param.softmaxScale;
}

void SetParallelParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param)
{
    layerParam.tpRank = param.tpRank;
    layerParam.tpSize = param.tpSize;
    layerParam.tpDomain = param.tpDomain;
    layerParam.hasDP = param.hasDP;
    layerParam.dpRank = param.dpRank;
    layerParam.dpSize = param.dpSize;
    layerParam.dpDomain = param.dpDomain;
    layerParam.hasES = param.hasES;
    layerParam.esRank = param.esRank;
    layerParam.esSize = param.esSize;
    layerParam.esDomain = param.esDomain;
    layerParam.hasEP = param.hasEP;
    layerParam.epRank = param.epRank;
    layerParam.epSize = param.epSize;
    layerParam.epDomain = param.epDomain;
    layerParam.tpRankTableFile = param.rankTableFile;
    layerParam.dpRankTableFile = param.rankTableFile;
    layerParam.esRankTableFile = param.rankTableFile;
    layerParam.epRankTableFile = param.rankTableFile;
}

void DecoderModel::SetLayerParam(DecoderLayerParam &layerParam, int64_t layerId)
{
    layerParam.isFA = param_.isFA;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.isBF16 = param_.isBF16;
    layerParam.enableSwiGLU = param_.enableSwiGLU;
    layerParam.enableLcoc = param_.enableLcoc;
    layerParam.hasSharedExpert = param_.hasSharedExpert;
    layerParam.hasSharedExpertGate = param_.hasSharedExpertGate;
    layerParam.processLogits = param_.processLogits;
    layerParam.routedScalingFactor = param_.routedScalingFactor;
    layerParam.packQuantType = param_.packQuantType[layerId];
    layerParam.attnLinearQuantType = param_.attnLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = param_.mlpLinearQuantType[layerId];
    layerParam.moeLinearQuantType = param_.moeLinearQuantType[layerId];
    layerParam.attnLinearTransposeType = param_.attnLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = param_.mlpLinearTransposeType[layerId];
    layerParam.moeLinearTransposeType = param_.moeLinearTransposeType[layerId];
    layerParam.normEps = param_.normEps;
    layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
    layerParam.rank = param_.rank;
    layerParam.worldSize = param_.worldSize;
    layerParam.backend = param_.backend;
    layerParam.rankTableFile = param_.rankTableFile;
    layerParam.layerId = layerId;
    layerParam.numOfSelectedExperts = param_.numOfSelectedExperts;
    layerParam.expertParallelDegree = param_.expertParallelDegree;
    layerParam.numOfExperts = param_.numOfExperts;
    layerParam.maskStartIdx = param_.maskStartIdx;
    layerParam.firstKDenseReplace = param_.firstKDenseReplace;
    layerParam.numOfSharedExperts = param_.numOfSharedExperts;
    layerParam.routingMethod = param_.routingMethod;
    layerParam.numOfGroups = param_.numOfGroups;
    layerParam.topkGroups = param_.topkGroups;
    layerParam.enableAddNorm = param_.enableAddNorm;
    SetMlaParam(layerParam, param_);
    SetParallelParam(layerParam, param_);
}

atb::Status DecoderModel::AddLayer()
{
    atb::Operation *op = nullptr;
    for (uint32_t layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        atb_speed::Model::Node layerNode;
        DecoderLayerParam layerParam;
        SetLayerParam(layerParam, layerId);
        ATB_SPEED_LOG_DEBUG("start create Decoderlayer");
        CHECK_OPERATION_STATUS_RETURN(DecoderLayer(layerParam, &op));
        ATB_SPEED_LOG_DEBUG("Decoderlayer create success");
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        size_t inTensorId = 0;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(layerId, WEIGHT_COUNT_PER_LAYER) \
                 + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        AddLayerHostWeight(layerNode, inTensorId, layerId);
        layerNode.outTensors = {&graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"))};
        graph_.nodes.push_back(layerNode);
    }
    ATB_SPEED_LOG_DEBUG("DecoderModel build graph:fill intensors");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddLayerHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_zero_hot_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_final_state_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_cos_emb"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_sin_emb"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_seq_len"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_place_holder"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_token_offset"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_kvcache_idx"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_block_tables"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_slots"));
    if (param_.hasDP) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_dp_input_indices_model"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_dp_gather_indices0_model"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_dp_gather_indices1_model"));
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    auto finalNormNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode->operation.reset(op);
    const size_t finalLayerNormWeightTensorId =
        this -> graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode->inTensors = {
        &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)
    };
    finalNormNode->outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"))
    };

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph:finalNormNode end");
    graph_.nodes.push_back(*finalNormNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    auto lmHeadNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param_.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param_.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.unpadInputs = !param_.isFA;
    if (param_.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.tpRank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.tpSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rankTableFile = param_.rankTableFile;
        lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain = param_.tpDomain;
    }
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    ATB_SPEED_LOG_DEBUG("DecoderModel build graph:create LMHead end");

    lmHeadNode->operation.reset(op);
    const size_t finalLinearWeightTensorId = this -> graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    uint32_t placeHolderIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_place_holder");
    lmHeadNode->inTensors = {
        &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_logits_indices"))
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode->outTensors = {&graph_.outTensors.at(0)};

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph success");
    graph_.nodes.push_back(*lmHeadNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        // 仅需在graph的intensor中bind一次
        return atb::NO_ERROR;
    }

    const uint32_t tokenOffsetTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_token_offset");
    if (tokenOffsetTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tokenOffsetTensorIdx).hostData = tokenOffset.data();
    }

    const uint32_t seqLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_seq_len");
    if (seqLenTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(seqLenTensorIdx).hostData = seqLen.data();
    }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");

    return atb::NO_ERROR;
}
} // namespace deepseekV2
} // namespace atb_speed
