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
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "models/internlm2/20b/layer/decoder_layer.h"
#include "models/internlm2/20b/model/decoder_model.h"

namespace atb_speed {
namespace internlm2_20b {

// Weight count
const uint32_t WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const uint32_t WEIGHT_COUNT_POST_NORM = 1;
const uint32_t WEIGHT_COUNT_LM_HEAD = 1;
const uint32_t WEIGHT_COUNT_LORA_PER_LAYER = 14;

// Operation count
uint32_t g_operationCountBeforeLayer = 0;
const uint32_t OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

// Speculate index
const uint32_t IN_TENSOR_Q_LEN_INDEX = 13;

void DecoderModel::Param::ParseBasicParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("skipWordEmbedding")) {
        skipWordEmbedding = paramJson["skipWordEmbedding"].get<bool>();
    }
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    lmHeadTransposeType = paramJson["lmHeadTransposeType"].get<int>();
    supportSwiGLU = paramJson["supportSwiGLU"].get<bool>();
    supportLcoc = paramJson["supportLcoc"].get<bool>();
    if (paramJson.contains("useImMask")) {
        useImMask = paramJson["useImMask"].get<bool>();
    }
    if (paramJson.contains("supportLora")) {
        supportLora = paramJson["supportLora"].get<bool>();
    }
    kvQuant = paramJson["kvQuant"].get<bool>();
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numAttentionHeadsPerRank = CheckPositive(paramJson["numAttentionHeadsPerRank"].get<int>());
    hiddenSizePerAttentionHead = CheckPositive(paramJson["hiddenSizePerAttentionHead"].get<int>());
    numHiddenLayers = CheckNumHiddenLayersValid(paramJson["numHiddenLayers"].get<int>());
    numKeyValueHeadsPerRank = CheckPositive(paramJson["numKeyValueHeadsPerRank"].get<int>());
    rank = paramJson["rank"].get<int>();
    worldSize = CheckPositive(paramJson["worldSize"].get<int>());
}

void DecoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR("parse param fail, please check param's format, error: " << e.what());
        throw std::runtime_error(ss.str());
    }
    ParseBasicParams(paramJson);
    if (rank >= worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater than rank, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    backend = paramJson["backend"].get<std::string>();
    splitWithStride = paramJson["splitWithStride"].get<bool>();
    if (paramJson.contains("rankTableFile")) {
        rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.contains("positionEmbeddingType")) {
        positionEmbeddingType = paramJson["positionEmbeddingType"].get<std::string>();
    }
    for (auto item : paramJson["packQuantType"]) {
        packQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckPackQuantParamsSufficient(packQuantType, numHiddenLayers);

    for (auto item : paramJson["linearQuantType"]) {
        linearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearPackParamsSufficient(linearQuantType, numHiddenLayers);

    for (auto item : paramJson["linearTransposeType"]) {
        linearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearPackParamsSufficient(linearTransposeType, numHiddenLayers);
    PrintParam();
}

void DecoderModel::Param::PrintParam()
{
    ATB_SPEED_LOG_DEBUG("DecoderModel param" << ", isFA:" << isFA
                  << ", skipWordEmbedding: " << skipWordEmbedding
                  << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                  << isLmHeadParallel << ", lmHeadTransposeType: " << lmHeadTransposeType <<", supportSwiGLU: "
                  << supportSwiGLU << "supportLcoc" << supportLcoc
                  << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:"
                  << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", splitWithStride:" << splitWithStride
                  << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
                  << ", rankTableFile:" << rankTableFile
                  << ", positionEmbeddingType:" << positionEmbeddingType
                  << ", supportLora:" << supportLora
                  << ", useImMask:" << useImMask);
}

DecoderModel::DecoderModel(const std::string &param) : Model("DecoderModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

DecoderModel::~DecoderModel() {}

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
    uint64_t loraSize = 0;
    if (param_.supportLora) {
        loraSize = CheckIntMulOverFlow(WEIGHT_COUNT_LORA_PER_LAYER, param_.numHiddenLayers);
    }
    CHECK_TENSORDESC_DIMNUM_VALID(
        graph_.weightTensors.at(graph_.weightTensors.size() - loraSize - 1).desc.shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dimNum);
    const int64_t vocabSizePerRank = \
        graph_.weightTensors.at(graph_.weightTensors.size() - loraSize - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - loraSize - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = \
        param_.skipWordEmbedding ? inTensorDescs.at(1).shape.dimNum : (inTensorDescs.at(0).shape.dimNum + 1);
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    if (param_.isFA) {  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] = param_.isPrefill ? \
            inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0] : 1;
    } else {  // unpadInputs = true
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
        }
    }
    if (outTensorDescs.at(0).shape.dimNum <= 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = \
            CheckIntMulOverFlow(vocabSizePerRank, param_.worldSize);
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    return atb::NO_ERROR;
}

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 1;

int64_t DecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    // wordEmbeddingNode
    if (!param_.skipWordEmbedding) {
        atb_speed::Model::Node wordEmbeddingNode;
        atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
        wordEmbeddingParam.unpadInputs = !param_.isFA;
        if (param_.isEmbeddingParallel) {
            wordEmbeddingParam.tensorParallelInfo = {
                param_.rank, param_.worldSize, param_.backend, param_.rankTableFile
            };
        };
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
        wordEmbeddingNode.operation.reset(op);
        wordEmbeddingNode.inTensors = {
            &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
            &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)
        };
        wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES)};
        graph_.nodes.push_back(wordEmbeddingNode);
    }

    return atb::NO_ERROR;
}


int64_t DecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    if (param_.positionEmbeddingType == "ROPE") {
        atb_speed::Model::Node positionalEmbeddingGatherNode;
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
        positionalEmbeddingGatherNode.operation.reset(op);
        positionalEmbeddingGatherNode.inTensors = {
            &graph_.inTensors.at(IN_TENSOR_POSITION_IDS),
            &graph_.inTensors.at(IN_TENSOR_COS_TABLE),
            &graph_.inTensors.at(IN_TENSOR_SIN_TABLE),
        };
        positionalEmbeddingGatherNode.outTensors = {
            &graph_.internalTensors.at(INTERNAL_TENSOR_COS_EMB),
            &graph_.internalTensors.at(INTERNAL_TENSOR_SIN_EMB)
        };
        graph_.nodes.push_back(positionalEmbeddingGatherNode);
    }

    return atb::NO_ERROR;
}

void DecoderModel::SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId)
{
    layerParam.isFA = param_.isFA;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.isBF16 = param_.isBF16;
    layerParam.supportSwiGLU = param_.supportSwiGLU;
    layerParam.supportLcoc = param_.supportLcoc;
    layerParam.supportLora = param_.supportLora;
    layerParam.useImMask = param_.useImMask;
    layerParam.packQuantType = param_.packQuantType[layerId];
    layerParam.linearQuantType = param_.linearQuantType[layerId];
    layerParam.linearTransposeType = param_.linearTransposeType[layerId];
    layerParam.kvQuant = param_.kvQuant;
    layerParam.rmsNormEps = param_.rmsNormEps;
    layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
    layerParam.splitWithStride = param_.splitWithStride;
    layerParam.tensorParallelInfo = {param_.rank, param_.worldSize, param_.backend, param_.rankTableFile};
    if (param_.positionEmbeddingType == "ROPE") {
        layerParam.positionEmbeddingType = atb_speed::internlm2_20b::ROPE;
    }
}

void DecoderModel::AddLoraLayerIntensor(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    if (param_.supportLora) {
        for (uint32_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_LORA_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(param_.numHiddenLayers, weightCountPerLayer) +
                WEIGHT_COUNT_WORD_EMBEDDINGNODE + WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD +
                CheckIntMulOverFlow(layerId, WEIGHT_COUNT_LORA_PER_LAYER) + weightTensorId);
        }
    } else {
        for (uint32_t i = 0; i < WEIGHT_COUNT_LORA_PER_LAYER; i++) {
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER);
        }
    }
}

int64_t DecoderModel::AddLayer()
{
    atb::Operation *op = nullptr;
    for (uint32_t layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        atb_speed::Model::Node layerNode;
        atb_speed::internlm2_20b::DecoderLayerParam layerParam;
        SetLayerParam(layerParam, layerId);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::internlm2_20b::DecoderLayer(layerParam, &op));

        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        uint32_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = param_.skipWordEmbedding ? \
            &graph_.inTensors.at(IN_TENSOR_INPUT_EMBEDDING) : \
            &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES);
        for (uint32_t weightTensorId = 0; weightTensorId < weightCountPerLayer; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(layerId, weightCountPerLayer) + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        AddLoraLayerIntensor(layerNode, layerId, inTensorId);
        if (!param_.kvQuant) {
            for (int i = 0; i < 8; i++) {  // 8: KV_QUANT has 8 params
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER);
            }
        }
        if (param_.positionEmbeddingType == "ROPE") {
            layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_TENSOR_COS_EMB);
            layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_TENSOR_SIN_EMB);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KV_CACHE_IDX);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_LORA_IM_MASK);

        layerNode.outTensors = {param_.skipWordEmbedding ? \
            &graph_.inTensors.at(IN_TENSOR_INPUT_EMBEDDING) : \
            &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES)};
        graph_.nodes.push_back(layerNode);
    }

    return atb::NO_ERROR;
}

int64_t DecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node finalNormNode;
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode.operation.reset(op);
    uint32_t finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    if (param_.supportLora) {
        finalLayerNormWeightTensorId -= CheckIntMulOverFlow(WEIGHT_COUNT_LORA_PER_LAYER, param_.numHiddenLayers);
    }
    finalNormNode.inTensors = {
        param_.skipWordEmbedding ? \
        &graph_.inTensors.at(IN_TENSOR_INPUT_EMBEDDING) : \
        &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)
    };
    finalNormNode.outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        param_.skipWordEmbedding ? \
        &graph_.inTensors.at(IN_TENSOR_INPUT_EMBEDDING) : \
        &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES)
    };
    graph_.nodes.push_back(finalNormNode);

    return atb::NO_ERROR;
}

int64_t DecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node lmHeadNode;
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param_.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param_.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.unpadInputs = !param_.isFA;
    if (param_.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo = {
            param_.rank, param_.worldSize, param_.backend, param_.rankTableFile
        };
    }
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    lmHeadNode.operation.reset(op);
    uint64_t finalLinearWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    if (param_.supportLora) {
        finalLinearWeightTensorId -= CheckIntMulOverFlow(WEIGHT_COUNT_LORA_PER_LAYER, param_.numHiddenLayers);
    }
    lmHeadNode.inTensors = {
        param_.skipWordEmbedding ? \
        &graph_.inTensors.at(IN_TENSOR_INPUT_EMBEDDING) : \
        &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
    graph_.nodes.push_back(lmHeadNode);

    return atb::NO_ERROR;
}

int64_t DecoderModel::BuildGraph()
{
    if (param_.kvQuant) {
        weightCountPerLayer += 8;  // 8: kv cache int8 多8个inTensor
    }
    // set size
    uint64_t weightTensorSize =
        WEIGHT_COUNT_WORD_EMBEDDINGNODE +
        CheckIntMulOverFlow(weightCountPerLayer, param_.numHiddenLayers) +
        WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    if (param_.supportLora) {
        weightTensorSize += CheckIntMulOverFlow(WEIGHT_COUNT_LORA_PER_LAYER, param_.numHiddenLayers);
    }
    graph_.weightTensors.resize(weightTensorSize);

    uint64_t inTensorCount = IN_TENSOR_COUNT;
    graph_.inTensors.resize(inTensorCount);
    graph_.outTensors.resize(OUT_TENSOR_COUNT);

    int internelTensorIdx = 1;
    if (param_.positionEmbeddingType == "ROPE") {
        internelTensorIdx = 3;  // 3: Internal tensor的数量
    }
    graph_.internalTensors.resize(internelTensorIdx);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    if (param_.skipWordEmbedding && param_.positionEmbeddingType != "ROPE") {
        ATB_SPEED_LOG_ERROR("If skipWordEmbedding is True, positionEmbeddingType must is ROPE");
        return atb::ERROR_INVALID_PARAM;
    }
    if (param_.positionEmbeddingType == "ROPE") {
        if (param_.skipWordEmbedding) {
            g_operationCountBeforeLayer = 1;  // 1: Layer前Operation的数量
        } else {
            g_operationCountBeforeLayer = 2;  // 2: Layer前Operation的数量
        }
    }

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph begin");

    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph success");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::ParseParam(const std::string &param)
{
    ATB_SPEED_LOG_DEBUG("ParseParam start.");
    CHECK_PARAM_LT(param.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR("parse param fail, please check param's format, error: " << e.what());
        throw std::runtime_error(ss.str());
    }
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        int tokenOffset = item.get<int>();
        CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
        tokenOffset_.push_back(tokenOffset);
        ATB_SPEED_LOG_DEBUG("token offset value: " << item);
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        int seqLen = item.get<int>();
        CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
        seqLen_.push_back(seqLen);
        ATB_SPEED_LOG_DEBUG("seqLen value: " << item << "Prefill" << paramJson["isPrefill"]);
    }

    qLen_.clear();
    for (auto item : paramJson["qLen"]) {
        int qLen = item.get<int>();
        CHECK_PARAM_LT(qLen, MAX_PARAM_VALUE);
        qLen_.push_back(qLen);
        ATB_SPEED_LOG_DEBUG("qLen value: " << item);
    }
    ATB_SPEED_LOG_DEBUG("ParseParam end.");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor");
    ATB_SPEED_LOG_DEBUG("nodeId = " << nodeId);

    if (param_.positionEmbeddingType == "ROPE") {
        if (param_.skipWordEmbedding) {
            g_operationCountBeforeLayer = 1;  // 1: Layer前Operation的数量
        } else {
            g_operationCountBeforeLayer = 2;  // 2: Layer前Operation的数量
        }
    }

    if (nodeId < g_operationCountBeforeLayer || \
        nodeId >= (g_operationCountBeforeLayer + param_.numHiddenLayers)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = DecoderLayerTensorIdx::IN_TOKEN_OFFSET;
    const uint32_t seqLenTensorId = DecoderLayerTensorIdx::IN_SEQ_LEN;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}
} // namespace internlm2_20b
} // namespace atb_speed