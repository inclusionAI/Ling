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
#include "nlohmann/json.hpp"
#include "vector"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "models/qwen/layer/moe_decoder_layer.h"
#include "models/qwen/model/moe_decoder_model.h"

namespace atb_speed {
namespace qwen {

// Weight count
const int WEIGHT_COUNT_PER_LAYER = 68;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 2;  // wte(wordEmbed) + gather(cos/sin embedding)
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

constexpr size_t ATTN_LINEAR_TYPE_LENGTH = 6;
constexpr size_t MLP_LINEAR_TYPE_LENGTH = 4;
constexpr size_t MOE_LINEAR_TYPE_LENGTH = 4;

enum InTensorId : int {
    IN_TENSOR_INPUT_IDS = 0,
    IN_TENSOR_POSITION_IDS,
    IN_TENSOR_COS_TABLE,
    IN_TENSOR_SIN_TABLE,
    IN_TENSOR_ATTENTION_MASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_KVCACHE_IDX,
    IN_TENSOR_TOKEN_OFFSET,
    IN_TENSOR_PLACE_HOLDER,
    IN_TENSOR_SEQ_LEN,
    IN_TENSOR_LOGITS_INDICES,
    IN_EXPERT_ARRAY_MODEL,
    IN_EXPERT_GROUP_MODEL,
    IN_ONE_HOT_MODEL,
    IN_ZERO_HOT_MODEL,
    IN_TENSOR_NUM,
};

enum InternalTensorId : int {
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    INTERNAL_TENSOR_COS_EMB,
    INTERNAL_TENSOR_SIN_EMB,
    INTERNAL_TENSOR_LAYER_OUT_BASE,
};

void MoeDecoderModel::Param::ParseBasicParams(const nlohmann::json &paramJson)
{
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    lmHeadTransposeType = paramJson["lmHeadTransposeType"].get<int>();
    supportSwiGLU = paramJson["supportSwiGLU"].get<bool>();
    supportLcoc = paramJson["supportLcoc"].get<bool>();
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numAttentionHeadsPerRank = CheckPositive(paramJson["numAttentionHeadsPerRank"].get<int>());
    hiddenSizePerAttentionHead = CheckPositive(paramJson["hiddenSizePerAttentionHead"].get<int>());
    numHiddenLayers = CheckPositive(paramJson["numHiddenLayers"].get<int>());
    numKeyValueHeadsPerRank = CheckPositive(paramJson["numKeyValueHeadsPerRank"].get<int>());
    rank = paramJson["rank"].get<int>();
    worldSize = CheckPositive(paramJson["worldSize"].get<int>());
}

void MoeDecoderModel::Param::AddLogInfo()
{
    ATB_SPEED_LOG_DEBUG("MoeDecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                  << isLmHeadParallel << ", lmHeadTransposeType: " << lmHeadTransposeType
                  << ", supportSwiGLU: " << supportSwiGLU << "supportLcoc" << supportLcoc
                  << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:"
                  << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen << ", rankTableFile:" << rankTableFile
                  << ", numOfExperts:" << numOfExperts << ", expertParallelDegree:" << expertParallelDegree
                  << ", numOfSelectedExperts:" << numOfSelectedExperts << "routingMethod: " << routingMethod
                  << ", packQuantType: " << packQuantType << ", attnLinearQuantType" << attnLinearQuantType
                  << ", mlpLinearQuantType: " << mlpLinearQuantType << ", moeLinearQuantType: " << moeLinearQuantType
                  << ", attnLinearTransposeTyp: " << attnLinearTransposeType
                  << ", mlpLinearTransposeType: " << mlpLinearTransposeType
                  << ", moeLinearTransposeType: " << moeLinearTransposeType);
}

void MoeDecoderModel::Param::AddParamJson(const std::string &param)
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
    if (paramJson.contains("rankTableFile")) {
        rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    for (auto item : paramJson["packQuantType"]) {
        packQuantType.push_back(item.get<std::vector<int>>());
    }
}

void MoeDecoderModel::Param::AddParamJsonMoE(const std::string &param)
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
    if (paramJson.contains("numOfExperts")) {
        numOfExperts = paramJson["numOfExperts"].get<int>();
    }
    if (paramJson.contains("routingMethod")) {
        routingMethod = paramJson["routingMethod"].get<std::string>();
    }
    if (paramJson.contains("expertParallelDegree")) {
        expertParallelDegree = paramJson["expertParallelDegree"].get<int>();
    }
    for (auto item : paramJson["numOfSelectedExperts"]) {
        numOfSelectedExperts.push_back(item.get<int>());
    }
}

void MoeDecoderModel::Param::FromString(const std::string &param)
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
    ParseBasicParams(paramJson);
    if (rank >= worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater than rank, please check." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    backend = paramJson["backend"].get<std::string>();
    AddParamJson(param);
    AddParamJsonMoE(param);
    CheckPackQuantParamsSufficient(packQuantType, numHiddenLayers);

    for (auto item : paramJson["attnLinearQuantType"]) {
        attnLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(attnLinearQuantType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);
    for (auto item : paramJson["mlpLinearQuantType"]) {
        mlpLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(mlpLinearQuantType, numHiddenLayers, MLP_LINEAR_TYPE_LENGTH);
    for (auto item : paramJson["moeLinearQuantType"]) {
        moeLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(moeLinearQuantType, numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);

    for (auto item : paramJson["attnLinearTransposeType"]) {
        attnLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(attnLinearTransposeType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);
    for (auto item : paramJson["mlpLinearTransposeType"]) {
        mlpLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(mlpLinearTransposeType, numHiddenLayers, MLP_LINEAR_TYPE_LENGTH);
    for (auto item : paramJson["moeLinearTransposeType"]) {
        moeLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(moeLinearTransposeType, numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);

    AddLogInfo();
}

MoeDecoderModel::MoeDecoderModel(const std::string &param) : Model("MoeDecoderModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

MoeDecoderModel::~MoeDecoderModel() {}

uint32_t MoeDecoderModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t MoeDecoderModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status MoeDecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter MoeDecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    if (param_.isFA) {  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] =
            param_.isPrefill ? inTensorDescs.at(IN_TENSOR_LOGITS_INDICES).shape.dims[0] : 1;
    } else {  // unpadInputs = true
        if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGITS_INDICES).shape.dims[0];
        }
    }

    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank * param_.worldSize;
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    return atb::NO_ERROR;
}

int64_t MoeDecoderModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_WORD_EMBEDDINGNODE +
                                 WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers +
                                 WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_NUM);
    graph_.outTensors.resize(1);
    graph_.internalTensors.resize(INTERNAL_TENSOR_LAYER_OUT_BASE);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    ATB_SPEED_LOG_DEBUG("MoeDecoderModel build graph begin");

    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    return atb::NO_ERROR;
}

int64_t MoeDecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    auto wordEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    wordEmbeddingParam.unpadInputs = !param_.isFA;
    if (param_.isEmbeddingParallel) {
        wordEmbeddingParam.tensorParallelInfo = {param_.rank, param_.worldSize, param_.backend, param_.rankTableFile};
    };
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
    wordEmbeddingNode->operation.reset(op);
    wordEmbeddingNode->inTensors = {
        &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
        &graph_.inTensors.at(IN_TENSOR_INPUT_IDS)
    };
    wordEmbeddingNode->outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES)};
    ATB_SPEED_LOG_DEBUG("wordEmbeddingNode is doing!");
    graph_.nodes.push_back(*wordEmbeddingNode);
    return atb::NO_ERROR;
}

int64_t MoeDecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    auto posEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    posEmbeddingNode->operation.reset(op);
    posEmbeddingNode->inTensors = {
        &graph_.inTensors.at(IN_TENSOR_POSITION_IDS),
        &graph_.inTensors.at(IN_TENSOR_COS_TABLE),
        &graph_.inTensors.at(IN_TENSOR_SIN_TABLE),
    };
    posEmbeddingNode->outTensors = {
        &graph_.internalTensors.at(INTERNAL_TENSOR_COS_EMB),
        &graph_.internalTensors.at(INTERNAL_TENSOR_SIN_EMB)
    };
    graph_.nodes.push_back(*posEmbeddingNode);
    ATB_SPEED_LOG_DEBUG("posEmbeddingNode is doing!");
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::SetLayerParam(atb_speed::qwen::MoeDecoderLayerParam &layerParam, const int layerId)
{   // 26
    layerParam.isFA = param_.isFA;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.isBF16 = param_.isBF16;
    layerParam.supportSwiGLU = param_.supportSwiGLU;
    layerParam.supportLcoc = param_.supportLcoc;
    layerParam.packQuantType = param_.packQuantType[layerId];
    layerParam.attnLinearQuantType = param_.attnLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = param_.mlpLinearQuantType[layerId];
    layerParam.moeLinearQuantType = param_.moeLinearQuantType[layerId];
    layerParam.attnLinearTransposeType = param_.attnLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = param_.mlpLinearTransposeType[layerId];
    layerParam.moeLinearTransposeType = param_.moeLinearTransposeType[layerId];
    layerParam.rmsNormEps = param_.rmsNormEps;
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
    layerParam.routingMethod = param_.routingMethod;
    layerParam.hasSharedExpertGate = true;
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddLayer()
{
    atb::Operation *op = nullptr;
    atb::Tensor *firstInTensor = &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES);
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto layerNode = std::make_unique<atb_speed::Model::Node>();
        atb_speed::qwen::MoeDecoderLayerParam layerParam;
        SetLayerParam(layerParam, layerId);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::qwen::MoeDecoderLayer(layerParam, &op));
        layerNode->operation.reset(op);
        layerNode->inTensors.resize(layerNode->operation->GetInputNum());
        size_t inTensorId = 0;
        layerNode->inTensors.at(inTensorId++) = firstInTensor;

        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode->inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
        }
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_EXPERT_ARRAY_MODEL);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_EXPERT_GROUP_MODEL);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ONE_HOT_MODEL);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ZERO_HOT_MODEL);
        layerNode->inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_TENSOR_COS_EMB);
        layerNode->inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_TENSOR_SIN_EMB);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTION_MASK);
        layerNode->inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode->inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LEN);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KVCACHE_IDX);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode->inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode->outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES)};
        firstInTensor = layerNode->outTensors.at(0);
        ATB_SPEED_LOG_DEBUG("layerNode_" << layerId << " is doing!");
        graph_.nodes.push_back(*layerNode);
    }
    ATB_SPEED_LOG_DEBUG("layerNode is doing!");
    return atb::NO_ERROR;
}

int64_t MoeDecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    auto finalNormNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode->operation.reset(op);
    const size_t finalLayerNormWeightTensorId =
        this -> graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode->inTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES),
     &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode->outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES)
    };

    ATB_SPEED_LOG_DEBUG("finalNormNode is doing!");
    graph_.nodes.push_back(*finalNormNode);
    return atb::NO_ERROR;
}

int64_t MoeDecoderModel::AddLmhead()
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
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.worldSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rankTableFile = param_.rankTableFile;
    }
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    ATB_SPEED_LOG_DEBUG("lmHeadNode is doing!");

    lmHeadNode->operation.reset(op);
    const size_t finalLinearWeightTensorId = this -> graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode->inTensors = {
        &graph_.internalTensors.at(INTERNAL_TENSOR_HIDDEN_STATES),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_PLACE_HOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGITS_INDICES)
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode->outTensors = {&graph_.outTensors.at(0)};

    ATB_SPEED_LOG_DEBUG("MoeDecoderModel build graph success");
    graph_.nodes.push_back(*lmHeadNode);
    return atb::NO_ERROR;
}


atb::Status MoeDecoderModel::ParseParam(const std::string &param)
{
    ATB_SPEED_LOG_DEBUG("ParseParam start.");
    CHECK_PARAM_LT(param.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        int tokenOffset = item.get<int>();
        CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
        tokenOffset_.push_back(tokenOffset);
        ATB_SPEED_LOG_DEBUG("tokenOffset value: " << item);
    }
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        int seqLen = item.get<int>();
        CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
        seqLen_.push_back(seqLen);
        ATB_SPEED_LOG_DEBUG("Prefill" << paramJson["isPrefill"] << "seqLen value: " << item);
    }
    ATB_SPEED_LOG_DEBUG("ParseParam end.");
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor");
    ATB_SPEED_LOG_DEBUG("nodeId = " << nodeId);

    auto upperBound = OPERATION_COUNT_BEFORE_LAYER;
    auto lowerBound = upperBound + param_.numHiddenLayers;
    if (nodeId < static_cast<uint32_t>(upperBound) || nodeId >= static_cast<uint32_t>(lowerBound)) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = DecoderLayerTensorId::IN_TOKEN_OFFSET;
    const uint32_t seqLenTensorId = DecoderLayerTensorId::IN_SEQ_LEN;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}
} // namespace qwen
} // namespace atb_speed
