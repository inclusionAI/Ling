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
#include "pa_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "atb_speed/log.h"
#include "atb_speed/utils/model_factory.h"
#include "models/gptneox/20b/layer/decoder_layer.h"
#include "operations/fusion/parallel_lmhead.h"

namespace atb_speed {
namespace gptneox_20b {

REGISTER_MODEL(gptneox_20b, PAModel);

const int WEIGHT_COUNT_PER_LAYER = 12;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int OUT_TENSOR_DIM_NUM = 3;
const int LAYER_NORM_AXIS_NUM = 1;

void PAModel::Param::FromString(const std::string &param)
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
    layerNormEps = paramJson["layerNormEps"].get<float>();
    headNum = CheckPositive(paramJson["headNum"].get<int>());
    dk = CheckPositive(paramJson["dk"].get<int>());
    layerNum = CheckNumHiddenLayersValid(paramJson["layerNum"].get<int>());
    rotaryPct = paramJson["rotaryPct"].get<float>();
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = CheckPositive(paramJson["rankSize"].get<int>());
    }
    if (paramJson.contains("qScale")) {
        qScale = paramJson["qScale"].get<float>();
    }
    if (paramJson.contains("qkScale")) {
        qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }

    ATB_SPEED_LOG_DEBUG("GptNeox20BModel param layerNormEps:"
                        << layerNormEps << ", headNum:" << headNum << ", dk:" << dk << ", layerNum:" << layerNum
                        << ", rotaryPct:" << rotaryPct << ", backend: " << backend);
}

PAModel::PAModel(const std::string &param) : Model("GptNeoX_20B_MODEL", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PAModel::~PAModel() {}

uint32_t PAModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t PAModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status PAModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outTensorLastDimSize = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    auto outDimNum = inTensorDescs.at(0).shape.dimNum + 1;
    for (uint i = 0; i < outDimNum - 1; i++) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    CHECK_TENSORDESC_DIMNUM_VALID(outDimNum);
    outTensorDescs.at(0).shape.dims[outDimNum - 1] = CheckIntMulOverFlow(outTensorLastDimSize, param_.rankSize);

    // change first dim
    if (param_.isPrefill) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }

    return atb::NO_ERROR;
}

int64_t PAModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    // wordEmbeddingNode
    atb_speed::Model::Node wordEmbeddingNode;
    atb::infer::GatherParam wordEmbeddingParam;
    CREATE_OPERATION(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_ID)};
    graph_.nodes.push_back(wordEmbeddingNode);

    return atb::NO_ERROR;
}

int64_t PAModel::AddLayer()
{
    atb::Operation *op = nullptr;
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        atb_speed::Model::Node layerNode;
        atb_speed::gptneox_20b::PALayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rotaryPct = param_.rotaryPct;
        opParam.model = "gptneox_20b";
        opParam.isPrefill = param_.isPrefill;
        opParam.qScale = param_.qScale;
        opParam.qkScale = param_.qkScale;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        CHECK_OPERATION_STATUS_RETURN(atb_speed::gptneox_20b::PALayer(opParam, &op));
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        uint32_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_TENSOR_ID);
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_INPUT_LENGTHS);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_ID)};
        graph_.nodes.push_back(layerNode);
    }
    return atb::NO_ERROR;
}

int64_t PAModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node finalNormNode;
    atb::infer::LayerNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.layerNormEps;
    finalNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_NUM;
    finalNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_NUM;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const uint32_t finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const uint32_t finalLayerNormBiasTensorId =
        graph_.weightTensors.size() - (FINALNORMNODE_WEIGHT_COUNT - 1) - OUT_LM_HEAD_WEIGHT_COUNT;
    finalNormNode.inTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_ID),
                               &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_ID)};

    graph_.nodes.push_back(finalNormNode);

    return atb::NO_ERROR;
}

int64_t PAModel::AddLmhead()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node lmHeadNode;
    atb_speed::common::ParallelLmHeadParam lmHeadParam;
    lmHeadParam.rank = param_.rank;
    lmHeadParam.rankSize = param_.rankSize;
    lmHeadParam.unpadInputs = true;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.backend = param_.backend;
    CHECK_OPERATION_STATUS_RETURN(ParallelLmHead(lmHeadParam, &op));
    lmHeadNode.operation.reset(op);
    const uint64_t finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;

    if (param_.isPrefill) {
        lmHeadNode.inTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_ID),
                                &graph_.weightTensors.at(finalLinearWeightTensorId),
                                &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)};
    } else {
        lmHeadNode.inTensors = {&graph_.internalTensors.at(INTERNAL_TENSOR_ID),
                                &graph_.weightTensors.at(finalLinearWeightTensorId)};
    }
    lmHeadNode.outTensors = {&graph_.outTensors.at(OUT_TENSOR_ID)};
    graph_.nodes.push_back(lmHeadNode);

    return atb::NO_ERROR;
}

int64_t PAModel::BuildGraph()
{
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);
    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);
    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);
    graph_.internalTensors.resize(INTERNAL_TENSOR_MAX);

    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    return atb::NO_ERROR;
}

atb::Status PAModel::ParseParam(const std::string &param)
{
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
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        int seqLen = item.get<int>();
        CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
        seqLen_.push_back(seqLen);
    }
    ATB_SPEED_LOG_DEBUG("PAModel ParseParam seqLen: " << seqLen_.capacity());
    return atb::NO_ERROR;
}

atb::Status PAModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER) {
        return atb::NO_ERROR;
    } else if (nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = LayerPATensorId::IN_INPUT_LENGTHS;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    return atb::NO_ERROR;
}
} // namespace gptneox_20b
} // namespace atb_speed
