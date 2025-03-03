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
#include "flash_attention_model.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "atb_speed/log.h"
#include "atb_speed/utils/model_factory.h"
#include "models/vlmo/2b/layer/encoder_layer.h"
#include "models/vlmo/2b/layer/encoder_vl_layer.h"
#include "operations/fusion/parallel_layer_v2.h"

namespace atb_speed {
namespace vlmo {
REGISTER_MODEL(vlmo, FlashAttentionModel);

const int WEIGHT_COUNT_PER_LAYER = 20;
const int WEIGHT_COUNT_PER_VL_LAYER = 14;

enum InTensorId : int {
    IN_TENSOR_X = 0,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_HOLDER,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_LAYEROUT = 0,
    OUT_TENSOR_MAX,
};

void FlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ATB_SPEED_LOG_ERROR("vlmo model parse param fail, please check param's format, error: " << e.what());
        ss << "parse   param fail,  please check param's format, error: " << e.what() << std::endl;
        throw std::runtime_error(ss.str());
    }
    layerNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = CheckPositive(paramJson["headNum"].get<int>());
    dk = CheckPositive(paramJson["dk"].get<int>());
    layerNum = CheckPositive(paramJson["layerNum"].get<int>());
    if (paramJson.contains("rank")) {
        rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        rankSize = CheckPositive(paramJson["rankSize"].get<int>());
    }
    if (paramJson.contains("backend")) {
        backend = paramJson["backend"];
    }
    if (paramJson.contains("maxTextLen")) {
        maxTextLen = paramJson["maxTextLen"];
    }
    if (paramJson.contains("vlLayerIndex")) {
        vlLayerIndex = paramJson["vlLayerIndex"];
    }
}

FlashAttentionModel::FlashAttentionModel(const std::string &param) : Model("FlashAttentionModel", param)
{
    ATB_SPEED_LOG_DEBUG("FlashAttentionModel constructor function begin");
    param_.FromString(param);
}

FlashAttentionModel::~FlashAttentionModel() = default;

uint32_t FlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

uint32_t FlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

atb::Status FlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                            std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::ConstructLayerInTensors(size_t &inTensorId, atb::Tensor *&firstInTensor,
                                                     atb_speed::Model::Node &layerNode, const int &layerId)
{
    layerNode.inTensors.at(inTensorId++) = firstInTensor;
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::ADDLayer(int &layerId, atb::Tensor *&firstInTensor)
{
    atb::Operation *op = nullptr;
    for (; layerId < param_.vlLayerIndex; ++layerId) {
        ATB_SPEED_LOG_DEBUG(__func__ << " layerId " << layerId << " create node");
        atb_speed::Model::Node layerNode;
        atb_speed::vlmo::EncoderLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.maxTextLen = param_.maxTextLen;
        CHECK_OPERATION_STATUS_RETURN(atb_speed::vlmo::EncoderLayer(opParam, &op));
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        ATB_SPEED_LOG_DEBUG("FlashAttentionModel Layer Construction Begins");
        size_t inTensorId = 0;
        ConstructLayerInTensors(inTensorId, firstInTensor, layerNode, layerId);
        for (int weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            ATB_SPEED_LOG_DEBUG(__func__ << " layerId " << layerId << " weightID" << weightTensorId
                                         << " -> in weight ID"
                                         << (CheckIntMulOverFlow(layerId, WEIGHT_COUNT_PER_LAYER) + weightTensorId));
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(CheckIntMulOverFlow(layerId, WEIGHT_COUNT_PER_LAYER) + weightTensorId);
        }

        layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
        for (int i = 0; i < OUT_TENSOR_MAX; i++) {
            layerNode.outTensors.at(i) = &graph_.internalTensors.at((layerId * 1) + i);
        }
        firstInTensor = layerNode.outTensors.at(0);
        graph_.nodes.push_back(layerNode);
        ATB_SPEED_LOG_DEBUG(" graph_ nodes add " << graph_.nodes.size());
    }
    return atb::NO_ERROR;
}

int64_t FlashAttentionModel::ADDVlLayer(int &layerId, atb::Tensor *&firstInTensor)
{
    atb::Operation *op = nullptr;
    for (; layerId < param_.layerNum; ++layerId) {
        ATB_SPEED_LOG_DEBUG(__func__ << " layerId " << layerId << " create node");
        atb_speed::Model::Node layerNode;
        atb_speed::vlmo::EncoderVllayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.maxTextLen = param_.maxTextLen;
        atb_speed::vlmo::EncoderVlLayer(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        ATB_SPEED_LOG_DEBUG("FlashAttentionModel Layer Construction Begins");
        size_t inTensorId = 0;
        ConstructLayerInTensors(inTensorId, firstInTensor, layerNode, layerId);
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_VL_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                (CheckIntMulOverFlow(WEIGHT_COUNT_PER_LAYER, param_.vlLayerIndex)) +
                CheckIntMulOverFlow((layerId - param_.vlLayerIndex), WEIGHT_COUNT_PER_VL_LAYER) + weightTensorId);
        }

        layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
        if (layerId + 1 == param_.layerNum) {
            for (int i = 0; i < OUT_TENSOR_MAX; i++) {
                layerNode.outTensors.at(i) = &graph_.outTensors.at(i);
            }
        } else {
            for (int i = 0; i < OUT_TENSOR_MAX; i++) {
                layerNode.outTensors.at(i) = &graph_.internalTensors.at((layerId * OUT_TENSOR_MAX) + i);
            }
            firstInTensor = layerNode.outTensors.at(0);
        }
        graph_.nodes.push_back(layerNode);
        ATB_SPEED_LOG_DEBUG(" graph_ nodes add " << graph_.nodes.size());
    }
    return atb::NO_ERROR;
}
int64_t FlashAttentionModel::BuildGraph()
{
    const int weightTensorSize =
        CheckIntMulOverFlow(WEIGHT_COUNT_PER_LAYER, param_.vlLayerIndex) +
        CheckIntMulOverFlow(WEIGHT_COUNT_PER_VL_LAYER, (param_.layerNum - param_.vlLayerIndex));
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum * 2); // 2是layer层数的倍数值
    graph_.outTensors.resize(OUT_TENSOR_MAX);
    const int internalTensorSize = (param_.layerNum - 1) * OUT_TENSOR_MAX;
    graph_.internalTensors.resize(internalTensorSize);
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);
    int layerId = 0;
    ADDLayer(layerId, firstInTensor);
    ADDVlLayer(layerId, firstInTensor);
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::ParseParam(const std::string &param)
{
    CHECK_PARAM_LT(param.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson;
    ATB_SPEED_LOG_DEBUG("paramJson begin");
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse  param fail, please check param's format, error: " << e.what() << std::endl;
        throw std::runtime_error(ss.str());
    }

    tokenOffset_.clear();
    ATB_SPEED_LOG_DEBUG("tokenOffset get begin");
    for (auto item : paramJson["tokenOffset"]) {
        int tokenOffset = item.get<int>();
        CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
        tokenOffset_.push_back(tokenOffset);
    }
    ATB_SPEED_LOG_DEBUG("seqLen get begin");
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        int seqLen = item.get<int>();
        CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
        seqLen_.push_back(seqLen);
    }
    return atb::NO_ERROR;
}

atb::Status FlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    uint32_t layerNum = static_cast<uint32_t>(param_.layerNum);
    if (nodeId >= layerNum) {
        return atb::NO_ERROR;
    }

    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = 4;
    const uint32_t seqLenTensorId = 5;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}
} // namespace vlmo
} // namespace atb_speed