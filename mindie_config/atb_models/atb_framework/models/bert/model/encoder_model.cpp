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

#include "atb_speed/utils/model_factory.h"
#include "models/bert/operation/embedding.h"
#include "models/bert/layer/encoder_layer.h"
#include "models/bert/model/encoder_model.h"


namespace atb_speed::bert {

    REGISTER_MODEL(bert, EncoderModel);

    enum InTensorId : int {
        IN_INPUT_IDS = 0,
        IN_POSITION_IDS,
        IN_TOKENTYPE_IDS,
        IN_ATTENTIONMASK,
        IN_PASTKEY,
        IN_PASTVALUE,
        IN_TOKENOFFSET,
        IN_SEQLEN,
        IN_TENSOR_MAX
    };

    enum OutTensorId : int {
        OUT_HIDDENSTATES = 0,
        OUT_TENSOR_MAX
    };

    const int IN_TENSOR_INPUTIDS_ID = 0;
    const int WORDEMBEDDING_WEIGHT_ID = 0;
    const int INTERNAL_TENSOR_EMBEDDING_OUT_ID = 0;
    const int OUT_TENSOR_HIDDENSTATES_ID = 0;

    const int IN_TENSOR_COUNT = 3;
    const int EMBEDDING_WEIGHT_COUNT = 5;
    const int WEIGHT_COUNT_PER_LAYER = 16;

    const int INTERNAL_TENSOR_COUNT_BEFORE_LAYER = 1;
    const int OPERATION_COUNT_BEFORE_LAYER = 1;

    const int OUT_TENSOR_HIDDENSTATES_DIM_NUM = 3;  // [batch_size, seq_len, hidden_size]

    const uint64_t LAYER_IN_TOKENOFFSET_ID = 20;
    const uint64_t LAYER_IN_SEQLEN_ID = 21;

    void EncoderModel::Param::FromString(const std::string &param)
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
        dk = CheckPositive(paramJson["dk"].get<int>());
        geluApproximate = paramJson["geluApproximate"].get<int>();
        headNum = CheckPositive(paramJson["headNum"].get<int>());
        layerNormEps = paramJson["layerNormEps"].get<float>();
        layerNormImplMode = CheckPositive(paramJson["layerNormImplMode"].get<int>());
        layerNum = CheckNumHiddenLayersValid(paramJson["layerNum"].get<int>());
        if (paramJson.contains("rank")) {
            rank = paramJson["rank"].get<int>();
        }
        if (paramJson.contains("rankSize")) {
            rankSize = CheckPositive(paramJson["rankSize"].get<int>());
        }
    }

    EncoderModel::EncoderModel(const std::string &param) : Model("FlashAttentionModel", param)
    {
        param_.FromString(param);
    }

    EncoderModel::~EncoderModel() = default;

    uint32_t EncoderModel::GetInputNum() const
    {
        return graph_.inTensors.size();
    }

    uint32_t EncoderModel::GetOutputNum() const
    {
        return graph_.outTensors.size();
    }

    atb::Status EncoderModel::InferShape(
        const std::vector<atb::TensorDesc> &inTensorDescs,
        std::vector<atb::TensorDesc> &outTensorDescs
    )
    {
        if (outTensorDescs.size() != GetOutputNum()) {
            return atb::ERROR_INVALID_GRAPH;
        }
        const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
        outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID) = graph_.weightTensors.at(WORDEMBEDDING_WEIGHT_ID).desc;
        outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dimNum = OUT_TENSOR_HIDDENSTATES_DIM_NUM;
        CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dimNum);

        size_t inTensorShapeDimIndex = 0;
        size_t outTensorShapeDimIndex = 0;

        outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
            inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
        outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] =
            inTensorDescs.at(IN_TENSOR_INPUTIDS_ID).shape.dims[inTensorShapeDimIndex++];
        outTensorDescs.at(OUT_TENSOR_HIDDENSTATES_ID).shape.dims[outTensorShapeDimIndex++] = outDim;
        ATB_SPEED_LOG_INFO("EncoderModel::InferShape end");

        return atb::NO_ERROR;
    }

    int64_t EncoderModel::Embedding()
    {
        atb::Operation *op = nullptr;
        atb_speed::Model::Node embeddingNode;

        atb_speed::bert::EmbeddingParam embeddingParam;
        embeddingParam.layerNormEps = param_.layerNormEps;
        embeddingParam.layerNormImplMode = param_.layerNormImplMode;
        atb_speed::bert::EmbeddingLayer(embeddingParam, &op);
        embeddingNode.operation.reset(op);
        embeddingNode.inTensors.resize(IN_TENSOR_COUNT + EMBEDDING_WEIGHT_COUNT);
        size_t embInTensorId = 0;
        embeddingNode.inTensors.at(embInTensorId++) = &graph_.inTensors.at(IN_INPUT_IDS);
        embeddingNode.inTensors.at(embInTensorId++) = &graph_.inTensors.at(IN_POSITION_IDS);
        embeddingNode.inTensors.at(embInTensorId++) = &graph_.inTensors.at(IN_TOKENTYPE_IDS);
        for (int embWeightTensorId = 0; embWeightTensorId < EMBEDDING_WEIGHT_COUNT; ++embWeightTensorId) {
            embeddingNode.inTensors.at(embInTensorId++) = &graph_.weightTensors.at(embWeightTensorId);
        }
        embeddingNode.outTensors = { &graph_.internalTensors.at(INTERNAL_TENSOR_EMBEDDING_OUT_ID) };
        graph_.nodes.push_back(embeddingNode);

        return atb::NO_ERROR;
    }

    int64_t EncoderModel::Layer()
    {
        atb::Operation *op = nullptr;

        for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
            atb_speed::Model::Node layerNode;
            atb_speed::bert::EncoderLayerParam layerParam;
            layerParam.dk = param_.dk;
            layerParam.geluApproximate = param_.geluApproximate;
            layerParam.headNum = param_.headNum;
            layerParam.layerNormEps = param_.layerNormEps;
            layerParam.layerNormImplMode = param_.layerNormImplMode;
            atb_speed::bert::EncoderLayer(layerParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation -> GetInputNum());
            size_t layerInTensorId = 0;
            int lastInTensorId = INTERNAL_TENSOR_COUNT_BEFORE_LAYER + layerId - 1;
            layerNode.inTensors.at(layerInTensorId++) = &graph_.internalTensors.at(lastInTensorId);
            for (int layerWeightTensorId = 0; layerWeightTensorId < WEIGHT_COUNT_PER_LAYER; ++layerWeightTensorId) {
                uint64_t tensorId =
                    EMBEDDING_WEIGHT_COUNT +
                    CheckIntMulOverFlow(layerId, WEIGHT_COUNT_PER_LAYER) +
                    layerWeightTensorId;
                layerNode.inTensors.at(layerInTensorId++) = &graph_.weightTensors.at(tensorId);
            }
            layerNode.inTensors.at(layerInTensorId++) = &graph_.inTensors.at(IN_ATTENTIONMASK);
            layerNode.inTensors.at(layerInTensorId++) = &graph_.inTensors.at(IN_PASTKEY);
            layerNode.inTensors.at(layerInTensorId++) = &graph_.inTensors.at(IN_PASTVALUE);
            layerNode.inTensors.at(layerInTensorId++) = &graph_.inTensors.at(IN_TOKENOFFSET);
            layerNode.inTensors.at(layerInTensorId++) = &graph_.inTensors.at(IN_SEQLEN);
            layerNode.inTensors.at(layerInTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
            if (layerId == param_.layerNum - 1) {
                layerNode.outTensors = { &graph_.outTensors.at(OUT_HIDDENSTATES) };
            } else {
                layerNode.outTensors = { &graph_.internalTensors.at(lastInTensorId + 1) };
            }
            graph_.nodes.push_back(layerNode);
        }

        return atb::NO_ERROR;
    }

    int64_t EncoderModel::BuildGraph()
    {
        ATB_SPEED_LOG_INFO(__func__ << " called");
        const uint64_t weightTensorSize =
            EMBEDDING_WEIGHT_COUNT +
            CheckIntMulOverFlow(WEIGHT_COUNT_PER_LAYER, param_.layerNum);
        graph_.weightTensors.resize(weightTensorSize);
        graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
        graph_.outTensors.resize(OUT_TENSOR_MAX);

        const int internalTensorSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum - 1;
        graph_.internalTensors.resize(internalTensorSize);

        CHECK_OPERATION_STATUS_RETURN(Embedding());
        CHECK_OPERATION_STATUS_RETURN(Layer());

        return atb::NO_ERROR;
    }

    atb::Status EncoderModel::ParseParam(const std::string &param)
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

        tokenOffset_.clear();
        for (const auto &item : paramJson["tokenOffset"]) {
            int tokenOffset = item.get<int>();
            CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
            tokenOffset_.push_back(tokenOffset);
        }
        seqLen_.clear();
        for (const auto &item : paramJson["seqLen"]) {
            int seqLen = item.get<int>();
            CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
            seqLen_.push_back(seqLen);
        }
        return atb::NO_ERROR;
    }

    atb::Status EncoderModel::BindParamHostTensor(uint32_t nodeId)
    {
        if (
            nodeId < OPERATION_COUNT_BEFORE_LAYER ||
            nodeId >= static_cast<uint32_t>(OPERATION_COUNT_BEFORE_LAYER + param_.layerNum)
        ) {
            return atb::NO_ERROR;
        }
        auto &node = graph_.nodes.at(nodeId);

        const uint64_t tokenOffsetTensorId = LAYER_IN_TOKENOFFSET_ID;
        const uint64_t seqLenTensorId = LAYER_IN_SEQLEN_ID;

        node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();

        ATB_SPEED_LOG_INFO("EncoderModel::BindParamHostTensor end");
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::bert
