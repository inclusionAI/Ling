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

#include "atb_speed/log.h"
#include "operations/aclnn/ops/layer_norm_operation.h"
#include "models/bert/operation/embedding.h"


namespace atb_speed::bert {

    enum EmbeddingLayerTensorId : int {
        // input tensors
        IN_INPUT_IDS = 0,
        IN_POSITION_IDS,
        IN_TOKENTYPE_IDS,
        IN_WORDEMBED_WEIGHT,
        IN_POSEMBED_WEIGHT,
        IN_TOKENTYPEEMBED_WEIGHT,
        IN_EMBEDNORM_WEIGHT,
        IN_EMBEDNORM_BIAS,
        // output tensors
        OUT_EMBED_RESULT,
        // intermediate tensors
        INTERMEDIATE_WORDEMBED_OUT,
        INTERMEDIATE_TOKENTYPEEMBED_OUT,
        INTERMEDIATE_FIRSTADD_OUT,
        INTERMEDIATE_POSEMBED_OUT,
        INTERMEDIATE_LASTADD_OUT
    };

    static const uint64_t IN_TENSOR_COUNT = 8;
    static const uint64_t OUT_TENSOR_COUNT = 1;
    static const uint64_t INTERNAL_TENSOR_COUNT = 5;
    static const uint64_t NODE_COUNT = 6;

    atb::Status EmbeddingLayer(const EmbeddingParam &param, atb::Operation **operation)
    {
        ATB_SPEED_LOG_INFO(__func__ << " called");
        atb::GraphParam opGraph;
        opGraph.name = "EmbeddingLayer";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
        opGraph.nodes.resize(NODE_COUNT);

        size_t nodeId = 0;
        atb::Node &wordEmbeddingNode = opGraph.nodes.at(nodeId++);
        atb::Node &tokenTypeEmbeddingNode = opGraph.nodes.at(nodeId++);
        atb::Node &firstAddNode = opGraph.nodes.at(nodeId++);
        atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
        atb::Node &lastAddNode = opGraph.nodes.at(nodeId++);
        atb::Node &embLayerNormNode = opGraph.nodes.at(nodeId++);

        atb::infer::GatherParam embeddingParam;
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        atb_speed::common::AclNNLayerNormParam embLayerNormParam;
        embLayerNormParam.layerNormEps = param.layerNormEps;
        embLayerNormParam.layerNormImplMode = param.layerNormImplMode;

        // Word Embeddings
        CREATE_OPERATION(embeddingParam, &wordEmbeddingNode.operation);
        wordEmbeddingNode.inTensorIds = { IN_WORDEMBED_WEIGHT, IN_INPUT_IDS };
        wordEmbeddingNode.outTensorIds = { INTERMEDIATE_WORDEMBED_OUT };

        // TokenType Embeddings
        CREATE_OPERATION(embeddingParam, &tokenTypeEmbeddingNode.operation);
        tokenTypeEmbeddingNode.inTensorIds = { IN_TOKENTYPEEMBED_WEIGHT, IN_TOKENTYPE_IDS };
        tokenTypeEmbeddingNode.outTensorIds = { INTERMEDIATE_TOKENTYPEEMBED_OUT };

        // Word Embeddings + TokenType Embeddings
        CREATE_OPERATION(addParam, &firstAddNode.operation);
        firstAddNode.inTensorIds = { INTERMEDIATE_WORDEMBED_OUT, INTERMEDIATE_TOKENTYPEEMBED_OUT };
        firstAddNode.outTensorIds = { INTERMEDIATE_FIRSTADD_OUT };

        // Position Embeddings
        CREATE_OPERATION(embeddingParam, &positionEmbeddingNode.operation);
        positionEmbeddingNode.inTensorIds = { IN_POSEMBED_WEIGHT, IN_POSITION_IDS };
        positionEmbeddingNode.outTensorIds = { INTERMEDIATE_POSEMBED_OUT };

        // Embeddings + Position Embeddings
        CREATE_OPERATION(addParam, &lastAddNode.operation);
        lastAddNode.inTensorIds = { INTERMEDIATE_FIRSTADD_OUT, INTERMEDIATE_POSEMBED_OUT };
        lastAddNode.outTensorIds = { INTERMEDIATE_LASTADD_OUT };

        // Layer Norm
        embLayerNormNode.operation = new atb_speed::common::LayerNormOperation(
            "embLayerNormNode",
            embLayerNormParam
        );
        embLayerNormNode.inTensorIds = { INTERMEDIATE_LASTADD_OUT, IN_EMBEDNORM_WEIGHT, IN_EMBEDNORM_BIAS };
        embLayerNormNode.outTensorIds = { OUT_EMBED_RESULT };

        CREATE_OPERATION(opGraph, operation);
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::bert
