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

#include "atb/atb_infer.h"
#include "models/bloom/model/bloom_decoder_model.h"

namespace atb_speed {
namespace bloom {

BloomDecoderModel::BloomDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->weightCountWordEmbedding = 3;  // 3: wordembedding weight, first norm weight, first norm bias
    this->weightCountFinalNorm = 2;      // 2: final nrom weight, final norm bias
    this->inTensorCandidates = {
        {"default", {
            "input_ids", "positional_ids", "cosine_table", "sine_table", "attention_mask",
            "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"}
        },
    };
    this->internalTensorCandidates = {
        {"default", {"hidden_states"}},
    };
}


atb::Status BloomDecoderModel::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(this->AddFirstNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}


atb::Status BloomDecoderModel::AddFirstNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node firstNormNode;
    atb::infer::LayerNormParam firstNormParam;
    this->SetFinalNormParam(firstNormParam);  // first/final set param 可以复用
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(firstNormParam, &op));
    firstNormNode.operation.reset(op);
    firstNormNode.inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")),
        &graph_.weightTensors.at(1),
        &graph_.weightTensors.at(2)
    };
    firstNormNode.outTensors = {firstNormNode.inTensors.at(0)};  // 输出原地写在输入上
    graph_.nodes.push_back(firstNormNode);

    return atb::NO_ERROR;
}


atb::Status BloomDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    BloomDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace bloom
} // namespace atb_speed
