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
#ifndef ATB_SPEED_MODELS_QWEN_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_MOE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace qwen {
struct MoeDecoderLayerParam {
    bool isFA = false;
    bool isPrefill = false;
    bool isBF16 = true;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportLcoc = false;
    bool enableTopKSoftmax = false; // whether or not to use integrated aclnn operator
    int quantType = 0;
    float rmsNormEps = 0;
    bool transpose = true;
    int numOfExperts = 60;            // num of total experts
    int expertParallelDegree = 1;
    int maskStartIdx = 0;
    int layerId = 0;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    int rank = 0;
    int worldSize = 1;
    bool hasSharedExpertGate = true;
    std::string routingMethod = "softMaxTopK";
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> packQuantType = {};
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> mlpLinearQuantType = {};
    std::vector<int> moeLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    std::vector<int> mlpLinearTransposeType = {};
    std::vector<int> moeLinearTransposeType = {};
    atb::SVector<int32_t> numOfSelectedExperts = {}; // num of selected experts
};

enum DecoderLayerTensorId : int {
    IN_HIDDEN_STATES = 0,               // shape: FA: [batchSize, seqLen, maxPositionEmbeddings]
    // input_norm
    IN_INPUT_NORM_WEIGHT,               // shape: [hiddenSize]
    IN_INPUT_NORM_BIAS,
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,
    // q
    IN_QKV_WEIGHT_0,                    // Pack: shape:
    IN_QKV_BIAS_0,                  // Quant所需权重
    IN_QKV_DESCALE_0,                   // Quant所需权重
    IN_QKV_OFFSET_0,                    // Quant所需权重
    IN_QKV_SCALE_0,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_0,
    // k
    IN_QKV_WEIGHT_1,                    // Pack: no usage;  No pack: (K)
    IN_QKV_BIAS_1,                  // Quant所需权重
    IN_QKV_DESCALE_1,                   // Quant所需权重
    IN_QKV_OFFSET_1,                    // Quant所需权重
    IN_QKV_SCALE_1,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_1,
    // v
    IN_QKV_WEIGHT_2,                    // Pack: no usage; No pack: (V)
    IN_QKV_BIAS_2,                  // Quant所需权重
    IN_QKV_DESCALE_2,                   // Quant所需权重
    IN_QKV_OFFSET_2,                    // Quant所需权重
    IN_QKV_SCALE_2,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_2,
    // o_proj
    IN_ATTENTION_OUT_WEIGHT,            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_BIAS,          // Quant所需权重
    IN_ATTENTION_OUT_DESCALE,           // Quant所需权重
    IN_ATTENTION_OUT_OFFSET,            // Quant所需权重
    IN_ATTENTION_OUT_SCALE,             // Quant所需权重
    IN_ATTENTION_OUT_COMPRESS_IDX,
    // post_norm
    IN_SELFATTENTION_OUT_NORM_WEIGHT,
    IN_SELFATTENTION_OUT_NORM_BIAS,
    IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT,
    IN_SELFATTENTION_OUT_NEW_NORM_BIAS,
    // gate
    IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
    IN_BLOCK_SPARSE_MOE_GATE_BIAS,
    IN_BLOCK_SPARSE_MOE_GATE_DESCALE,
    IN_BLOCK_SPARSE_MOE_GATE_OFFSET,
    IN_BLOCK_SPARSE_MOE_GATE_SCALE,
    IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX,
    // gate_up
    IN_MLP_GATEUP_WEIGHT_EXPERT,
    IN_MLP_GATEUP_BIAS_EXPERT,
    IN_MLP_GATEUP_DESCALE_EXPERT,
    IN_MLP_GATEUP_OFFSET_EXPERT,
    IN_MLP_GATEUP_SCALE_EXPERT,
    IN_MLP_GATEUP_COMPRESS_IDX_EXPERT,
    // down
    IN_MLP_DOWN_WEIGHT_EXPERT,
    IN_MLP_DOWN_BIAS_EXPERT,
    IN_MLP_DOWN_DESCALE_EXPERT,
    IN_MLP_DOWN_OFFSET_EXPERT,
    IN_MLP_DOWN_SCALE_EXPERT,
    IN_MLP_DOWN_COMPRESS_IDX_EXPERT,
    // shared_expert_gate_up
    IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
    IN_MLP_GATEUP_BIAS_SHARED_EXPERT,
    IN_MLP_GATEUP_DESCALE_SHARED_EXPERT,
    IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
    IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
    IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT,
    // shared_expert_down
    IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
    IN_MLP_DOWN_BIAS_SHARED_EXPERT,
    IN_MLP_DOWN_DESCALE_SHARED_EXPERT,
    IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
    IN_MLP_DOWN_SCALE_SHARED_EXPERT,
    IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT,
    // shared_expert_gate
    IN_MLP_GATE_WEIGHT_SHARED_EXPERT,
    IN_MLP_GATE_BIAS_SHARED_EXPERT,
    IN_MLP_GATE_DESCALE_SHARED_EXPERT,
    IN_MLP_GATE_OFFSET_SHARED_EXPERT,
    IN_MLP_GATE_SCALE_SHARED_EXPERT,
    IN_MLP_GATE_COMPRESS_IDX_SHARED_EXPERT,
    // input_tensor
    IN_EXPERT_ARRAY,
    IN_EXPERT_GROUP,
    IN_ONE_HOT,
    IN_ZERO_HOT,
    IN_COS_TABLE,                       // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead]
    IN_SIN_TABLE,                       // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead]
    IN_ATTENTION_MASK,                  // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    IN_K_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings,
    IN_V_CACHE,                         // shape: FA: [batchSize, maxPositionEmbeddings,
    IN_SEQ_LEN,                         // shape: [batchSize]
    IN_PLACE_HOLDER,                    // shape: [1]
    IN_TOKEN_OFFSET,                    // shape: [batchSize]; FA所需参数
    IN_LAYER_ID,                        // shape: [1]; FA所需参数
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PA所需参数
    IN_SLOTS,                           // shape: [seqLen]; PA所需参数
    // out_tensor
    OUT_DECODER_LAYER,                  // shape: FA: [batchSize, seqLen, maxPositionEmbeddings]
    // internal_tensor
    INTERMEDIATE_ATTENTION_OUT,
    INTERMIDATE_SELFATTENTION_NORM_OUT,
    INTERMIDATE_MOE_OUT,
    INTERMEDIATE_SHARE_EXPERTS_OUT,
    INTERMEDIATE_MLP_OUT,
};

atb::Status MoeDecoderLayer(const MoeDecoderLayerParam &param, atb::Operation **operation);

class MoeDecoderLayer : public HostTensorBinder {
public:
    MoeDecoderLayer();
    ~MoeDecoderLayer() override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
}  // namespace qwen
}  // namespace atb_speed
#endif