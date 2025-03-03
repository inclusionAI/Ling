// Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_INTERNLM2_20B_PAGE_ATTENTION_LAYER_H
#define ATB_SPEED_MODELS_INTERNLM2_20B_PAGE_ATTENTION_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"

#include "operations/fusion/linear/linear_parallel.h"


namespace atb_speed {
namespace internlm2_20b {
enum PositionEmbeddingType : uint32_t {
    ROPE = 0,
};

struct DecoderLayerParam {
    bool isFA = true;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isPack = true;
    bool supportSwiGLU = false;
    bool supportLcoc = false;
    bool supportLora = false;
    bool useImMask = false;
    bool kvQuant = false;
    int quantType = 0;
    float rmsNormEps = 0;
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    int numKeyValueHeadsPerRank = 0;
    bool splitWithStride = true;
    PositionEmbeddingType positionEmbeddingType = ROPE;
    atb_speed::common::TensorParallelInfo tensorParallelInfo;
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> packQuantType = {};  // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearQuantType = {};
    std::vector<int> linearTransposeType;
};

enum DecoderLayerTensorIdx : uint32_t {
    // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    IN_HIDDEN_STATES = 0,
    IN_INPUT_NORM_WEIGHT,               // shape: [hiddenSize]
    IN_INPUT_NORM_BIAS,
    IN_INPUT_NORM_NEW_WEIGHT,
    IN_INPUT_NORM_NEW_BIAS,
    // Pack:
    // MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    // GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
    // No pack:
    // (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_WEIGHT_0,
    IN_QKV_BIAS_0,                      // Quant所需权重
    IN_QKV_DESCALE_0,                   // Quant所需权重
    IN_QKV_OFFSET_0,                    // Quant所需权重
    IN_QKV_SCALE_0,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_0,
    // Pack: no usage;
    // No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_WEIGHT_1,
    IN_QKV_BIAS_1,                      // Quant所需权重
    IN_QKV_DESCALE_1,                   // Quant所需权重
    IN_QKV_OFFSET_1,                    // Quant所需权重
    IN_QKV_SCALE_1,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_1,
    // Pack: no usage;
    // No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
    IN_QKV_WEIGHT_2,
    IN_QKV_BIAS_2,                      // Quant所需权重
    IN_QKV_DESCALE_2,                   // Quant所需权重
    IN_QKV_OFFSET_2,                    // Quant所需权重
    IN_QKV_SCALE_2,                     // Quant所需权重
    IN_QKV_COMPRESS_IDX_2,
    // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
    IN_ATTENTION_OUT_WEIGHT,
    IN_ATTENTION_OUT_BIAS,              // Quant所需权重
    IN_ATTENTION_OUT_DESCALE,           // Quant所需权重
    IN_ATTENTION_OUT_OFFSET,            // Quant所需权重
    IN_ATTENTION_OUT_SCALE,             // Quant所需权重
    IN_ATTENTION_OUT_COMPRESS_IDX,
    IN_ATTENTION_NORM_WEIGHT,           // shape: [hiddenSize]
    IN_ATTENTION_NORM_BIAS,
    IN_ATTENTION_NORM_NEW_WEIGHT,
    IN_ATTENTION_NORM_NEW_BIAS,
    // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
    // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_WEIGHT_0,
    IN_MLP_BIAS_0,                      // Quant所需权重
    IN_MLP_DESCALE_0,                   // Quant所需权重
    IN_MLP_OFFSET_0,                    // Quant所需权重
    IN_MLP_SCALE_0,                     // Quant所需权重
    IN_MLP_COMPRESS_IDX_0,
    // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
    IN_MLP_WEIGHT_1,
    IN_MLP_BIAS_1,                      // Quant所需权重
    IN_MLP_DESCALE_1,                   // Quant所需权重
    IN_MLP_OFFSET_1,                    // Quant所需权重
    IN_MLP_SCALE_1,                     // Quant所需权重
    IN_MLP_COMPRESS_IDX_1,
    IN_MLP_DOWN_WEIGHT,                 // shape: [hiddenSize, intermediateSizePerRank]
    IN_MLP_DOWN_BIAS,                   // Quant所需权重
    IN_MLP_DOWN_DESCALE,                // Quant所需权重
    IN_MLP_DOWN_OFFSET,                 // Quant所需权重
    IN_MLP_DOWN_SCALE,                  // Quant所需权重
    IN_MLP_DOWN_COMPRESS_IDX,
    // lora
    IN_QKV_LORA_A_0,
    IN_QKV_LORA_B_0,
    IN_QKV_LORA_A_1,
    IN_QKV_LORA_B_1,
    IN_QKV_LORA_A_2,
    IN_QKV_LORA_B_2,
    IN_DENSE_LORA_A,
    IN_DENSE_LORA_B,
    IN_MLP_LORA_A_0,
    IN_MLP_LORA_B_0,
    IN_MLP_LORA_A_1,
    IN_MLP_LORA_B_1,
    IN_MLP_DOWN_LORA_A,
    IN_MLP_DOWN_LORA_B,
    // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_K_QUANT_SCALE,                  // kv quant
    IN_K_DEQUANT_SCALE,                // kv quant
    IN_V_QUANT_SCALE,                  // kv quant
    IN_V_DEQUANT_SCALE,                // kv quant
    IN_K_QUANT_OFFSET,                 // kv quant
    IN_K_DEQUANT_OFFSET,               // kv quant
    IN_V_QUANT_OFFSET,                 // kv quant
    IN_V_DEQUANT_OFFSET,               // kv quant

    IN_COS_TABLE,
    // shape: FA: [batchSize * seqLen, hiddenSizePerAttentionHead] PA: [seqLen, hiddenSizePerAttentionHead]
    IN_SIN_TABLE,
    // shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings] PA: [seqLen, seqLen]
    IN_ATTENTION_MASK,
    // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    // PA: [blockNum, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_K_CACHE,
    // shape: FA: [batchSize, maxPositionEmbeddings, numKeyValueHeadsPerRank * hiddenSizePerAttentionHead]
    // PA: [2622, hiddenSizePerAttentionHead, numAttentionHeadsPerRank, hiddenSizePerAttentionHead]
    IN_V_CACHE,
    IN_SEQ_LEN,                         // shape: [batchSize]
    IN_PLACE_HOLDER,                    // shape: [1]
    IN_TOKEN_OFFSET,                    // shape: [batchSize]; FA所需参数
    IN_LAYER_ID,                        // shape: [1]; FA所需参数
    IN_BLOCK_TABLES,                    // shape: [seqLen, seqLen]; PA所需参数
    IN_SLOTS,                           // shape: [seqLen]; PA所需参数
    IN_LORA_IM_MASK,                    // shape: [seqLen, 1]; Lora所需参数，首token推理只关注图像部分

    // shape: FA: [batchSize, seqLen, maxPositionEmbeddings] PA: [seqLen, hiddenSize]
    OUT_DECODER_LAYER,

    INTERMEDIATE_ATTENTION_OUT,         // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_RESIDUAL_ADD_OUT,      // shape: PA: [seqLen, hiddenSize]
    INTERMEDIATE_MLP_OUT,               // shape: PA: [seqLen, hiddenSize]
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

}  // namespace internlm2_20b
}  // namespace atb_speed
#endif