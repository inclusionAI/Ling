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
#ifndef ATB_SPEED_BASE_PARAM_H
#define ATB_SPEED_BASE_PARAM_H
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace base {
enum PositionEmbeddingType : uint32_t {
    ROPE = 0,
    ALIBI,
    ABSOLUTE,
};

enum NormType : uint32_t {
    RMS_NORM = 0,
    LAYER_NORM,
};

enum HasBias : uint32_t {
    // linearHasBias的四个元素
    QKV_HASBIAS = 0,
    SELFATTENTION_HASBIAS,
    GATEUP_HASBIAS,
    DOWN_HASBIAS,
};

class Param {
public:
    // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
    bool isFA = true;
    // batchSize和seqLen是否合轴
    bool isUnpadInputs = true;
    // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
    bool isPrefill = false;
    // isBF16为true时采用BF16精度; 反之，则采用FP16精度
    bool isBF16 = false;
    // isLite为true时表示使用lite设备
    bool isLite = false;
    // MLP是否使用SwiGLU，若为true时，则使用；反之，使用swish
    bool enableSwiGLU = false;
    // 是否支持通信计算掩盖
    bool enableLcoc = false;
    // 是否并行解码
    bool enableSpeculate = false;
    // 是否RA
    bool enableCompressHead = false;
    // 是否开启split fuse功能
    bool enableSplitFuse = false;
    // 是否支持lora
    bool enableLora = false;
    bool loraEnableGMM = false;
    // 是否使用kv cache int8 量化
    bool enableKvQuant = false;
    bool kvQuantHasOffset = true;
    // 是否使用FA3量化
    bool enableFA3 = false;
    // 是否使用lccl all reduce量化
    bool enableReduceQuant = false;
    // 是否使用AddNorm融合算子 (仅Layer内的残差连接)
    bool enableAddNorm = false;
    // 是否使用Prefix Cache
    bool enablePrefixCache = false;
    atb_speed::common::OpBackend attnBackend = atb_speed::common::OpBackend::ATB;
    PositionEmbeddingType positionEmbeddingType = PositionEmbeddingType::ROPE;
    float normEps = 0;
    NormType normType = NormType::RMS_NORM;
    uint32_t quantGroupSize = 0;
    uint32_t numAttentionHeadsPerRank = 0;
    uint32_t hiddenSizePerAttentionHead = 0;
    uint32_t numKeyValueHeadsPerRank = 0;
    std::string weightQuantType = "";

    Param() {};
    virtual ~Param() {};

    virtual void PrintParam()
    {
        ATB_SPEED_LOG_DEBUG("Param: " << "isFA: " << isFA
                      << ", isUnpadInputs: " << isUnpadInputs
                      << ", isPrefill: " << isPrefill
                      << ", isBF16: " << isBF16
                      << ", isLite: " << isLite
                      << ", enableSwiGLU: " << enableSwiGLU
                      << ", enableLcoc: " << enableLcoc
                      << ", enableSpeculate: " << enableSpeculate
                      << ", enableCompressHead: " << enableCompressHead
                      << ", enableSplitFuse: " << enableSplitFuse
                      << ", enableLora: " << enableLora
                      << ", loraEnableGMM: " << loraEnableGMM
                      << ", enableKvQuant: " << enableKvQuant
                      << ", enableReduceQuant: " << enableReduceQuant
                      << ", enableAddNorm: " << enableAddNorm
                      << ", enablePrefixCache: " << enablePrefixCache
                      << ", attnBackend: " << attnBackend
                      << ", positionEmbeddingType: " << positionEmbeddingType
                      << ", normType: " << normType
                      << ", normEps: " << normEps
                      << ", quantGroupSize: " << quantGroupSize
                      << ", numAttentionHeadsPerRank: " << numAttentionHeadsPerRank
                      << ", hiddenSizePerAttentionHead: " << hiddenSizePerAttentionHead
                      << ", numKeyValueHeadsPerRank: " << numKeyValueHeadsPerRank
                      << ", weightQuantType: " << weightQuantType);
    }
    virtual void CheckParam() {};
};
} // namespace base
} // namespace atb_speed


#endif