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

#ifndef ATB_SPEED_MODELS_COMMON_LATENT_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_LATENT_ATTENTION_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"

namespace atb_speed {
namespace common {

constexpr uint64_t Q_PROJ_A_LINEAR_INDEX = 0;
constexpr uint64_t Q_PROJ_B_LINEAR_INDEX = 1;
constexpr uint64_t KV_PROJ_A_LINEAR_INDEX = 2;
constexpr uint64_t KV_PROJ_B_FOR_Q_LINEAR_INDEX = 3;
constexpr uint64_t KV_PROJ_B_FOR_V_LINEAR_INDEX = 4;
constexpr uint64_t O_LINEAR_INDEX = 5;

template <typename NormParamType>
struct LatentAttentionParam {
    // QKV linear param
    bool isGroupedQueryAttention = false;
    bool isBF16 = false;
    bool splitWithStride = false;
    bool qkvHasBias = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableNormQuantOp = true;
    int quantGroupSize = 0;
    // MLA Param
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;

    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    NormParamType normParamType;
    NormParamType normQuantParamType;
    // rope param
    atb_speed::common::RotaryType rotaryType;
    atb::infer::RopeParam ropeParam;
    // self attention param
    bool enableLogN = false;
    bool isFA = true;
    bool isPrefill = false;
    int headDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    // self out linear param
    bool selfAttnHasBias = false;
    bool enableLcoc = false;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
};

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

template <typename NormParamType>
atb::Status Attention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif