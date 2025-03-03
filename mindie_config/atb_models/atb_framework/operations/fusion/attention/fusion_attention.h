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

#ifndef ATB_SPEED_MODELS_COMMON_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_ATTENTION_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/aclnn/ops/attn_operation.h"

namespace atb_speed {
namespace common {

enum AttnInTensorCategory : unsigned int {
    ATTN_DEFAULT = 0,
    ATTN_ALIBI_MASK_COMPRESS,
    ATTN_COMPRESS_HEAD_ALIBI,
    ATTN_COMPRESS_HEAD_ROPE,
    ATTN_SPECULATE,
    ATTN_KV_QUANT_SCALE,
    ATTN_KV_QUANT_OFFSET,
    ATTN_FA3,
    ATTN_LORA_MASK,
    ATTN_LORA,
    ATTN_REDUCE_QUANT,
    ATTN_LOG_N_SCALE,
    ATTN_END
};

const uint64_t Q_LINEAR_INDEX = 0;
const uint64_t K_LINEAR_INDEX = 1;
const uint64_t V_LINEAR_INDEX = 2;
const uint64_t DENSE_LINEAR_INDEX = 3;

template <typename NormParamType>
struct FusionAttentionParam {
    // QKV linear param
    bool isGroupedQueryAttention = false;
    bool isBF16 = false;
    bool splitWithStride = false;
    bool qkvHasBias = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableNormQuantOp = true;
    bool supportLora = false;
    bool useImMask = false;
    bool loraEnableGMM = false;
    int attnBackend = atb_speed::common::OpBackend::ATB;
    int quantGroupSize = 0;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    std::vector<int> layerLinearQuantType = {};
    std::vector<int> layerLinearTransposeType = {};
    NormParamType normParamType;
    NormParamType normQuantParamType;
    // rope param
    atb_speed::common::RotaryType rotaryType;
    atb::infer::RopeParam ropeParam;
    // self attention param
    bool enableLogN = false;
    bool enableQScale = false;
    bool enableSplitFuse = false;
    bool isFA = true;
    bool isPrefill = false;
    int headDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    atb_speed::common::AclNNAttnParam aclnnIncreAttentionParam;
    // self out linear param
    bool selfAttnHasBias = false;
    bool supportLcoc = false;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
};

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(const FusionAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif