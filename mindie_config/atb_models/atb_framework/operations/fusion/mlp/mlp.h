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
#ifndef ATB_SPEED_MODELS_COMMON_MLP_OPERATION_H
#define ATB_SPEED_MODELS_COMMON_MLP_OPERATION_H
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

enum MlpInTensorCategory : unsigned int {
    MLP_DEFAULT = 0,
    MLP_ADD_NORM,
    MLP_LORA_MASK,
    MLP_LORA,
    MLP_REDUCE_QUANT,
    MLP_END
};

enum MlpPackType : unsigned int {
    GATE_UP_WEIGHT_PACK = 0,
    GATE_UP_WEIGHT_NO_PACK = 1,
    UP_WEIGHT_ONLY = 2,
};

const uint64_t GATE_LINEAR_INDEX = 4;
const uint64_t UP_LINEAR_INDEX = 5;
const uint64_t DOWN_LINEAR_INDEX = 6;

template <typename NormParamType>
struct MlpParam {
    bool isBF16 = false;
    bool isLite = false;
    bool gateUpHasBias = false;
    bool downHasBias = false;
    bool supportLcoc = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableAddNorm = false;
    bool enableNormQuantOp = true;
    bool supportLora = false;
    bool useImMask = false;
    bool loraEnableGMM = false;
    MlpPackType mlpPackType = GATE_UP_WEIGHT_PACK;
    std::vector<int> layerLinearQuantType = {};
    std::vector<int> layerLinearTransposeType = {};
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int quantGroupSize = 0;
    NormParamType normParamType;
    NormParamType normQuantParamType;
    atb::infer::ActivationParam activationParam;
    int downQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    atb_speed::common::TensorParallelInfo downLinearTensorParallelInfo;
};

MlpPackType GetMlpPackType(const int &packQuantType, bool upWeightOnly);
template <typename NormParamType>
atb::Status Mlp(const MlpParam<NormParamType> &param, atb::Operation **operation);
template <typename NormParamType>
atb::Status MlpSwiGLU(const MlpParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif