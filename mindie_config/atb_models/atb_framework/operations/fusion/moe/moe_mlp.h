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
#ifndef ATB_SPEED_MODELS_MOE_MLP_OPERATION_H
#define ATB_SPEED_MODELS_MOE_MLP_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
struct MoeMlpParam {
    bool transpose = true;
    bool supportSwiGLU = true;
    int32_t topk = 2;
    int numOfExperts = 8;
    int gmmQuantType = 0;
    std::vector<int> moeLinearQuantType = {};
    bool hasBias = false;
    bool isBF16 = false;
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    bool enableFusedRouting = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
};

atb::Status CreateMoeMlpOperation(const MoeMlpParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif