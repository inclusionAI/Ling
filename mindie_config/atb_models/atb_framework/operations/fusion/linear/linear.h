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
#ifndef ATB_SPEED_MODELS_COMMON_LINEAR_H
#define ATB_SPEED_MODELS_COMMON_LINEAR_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {

struct FusionLinearParam {
    LinearQuantType quantType = NO_QUANT;
    bool isBF16 = false;
    bool hasBias = false;
    bool supportLora = false;
    bool useImMask = false;
    bool loraEnableGMM = false;
    int transposeType = TRANSPOSE;
    int quantGroupSize = 0;
};

atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif