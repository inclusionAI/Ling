/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */

#ifndef ATB_SPEED_MODELS_MOE_SHARED_EXPERT_H
#define ATB_SPEED_MODELS_MOE_SHARED_EXPERT_H

#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

constexpr uint64_t SHARED_MOE_GATE_LINEAR_INDEX = 0;
constexpr uint64_t SHARED_MOE_UP_LINEAR_INDEX = 1;
constexpr uint64_t SHARED_MOE_DOWN_LINEAR_INDEX = 2;
constexpr uint64_t SHARED_MOE_SHAREGATE_LINEAR_INDEX = 3;

struct SharedExpertParam {
    bool transposeGateup = true;
    bool transposeDown = false;
    bool hasSharedExpertGate = true;
    bool supportSwiGLU = true;
    bool isBF16 = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    std::vector<int> mlpLinearQuantType = {};
    std::vector<int> mlpLinearTransposeType = {};
};

std::map<std::string, uint32_t> ConstructTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

atb::Status CreateSharedExpertOperation(
    const SharedExpertParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif