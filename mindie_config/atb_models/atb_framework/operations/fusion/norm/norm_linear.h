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

#ifndef ASCEND_SPEED_INFERENCE_COMMON_NORM_LINEAR_H
#define ASCEND_SPEED_INFERENCE_COMMON_NORM_LINEAR_H

#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {

template <typename NormParamType>
struct NormLinearParam {
    bool isAntiOutlier = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableAddNorm = false;
    NormParamType normParamType;
    NormParamType normQuantParamType;
    atb_speed::common::FusionLinearParam fusionLinearParam;
};

LinearQuantType GetLinearQuantType(const int &packQuantType, const int &linearType, bool hasNorm);
template <typename NormParamType>
atb::Status NormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation);
template <typename NormParamType>
int64_t InsertNorm(atb::GraphParam &opGraph, const NormLinearParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap);

} // namespace common
} // namespace atb_speed

#endif
