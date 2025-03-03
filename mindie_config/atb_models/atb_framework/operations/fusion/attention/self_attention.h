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

#ifndef ATB_SPEED_MODELS_COMMON_SELF_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_SELF_ATTENTION_H

#include <map>
#include <atb/atb_infer.h>
#include "operations/fusion/attention/fusion_attention.h"

namespace atb_speed {
namespace common {

template <typename NormParamType>
int64_t AddSelfAttention(
    atb::GraphParam &opGraph, const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap);
    
} // namespace common
} // namespace atb_speed
#endif