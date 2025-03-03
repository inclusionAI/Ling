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
#ifndef ATB_SPEED_MODELS_COMMON_LMHEAD_H
#define ATB_SPEED_MODELS_COMMON_LMHEAD_H

#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
struct LmHeadParam {
    bool gatherAhead = false;  // Prefill阶段使用gatherAhead，只获取所需的token，以此减少显存占用
    bool unpadInputs = false;
    int hiddenSizePerAttentionHead = 0;  // 当Parallel的类型为ROW PARALLEL时，需要使用此参数对中间tensor进行切分
    atb_speed::common::LinearParallelParam linearParallelParam;
};

atb::Status LmHead(const LmHeadParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif
