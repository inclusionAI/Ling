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

#ifndef ATB_SPEED_MODELS_COMMON_INFER_SHAPE_FUNCTIONS_H
#define ATB_SPEED_MODELS_COMMON_INFER_SHAPE_FUNCTIONS_H

#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"


namespace atb_speed {
namespace common {

void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape);
void UnsqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape, int32_t headNum, int32_t headDim);
void UnsqueezeAxis(const atb::Dims &oldShape, atb::Dims &newShape, int32_t axis);
void SqueezeBatchAndHiddenSize(const atb::Dims& oldShape, atb::Dims& newShape);
void InternlmV2QKVSplit(
    const atb::Dims& oldShape, atb::Dims& newShape, int32_t headNum, int32_t kvHeadNum, int32_t headDim);

} // namespace common
} // namespace atb_speed
#endif