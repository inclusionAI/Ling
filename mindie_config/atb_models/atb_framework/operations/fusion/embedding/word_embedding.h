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
#ifndef ATB_SPEED_MODELS_COMMON_LAYER_WORD_EMBEDDING_H
#define ATB_SPEED_MODELS_COMMON_LAYER_WORD_EMBEDDING_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
struct WordEmbeddingParam {
    bool unpadInputs = false;
    int axis = 0;
    // 若embedding权重按列切分，则需调用all gather算子并传入rank相关的信息
    atb_speed::common::TensorParallelInfo tensorParallelInfo;
};

atb::Status WordEmbedding(const WordEmbeddingParam &param, atb::Operation **operation);
}  // namespace common
}  // namespace atb_speed
#endif
