/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef ATB_SPEED_UTILS_TENSOR_UTIL_H
#define ATB_SPEED_UTILS_TENSOR_UTIL_H
#include <string>
#include <atb/types.h>

namespace atb_speed {
class TensorUtil {
public:
    static std::string TensorToString(const atb::Tensor &tensor);
    static std::string TensorDescToString(const atb::TensorDesc &tensorDesc);
    static uint64_t GetTensorNumel(const atb::Tensor &tensor);
    static uint64_t GetTensorNumel(const atb::TensorDesc &tensorDesc);
    static bool TensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB);
};
} // namespace atb_speed
#endif