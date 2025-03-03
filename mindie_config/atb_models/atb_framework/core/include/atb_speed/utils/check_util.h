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
#ifndef ATB_SPEED_UTILS_CHECK_H
#define ATB_SPEED_UTILS_CHECK_H
#include <vector>
#include <cstddef>
#include <sstream>
#include <limits>

#include "nlohmann/json.hpp"

namespace atb_speed {

using Json = nlohmann::json;

template<typename T, typename U>
typename std::common_type<T, U>::type CheckIntMulOverFlow(const T a, const U b)
{
    if (std::is_signed<T>::value != std::is_signed<U>::value) {
        throw std::runtime_error("Multiplication between signed and unsigned integer not supported, it's not safe");
    }
    using PromotedType = typename std::common_type<T, U>::type;
    if (a == 0 || b == 0) {
        return 0;
    }

    PromotedType pa = static_cast<PromotedType>(a);
    PromotedType pb = static_cast<PromotedType>(b);

    if constexpr (std::is_signed<PromotedType>::value) {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        const PromotedType minVal = std::numeric_limits<PromotedType>::min();
        if (pa > 0 && pb > 0) {
            if (pa > maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < 0 && pb < 0) {
            if (pa < maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa > 0 && pb < 0) {
            if (pa > minVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < minVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    } else {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        if (pa > maxVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    }
    return pa * pb;
}
int CheckParamRange(const int &intParam, int min, int max);
int CheckNumHiddenLayersValid(const int &numHiddenLayers);
int CheckPositive(const int &intParam);
template <typename T>
void CheckLinearParamsSufficient(const std::vector<std::vector<T>> &linearParam, \
    size_t numHiddenLayers, size_t thershold);
void CheckPackQuantParamsSufficient(const std::vector<std::vector<int>> &packQuantType, size_t numHiddenLayers);
void CheckLinearPackParamsSufficient(const std::vector<std::vector<int>> &linearPackType, size_t numHiddenLayers);
void CheckLinearHasBiasSufficient(const std::vector<std::vector<bool>> &linearHasBias, size_t numHiddenLayers);
bool IsWithinRange(const std::string& integerType, const Json& jsonValue);
uint32_t GetIntegerParamValue(const Json& jsonData, const std::string& configName, uint32_t defaultVal);
} // namespace atb_speed
#endif