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
#ifndef ATB_SPEED_BASE_LAYER_PARAM_H
#define ATB_SPEED_BASE_LAYER_PARAM_H
#include <vector>
#include <nlohmann/json.hpp>
#include "models/base/param/param.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace base {

class LayerParam : public Param {
public:
    LayerParam() {};
    ~LayerParam() override {};
    void PrintParam() override;
    void CheckParam() override;

    atb_speed::common::TensorParallelInfo tensorParallelInfo;
    std::vector<int> packQuantType = {};
    std::vector<int> linearQuantType = {};
    std::vector<int> linearTransposeType = {};
    std::vector<bool> linearHasBias = {false, false, false, false};
};
} // namespace base
} // namespace atb_speed
#endif