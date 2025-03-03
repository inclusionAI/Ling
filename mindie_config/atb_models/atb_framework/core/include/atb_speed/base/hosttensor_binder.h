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
#ifndef ATB_SPEED_HOSTTENSOR_BINDER_H
#define ATB_SPEED_HOSTTENSOR_BINDER_H
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>

namespace atb_speed {
class HostTensorBinder {
public:
    HostTensorBinder() = default;
    virtual ~HostTensorBinder() = default;
    virtual void ParseParam(const nlohmann::json &paramJson) = 0;
    virtual void BindTensor(atb::VariantPack &variantPack) = 0;
};
} // namespace atb_speed
#endif