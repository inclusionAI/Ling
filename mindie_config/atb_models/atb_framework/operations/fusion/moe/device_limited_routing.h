/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_DEVICE_LIMITED_ROUTING_OPERATION_H
#define ATB_SPEED_MODELS_DEVICE_LIMITED_ROUTING_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace deviceLimitedRouting {
struct DeviceLimitedRoutingParam {
    int numOfExperts = 64; // number of experts in total
    int numOfGroups = 8; // number of groups/device in total
    atb::SVector<int32_t> topkGroups = {3}; // number of groups/device selected
};

atb::Status CreateDeviceLimitedRoutingOperation(const DeviceLimitedRoutingParam &param, atb::Operation **operation);

}
} // namespace atb_speed
#endif