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
#include "atb_speed/utils/speed_probe.h"

namespace atb_speed {

bool SpeedProbe::IsReportModelTopoInfo(const std::string &modelName)
{
    (void)modelName;
    return false;
}

void SpeedProbe::ReportModelTopoInfo(const std::string &modelName, const std::string &graph)
{
    (void)modelName;
    (void)graph;
    return;
}

} // namespace atb_speed