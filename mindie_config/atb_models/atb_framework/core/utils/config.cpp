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
#include "atb_speed/utils/config.h"
#include <string>
#include <iostream>
#include <thread>
#include <atb_speed/utils/match.h>
#include <atb_speed/utils/str_split.h>
#include "atb_speed/log.h"

namespace atb_speed {
Config::Config()
{
    isConvertNCHWToND_ = true;
    isTorchTensorFormatCast_ = true;
    isUseTilingCopyStream_ = IsEnable("ATB_USE_TILING_COPY_STREAM");
    isLayerInternalTensorReuse_ = true;
    ATB_SPEED_LOG_DEBUG(" \nIsConvertNCHWToND:" << isConvertNCHWToND_
                   << "\nIsTorchTensorFormatCast:" << isTorchTensorFormatCast_
                   << "\nIsLayerInternalTensorReuse:" << isLayerInternalTensorReuse_);
}

Config::~Config() {}

bool Config::IsEnable(const char *env, bool enable)
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsTorchTensorFormatCast() const { return isTorchTensorFormatCast_; };

bool Config::IsConvertNCHWToND() const { return isConvertNCHWToND_; }

bool Config::IsUseTilingCopyStream() const { return isUseTilingCopyStream_; }

bool Config::IsLayerInternalTensorReuse() const
{
    return isLayerInternalTensorReuse_;
}
} // namespace atb_speed