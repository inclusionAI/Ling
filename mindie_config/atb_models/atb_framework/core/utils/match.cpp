/**
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
#include "atb_speed/utils/match.h"
#include <cstring>

namespace atb_speed {
bool StartsWith(const std::string &text, const std::string &prefix)
{
    if (text.data() == nullptr || prefix.data() == nullptr) {
        return false;
    }
    return prefix.empty() || (text.size() >= prefix.size() && memcmp(text.data(), prefix.data(), prefix.size()) == 0);
}

bool EndsWith(const std::string &text, const std::string &suffix)
{
    if (text.data() == nullptr || suffix.data() == nullptr) {
        return false;
    }
    return suffix.empty() || (text.size() >= suffix.size() &&
                              memcmp(text.data() + (text.size() - suffix.size()), suffix.data(), suffix.size()) == 0);
}
} // namespace atb_speed
