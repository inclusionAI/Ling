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
#include "atb_speed/utils/timer.h"
#include <sys/time.h>

namespace atb_speed {
const uint64_t MICRSECOND_PER_SECOND = 1000000;

Timer::Timer() { startTimepoint_ = GetCurrentTimepoint(); }

Timer::~Timer() {}

uint64_t Timer::ElapsedMicroSecond()
{
    uint64_t now = GetCurrentTimepoint();
    uint64_t use = now - startTimepoint_;
    startTimepoint_ = now;
    return use;
}

void Timer::Reset() { startTimepoint_ = GetCurrentTimepoint(); }

uint64_t Timer::GetCurrentTimepoint() const
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    uint64_t ret =
        static_cast<uint64_t>(tv.tv_sec * MICRSECOND_PER_SECOND + tv.tv_usec);
    return ret;
}
} // namespace atb_speed