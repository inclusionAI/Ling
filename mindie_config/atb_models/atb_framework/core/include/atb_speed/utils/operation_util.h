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
#ifndef ATB_SPEED_UTILS_OPERATION_H
#define ATB_SPEED_UTILS_OPERATION_H
#include <atb/atb_infer.h>

namespace atb_speed {
#define CREATE_OPERATION(param, operation) \
    do { \
        atb::Status atbStatus = atb::CreateOperation(param, operation); \
        if (atbStatus != atb::NO_ERROR) { \
            return atbStatus; \
        } \
    } while (0)

#define CHECK_OPERATION_STATUS_RETURN(atbStatus) \
    do { \
        if ((atbStatus) != atb::NO_ERROR) { \
            return (atbStatus); \
        } \
    } while (0)

#define CHECK_PARAM_LT(param, thershold) \
    do { \
        if ((param) >= (thershold)) { \
            ATB_SPEED_LOG_ERROR("param should be less than " << (thershold) << ", please check"); \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_PARAM_GT(param, thershold) \
    do { \
        if ((param) <= (thershold)) { \
            ATB_SPEED_LOG_ERROR("param should be greater than " << (thershold) << ", please check"); \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_PARAM_NE(param, value) \
    do { \
        if ((param) == (value)) { \
            ATB_SPEED_LOG_ERROR("param should not be equal to " << (value) << ", please check"); \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define CHECK_TENSORDESC_DIMNUM_VALID(dimNum) \
    do { \
        if ((dimNum) > (8) || (dimNum) == (0) ) { \
            ATB_SPEED_LOG_ERROR("dimNum should be less or equal to 8 and cannot be 0, please check"); \
            return atb::ERROR_INVALID_PARAM; \
        } \
    } while (0)
} // namespace atb_speed
#endif