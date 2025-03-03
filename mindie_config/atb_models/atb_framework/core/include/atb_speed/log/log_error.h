/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#ifndef ATB_SPEED_LOG_ERROR_H
#define ATB_SPEED_LOG_ERROR_H

namespace atb_speed {

constexpr int LOG_OK = 0;
constexpr int LOG_CREATE_LOGGER_FAILED = 1;
constexpr int LOG_CREATE_INNER_LOGGER_FAILED = 2;
constexpr int LOG_INVALID_PARAM = 3;
constexpr int LOG_NOT_INIT = 4;
constexpr int LOG_SET_PARAM_FAILED = 5;
constexpr int LOG_CONFIG_INIT_FAILED = 6;

} // namespace atb_speed

#endif // ATB_SPEED_LOG_ERROR_H
