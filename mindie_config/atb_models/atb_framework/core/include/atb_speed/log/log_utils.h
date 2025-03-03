/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#ifndef ATB_SPEED_LOG_UTILS_H
#define ATB_SPEED_LOG_UTILS_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include "atb_speed/log/log_config.h"

namespace atb_speed {

class LogUtils {
public:
    static bool SetMindieEnvFlag(bool& envLogFlag, const std::string& envVar);

    static LogLevel SetMindieEnvLevel(LogLevel& envLogLevel, const std::string& envVar);

    static std::unordered_map<std::string, std::string> ParseEnvToDict(
        const std::string& mindieEnv,
        const std::unordered_set<std::string>& validKeys = {});

    static std::string SetMindieLogPath(const std::string& envVar);

    static std::string SetMindieLogRotate(const std::string& envVar);
    static std::string Trim(std::string str);

    static std::vector<std::string> Split(const std::string &str, char delim = ' ');

    static void UpdateLogFileParam(
        std::string rotateConfig, uint32_t& maxFileSize, uint32_t& maxFiles);
};

} // namespace atb_speed

#endif // ATB_SPEED_LOG_UTILS_H