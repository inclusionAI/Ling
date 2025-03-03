/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <fcntl.h>
#include <iostream>
#include <unordered_set>

#include "atb_speed/log/log_utils.h"

#include "nlohmann/json.hpp"

#include "atb_speed/log/file_utils.h"
#include "atb_speed/log/log_config.h"
#include "atb_speed/log.h"

namespace atb_speed {

bool LogUtils::SetMindieEnvFlag(bool& envLogFlag, const std::string& envVar)
{
    const std::unordered_set<std::string> validBools = {"true", "false", "1", "0"};
    std::unordered_map<std::string, std::string> logBoolMap =
        LogUtils::ParseEnvToDict(envVar, validBools);
    static std::unordered_map<std::string, bool> printScreenMap = {
        { "0", false },
        { "1", true },
        { "false", false },
        { "true", true },
    };

    auto allCompIt = logBoolMap.find(ALL_COMPONENT_NAME);
    auto compIt = logBoolMap.find(COMPONENT_NAME);
    if (compIt != logBoolMap.end()) {
        envLogFlag = printScreenMap[compIt->second];
        return envLogFlag;
    } else if (allCompIt != logBoolMap.end()) {
        envLogFlag = printScreenMap[allCompIt->second];
        return envLogFlag;
    }
    return false;
}

LogLevel LogUtils::SetMindieEnvLevel(LogLevel& envLogLevel, const std::string& envVar)
{
    const std::unordered_set<std::string> validLevel = {"debug", "info", "warn", "error", "critical"};
    std::unordered_map<std::string, std::string> logBoolMap =
        LogUtils::ParseEnvToDict(envVar, validLevel);
    static std::unordered_map<std::string, LogLevel> logLevelMap = {
        { "off", LogLevel::off }, { "debug", LogLevel::debug },
        { "info", LogLevel::info }, { "warn", LogLevel::warn },
        { "error", LogLevel::err }, { "critical", LogLevel::critical },
    };

    auto allCompIt = logBoolMap.find(ALL_COMPONENT_NAME);
    auto compIt = logBoolMap.find(COMPONENT_NAME);
    if (compIt != logBoolMap.end()) {
        envLogLevel = logLevelMap[compIt->second];
        return envLogLevel;
    } else if (allCompIt != logBoolMap.end()) {
        envLogLevel = logLevelMap[allCompIt->second];
        return envLogLevel;
    }
    return LogLevel::info;
}

std::string LogUtils::SetMindieLogPath(const std::string& envVar)
{
    std::unordered_map<std::string, std::string> logFileMap = LogUtils::ParseEnvToDict(envVar);
    auto allCompIt = logFileMap.find(ALL_COMPONENT_NAME);
    auto compIt = logFileMap.find(COMPONENT_NAME);
    if (compIt != logFileMap.end()) {
        return compIt->second;
    } else if (allCompIt != logFileMap.end()) {
        return allCompIt->second;
    }
    return "";
}

std::unordered_map<std::string, std::string> LogUtils::ParseEnvToDict(
    const std::string& mindieEnv,
    const std::unordered_set<std::string>& validKeys)
{
    std::unordered_map<std::string, std::string> logFlag;
    std::vector<std::string> modules = LogUtils::Split(mindieEnv, ';');

    for (auto& module : modules) {
        module = LogUtils::Trim(module);

        size_t colonPos = module.find(':');
        if (colonPos != std::string::npos) {
            std::string moduleName = module.substr(0, colonPos);
            std::string flag = module.substr(colonPos + 1);

            moduleName = LogUtils::Trim(moduleName);
            flag = LogUtils::Trim(flag);

            std::transform(flag.begin(), flag.end(), flag.begin(), ::tolower);

            if (validKeys.empty() || validKeys.find(flag) != validKeys.end()) {
                logFlag[moduleName] = flag;
            }
        } else {
            std::string moduleLower = module;
            std::transform(moduleLower.begin(), moduleLower.end(), moduleLower.begin(), ::tolower);

            if (validKeys.empty() || validKeys.find(moduleLower) != validKeys.end()) {
                logFlag[ALL_COMPONENT_NAME] = moduleLower;
            }
        }
    }

    return logFlag;
}

std::string LogUtils::Trim(std::string str)
{
    if (str.empty()) {
        std::cout << "str is empty." << std::endl;
        return str;
    }

    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);
    return str;
}

std::vector<std::string> LogUtils::Split(const std::string &str, char delim)
{
    std::vector<std::string> tokens;
    // 1. check empty string
    if (str.empty()) {
        std::cout << "str is empty." << std::endl;
        return tokens;
    }

    auto stringFindFirstNot = [str, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < str.size(); i++) {
            if (str[i] != delim) {
                return i;
            }
        }
        return std::string::npos;
    };

    size_t lastPos = stringFindFirstNot(0);
    size_t pos = str.find(delim, lastPos);
    while (lastPos != std::string::npos) {
        tokens.emplace_back(str.substr(lastPos, pos - lastPos));
        lastPos = stringFindFirstNot(pos);
        pos = str.find(delim, lastPos);
    }
    return tokens;
}

void LogUtils::UpdateLogFileParam(
    std::string rotateConfig, uint32_t& maxFileSize, uint32_t& maxFiles)
{
    if (rotateConfig.empty()) {
        return;
    }
    std::istringstream configStream(rotateConfig);
    std::string option;
    std::string value;

    auto isNumeric = [](const std::string& str) {
        return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
    };

    while (configStream >> option) {
        if (!(configStream >> value)) {
            continue;
        }
        if (option == "-fs" && isNumeric(value)) {
            maxFileSize = static_cast<uint32_t>(std::stoi(value));
        } else if (option == "-r" && isNumeric(value)) {
            maxFiles = static_cast<uint32_t>(std::stoi(value));
        }
    }
}

std::string LogUtils::SetMindieLogRotate(const std::string& envVar)
{
    std::unordered_map<std::string, std::string> logRotateMap = LogUtils::ParseEnvToDict(envVar);
    auto allCompIt = logRotateMap.find(ALL_COMPONENT_NAME);
    auto compIt = logRotateMap.find(COMPONENT_NAME);

    std::string rotateConfig;
    if (compIt != logRotateMap.end()) {
        rotateConfig = compIt->second;
    } else if (allCompIt != logRotateMap.end()) {
        rotateConfig = allCompIt->second;
    } else {
        return "";
    }
    return rotateConfig;
}

} // namespace atb_speed