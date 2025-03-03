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

#ifndef ATB_SPEED_LOG_H
#define ATB_SPEED_LOG_H

#include <iostream>
#include <sstream>
#include <mutex>

#include "spdlog/common.h"
#include "spdlog/spdlog.h"

#include "log/log_config.h"

namespace atb_speed {

const std::map<std::string, std::string> ERROR_CODE_MAPPING = {
    // BACKEND
    {"BACKEND_CONFIG_VAL_FAILED", "MIE05E020000"},
    {"BACKEND_INIT_FAILED", "MIE05E020001"},
    // LLM_MANAGER
    {"LLM_MANAGER_CONFIG_FAILED", "MIE05E030000"},
    {"LLM_MANAGER_INIT_FAILED", "MIE05E030001"},
    // TEXT_GENERATOR
    {"TEXT_GENERATOR_PLUGIN_NAME_INVALID", "MIE05E010000"},
    {"TEXT_GENERATOR_FEAT_COMPAT_INVALID", "MIE05E010001"},
    {"TEXT_GENERATOR_REQ_ID_INVALID", "MIE05E010002"},
    {"TEXT_GENERATOR_TEMP_ZERO_DIV_ERR", "MIE05E010003"},
    {"TEXT_GENERATOR_REQ_PENALTY_ZERO_DIV_ERR", "MIE05E010004"},
    {"TEXT_GENERATOR_ZERO_ITER_ERR", "MIE05E010005"},
    {"TEXT_GENERATOR_ZERO_TIME_ERR", "MIE05E010006"},
    {"TEXT_GENERATOR_REQ_ID_UNUSED", "MIE05E010007"},
    {"TEXT_GENERATOR_GENERATOR_BACKEND_INVALID", "MIE05E010008"},
    {"TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH", "MIE05E010009"},
    {"TEXT_GENERATOR_MISSING_PREFILL_OR_INVALID_DECODE_REQ", "MIE05E010010"},
    {"TEXT_GENERATOR_MAX_BLOCK_SIZE_INVALID", "MIE05E010011"},
    {"TEXT_GENERATOR_EOS_TOKEN_ID_TYPE_INVALID", "MIE05E010012"},
    // ATB_MODELS
    {"ATB_MODELS_PARAM_OUT_OF_RANGE", "MIE05E000000"},
    {"ATB_MODELS_MODEL_PARAM_JSON_INVALID", "MIE05E000001"},
    {"ATB_MODELS_EXECUTION_FAILURE", "MIE05E000002"},
};

using LogLevel = spdlog::level::level_enum;

class Log {
public:
    static std::shared_ptr<Log> GetInstance();

    static int CreateInstance();

    static int CreateInstance(const std::string& loggerName, const std::shared_ptr<LogConfig> logConfig);

    static const std::shared_ptr<LogConfig> GetLogConfig();

    static int SetLogConfig(const std::shared_ptr<LogConfig> logConfig);

    static void LogMessage(LogLevel level, const std::string& prefix, const std::string& message,
                           bool validate = true);

    static void Flush();

    static void GetErrorCode(std::ostringstream& oss, const std::string& args);

    static void FormatLog(std::ostringstream& oss, LogLevel level);

    ~Log() = default;

private:

    explicit Log(const std::shared_ptr<LogConfig> logConfig);

    void SetHandlesCallback(spdlog::file_event_handlers &handlers);
    int Initialize(const std::string &loggerName);

private:
    static std::once_flag initFlag;
    static std::shared_ptr<Log> logger;
    static std::shared_ptr<Log> audiLogger;

    std::shared_ptr<spdlog::logger> innerLogger_;
    std::shared_ptr<LogConfig> logConfig_;
};

}  // namespace atb_speed

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto& el : vec) {
        os << el << ',';
    }
    return os;
}

#ifndef LOG_FILENAME
#define LOG_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define ATB_SPEED_LOG(level, msg, ...)                                                                      \
    do {       \
        if (atb_speed::Log::GetInstance() != nullptr &&                     \
            atb_speed::Log::GetLogConfig()->logLevel_ <= (level)) {         \
            std::ostringstream oss;                                          \
            ATB_SPEED_FORMAT_LOG(oss, level, msg, ##__VA_ARGS__);           \
            atb_speed::Log::LogMessage(level, "{}", oss.str(), false);      \
        }                                                                    \
    } while (0)

#define ATB_SPEED_FORMAT_LOG(oss, level, msg, ...)                           \
    do {                                                                     \
        atb_speed::Log::FormatLog(oss, level);                              \
        oss << "[" << LOG_FILENAME << ":" << __LINE__ << "] ";              \
        atb_speed::Log::GetErrorCode(oss, #__VA_ARGS__);                     \
        oss << msg;                                                          \
    } while (0)

#define ATB_SPEED_LOG_DEBUG(msg, ...) ATB_SPEED_LOG(spdlog::level::debug, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_INFO(msg, ...) ATB_SPEED_LOG(spdlog::level::info, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_WARN(msg, ...) ATB_SPEED_LOG(spdlog::level::warn, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_ERROR(msg, ...) ATB_SPEED_LOG(spdlog::level::err, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_FATAL(msg, ...) ATB_SPEED_LOG(spdlog::level::critical, msg, __VA_ARGS__)

#endif