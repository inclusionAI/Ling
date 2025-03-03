/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#ifndef ATB_SPEED_LOG_CONFIG_H
#define ATB_SPEED_LOG_CONFIG_H

#include <sys/stat.h>

#include "nlohmann/json.hpp"
#include "spdlog/common.h"

namespace atb_speed {

using Json = nlohmann::json;
using LogLevel = spdlog::level::level_enum;

// Default log settings
const std::string DEFAULT_LOGGER_NAME = "atb";
const bool DEFAULT_LOG_TO_STDOUT = false;
const bool DEFAULT_LOG_TO_FILE = true;
const LogLevel DEFAULT_LOG_LEVEL = LogLevel::info;
const bool DEFAULT_LOG_VERBOSE = true;
const std::string DEFAULT_LOG_PATH = "mindie/log";
constexpr uint32_t DEFAULT_LOG_FILE_COUNT = 20U;
constexpr uint32_t DEFAULT_LOG_FILE_SIZE = 20U; // 20 MB

constexpr char const *COMPONENT_NAME = "llm";
constexpr char const *ALL_COMPONENT_NAME = "*";

// Log setting limits
constexpr size_t MAX_PATH_LENGTH = 4096;
constexpr uint32_t MAX_ROTATION_FILE_COUNT_LIMIT = 64;
constexpr uint32_t MIN_ROTATION_FILE_COUNT_LIMIT = 1;
constexpr uint32_t MAX_ROTATION_FILE_SIZE_LIMIT = 500 * 1024 * 1024; // 500 MB
const mode_t MAX_LOG_DIR_PERM = S_IRWXU | S_IRGRP | S_IXGRP; // 750
const mode_t MAX_OPEN_LOG_FILE_PERM = S_IRUSR | S_IWUSR | S_IRGRP; // 640
const mode_t MAX_CLOSE_LOG_FILE_PERM = S_IRUSR | S_IRGRP; // 440
constexpr int MAX_LOG_LEVEL_LIMIT = 5;

const std::unordered_map<std::string, LogLevel> LOG_LEVEL_MAP {
    { "OFF", LogLevel::off }, { "NONE", LogLevel::off },
    { "TRACE", LogLevel::trace }, { "VERBOSE", LogLevel::trace },
    { "DEBUG", LogLevel::debug }, { "INFO", LogLevel::info },
    { "WARN", LogLevel::warn }, { "WARNING", LogLevel::warn },
    { "ERROR", LogLevel::err }, { "CRITICAL", LogLevel::critical },
};

class LogConfig {
public:
    LogConfig() = default;
    LogConfig(const LogConfig& config);
    LogConfig& operator=(const LogConfig& config) = delete;
    ~LogConfig() = default;

    int Init();
    int ValidateSettings();
    void MakeDirsWithTimeOut(const std::string parentPath) const;

public:
    bool logToStdOut_ = DEFAULT_LOG_TO_STDOUT;
    bool logToFile_ = DEFAULT_LOG_TO_FILE;
    bool logVerbose_ = DEFAULT_LOG_VERBOSE;
    LogLevel logLevel_ = DEFAULT_LOG_LEVEL;
    std::string logFilePath_;
    std::string logRotateConfig_;
    uint32_t logFileSize_ = DEFAULT_LOG_FILE_SIZE * 1024 * 1024; // 1 MB = 1024 KB = 1024 * 1024 B
    uint32_t logFileCount_ = DEFAULT_LOG_FILE_COUNT;

private:
    void InitLogToStdout();
    void InitLogToFile();
    void InitLogLevel();
    void InitLogFilePath();

    bool CheckAndGetLogPath(const std::string& configLogPath);

private:
    Json configJsonData_{};
};

} // namespace atb_speed

#endif // ATB_SPEED_LOG_CONFIG_H