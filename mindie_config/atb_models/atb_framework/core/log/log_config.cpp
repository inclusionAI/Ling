/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "atb_speed/log/log_config.h"

#include <fcntl.h>
#include <iostream>
#include <unordered_set>

#include "nlohmann/json.hpp"
#include "spdlog/details/os.h"

#include "atb_speed/log/file_utils.h"
#include "atb_speed/log/log_utils.h"
#include "atb_speed/utils/check_util.h"
#include "atb_speed/utils/file_system.h"
#include "atb_speed/log/log_error.h"

namespace atb_speed {

LogConfig::LogConfig(const LogConfig& config)
    : logToStdOut_(config.logToStdOut_),
      logToFile_(config.logToFile_),
      logLevel_(config.logLevel_),
      logFilePath_(config.logFilePath_),
      logFileSize_(config.logFileSize_),
      logFileCount_(config.logFileCount_) {}

int LogConfig::Init()
{
    InitLogToStdout();
    InitLogToFile();
    InitLogLevel();
    InitLogFilePath();
    return LOG_OK;
}

void LogConfig::InitLogToStdout()
{
    const char *mindieLogsToStdout = std::getenv("MINDIE_LOG_TO_STDOUT");
    if (mindieLogsToStdout != nullptr) {
        logToStdOut_ = LogUtils::SetMindieEnvFlag(logToStdOut_, mindieLogsToStdout);
        return;
    }
    const char *envToStdout = std::getenv("ATB_LOG_TO_STDOUT");
    // Avoid race conditions:
    std::string toStdOut;
    if (envToStdout != nullptr) {
        toStdOut = envToStdout;
        logToStdOut_ = (toStdOut == "1");
        return;
    }
}

void LogConfig::InitLogToFile()
{
    const char *mindieLogsToFile = std::getenv("MINDIE_LOG_TO_FILE");
    if (mindieLogsToFile != nullptr) {
        logToFile_ = LogUtils::SetMindieEnvFlag(logToFile_, mindieLogsToFile);
        return;
    }
    const char *envToFile = std::getenv("ATB_LOG_TO_FILE");
    // Avoid race conditions:
    std::string toFile;
    if (envToFile != nullptr) {
        toFile = envToFile;
        logToFile_ = (toFile == "1");
        return;
    }
}

void LogConfig::InitLogLevel()
{
    const char *mindieLogsLevel = std::getenv("MINDIE_LOG_LEVEL");
    if (mindieLogsLevel != nullptr) {
        logLevel_ = LogUtils::SetMindieEnvLevel(logLevel_, mindieLogsLevel);
        return;
    }
    const char *envLevel = std::getenv("ATB_LOG_LEVEL");
    if (envLevel != nullptr) {
        std::string envLevelStr(envLevel);
        std::transform(envLevelStr.begin(), envLevelStr.end(), envLevelStr.begin(), ::toupper);
        auto iter = LOG_LEVEL_MAP.find(envLevelStr);
        if (iter != LOG_LEVEL_MAP.end()) {
            logLevel_ =  iter->second;
            return;
        }
    }
}

void LogConfig::InitLogFilePath()
{
    const char *mindieLogPath = std::getenv("MINDIE_LOG_PATH");
    if (mindieLogPath != nullptr) {
        std::string envLogDir = LogUtils::SetMindieLogPath(mindieLogPath);
        if (envLogDir[0] != '/') {
            logFilePath_ = DEFAULT_LOG_PATH + "/" + envLogDir + "/debug";
        } else {
            logFilePath_ = envLogDir + "/debug";
        }
    } else {
        logFilePath_ = DEFAULT_LOG_PATH + "/debug";
    }
    int pid = spdlog::details::os::pid();
    std::time_t now = std::time(nullptr);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", std::localtime(&now));
    logFilePath_ += "/mindie-llm_atbmodels_c_" + std::to_string(pid) + "_" + std::string(buffer) + ".log";
}

void LogConfig::MakeDirsWithTimeOut(const std::string parentPath) const
{
    uint32_t limitTime = 500;
    auto start = std::chrono::steady_clock::now();
    std::chrono::milliseconds timeout(limitTime);
    while (!FileSystem::Exists(parentPath)) {
        auto it = FileSystem::Makedirs(parentPath, MAX_LOG_DIR_PERM);
        if (it) {
            break;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            std::cout << "Create dirs failed : timed out!" << std::endl;
            break;
        }
    }
}

int LogConfig::ValidateSettings()
{
    if (!logToFile_) {
        return LOG_OK;
    }
    if (!CheckAndGetLogPath(logFilePath_)) {
        std::cout << "Cannot get the log path." << std::endl;
        return LOG_INVALID_PARAM;
    }
    if (logFileSize_ > MAX_ROTATION_FILE_SIZE_LIMIT) {
        std::cout << "Invalid max file size, which should be smaller than "
            << MAX_ROTATION_FILE_SIZE_LIMIT << " bytes." << std::endl;
        return LOG_INVALID_PARAM;
    }
    if (logFileCount_ > MAX_ROTATION_FILE_COUNT_LIMIT ||
        logFileCount_ < MIN_ROTATION_FILE_COUNT_LIMIT) {
        std::cout << "Invalid max file count, which should be greater than " <<
                     MIN_ROTATION_FILE_COUNT_LIMIT << " and less than " << MAX_ROTATION_FILE_COUNT_LIMIT;
        return LOG_INVALID_PARAM;
    }
    return LOG_OK;
}

bool LogConfig::CheckAndGetLogPath(const std::string& configLogPath)
{
    if (configLogPath.empty()) {
        std::cout << "The path of log in config is empty." << std::endl;
        return false;
    }

    std::string filePath = configLogPath;
    std::string baseDir = "/";
    if (configLogPath[0] != '/') { // The configLogPath is relative.
        std::string homePath;
        if (!GetHomePath(homePath).IsOk()) {
            std::cout << "Failed to get home path." << std::endl;
            return false;
        }
        baseDir = homePath;
        filePath = homePath + "/" + configLogPath;
    }

    if (filePath.length() > MAX_PATH_LENGTH) {
        std::cout << "The path of log is too long: " << filePath << std::endl;
        return false;
    }
    size_t lastSlash = filePath.rfind('/', filePath.size() - 1);
    if (lastSlash == std::string::npos) {
        std::cout << "The form of logPath is invalid: " << filePath << std::endl;
        return false;
    }

    std::string parentPath = filePath.substr(0, lastSlash);
    std::string errMsg;

    MakeDirsWithTimeOut(parentPath);

    if (!FileUtils::IsFileValid(parentPath.c_str(), errMsg, true, MAX_LOG_DIR_PERM, MAX_ROTATION_FILE_SIZE_LIMIT)) {
        throw std::runtime_error(errMsg);
    }

    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT, MAX_OPEN_LOG_FILE_PERM);
    if (fd == -1) {
        throw std::runtime_error("Creating log file error: " + filePath);
    }
    close(fd);

    if (!FileUtils::RegularFilePath(filePath, baseDir, errMsg, true) ||
        !FileUtils::IsFileValid(filePath, errMsg, false, MAX_OPEN_LOG_FILE_PERM, MAX_ROTATION_FILE_SIZE_LIMIT)) {
        std::cerr << errMsg << std::endl;
        return false;
    }
    logFilePath_ = filePath;
    return true;
}

} // namespace atb_speed
