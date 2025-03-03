/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#include "atb_speed/log.h"

#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <regex>
#include <stdexcept>

#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/details/os.h"

#include "atb_speed/log/log_config.h"
#include "atb_speed/log/log_error.h"
#include "atb_speed/log/file_utils.h"

namespace atb_speed {

std::once_flag Log::initFlag{};
std::shared_ptr<Log> Log::logger{};
std::shared_ptr<Log> Log::audiLogger{};

Log::Log(const std::shared_ptr<LogConfig> logConfig)
    : logConfig_(logConfig ? std::make_shared<LogConfig>(*logConfig) : nullptr) {}

std::shared_ptr<Log> Log::GetInstance()
{
    std::call_once(initFlag, [] { CreateInstance(); });
    return logger;
}

int Log::CreateInstance()
{
    if (logger != nullptr) {
        return LOG_OK;
    }
    std::shared_ptr<LogConfig> logConfig = std::make_shared<LogConfig>();
    if (!logConfig || logConfig->Init() != LOG_OK) {
        throw std::runtime_error("ATB failed to init the logConfig.");
    }
    int ret = Log::CreateInstance(DEFAULT_LOGGER_NAME, logConfig);
    if (ret != LOG_OK) {
        throw std::runtime_error("ATB create log instance fail.");
    }
    return LOG_OK;
}

int Log::CreateInstance(const std::string& loggerName, const std::shared_ptr<LogConfig> logConfig)
{
    if (logger != nullptr) {
        return LOG_OK;
    }

    if (logConfig == nullptr) {
        return LOG_INVALID_PARAM;
    }
    int ret = logConfig->ValidateSettings();
    if (ret != LOG_OK) {
        throw std::runtime_error("ATB log params validation failed.");
    }

    auto loggerPtr = new Log(logConfig);
    logger = std::shared_ptr<Log>(loggerPtr);
    if (logger == nullptr) {
        throw std::runtime_error("ATB failed to create logger.");
    }
    ret = logger->Initialize(loggerName);
    if (ret != LOG_OK) {
        throw std::runtime_error("ATB failed to initialize inner logger.");
    }
    return LOG_OK;
}

const std::shared_ptr<LogConfig> Log::GetLogConfig()
{
    if (logger != nullptr) {
        return logger->logConfig_;
    }
    throw std::runtime_error("ATB logger is null, failed to get LogConfig.");
}

int Log::SetLogConfig(const std::shared_ptr<LogConfig> logConfig)
{
    if (!logger) {
        return LOG_NOT_INIT;
    }
    if (logConfig == nullptr) {
        return LOG_INVALID_PARAM;
    }
    int ret = logConfig->ValidateSettings();
    if (ret != 0) {
        throw std::runtime_error("ATB failed to validate logConfig.");
    }
    logger->logConfig_ = std::make_shared<LogConfig>(*logConfig);
    return LOG_OK;
}

void Log::LogMessage(LogLevel level, const std::string& prefix, const std::string& message, bool validate)
{
    if (logger == nullptr || logger->innerLogger_ == nullptr) {
        throw std::runtime_error("ATB logger is null.");
    }
    if (validate && (prefix.empty() || message.empty() || level < 0 || level > MAX_LOG_LEVEL_LIMIT)) {
        throw std::runtime_error("ATB invalid log params.");
    }

    logger->innerLogger_->log(level, "{}", message.c_str());
}

void Log::SetHandlesCallback(spdlog::file_event_handlers &handlers)
{
    handlers.after_open = [](spdlog::filename_t filename, const std::FILE *fstream) {
        std::string baseDir = "/";
        std::string errMsg;
        if (!FileUtils::RegularFilePath(filename, baseDir, errMsg, true)) {
            std::cerr << "Regular file failed by " << errMsg << std::endl;
            throw std::runtime_error("atb models Regular log file path failed");
        }
        chmod(filename.c_str(), S_IRUSR | S_IWUSR | S_IRGRP);
        (void) fstream;
    };
    handlers.before_close = [](spdlog::filename_t filename, const std::FILE *fstream) {
        std::string baseDir = "/";
        std::string errMsg;
        if (!FileUtils::RegularFilePath(filename, baseDir, errMsg, true)) {
            std::cerr << "Regular file failed by " << errMsg << std::endl;
            throw std::runtime_error("atb models Regular log file path failed");
        }
        chmod(filename.c_str(), S_IRUSR | S_IRGRP);
        (void) fstream;
    };
}

int Log::Initialize(const std::string &loggerName)
{
    std::vector<spdlog::sink_ptr> sinks;
    try {
        if (logConfig_->logToStdOut_) {
            auto stdoutSink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
            stdoutSink->set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%t] [llm] [%l] %v");
            sinks.push_back(stdoutSink);
        }
        if (logConfig_->logToFile_) {
            spdlog::file_event_handlers handlers;
            SetHandlesCallback(handlers);
            auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                logConfig_->logFilePath_, logConfig_->logFileSize_, logConfig_->logFileCount_, false, handlers);
            sinks.push_back(fileSink);
            spdlog::flush_every(std::chrono::seconds(1));
        }
        innerLogger_ = std::make_shared<spdlog::logger>(loggerName, sinks.begin(), sinks.end());
        spdlog::register_logger(innerLogger_);

        innerLogger_->set_level(static_cast<spdlog::level::level_enum>(logConfig_->logLevel_));
        innerLogger_->set_pattern("%v");
        innerLogger_->info("", "");
        innerLogger_->set_pattern("%Y-%m-%d %H:%M:%S.%f %t %v");
        innerLogger_->info(
            "LLM log default format: [yyyy-mm-dd hh:mm:ss.uuuuuu]"
            "[processid] [threadid] [llm] [loglevel]"
            "[file:line] [error code] msg"
        );
        innerLogger_->set_pattern("[%Y-%m-%d %H:%M:%S.%f] %v");
        innerLogger_->flush_on(spdlog::level::err);
    } catch (const spdlog::spdlog_ex& e) {
        std::stringstream errMsg;
        errMsg << "Failed to create inner logger: " << e.what();
        throw std::runtime_error(errMsg.str());
    }
    return LOG_OK;
}

void Log::Flush()
{
    if (logger != nullptr) {
        logger->innerLogger_->flush();
    }
    if (audiLogger != nullptr) {
        audiLogger->innerLogger_->flush();
    }
}

void Log::GetErrorCode(std::ostringstream& oss, const std::string& args)
{
    std::ostringstream errorcode;
    errorcode << args;
    std::string errorcodeStr = errorcode.str();
    if (errorcodeStr.size() > 0) {
        auto it = ERROR_CODE_MAPPING.find(errorcodeStr);
        if (it != ERROR_CODE_MAPPING.end()) {
            oss << "[" << it->second << "] ";
        } else {
            std::cout << "ErrorCode not found in errorCodeMap!" << std::endl;
        }
    }
}

void Log::FormatLog(std::ostringstream& oss, LogLevel level)
{
    int pid = spdlog::details::os::pid();
    auto tid = std::this_thread::get_id();
    oss << "[" << std::to_string(pid) << "] [" << tid << "] [llm] [";
    std::string levelUpper = spdlog::level::to_string_view(level).data();
    std::transform(levelUpper.begin(), levelUpper.end(), levelUpper.begin(), ::toupper);
    oss << levelUpper << "] ";
}

}
