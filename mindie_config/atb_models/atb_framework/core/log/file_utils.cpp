/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#include <iostream>
#include <unistd.h>
#include <climits>
#include <sys/stat.h>
#include <cerrno>
#include <regex>
#include "securec.h"
#include "atb_speed/log/file_utils.h"

namespace {
    constexpr long MIN_MALLOC_SIZE = 1;
    constexpr uint64_t DEFAULT_MAX_DATA_SIZE = 4294967296;
    constexpr int PER_PERMISSION_MASK_RWX = 0b111;
}

namespace atb_speed {

static uint64_t g_defaultMaxDataSize = DEFAULT_MAX_DATA_SIZE;
static const mode_t FILE_MODE = 0740;

static size_t GetFileSize(const std::string &filePath)
{
    if (!FileUtils::CheckFileExists(filePath)) {
        std::cerr << "File does not exist!" << std::endl;
        return 0;
    }
    std::string baseDir = "/";
    std::string errMsg;
    if (!FileUtils::RegularFilePath(filePath, baseDir, errMsg, true)) {
        std::cerr << "Regular file failed by " << errMsg << std::endl;
        return 0;
    }

    FILE *fp = fopen(filePath.c_str(), "rb");
    if (fp == nullptr) {
        std::cerr << "Failed to open file." << std::endl;
        return 0;
    }
    int res = fseek(fp, 0, SEEK_END);
    if (res != 0) {
        std::cerr << "Failed to fseek SEEK_END." << std::endl;
        if (fclose(fp) != 0) {
            std::cerr << "File close failed." << std::endl;
        }
        return 0;
    }
    size_t fileSize = static_cast<size_t>(ftell(fp));
    res = fseek(fp, 0, SEEK_SET);
    if (res != 0) {
        std::cerr << "Failed to fseek SEEK_SET." << std::endl;
        if (fclose(fp) != 0) {
            std::cerr << "File close failed." << std::endl;
        }
        return 0;
    }
    res = fclose(fp);
    if (res != 0) {
        std::cerr << "File close failed." << std::endl;
        return 0;
    }
    return fileSize;
}

static bool CheckDataSize(uint64_t size, uint64_t maxFileSize = DEFAULT_MAX_DATA_SIZE)
{
    if (maxFileSize <= MIN_MALLOC_SIZE || maxFileSize > g_defaultMaxDataSize) {
        return false;
    }
    if ((size > maxFileSize) || (size < MIN_MALLOC_SIZE)) {
        std::cerr << "Input data size(" << size << ") out of range["
                  << MIN_MALLOC_SIZE << "," << maxFileSize << "]." << std::endl;
        return false;
    }

    return true;
}

bool FileUtils::CheckFileExists(const std::string &filePath)
{
    struct stat buffer;
    return (stat(filePath.c_str(), &buffer) == 0);
}

bool FileUtils::CheckDirectoryExists(const std::string &dirPath)
{
    struct stat buffer;
    if (stat(dirPath.c_str(), &buffer) != 0) {
        return false;
    }
    return (S_ISDIR(buffer.st_mode) == 1);
}

bool FileUtils::IsSymlink(const std::string &filePath)
{
    struct stat buf;
    std::string normalizedPath = filePath;
    while (!normalizedPath.empty() && normalizedPath.back() == '/') {
        normalizedPath.pop_back();
    }
    if (lstat(normalizedPath.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISLNK(buf.st_mode);
}

bool FileUtils::RegularFilePath(const std::string &filePath, const std::string &baseDir, std::string &errMsg, bool flag)
{
    if (filePath.empty()) {
        errMsg = "The file path: " + filePath + " is empty.";
        return false;
    }
    if (baseDir.empty()) {
        errMsg = "The file path: " + filePath + " basedir is empty.";
        return false;
    }
    if (filePath.size() >= PATH_MAX) {
        errMsg = "The file path " + filePath + " exceeds the maximum value set by PATH_MAX.";
        return false;
    }
    if (flag) {
        if (IsSymlink(filePath)) {
            errMsg = "The file " + filePath + " is a link.";
            return false;
        }
    }
    
    char path[PATH_MAX] = { 0x00 };
    if (realpath(filePath.c_str(), path) == nullptr) {
        errMsg = "The path " + filePath + " realpath parsing failed.";
        if (errno == EACCES) {
            errMsg += " Make sure the path's owner has execute permission.";
        }
        return false;
    }
    std::string realFilePath(path, path + strlen(path));
    if (flag) {
        std::string dir = baseDir.back() == '/' ? baseDir : baseDir + "/";
        if (realFilePath.rfind(dir, 0) != 0) {
            errMsg = "The file " + filePath + " is invalid, it's not in baseDir " + baseDir + " directory.";
            return false;
        }
    }
    return true;
}

bool FileUtils::RegularFilePath(const std::string &filePath, std::string &errMsg)
{
    if (filePath.empty()) {
        errMsg = "The file path: " + filePath + " is empty.";
        return false;
    }

    if (filePath.size() > PATH_MAX) {
        errMsg = "The file path " + filePath + " exceeds the maximum value set by PATH_MAX.";
        return false;
    }
    if (IsSymlink(filePath)) {
        errMsg = "The file " + filePath + " is a link.";
        return false;
    }
    char path[PATH_MAX] = { 0x00 };
    if (realpath(filePath.c_str(), path) == nullptr) {
        errMsg = "The path " + filePath + " realpath parsing failed.";
        if (errno == EACCES) {
            errMsg += " Make sure the path's owner has execute permission.";
        }
        return false;
    }
    return true;
}

bool FileUtils::IsFileValid(const std::string &configFile, std::string &errMsg)
{
    if (!CheckFileExists(configFile)) {
        errMsg = "The input file is not a regular file or not exists";
        return false;
    }
    size_t fileSize = GetFileSize(configFile);
    if (fileSize == 0) {
        errMsg = "The input file is empty";
    } else if (!CheckDataSize(fileSize)) {
        errMsg = "Read input file failed, file is too large, file name " + configFile;
        return false;
    }
    return true;
}

bool FileUtils::IsFileValid(const std::string &filePath, std::string &errMsg, bool isFileExist,
                            mode_t mode, uint64_t maxfileSize)
{
    if (!CheckFileExists(filePath)) {
        errMsg = "The input file is not a regular file or not exists";
        return !isFileExist;
    }
    if (!CheckDirectoryExists(filePath)) {
        size_t fileSize = GetFileSize(filePath);
        if (fileSize == 0) {
            errMsg = "The input file is empty";
        } else if (!CheckDataSize(fileSize, maxfileSize)) {
            errMsg = "Read input file failed, file is too large";
            return false;
        }
    }
    if (!ConstrainOwner(filePath, errMsg) || !ConstrainPermission(filePath, mode, errMsg)) {
        return false;
    }
    return true;
}

bool FileUtils::ConstrainOwner(const std::string &filePath, std::string &errMsg)
{
    struct stat buf;
    int ret = stat(filePath.c_str(), &buf);
    if (ret != 0) {
        errMsg = "Get file stat failed.";
        return false;
    }
    if (buf.st_uid != getuid()) {
        errMsg = "owner id diff: current process user id is " + std::to_string(getuid()) + ", file owner id is " +
            std::to_string(buf.st_uid);
        return false;
    }
    return true;
}

bool FileUtils::ConstrainPermission(const std::string &filePath, const mode_t &mode, std::string &errMsg)
{
    struct stat buf;
    int ret = stat(filePath.c_str(), &buf);
    if (ret != 0) {
        errMsg = "Get file stat failed.";
        return false;
    }

    mode_t mask = PER_PERMISSION_MASK_RWX;
    const int perPermWidth = 3;
    std::vector<std::string> permMsg = { "Other group permission", "Owner group permission", "Owner permission" };
    for (int i = perPermWidth; i > 0; i--) {
        uint32_t curPerm = (buf.st_mode & (mask << ((i - 1) * perPermWidth))) >> ((i - 1) * perPermWidth);
        uint32_t maxPerm = (mode & (mask << ((i - 1) * perPermWidth))) >> ((i - 1) * perPermWidth);
        if ((curPerm | maxPerm) != maxPerm) {
            errMsg = " Check " + permMsg[i - 1] + " failed: Current permission is " + std::to_string(curPerm) +
                ", but required no greater than " + std::to_string(maxPerm) + ".";

            return false;
        }
    }
    return true;
}

bool CanonicalPath(std::string &path)
{
    if (path.empty() || path.size() >= PATH_MAX) {
        return false;
    }

    /* It will allocate memory to store path */
    char pathtemp[PATH_MAX] = { 0x00 };
    if (realpath(path.c_str(), pathtemp) == nullptr) {
        return false;
    }
    path = pathtemp;
    return true;
}

Error GetHomePath(std::string &outHomePath)
{
    const char *homePath = std::getenv("HOME");
    if (homePath == nullptr) {
        throw std::runtime_error("homePath is null, please check env HOME");
    } else {
        char realPath[PATH_MAX];
        errno_t ret = strcpy_s(realPath, PATH_MAX, homePath);
        if (ret != EOK) {
            std::cout << "strcpy_s failed" << ret << std::endl;
            return Error(Error::Code::ERROR, "ERROR: strcpy_s failed.");
        }
        outHomePath = realPath;
    }
    if (!CanonicalPath(outHomePath)) {
        std::cout << " Failed to get real path of home " << std::endl;
        return Error(Error::Code::ERROR, "ERROR: Failed to get real path of home.");
    }
    return Error(Error::Code::OK);
}

bool CheckAndGetLogPath(
    const std::string &logPath, long sizeLimit, std::string &outPath, const std::string &defaultPath)
{
    bool usingDefault = logPath == defaultPath;
    std::string usingDefaultNotice = !usingDefault ? " using default path instead" : "";

    if (logPath.empty()) {
        std::cout << "logPath is empty" << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    std::string path = logPath;
    std::string baseDir = "/";

    if (logPath[0] != '/') {
        // starts with '/', regarded as an absolute path
        // otherwise, regarded as a relative path
        std::string homePath{};
        if (!GetHomePath(homePath).IsOk()) {
            std::cout << "failed to get home path" << std::endl;
            return false;
        }
        baseDir = homePath;
        path = homePath + "/" + logPath;
    }

    std::regex reg(".{1,4096}");
    if (!std::regex_match(path, reg)) {
        std::cout << "The logPath is too long." << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }

    size_t lastSlash = path.size();
    lastSlash = path.rfind('/', lastSlash - 1);
    if (lastSlash == std::string::npos) {
        std::cout << "logPath is illegal,  must such as /xxx.log." << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    std::string parentPath = path.substr(0, lastSlash);
    if (!FileUtils::CheckDirectoryExists(parentPath)) {
        std::cout << "The parent path of logPath is not exist or not a dir." << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    std::string errMsg{};
    if (!FileUtils::RegularFilePath(path, baseDir, errMsg, true) ||
        !FileUtils::IsFileValid(path, errMsg, false, 0b110'100'000, sizeLimit)) {
        std::cerr << errMsg << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    outPath = path;
    return true;
}

} // namespace atb_speed