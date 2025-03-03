/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#ifndef ATB_SPEED_FILE_UTIL_H
#define ATB_SPEED_FILE_UTIL_H

#include <string>
#include "atb_speed/log/error.h"

namespace atb_speed {

class FileUtils {
public:
    /**
     * judge file exists
     * @param path: file full path
     * @param pattern: regex pattern
     */
    static bool CheckFileExists(const std::string& filePath);

    /**
     * is directory exists.
     * @param dir directory
     * @return
     */
    static bool CheckDirectoryExists(const std::string& dirPath);

    /** Check whether the destination path is a link
     * @param filePath raw file path
     * @return
     */
    static bool IsSymlink(const std::string& filePath);

    /** Regular the file path using realPath.
     * @param filePath raw file path
     * @param baseDir file path must in base dir
     * @param errMsg the err msg
     * @return
     */
    static bool RegularFilePath(const std::string& filePath, const std::string& baseDir, std::string &errMsg,
        bool flag);

    /** Regular the file path using realPath.
     * @param filePath raw file path
     * @param errMsg the err msg
     * @return
     */
    static bool RegularFilePath(const std::string& filePath, std::string &errMsg);

    /** Check the existence of the file and the size of the file.
     * @param configFile the input file path
     * @param errMsg the err msg
     * @param checkPermission check perm
     * @param onlyCurrentUserOp strict check, only current user can write or execute
     * @return
     */
    static bool IsFileValid(const std::string& configFile, std::string &errMsg);
    static bool IsFileValid(const std::string &filePath, std::string &errMsg,
                            bool isFileExist, mode_t mode, uint64_t maxfileSize);

    /** Check the file owner, file must be owner current user
     * @param filePath the input file path
     * @param errMsg error msg
     * @return
     */
    static bool ConstrainOwner(const std::string &filePath, std::string &errMsg);

    /** Check the file mode, file must be no greater than mode
     * @param filePath the input file path
     * @param mode file mode
     * @param errMsg error msg
     * @return
     */
    static bool ConstrainPermission(const std::string &filePath, const mode_t &mode, std::string &errMsg);
};
Error GetHomePath(std::string &outHomePath);

bool CheckAndGetLogPath(
    const std::string& logPath, long sizeLimit, std::string& outPath, const std::string& defaultPath);

bool CanonicalPath(std::string &path);
} // namespace atb_speed

#endif // ATB_FILE_UTIL_H
