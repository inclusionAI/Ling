/**
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
#include "atb_speed/utils/file_system.h"
#include <fstream>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

namespace atb_speed {

constexpr size_t MAX_PATH_LEN = 256;

bool FileSystem::Exists(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }
    return true;
}

bool FileSystem::IsDir(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }

    return S_ISDIR(st.st_mode);
}

std::string FileSystem::Join(const std::vector<std::string> &paths)
{
    std::string retPath;
    for (const auto &path : paths) {
        if (retPath.empty()) {
            retPath.append(path);
        } else {
            retPath.append("/" + path);
        }
    }
    return retPath;
}

int64_t FileSystem::FileSize(const std::string &filePath)
{
    struct stat st;
    if (stat(filePath.c_str(), &st) < 0) {
        return -1;
    }
    return st.st_size;
}

std::string FileSystem::BaseName(const std::string &filePath)
{
    std::string fileName;
    const char *str = strrchr(filePath.c_str(), '/');
    if (str) {
        fileName = str + 1;
    } else {
        fileName = filePath;
    }
    return fileName;
}

std::string FileSystem::DirName(const std::string &path)
{
    int32_t idx = path.size() - 1;
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    std::string sub = path.substr(0, idx);
    const char *str = strrchr(sub.c_str(), '/');
    if (str == nullptr) {
        return ".";
    }
    idx = str - sub.c_str() - 1;
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    if (idx < 0) {
        return "/";
    }
    return path.substr(0, idx + 1);
}

bool FileSystem::DeleteFile(const std::string &filePath)
{
    int ret = remove(filePath.c_str());
    return ret == 0;
}

bool FileSystem::MakeDir(const std::string &dirPath, int mode)
{
    int ret = mkdir(dirPath.c_str(), mode);
    return ret == 0;
}

bool FileSystem::Makedirs(const std::string &dirPath, const mode_t mode)
{
    int32_t offset = 0;
    int32_t pathLen = dirPath.size();
    do {
        const char *str = strchr(dirPath.c_str() + offset, '/');
        offset = (str == nullptr) ? pathLen : str - dirPath.c_str() + 1;
        std::string curPath = dirPath.substr(0, offset);
        if (!Exists(curPath)) {
            if (!MakeDir(curPath, mode)) {
                return false;
            }
        }
    } while (offset != pathLen);
    return true;
}
} // namespace atb_speed