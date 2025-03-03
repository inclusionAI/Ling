/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * AscendTransformerBoost is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef FUZZ_UTIL_H
#define FUZZ_UTIL_H
#include <cstdlib>
#include <ctime>
#include <acl/acl.h>
#include "atb_speed/log.h"
#include "atb/types.h"

namespace atb_speed {
namespace FuzzUtil {
    aclDataType GetRandomAclDataType(int input);
    aclFormat GetRandomAclFormat(int input);
    bool GetRandomBool(uint32_t &fuzzIndex);
    uint64_t GetRandomDimNum(uint32_t input);
    void GetRandomModelType(std::vector<std::vector<int>> &modelType, int len, int numLayers, int quantType);
}
}
#endif
