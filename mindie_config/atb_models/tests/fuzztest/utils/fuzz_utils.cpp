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
#include "fuzz_utils.h"
#include <climits>
#include <cfloat>
#include "secodeFuzz.h"

namespace atb_speed {
const int ACL_DATA_RANDOM_NUM = 28;
const int ACL_FORMAT_RANDOM_NUM = 36;
const int DIMNUM_RANDOM_NUM = 9;
const int BOOL_RANDOM_NUM = 2;
const int SINGLE_DIM_RANDOM_NUM = 2000;
aclDataType FuzzUtil::GetRandomAclDataType(int input)
{
    return aclDataType(input % ACL_DATA_RANDOM_NUM);
}

aclFormat FuzzUtil::GetRandomAclFormat(int input)
{
    return aclFormat(input % ACL_FORMAT_RANDOM_NUM);
}

uint64_t FuzzUtil::GetRandomDimNum(uint32_t input)
{
    return input % DIMNUM_RANDOM_NUM;
}

bool FuzzUtil::GetRandomBool(uint32_t &fuzzIndex)
{
    u16 randomNum = *(u16 *) DT_SetGetU16(&g_Element[fuzzIndex++], 0);
    return randomNum % BOOL_RANDOM_NUM;
}

void FuzzUtil::GetRandomModelType(std::vector<std::vector<int>> &modelType, int len, int numLayers, int quantType)
{
    if (quantType == 0) {
        return ;
    }
    std::vector<int> layerType;
    for (int i = 0; i < len; ++i) {
        layerType.push_back(std::rand() % quantType - 1);
    }

    for (int layerId = 0; layerId < numLayers; ++layerId) {
        modelType.push_back(layerType);
    }
}
}