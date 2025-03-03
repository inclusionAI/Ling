/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include <gtest/gtest.h>
#include <random>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "models/bloom/layer/bloom_decoder_layer.h"
#include "models/bloom/model/bloom_decoder_model.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

unsigned int GetRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(0, 65535);  // 0 ~ 65535
    return dis(gen);
}

nlohmann::json BloomGetParam(int fuzzi, int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json parameter;
    parameter["isUnpadInputs"] = static_cast<bool>(GetRandomNumber() % 2);  // 2: bool mod
    parameter["isFA"] = static_cast<bool>(GetRandomNumber() % 2);  // 2: bool mod
    parameter["isPrefill"] = static_cast<bool>(GetRandomNumber() % 2);  // 2: bool mod
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = static_cast<bool>(GetRandomNumber() % 2);  // 2: bool mod
    parameter["isLmHeadParallel"] = static_cast<bool>(GetRandomNumber() % 2);  // 2: bool mod
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["normEps"] = float(GetRandomNumber()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
    parameter["normType"] = GetRandomNumber() % 2; // 2: 0 or 1
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 1024);  // 128 1024: bound
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["numHiddenLayers"] = numHiddenLayers;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize + 2); // 2: rank > w.s.
    parameter["worldSize"] = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    
    std::vector<std::vector<int>> modelPackQuantType;
    FuzzUtil::GetRandomModelType(modelPackQuantType, 2, numHiddenLayers, 12); // 2: len, 12: quant
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<std::vector<int>> modelLinearQuantType;
    FuzzUtil::GetRandomModelType(modelLinearQuantType, 7, numHiddenLayers, 3); // 7: len, 3: quant
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<std::vector<int>> modelLinearTransposeType;
    FuzzUtil::GetRandomModelType(modelLinearTransposeType, 7, numHiddenLayers, 3); // 7: len, 3: quant

    for (int i = 0; i < 7; i++) {  // len: 7
        modelLinearTransposeType[0][i] = modelLinearTransposeType[0][i] == -1 ? 0 : modelLinearTransposeType[0][i];
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
    return parameter;
}

nlohmann::json BloomGetAclParam(int fuzzi, int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json aclParameter;

    int layerTokenOffset = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelTokenOffset;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelTokenOffset.push_back(layerTokenOffset);
    }
    aclParameter["tokenOffset"] = modelTokenOffset;

    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 1,20: bound
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; i++) {
        modelSeqLen.push_back(GetRandomNumber() % 100);  // 100: mod
    }
    aclParameter["seqLen"] = modelSeqLen;
    return aclParameter;
}

TEST(BloomModelDTFuzz, BloomModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("===== bloom fuzz begin =====");
    std::string fuzzName = "BloomModelDTFuzzBloomModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("bloom_fuzz");

    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
        nlohmann::json parameter, aclParameter;
        parameter = BloomGetParam(fuzzi, numHiddenLayers, fuzzIndex);
        aclParameter = BloomGetAclParam(fuzzi, numHiddenLayers, fuzzIndex);
        // Use "None" to raise exception
        std::string parameter_string = GetRandomNumber() % 10 ? parameter.dump(): "None";  // "None" (1/10)
        std::string acl_parameter_string = GetRandomNumber() % 20 ? aclParameter.dump() : "None";  // "None" (1/20)

        try {
            pybind11::object bloomFuzz = modelModule.attr("BloomFuzz");
            pybind11::object bloomFuzzIns = bloomFuzz("bloom_BloomDecoderModel");
            pybind11::object ret = bloomFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                bloomFuzzIns.attr("set_weight")();
                bloomFuzzIns.attr("set_kv_cache")();
                bloomFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}