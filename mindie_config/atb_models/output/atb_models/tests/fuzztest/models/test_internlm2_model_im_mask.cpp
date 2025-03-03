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
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "atb_speed/log.h"
#include "models/internlm2/20b/model/decoder_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void Internlm2PAImMaskGetParamType(int numHiddenLayers, nlohmann::json &parameter)
{
    std::vector<int> layerPackQuantType = {
        std::rand() % 12,
        std::rand() % 12
    };
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<int> layerLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearQuantType.push_back(layerLinearQuantType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<int> layerLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearTransposeType.push_back(layerLinearTransposeType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
}

nlohmann::json Internlm2PAImMaskGetParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json parameter;

    parameter["skipWordEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["supportSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["kvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["rmsNormEps"] = float(std::rand()) / RAND_MAX;
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    int hiddenSizeLowerBound = 128;
    int hiddenSizeUpperBound = 1024;
    parameter["hiddenSizePerAttentionHead"] =
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++],
                                          hiddenSizeLowerBound,
                                          hiddenSizeLowerBound,
                                          hiddenSizeUpperBound);
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank /
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["numHiddenLayers"] = numHiddenLayers;
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    parameter["splitWithStride"] = FuzzUtil::GetRandomBool(fuzzIndex);
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["rankTableFile"] = "";
    std::vector<std::string> positionEmbeddingTypeTable = {"ROPE", "ALIBI"};
    parameter["backend"] = positionEmbeddingTypeTable[
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["supportLora"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["useImMask"] = true;

    Internlm2PAImMaskGetParamType(numHiddenLayers, parameter);

    return parameter;
}

nlohmann::json Internlm2PAImMaskGetAclParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json aclParameter;

    int layerTokenOffset = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelTokenOffset;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelTokenOffset.push_back(layerTokenOffset);
    }
    aclParameter["tokenOffset"] = modelTokenOffset;

    int layerSeqLen = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelSeqLen;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelSeqLen.push_back(layerSeqLen);
    }
    aclParameter["seqLen"] = modelSeqLen;

    return aclParameter;
}

TEST(Internlm2PAImMaskModelDTFuzz, DecoderModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "Internlm2PAImMaskModelDTFuzzDecoderModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("internlm2_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);

        nlohmann::json parameter, aclParameter;
        parameter = Internlm2PAImMaskGetParam(numHiddenLayers, fuzzIndex);
        aclParameter = Internlm2PAImMaskGetAclParam(numHiddenLayers, fuzzIndex);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object llamaFuzz = modelModule.attr("InternLM2Fuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("internlm2_20b_DecoderModel");

            pybind11::object ret = llamaFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                llamaFuzzIns.attr("set_weight")();
                llamaFuzzIns.attr("set_kv_cache")();
                llamaFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

}
