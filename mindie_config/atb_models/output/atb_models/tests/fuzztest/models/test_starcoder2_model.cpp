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
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "models/starcoder2/layer/decoder_layer.h"
#include "models/starcoder2/model/decoder_model.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void SetStarcoder2IntParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is numattentionheads of per rank
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 1024); // 128, 1024 is hidden size of per head
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["headNum"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 200); // 200 is num of head
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8); // 8 is worldSize
    parameter["rank"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 0, 0, 10); // 10 is  num of rank
    parameter["worldSize"] = worldSize;
    parameter["normType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["positionEmbeddingType"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 0, 1, 1);

    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100 is max len of seqlen
    }
    aclParameter["seqLen"] = modelSeqLen;
}

void SetStarcoder2VectorParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 40);
    parameter["numHiddenLayers"] = numHiddenLayers;
    std::vector<int> layerPackQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
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

TEST(Starcoder2ModelDTFuzz, Starcoder2Model)
{
    std::srand(time(NULL));
    std::string fuzzName = "Starcoder2ModelDTFuzzStarcoder2Model";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("starcoder2_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        parameter["supportSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableAddNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["layerNormEps"] = double(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
        parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["normEps"] = float(std::rand()) / RAND_MAX;
        SetStarcoder2IntParam(fuzzIndex, parameter, aclParameter);
        SetStarcoder2VectorParam(fuzzIndex, parameter, aclParameter);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object starcoder2Fuzz = modelModule.attr("BaseFuzz");
            pybind11::object starcoder2FuzzIns = starcoder2Fuzz("starcoder2_DecoderModel");
            
            pybind11::object ret = starcoder2FuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                starcoder2FuzzIns.attr("set_weight")();
                starcoder2FuzzIns.attr("set_kv_cache")();
                starcoder2FuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}