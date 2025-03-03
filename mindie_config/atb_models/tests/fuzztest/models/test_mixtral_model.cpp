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
#include "models/moe/layer/decoder_layer.h"
#include "models/mixtral/model/decoder_model.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void SetMixtralIntParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is numattentionheads of per rank
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 1024); // 128, 1024 is hidden size of per head
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["numOfExperts"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is num of Experts
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8); // 8 is worldSize
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["expertParallelDegree"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 64/worldSize, 64/worldSize, 64/worldSize); // 64 is worldSize * expertParallelDegree
    parameter["worldSize"] = worldSize;
    parameter["normType"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 0, 1, 1);
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    std::vector<std::string> routingMethodEnumTable = {"softMaxTopK", ""};
    parameter["routingMethod"] = routingMethodEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    std::vector<std::string> rankTableFileEnumTable = {"", ""};
    parameter["rankTableFile"] = rankTableFileEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];

    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100 is max len of seqlen
    }
    aclParameter["seqLen"] = modelSeqLen;
    
    int tokenOffsetSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
    std::vector<int> tokenOffsetLen;
    for (int i = 0; i < tokenOffsetSize; ++i) {
        tokenOffsetLen.push_back(std::rand() % 100); // 100 is max len of token offset
    }
    aclParameter["tokenOffset"] = tokenOffsetLen;
}

void SetMixtralVectorParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 56);
    parameter["numHiddenLayers"] = numHiddenLayers;
    int layerNumOfSelectedExperts = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    atb::SVector<int32_t> modelNumOfSelectedExperts;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelNumOfSelectedExperts.push_back(layerNumOfSelectedExperts);
    }
    parameter["numOfSelectedExperts"] = modelNumOfSelectedExperts;
}

void SetMixtralSVectorpackParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 28);
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
}

void SetMixtralSVectorQuantTypeParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 28);
    parameter["numHiddenLayers"] = numHiddenLayers;

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

    std::vector<int> layermlpLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmlpLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmlpLinearQuantType.push_back(layermlpLinearQuantType);
    }
    parameter["mlpLinearQuantType"] = modelmlpLinearQuantType;

    std::vector<int> layermoeLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmoeLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmoeLinearQuantType.push_back(layermoeLinearQuantType);
    }
    parameter["moeLinearQuantType"] = modelmoeLinearQuantType;
}

void SetMixtralSVectorTransposeTypeParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 28);
    parameter["numHiddenLayers"] = numHiddenLayers;

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

    std::vector<int> layermlpLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmlpLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmlpLinearTransposeType.push_back(layermlpLinearTransposeType);
    }
    parameter["mlpLinearTransposeType"] = modelmlpLinearTransposeType;

    std::vector<int> layermoeLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmoeLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmoeLinearTransposeType.push_back(layermoeLinearTransposeType);
    }
    parameter["moeLinearTransposeType"] = modelmoeLinearTransposeType;
}

TEST(MixtralModelDTFuzz, MixtralModel)
{
    std::srand(time(NULL));
    std::string fuzzName = "MixtralModelDTFuzzMixtralModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("mixtral_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["qkvHasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        parameter["supportSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["normEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
        parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["normHasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableFuseRouting"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableAddNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
        SetMixtralIntParam(fuzzIndex, parameter, aclParameter);
        SetMixtralVectorParam(fuzzIndex, parameter, aclParameter);
        SetMixtralSVectorpackParam(fuzzIndex, parameter, aclParameter);
        SetMixtralSVectorQuantTypeParam(fuzzIndex, parameter, aclParameter);
        SetMixtralSVectorTransposeTypeParam(fuzzIndex, parameter, aclParameter);
        
        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object mixtralFuzz = modelModule.attr("BaseFuzz");
            pybind11::object mixtralFuzzIns = mixtralFuzz("mixtral_DecoderModel");
            
            pybind11::object ret = mixtralFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                mixtralFuzzIns.attr("set_weight")();
                mixtralFuzzIns.attr("set_kv_cache")();
                mixtralFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}