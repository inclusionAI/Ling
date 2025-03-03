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
#include "models/skywork/layer/decoder_layer.h"
#include "models/skywork/model/decoder_model.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void SetSkyworkOpsIntParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    parameter["numAttentionHeadsPerRank"] = 5; // 5 is numAttentionHeadsPerRank
    parameter["hiddenSizePerAttentionHead"] = 128; // 128 is hidden size of per head
    parameter["numKeyValueHeadsPerRank"] = 5; // 5 is numKeyValueHeadsPerRank
    parameter["numOfExperts"] = 16; // 16 is num of Experts
    parameter["numOfSelectedExperts"] = 2; // 2 is num of SelectedExperts
    parameter["rank"] = 0;
    parameter["expertParallelDegree"] = 1;
    parameter["worldSize"] = 1;
    parameter["normType"] = 0;
    parameter["backend"] = "lccl";
    parameter["routingMethod"] = "integratedSoftmaxTopK";
    parameter["rankTableFile"] = "";

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

void SetSkyworkOpsSVectorpackParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = 1;
    parameter["numHiddenLayers"] = numHiddenLayers;

    std::vector<int> layerPackQuantType = {1, 1};
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;
}

void SetSkyworkOpsSVectorQuantTypeParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = 1;
    parameter["numHiddenLayers"] = numHiddenLayers;

    std::vector<int> layerLinearQuantType = {0, -1, -1, 0, -1, -1, -1};
    std::vector<std::vector<int>> modelLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearQuantType.push_back(layerLinearQuantType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<int> layermlpLinearQuantType = {-1, -1, -1, -1};
    std::vector<std::vector<int>> modelmlpLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmlpLinearQuantType.push_back(layermlpLinearQuantType);
    }
    parameter["mlpLinearQuantType"] = modelmlpLinearQuantType;

    std::vector<int> layermoeLinearQuantType = {0, 0, -1, 0};
    std::vector<std::vector<int>> modelmoeLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmoeLinearQuantType.push_back(layermoeLinearQuantType);
    }
    parameter["moeLinearQuantType"] = modelmoeLinearQuantType;
}

void SetSkyworkOpsSVectorTransposeTypeParam(\
    uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = 1;
    parameter["numHiddenLayers"] = numHiddenLayers;

    std::vector<int> layerLinearTransposeType = {1, -1, -1, 0, -1, -1, -1};
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearTransposeType.push_back(layerLinearTransposeType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;

    std::vector<int> layermlpLinearTransposeType = {-1, -1, -1, -1};
    std::vector<std::vector<int>> modelmlpLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmlpLinearTransposeType.push_back(layermlpLinearTransposeType);
    }
    parameter["mlpLinearTransposeType"] = modelmlpLinearTransposeType;

    std::vector<int> layermoeLinearTransposeType = {1, -1, -1, -1};
    std::vector<std::vector<int>> modelmoeLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmoeLinearTransposeType.push_back(layermoeLinearTransposeType);
    }
    parameter["moeLinearTransposeType"] = modelmoeLinearTransposeType;
}

TEST(SkyworkOpsModelDTFuzz, SkyworkOpsModel)
{
    std::srand(time(NULL));
    std::string fuzzName = "SkyworkOpsModelDTFuzzSkyworkOpsModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("skywork_ops_fuzz");

    DT_FUZZ_START(0, 2, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["isFA"] = false;
        parameter["isPrefill"] = false;
        parameter["isBF16"] = false;
        parameter["isEmbeddingParallel"] = false;
        parameter["isLmHeadParallel"] = true;
        parameter["lmHeadTransposeType"] = 1;
        parameter["enableSwiGLU"] = true;
        parameter["enableLcoc"] = false;
        parameter["normEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
        parameter["isUnpadInputs"] = true;
        parameter["normHasBias"] = false;
        parameter["enableFusedRouting"] = false;
        parameter["enableAddNorm"] = false;
        parameter["skipWordEmbedding"] = false;
        SetSkyworkOpsIntParam(fuzzIndex, parameter, aclParameter);
        SetSkyworkOpsSVectorpackParam(fuzzIndex, parameter, aclParameter);
        SetSkyworkOpsSVectorQuantTypeParam(fuzzIndex, parameter, aclParameter);
        SetSkyworkOpsSVectorTransposeTypeParam(fuzzIndex, parameter, aclParameter);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object skyworkopsFuzz = modelModule.attr("BaseFuzz");
            pybind11::object skyworkopsFuzzIns = skyworkopsFuzz("skywork_DecoderModel");
            
            pybind11::object ret = skyworkopsFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                skyworkopsFuzzIns.attr("set_weight")();
                skyworkopsFuzzIns.attr("set_kv_cache")();
                skyworkopsFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}