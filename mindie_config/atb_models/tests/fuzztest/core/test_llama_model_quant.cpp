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
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void llamaQuantGetLayerFuzzParam(nlohmann::json &parameter, int quant)
{
    std::vector<int> layerPackQuantType;
    if (quant == 0) {
        layerPackQuantType = {12, 12}; // w8a16
    } else if (quant == 1) {
        layerPackQuantType = {13, 13}; // w4a16
    } else {
        layerPackQuantType = {18, 18}; // w8a8
    }
    
    std::vector<std::vector<int>> modelPackQuantType;
    modelPackQuantType.push_back(layerPackQuantType);
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<int> layerLinearQuantType;
    // quant == 0: w8a16; quant == 1: w4a16; quant == 2: w8a8
    if (quant != 2) {
        layerLinearQuantType = {1, -1, -1, 1, 1, -1, 1}; // w4a16 w8a16
    } else {
        layerLinearQuantType = {1, -1, -1, 1, 1, -1, 0}; // w8a8
    }
    std::vector<std::vector<int>> modelLinearQuantType;
    modelLinearQuantType.push_back(layerLinearQuantType);
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<int> layerLinearTransposeType;
    // quant == 0: w8a16; quant == 1: w4a16; quant == 2: w8a8
    if (quant != 2) {
        layerLinearTransposeType = {0, -1, -1, 0, 0, -1, 0}; // w4a16 w8a16
    } else {
        layerLinearTransposeType = {1, -1, -1, 1, 1, -1, 1}; // w8a8
    }
    std::vector<std::vector<int>> modelLinearTransposeType;
    modelLinearTransposeType.push_back(layerLinearTransposeType);
    parameter["linearTransposeType"] = modelLinearTransposeType;
}

int llamaQuantGetFuzzParam(uint32_t &fuzzIndex, int &index, nlohmann::json &parameter)
{
    int quant = index % 3;
    parameter["skipWordEmbedding"] = false;
    parameter["isFA"] = false;
    parameter["isPrefill"] = false;
    parameter["isUnpadInputs"] = true;
    parameter["isBF16"] = false;
    parameter["isEmbeddingParallel"] = true;
    parameter["isLmHeadParallel"] = true;
    parameter["lmHeadTransposeType"] = 1;
    parameter["enableSwiGLU"] = true;
    parameter["enableLcoc"] = false;
    parameter["gemma"] = false;
    parameter["enableKvQuant"] = false;
    parameter["enableSpeculate"] = false;
    parameter["normEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
    parameter["normType"] = 0;
    // quant == 0: w8a16; quant == 1: w4a16; quant == 2: w8a8
    parameter["numAttentionHeadsPerRank"] = (quant != 2) ? 64 : 32; // 64, 32: attentionheads of per rank
    parameter["hiddenSizePerAttentionHead"] = 128; // 128 is hidden size of per head
    parameter["numKeyValueHeadsPerRank"] = 8; // 8 is key value head of per rank
    parameter["numHiddenLayers"] = 1; // numhiddenlayers = 1
    // quant == 0: w8a16; quant == 1: w4a16; quant == 2: w8a8
    parameter["hiddenSize"] = (quant != 2) ? 8192 : 4096; // 8192, 4096: hidden size e.g. llama3-8B
    parameter["rank"] = 0;
    parameter["worldSize"] = 1;
    parameter["attnBackend"] = 1;
    parameter["backend"] = "lccl";
    parameter["positionEmbeddingType"] = 0;
    index++;
    llamaQuantGetLayerFuzzParam(parameter, quant);
    return quant;
}

void llamaQuantGetAclFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter, bool supportSpeculate)
{
    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100 is max len of seqlen
    }
    parameter["seqLen"] = modelSeqLen;
    int tokenOffsetSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
    std::vector<int> tokenOffsetLen;
    for (int i = 0; i < tokenOffsetSize; ++i) {
        tokenOffsetLen.push_back(std::rand() % 100); // 100 is max len of token offset
    }
    parameter["tokenOffset"] = tokenOffsetLen;
    if (supportSpeculate) {
        int qSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
        std::vector<int> qLen;
        for (int i = 0; i < qSize; ++i) {
            qLen.push_back(std::rand() % 100); // 100 is max len of qlen
        }
        parameter["qLen"] = qLen;
    }
    int blockNumsListSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
    std::vector<int> blockNumsList;
    for (int i = 0; i < blockNumsListSize; ++i) {
        blockNumsList.push_back(std::rand() % 100); // 100 is max len of token offset
    }
    parameter["blockNumsList"] = blockNumsList;
}

TEST(LlamaModelQuantDTFuzz, LlamaModelQuant)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "LlamaModelQuantDTFuzzLlamaModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("llama_fuzz");
    int index = 0;
    DT_FUZZ_START(0, 9, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        
        nlohmann::json parameter, aclParameter;
        int quant = llamaQuantGetFuzzParam(fuzzIndex, index, parameter);
        llamaQuantGetAclFuzzParam(fuzzIndex, aclParameter, parameter["enableSpeculate"]);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object llamaFuzz = modelModule.attr("LlamaFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("llama_LlamaDecoderModel");
            
            pybind11::object ret = llamaFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                if (index == 1) { // 1: a random attempt to parse a not json string
                    llamaFuzzIns.attr("execute_quant")(quant, "notajsonstring", int(parameter["enableSpeculate"]));
                } else {
                    llamaFuzzIns.attr("execute_quant")(quant, acl_parameter_string, int(parameter["enableSpeculate"]));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}