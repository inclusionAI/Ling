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
#include <atb/atb_infer.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "atb_speed/log.h"
#include "models/gptneox/20b/layer/decoder_layer.h"
#include "models/gptneox/20b/model/pa_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(GPTNeoxModelDTFuzz, PAModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "GPTNeoxModelDTFuzzPAModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("gptnoex_fuzz");

    DT_FUZZ_START(0, 100, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["headNum"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        int dk = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["dk"] = dk;
        int layerNum = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
        parameter["layerNum"] = layerNum;
        parameter["layerNormEps"] = 0.1;
        parameter["rotaryPct"] = float(std::rand()) / RAND_MAX;
        int rankSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, rankSize);
        parameter["rankSize"] = rankSize;
        parameter["rmsNormEps"] = 0.1;
        parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["qScale"] = float(std::rand()) / RAND_MAX;
        parameter["qkScale"] = float(std::rand()) / RAND_MAX;
        std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
        parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];

        int layerSeqLen = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        std::vector<int> modelSeqLen;
        for (int layerId = 0; layerId < layerNum; layerId++) {
            modelSeqLen.push_back(layerSeqLen);
        }
        aclParameter["seqLen"] = modelSeqLen;

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();
        try {
            pybind11::object llamaFuzz = modelModule.attr("GPTNoexFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("gptneox_20b_PAModel");

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