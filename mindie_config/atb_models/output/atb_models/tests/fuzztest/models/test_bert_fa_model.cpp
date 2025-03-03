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
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void bertGetFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    int dk = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 64);
    parameter["dk"] = dk;
    int geluApproximate = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], -1, -1, 1);
    parameter["geluApproximate"] = geluApproximate;
    int headNum = 16;
    parameter["headNum"] = headNum;
    float layerNormEps = float(std::rand()) / RAND_MAX;
    parameter["layerNormEps"] = layerNormEps;
    int layerNormImplMode = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 2);
    parameter["layerNormImplMode"] = layerNormImplMode;
    int layerNum = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 24);
    parameter["layerNum"] = layerNum;
    int rankSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, rankSize);
    parameter["rankSize"] = rankSize;
}

TEST(BertModelDTFuzz, FlashAttentionModel)
{
    std::srand(time(nullptr));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "BertModelDTFuzzFlashAttentionModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("bert_fuzz");

    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        bertGetFuzzParam(fuzzIndex, parameter);

        std::string parameter_string = parameter.dump();

        try{
            pybind11::object bertFuzz = modelModule.attr("BertFuzz");
            pybind11::object bertFuzzIns = bertFuzz("bert_EncoderModel");

            pybind11::object ret = bertFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                bertFuzzIns.attr("execute_fa")();
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}