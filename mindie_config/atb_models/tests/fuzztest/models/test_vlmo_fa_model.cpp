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
#include "atb_speed/log.h"
#include "models/vlmo/2b/layer/encoder_layer.h"
#include "models/vlmo/2b/layer/encoder_vl_layer.h"
#include "models/vlmo/2b/model/flash_attention_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(VlmoModelDTFuzz, FlashAttentionModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "VlmoModelDTFuzzFlashAttentionModel";

    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["rmsNormEps"] = 0.01;
        parameter["headNum"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        int dk = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["dk"] = dk;
        int layerNum = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
        parameter["layerNum"] = layerNum;
        int rankSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, rankSize);
        parameter["rankSize"] = rankSize;
        int maxTextLen = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
        parameter["maxTextLen"] = maxTextLen;
        int vlLayerIndex = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        parameter["vlLayerIndex"] = vlLayerIndex;
        parameter["backend"] = "vlmo";
        std::string parameter_string = parameter.dump();

        try {
            auto model = new atb_speed::vlmo::FlashAttentionModel(parameter_string);
            model->Init(nullptr, nullptr, nullptr);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}