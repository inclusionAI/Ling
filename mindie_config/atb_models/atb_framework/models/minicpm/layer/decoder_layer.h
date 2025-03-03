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
#ifndef ATB_SPEED_MODELS_MINICPM_DECODER_LAYER_H
#define ATB_SPEED_MODELS_MINICPM_DECODER_LAYER_H

#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "operations/fusion/linear/linear_parallel.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace minicpm {

class MiniCPMLayerParam : public atb_speed::base::LayerParam {
public:
    void PrintParam() override;

    float numHiddenLayers = 52;
    float scaleDepth = 1.4 ;
};

class MiniCPMDecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit MiniCPMDecoderLayer(const MiniCPMLayerParam &param);
    ~MiniCPMDecoderLayer() override {};
    uint32_t GetLayerParamIndex(const std::string &param);
protected:
    atb::Status AddOperationToGraph() override;

    MiniCPMLayerParam param;
};


}  // namespace minicpm
}  // namespace atb_speed
#endif