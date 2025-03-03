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
#ifndef ATB_SPEED_MODELS_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_MOE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {

class MoeLayerParam : public atb_speed::base::LayerParam {
public:
    MoeLayerParam() = default;
    virtual ~MoeLayerParam() override = default;
    void PrintParam() override;

    MoeLayerParam(const MoeLayerParam&) = default;
    MoeLayerParam& operator=(const MoeLayerParam&) = default;

    MoeLayerParam(MoeLayerParam&&) = default;
    MoeLayerParam& operator=(MoeLayerParam&&) = default;
    bool enableTopKSoftmax = false;
    bool transpose = true;
    int numOfExperts = 8;
    int expertParallelDegree = 1;
    bool normHasBias = false;
    bool enableFusedRouting = false;
    std::string routingMethod = "softMaxTopK";
    std::string processLogits = "normalization";
    std::vector<int> moeLinearQuantType = {};
    std::vector<int> mlpLinearQuantType = {};
    std::vector<int> moeLinearTransposeType = {};
    std::vector<int> mlpLinearTransposeType = {};
    atb::SVector<int32_t> numOfSelectedExperts = {2}; // num of selected experts
};

template <typename NormType>
class MoeDecoderLayer : public atb_speed::base::DecoderLayer<NormType> {
public:
    explicit MoeDecoderLayer(const MoeLayerParam &param);
    ~MoeDecoderLayer() override {};

protected:
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    std::map<std::string, uint32_t> ConstructNormTensorMap() const;
    virtual void SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam);
    atb::Status AddOperationToGraph() override;
    virtual void SetSelfNormParam(atb_speed::common::NormLinearParam<NormType> &selfNormParam);
    virtual atb::Status AddSelfNorm();
    virtual atb::Status AddMoe();
    virtual atb::Status AddMoeAllReduce();

    MoeLayerParam param;
};
}  // namespace moe
}  // namespace atb_speed
#endif