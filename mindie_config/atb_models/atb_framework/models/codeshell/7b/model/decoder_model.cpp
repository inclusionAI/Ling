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
#include "atb_speed/utils/check_util.h"
#include "models/codeshell/7b/model/decoder_model.h"

namespace atb_speed {
namespace codeshell_7b {

DecoderModel::DecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
}

atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    CHECK_OPERATION_STATUS_RETURN(atb_speed::base::DecoderModel::InferShape(inTensorDescs, outTensorDescs));
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    return atb::NO_ERROR;
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    DecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void DecoderModel::SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam)
{
    atb_speed::base::DecoderModel::SetLmHeadParam(lmHeadParam);
    lmHeadParam.hiddenSizePerAttentionHead = \
        CheckIntMulOverFlow(this->param.hiddenSizePerAttentionHead, this->param.numAttentionHeadsPerRank);
    if (this->param.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::ROW_PARALLEL;
    }
}

} // namespace codeshell_7b
} // namespace atb_speed
