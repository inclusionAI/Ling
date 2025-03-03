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
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_13b {

void BaichuanLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("BaichuanLayerParam: enableAlibiMaskFree: " << this->enableAlibiMaskFree);
}

PagedAttentionQuantLayer::PagedAttentionQuantLayer(
    const BaichuanLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.CheckParam();
    this->param.PrintParam();
    this->inTensorCandidates["alibi_mask_compress"] = {"in_slopes"};
}

void PagedAttentionQuantLayer::ConstructInTensorMap()
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::ConstructInTensorMap();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "alibi_mask_compress", this->inTensorList);
}

std::map<unsigned int, std::vector<std::string>> PagedAttentionQuantLayer::GetAttentionIntensor()
{
    std::map<unsigned int, std::vector<std::string>> attnInTensor = \
        DecoderLayer<atb::infer::RmsNormParam>::GetAttentionIntensor();
    if (this->param.enableAlibiMaskFree) {
        attnInTensor[common::AttnInTensorCategory::ATTN_ALIBI_MASK_COMPRESS] = \
                    this->inTensorCandidates["alibi_mask_compress"];
    }
    return attnInTensor;
}

void PagedAttentionQuantLayer::SetFusionAttentionNormParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionNormParam(fusionAttentionParam);
    fusionAttentionParam.normParamType.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
}

void PagedAttentionQuantLayer::SetFusionAttentionATBSelfAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionATBSelfAttentionParam(
        fusionAttentionParam);
    fusionAttentionParam.selfAttentionParam.kernelType = atb::infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION;
    if (this->param.enableAlibiMaskFree) {
        fusionAttentionParam.selfAttentionParam.maskType = \
        atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN;
    } else {
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
    }
}

void PagedAttentionQuantLayer::SetMlpNormParam(
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetMlpNormParam(mlpParam);
    mlpParam.normParamType.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
}

} // namespace baichuan2_13b
} // namespace atb_speed
