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
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include "atb/svector.h"

#include "operation_factory.h"
#include "operations/aclnn/ops/indexput_operation.h"
#include "operations/aclnn/ops/index_select_operation.h"
#include "operations/aclnn/ops/w8a16_operation.h"
#include "operations/aclnn/ops/w4a16_operation.h"

namespace atb {
namespace infer {
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::LayerNormParam::LayerNormType, {
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_UNDEFINED, "LAYER_NORM_UNDEFINED"},
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM, "LAYER_NORM_NORM"},
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_PRENORM, "LAYER_NORM_PRENORM"},
        {atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_POSTNORM, "LAYER_NORM_POSTNORM"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::RmsNormParam::RmsNormType, {
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_UNDEFINED, "RMS_NORM_UNDEFINED"},
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM, "RMS_NORM_NORM"},
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM, "RMS_NORM_PRENORM"},
        {atb::infer::RmsNormParam::RmsNormType::RMS_NORM_POSTNORM, "RMS_NORM_POSTNORM"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::RmsNormParam::PrecisionMode, {
        {atb::infer::RmsNormParam::PrecisionMode::HIGH_PRECISION_MODE, "HIGH_PRECISION_MODE"},
        {atb::infer::RmsNormParam::PrecisionMode::HIGH_PERFORMANCE_MODE, "HIGH_PERFORMANCE_MODE"},
    })
    
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::RmsNormParam::ModelType, {
        {atb::infer::RmsNormParam::ModelType::LLAMA_MODEL, "LLAMA_MODEL"},
        {atb::infer::RmsNormParam::ModelType::GEMMA_MODEL, "GEMMA_MODEL"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::DynamicQuantType, {
        {atb::infer::DynamicQuantType::DYNAMIC_QUANT_UNDEFINED, "DYNAMIC_QUANT_UNDEFINED"},
        {atb::infer::DynamicQuantType::DYNAMIC_QUANT_SYMMETRIC, "DYNAMIC_QUANT_SYMMETRIC"},
        {atb::infer::DynamicQuantType::DYNAMIC_QUANT_ASYMMETRIC, "DYNAMIC_QUANT_ASYMMETRIC"},
    })
    
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::InputLayout, {
        {atb::infer::InputLayout::TYPE_BSND, "TYPE_BSND"},
        {atb::infer::InputLayout::TYPE_BNSD, "TYPE_BNSD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::QuantType, {
        {atb::infer::QuantType::QUANT_UNDEFINED, "QUANT_UNDEFINED"},
        {atb::infer::QuantType::QUANT_INT4, "QUANT_INT4"},
        {atb::infer::QuantType::QUANT_INT8, "QUANT_INT8"},
        {atb::infer::QuantType::QUANT_INT16, "QUANT_INT16"},
        {atb::infer::QuantType::QUANT_FLOAT8, "QUANT_FLOAT8"},
        {atb::infer::QuantType::QUANT_FLOAT16, "QUANT_FLOAT16"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ActivationType, {
        {atb::infer::ActivationType::ACTIVATION_UNDEFINED, "ACTIVATION_UNDEFINED"},
        {atb::infer::ActivationType::ACTIVATION_RELU, "ACTIVATION_RELU"},
        {atb::infer::ActivationType::ACTIVATION_GELU, "ACTIVATION_GELU"},
        {atb::infer::ActivationType::ACTIVATION_FAST_GELU, "ACTIVATION_FAST_GELU"},
        {atb::infer::ActivationType::ACTIVATION_SWISH, "ACTIVATION_SWISH"},
        {atb::infer::ActivationType::ACTIVATION_LOG, "ACTIVATION_LOG"},
        {atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD, "ACTIVATION_SWIGLU_FORWARD"},
        {atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD, "ACTIVATION_SWIGLU_BACKWARD"},
        {atb::infer::ActivationType::ACTIVATION_SIGMOID, "ACTIVATION_SIGMOID"},
        {atb::infer::ActivationType::ACTIVATION_MAX, "ACTIVATION_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ActivationParam::GeLUMode, {
        {atb::infer::ActivationParam::GeLUMode::TANH_MODE, "TANH_MODE"},
        {atb::infer::ActivationParam::GeLUMode::NONE_MODE, "NONE_MODE"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::CommMode, {
        {atb::infer::CommMode::COMM_UNDEFINED, "COMM_UNDEFINED"},
        {atb::infer::CommMode::COMM_MULTI_PROCESS, "COMM_MULTI_PROCESS"},
        {atb::infer::CommMode::COMM_MULTI_THREAD, "COMM_MULTI_THREAD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ElewiseParam::ElewiseType, {
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_UNDEFINED, "ELEWISE_UNDEFINED"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST, "ELEWISE_CAST"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS, "ELEWISE_MULS"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_COS, "ELEWISE_COS"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_SIN, "ELEWISE_SIN"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_NEG, "ELEWISE_NEG"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT, "ELEWISE_QUANT"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, "ELEWISE_LOGICAL_NOT"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD, "ELEWISE_ADD"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL, "ELEWISE_MUL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, "ELEWISE_REALDIV"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, "ELEWISE_LOGICAL_AND"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, "ELEWISE_LOGICAL_OR"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_LESS, "ELEWISE_LESS"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, "ELEWISE_GREATER"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_SUB, "ELEWISE_SUB"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, "ELEWISE_EQUAL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL, "ELEWISE_QUANT_PER_CHANNEL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL, "ELEWISE_DEQUANT_PER_CHANNEL"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT, "ELEWISE_DYNAMIC_QUANT"},
        {atb::infer::ElewiseParam::ElewiseType::ELEWISE_TANH, "ELEWISE_TANH"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::LinearParallelParam::ParallelType, {
        {atb::infer::LinearParallelParam::ParallelType::UNDEFINED, "UNDEFINED"},
        {atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE, "LINEAR_ALL_REDUCE"},
        {atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER, "LINEAR_REDUCE_SCATTER"},
        {atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR, "ALL_GATHER_LINEAR"},
        {atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR, "PURE_LINEAR"},
        {atb::infer::LinearParallelParam::ParallelType::MAX, "MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::LinearParallelParam::QuantType, {
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED, "QUANT_TYPE_UNDEFINED"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TENSOR, "QUANT_TYPE_PER_TENSOR"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL, "QUANT_TYPE_PER_CHANNEL"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_GROUP, "QUANT_TYPE_PER_GROUP"},
        {atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX, "QUANT_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::AllReduceParam::QuantType, {
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED, "QUANT_TYPE_UNDEFINED"},
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_TENSOR, "QUANT_TYPE_PER_TENSOR"},
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL, "QUANT_TYPE_PER_CHANNEL"},
        {atb::infer::AllReduceParam::QuantType::QUANT_TYPE_MAX, "QUANT_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::CalcType, {
        {atb::infer::SelfAttentionParam::CalcType::UNDEFINED, "UNDEFINED"},
        {atb::infer::SelfAttentionParam::CalcType::ENCODER, "ENCODER"},
        {atb::infer::SelfAttentionParam::CalcType::DECODER, "DECODER"},
        {atb::infer::SelfAttentionParam::CalcType::PA_ENCODER, "PA_ENCODER"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::KernelType, {
        {atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_DEFAULT, "KERNELTYPE_DEFAULT"},
        {atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_HIGH_PRECISION, "KERNELTYPE_HIGH_PRECISION"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::ClampType, {
        {atb::infer::SelfAttentionParam::ClampType::CLAMP_TYPE_UNDEFINED, "CLAMP_TYPE_UNDEFINED"},
        {atb::infer::SelfAttentionParam::ClampType::CLAMP_TYPE_MIN_MAX, "CLAMP_TYPE_MIN_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::MaskType, {
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED, "MASK_TYPE_UNDEFINED"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM, "MASK_TYPE_NORM"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI, "MASK_TYPE_ALIBI"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM_COMPRESS, "MASK_TYPE_NORM_COMPRESS"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS, "MASK_TYPE_ALIBI_COMPRESS"},
        {atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS_SQRT, "MASK_TYPE_ALIBI_COMPRESS_SQRT"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::ScaleType, {
        {atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_TOR, "SCALE_TYPE_TOR"},
        {atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_LOGN, "SCALE_TYPE_LOGN"},
        {atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_MAX, "SCALE_TYPE_MAX"},
    })
    
    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::SelfAttentionParam::KvCacheCfg, {
        {atb::infer::SelfAttentionParam::KvCacheCfg::K_CACHE_V_CACHE, "K_CACHE_V_CACHE"},
        {atb::infer::SelfAttentionParam::KvCacheCfg::K_BYPASS_V_BYPASS, "K_BYPASS_V_BYPASS"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::MaskType, {
        {atb::infer::PagedAttentionParam::MaskType::UNDEFINED, "MASK_TYPE_UNDEFINED"},
        {atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM, "MASK_TYPE_NORM"},
        {atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI, "MASK_TYPE_ALIBI"},
        {atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC, "MASK_TYPE_SPEC"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::QuantType, {
        {atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_UNDEFINED, "TYPE_QUANT_UNDEFINED"},
        {atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION, "TYPE_DEQUANT_FUSION"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::CalcType, {
        {atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED, "CALC_TYPE_UNDEFINED"},
        {atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC, "CALC_TYPE_SPEC"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::CompressType, {
        {atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED, "COMPRESS_TYPE_UNDEFINED"},
        {atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD, "COMPRESS_TYPE_KVHEAD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::PagedAttentionParam::ScaleType, {
        {atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_TOR, "SCALE_TYPE_TOR"},
        {atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN, "SCALE_TYPE_LOGN"},
        {atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_MAX, "SCALE_TYPE_MAX"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ReshapeAndCacheParam::CompressType, {
        {atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED, "COMPRESS_TYPE_UNDEFINED"},
        {atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD, "COMPRESS_TYPE_KVHEAD"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::TransdataParam::TransdataType, {
        {atb::infer::TransdataParam::TransdataType::UNDEFINED, "UNDEFINED"},
        {atb::infer::TransdataParam::TransdataType::FRACTAL_NZ_TO_ND, "FRACTAL_NZ_TO_ND"},
        {atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ, "ND_TO_FRACTAL_NZ"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(atb::infer::ReduceParam::ReduceType, {
        {atb::infer::ReduceParam::ReduceType::REDUCE_UNDEFINED, "REDUCE_UNDEFINED"},
        {atb::infer::ReduceParam::ReduceType::REDUCE_MAX, "REDUCE_MAX"},
        {atb::infer::ReduceParam::ReduceType::REDUCE_MIN, "REDUCE_MIN"},
        {atb::infer::ReduceParam::ReduceType::REDUCE_SUM, "REDUCE_SUM"},
    })
}
}

NLOHMANN_JSON_SERIALIZE_ENUM(aclDataType, {
    {ACL_DT_UNDEFINED, "ACL_DT_UNDEFINED"},
    {ACL_FLOAT, "ACL_FLOAT"},
    {ACL_FLOAT16, "ACL_FLOAT16"},
    {ACL_INT8, "ACL_INT8"},
    {ACL_INT32, "ACL_INT32"},
    {ACL_UINT8, "ACL_UINT8"},
    {ACL_INT16, "ACL_INT16"},
    {ACL_UINT16, "ACL_UINT16"},
    {ACL_UINT32, "ACL_UINT32"},
    {ACL_INT64, "ACL_INT64"},
    {ACL_UINT64, "ACL_UINT64"},
    {ACL_DOUBLE, "ACL_DOUBLE"},
    {ACL_BOOL, "ACL_BOOL"},
    {ACL_STRING, "ACL_STRING"},
    {ACL_COMPLEX64, "ACL_COMPLEX64"},
    {ACL_COMPLEX128, "ACL_COMPLEX128"},
    {ACL_BF16, "ACL_BF16"},
    {ACL_INT4, "ACL_INT4"},
    {ACL_UINT1, "ACL_UINT1"},
    {ACL_COMPLEX32, "ACL_COMPLEX32"},
})

namespace atb_torch {

template <typename Param>
static atb::Operation *OperationCreate(Param &param, std::string operationName) {
    atb::Operation *operation = nullptr;
    atb::Status st = atb::CreateOperation(param, &operation);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("create atb " << operationName << " operation fail, error:" << st);
    }
    return operation;
}

static atb::Operation *ActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ActivationParam param;
    param.activationType = \
        paramJson.value("activationType", R"("ACTIVATION_UNDEFINED")"_json).get<atb::infer::ActivationType>();
    param.scale = paramJson.value("scale", 1.0f);
    param.dim = paramJson.value("dim", -1);
    param.geluMode = paramJson.value("geluMode", R"("TANH_MODE")"_json).get<atb::infer::ActivationParam::GeLUMode>();

    return OperationCreate(param, "Activation");
}

static atb::Operation *GatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::GatherParam param;
    param.axis = paramJson.value("axis", 0);
    param.batchDims = paramJson.value("batchDims", 0);

    return OperationCreate(param, "Gather");
}

static atb::Operation *SplitOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SplitParam param;
    param.splitDim = paramJson.value("splitDim", 0);
    param.splitNum = paramJson.value("splitNum", 2);  // 2: 默认值
    for (auto item : paramJson.value("splitSizes", R"([])"_json).get<std::vector<int32_t>>()) {
        param.splitSizes.push_back(item);
    }

    return OperationCreate(param, "Split");
}

static atb::Operation *ConcatOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ConcatParam param;
    param.concatDim = paramJson.value("concatDim", 0);

    return OperationCreate(param, "Concat");
}

static atb::Operation *SliceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SliceParam param;
    for (auto item : paramJson.value("offsets", R"([])"_json).get<std::vector<int64_t>>()) {
        param.offsets.push_back(item);
    }
    for (auto item : paramJson.value("size", R"([])"_json).get<std::vector<int64_t>>()) {
        param.size.push_back(item);
    }

    return OperationCreate(param, "Slice");
}

static atb::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson.value("perm", R"([])"_json).get<std::vector<int32_t>>()) {
        param.perm.push_back(item);
    }

    return OperationCreate(param, "Transpose");
}

static atb::Operation *ElewiseOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ElewiseParam param;
    param.elewiseType = \
        paramJson.value("elewiseType", R"("ELEWISE_UNDEFINED")"_json).get<atb::infer::ElewiseParam::ElewiseType>();
    param.quantParam.inputScale = paramJson.value("quantParam", R"({})"_json).value("inputScale", 1.0f);
    param.quantParam.inputOffset = paramJson.value("quantParam", R"({})"_json).value("inputOffset", 0);
    param.quantParam.asymmetric  = paramJson.value("quantParam", R"({})"_json).value("asymmetric", false);
    param.mulsParam.varAttr = paramJson.value("mulsParam", R"({})"_json).value("varAttr", 0.0f);
    param.outTensorType = paramJson.value("outTensorType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "Elewise");
}

static atb::Operation *ReshapeAndCacheOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReshapeAndCacheParam param;
    param.compressType = paramJson.value("compressType", \
        R"("COMPRESS_TYPE_UNDEFINED")"_json).get<atb::infer::ReshapeAndCacheParam::CompressType>();
    return OperationCreate(param, "ReshapeAndCache");
}

static atb::Operation *LayerNormOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LayerNormParam param;
    param.layerType = paramJson.value("layerType", \
        R"("LAYER_NORM_UNDEFINED")"_json).get<atb::infer::LayerNormParam::LayerNormType>();
    param.normParam.quantType = paramJson.value("normParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.normParam.epsilon = paramJson.value("normParam", R"({})"_json).value("epsilon", 1e-5);
    param.normParam.beginNormAxis = paramJson.value("normParam", R"({})"_json).value("beginNormAxis", 0);
    param.normParam.beginParamsAxis = \
        paramJson.value("normParam", R"({})"_json).value("beginParamsAxis", 0);
    param.normParam.dynamicQuantType = paramJson.value("normParam", \
        R"({})"_json).value("dynamicQuantType", \
        R"("DYNAMIC_QUANT_UNDEFINED")"_json).get<atb::infer::DynamicQuantType>();
    param.preNormParam.quantType = paramJson.value("preNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.preNormParam.epsilon = paramJson.value("preNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.preNormParam.opMode = paramJson.value("preNormParam", R"({})"_json).value("opMode ", 0);
    param.preNormParam.zoomScaleValue = paramJson.value("preNormParam", R"({})"_json).value("zoomScaleValue", 1.0f);
    param.postNormParam.quantType = paramJson.value("postNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.postNormParam.epsilon = paramJson.value("postNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.postNormParam.opMode = paramJson.value("postNormParam", R"({})"_json).value("opMode", 0);
    param.postNormParam.zoomScaleValue = paramJson.value("postNormParam", R"({})"_json).value("zoomScaleValue", 1.0f);

    return OperationCreate(param, "LayerNorm");
}

static atb::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RmsNormParam param;
    param.layerType = paramJson.value("layerType", \
        R"("RMS_NORM_UNDEFINED")"_json).get<atb::infer::RmsNormParam::RmsNormType>();
    param.normParam.quantType = paramJson.value("normParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.normParam.epsilon = paramJson.value("normParam", R"({})"_json).value("epsilon", 1e-5);
    param.normParam.layerNormEps = paramJson.value("normParam", R"({})"_json).value("layerNormEps", 1e-5);
    param.normParam.rstd = paramJson.value("normParam", R"({})"_json).value("rstd", false);
    param.normParam.precisionMode = paramJson.value("normParam", \
        R"({})"_json).value("precisionMode", \
        R"("HIGH_PRECISION_MODE")"_json).get<atb::infer::RmsNormParam::PrecisionMode>();
    param.normParam.modelType = paramJson.value("normParam", \
        R"({})"_json).value("modelType", R"("LLAMA_MODEL")"_json).get<atb::infer::RmsNormParam::ModelType>();
    param.normParam.dynamicQuantType = paramJson.value("normParam", \
        R"({})"_json).value("dynamicQuantType", \
        R"("DYNAMIC_QUANT_UNDEFINED")"_json).get<atb::infer::DynamicQuantType>();
    param.preNormParam.quantType = paramJson.value("preNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.preNormParam.epsilon = paramJson.value("preNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.preNormParam.hasBias = paramJson.value("preNormParam", R"({})"_json).value("hasBias", false);
    param.postNormParam.quantType = paramJson.value("postNormParam", \
        R"({})"_json).value("quantType", R"("QUANT_UNDEFINED")"_json).get<atb::infer::QuantType>();
    param.postNormParam.epsilon = paramJson.value("postNormParam", R"({})"_json).value("epsilon", 1e-5);
    param.postNormParam.hasBias = paramJson.value("postNormParam", R"({})"_json).value("hasBias", false);

    return OperationCreate(param, "RmsNorm");
}

static atb::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "AllGather");
}

static atb::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.allReduceType = paramJson.value("allReduceType", "sum");
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.commDomain = paramJson.value("commDomain", "");
    param.quantType = paramJson.value("quantType", \
        R"("QUANT_TYPE_UNDEFINED")"_json).get<atb::infer::AllReduceParam::QuantType>();
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "AllReduce");
}

static atb::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParam param;
    param.transposeA = paramJson.value("transposeA", false);
    param.transposeB = paramJson.value("transposeB", true);
    param.hasBias = paramJson.value("hasBias", true);
    param.enAccum = paramJson.value("enAccum", false);
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();

    return OperationCreate(param, "Linear");
}

static atb::Operation *LinearParallelOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParallelParam param;
    param.transWeight = paramJson.value("transWeight", true);
    param.rank = paramJson.value("rank", 0);
    param.rankSize = paramJson.value("rankSize", 0);
    param.rankRoot = paramJson.value("rankRoot", 0);
    param.hasResidual = paramJson.value("hasResidual", false);
    param.backend = paramJson.value("backend", "hccl");
    param.hcclComm = paramJson.value("hcclComm", nullptr);
    param.commMode = paramJson.value("commMode", R"("COMM_MULTI_PROCESS")"_json).get<atb::infer::CommMode>();
    param.rankTableFile = paramJson.value("rankTableFile", "");
    param.type = paramJson.value("type", \
        R"("LINEAR_ALL_REDUCE")"_json).get<atb::infer::LinearParallelParam::ParallelType>();
    param.keepIntermediate = paramJson.value("keepIntermediate", false);
    param.quantType = paramJson.value("quantType", \
        R"("QUANT_TYPE_UNDEFINED")"_json).get<atb::infer::LinearParallelParam::QuantType>();
    param.quantGroupSize = paramJson.value("quantGroupSize", 0);
    param.outDataType = paramJson.value("outDataType", R"("ACL_DT_UNDEFINED")"_json).get<aclDataType>();
    param.commDomain = paramJson.value("commDomain", "");

    return OperationCreate(param, "LinearParallel");
}

static atb::Operation *LinearSparseOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearSparseParam param;
    param.transposeA = paramJson.value("transposeA", false);
    param.transposeB = paramJson.value("transposeB", true);
    param.tilingK = paramJson.value("tilingK", 1);
    param.tilingN = paramJson.value("tilingN", 1);

    return OperationCreate(param, "LinearSparse");
}

static atb::Operation *RopeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RopeParam param;
    param.rotaryCoeff = paramJson.value("rotaryCoeff", 4);  // 4: 默认值
    param.cosFormat = paramJson.value("cosFormat", 0);

    return OperationCreate(param, "Rope");
}

static atb::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SelfAttentionParam param;
    param.headNum = paramJson.value("headNum", 0);
    param.kvHeadNum = paramJson.value("kvHeadNum", 0);
    param.qScale = paramJson.value("qScale", 1.0f);
    param.qkScale = paramJson.value("qkScale", 1.0f);
    param.batchRunStatusEnable = paramJson.value("batchRunStatusEnable", false);
    param.isTriuMask = paramJson.value("isTriuMask", 0);
    param.calcType = paramJson.value("calcType", \
        R"("UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::CalcType>();
    param.kernelType = paramJson.value("kernelType", \
        R"("KERNELTYPE_DEFAULT")"_json).get<atb::infer::SelfAttentionParam::KernelType>();
    param.clampType = paramJson.value("clampType", \
        R"("CLAMP_TYPE_UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::ClampType>();
    param.clampMin = paramJson.value("clampMin", 0);
    param.clampMax = paramJson.value("clampMax", 0);
    param.maskType = paramJson.value("maskType", \
        R"("MASK_TYPE_UNDEFINED")"_json).get<atb::infer::SelfAttentionParam::MaskType>();
    param.kvcacheCfg = paramJson.value("kvcacheCfg", \
        R"("K_CACHE_V_CACHE")"_json).get<atb::infer::SelfAttentionParam::KvCacheCfg>();
    param.scaleType = paramJson.value("scaleType", \
        R"("SCALE_TYPE_TOR")"_json).get<atb::infer::SelfAttentionParam::ScaleType>();
    param.inputLayout = paramJson.value("inputLayout", \
        R"("TYPE_BSND")"_json).get<atb::infer::InputLayout>();
    param.mlaVHeadSize = paramJson.value("mlaVHeadSize", 0);

    return OperationCreate(param, "SelfAttention");
}

static atb::Operation *PagedAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::PagedAttentionParam param;
    param.headNum = paramJson.value("headNum", 0);
    param.qkScale = paramJson.value("qkScale", 1.0f);
    param.kvHeadNum = paramJson.value("kvHeadNum", 0);
    param.maskType = paramJson.value("maskType", \
        R"("MASK_TYPE_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::MaskType>();
    param.batchRunStatusEnable = paramJson.value("batchRunStatusEnable", false);
    param.quantType = paramJson.value("quantType", \
        R"("TYPE_QUANT_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::QuantType>();
    param.hasQuantOffset = paramJson.value("hasQuantOffset", false);
    param.calcType = paramJson.value("calcType", \
        R"("CALC_TYPE_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::CalcType>();
    param.compressType = paramJson.value("compressType", \
        R"("COMPRESS_TYPE_UNDEFINED")"_json).get<atb::infer::PagedAttentionParam::CompressType>();
    param.scaleType = paramJson.value("scaleType", \
        R"("SCALE_TYPE_TOR")"_json).get<atb::infer::PagedAttentionParam::ScaleType>();
    param.inputLayout = paramJson.value("inputLayout", \
        R"("TYPE_BSND")"_json).get<atb::infer::InputLayout>();
    param.mlaVHeadSize = paramJson.value("mlaVHeadSize", 0);

    return OperationCreate(param, "PagedAttention");
}

static atb::Operation *TransdataOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransdataParam param;
    param.transdataType = paramJson.value("transdataType", \
        R"("UNDEFINED")"_json).get<atb::infer::TransdataParam::TransdataType>();
    for (auto item : paramJson.value("outCrops", R"([])"_json).get<std::vector<int64_t>>()) {
        param.outCrops.push_back(item);
    }

    return OperationCreate(param, "Transdata");
}

static atb::Operation *W8A16OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNWeightQuantBatchMatmulParam aclnnParam;
    aclnnParam.hasBias = paramJson.value("hasBias", false);
    aclnnParam.quantGroupSize = paramJson.value("quantGroupSize", 0);
    aclnnParam.transposeB = paramJson.value("transposeB", false);
    return new atb_speed::common::W8A16Operation("W8A16LinearNode", aclnnParam);
}

static atb::Operation *W4A16OperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNWeightQuantBatchMatmulParam aclnnParam;
    aclnnParam.hasBias = paramJson.value("hasBias", false);
    aclnnParam.quantGroupSize = paramJson.value("quantGroupSize", 0);
    aclnnParam.transposeB = paramJson.value("transposeB", false);
    return new atb_speed::common::W4A16Operation("W4A16LinearNode", aclnnParam);
}

static atb::Operation *IndexputOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::AclNNIndexputParam aclnnParam;
    aclnnParam.accumulate = paramJson.value("accumulate", false);
    aclnnParam.unsafe = paramJson.value("unsafe", true);
    return new atb_speed::common::IndexputOperation("IndexputNode", aclnnParam);
}

static atb::Operation *IndexSelectOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::common::IndexSelectParam indexSelectParam;
    indexSelectParam.dim = paramJson.value("dim", 0);
    return new atb_speed::common::IndexSelectOperation("IndexSelectOperation", indexSelectParam);
}

static atb::Operation *SoftmaxOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SoftmaxParam param;
    for (auto item : paramJson.value("axes", R"([])"_json).get<std::vector<int64_t>>()) {
        param.axes.push_back(item);
    }
    return OperationCreate(param, "Softmax");
}

static atb::Operation *SortOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SortParam param;
    for (auto item : paramJson.value("num", R"([])"_json).get<std::vector<int32_t>>()) {
        param.num.push_back(item);
    }
    return OperationCreate(param, "Sort");
}

static atb::Operation *ReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReduceParam param;
    param.reduceType = \
        paramJson.value("reduceType", R"("REDUCE_UNDEFINED")"_json).get<atb::infer::ReduceParam::ReduceType>();
    for (auto item : paramJson.value("axis", R"([])"_json).get<std::vector<int64_t>>()) {
        param.axis.push_back(item);
    }
    return OperationCreate(param, "Reduce");
}

REGISTER_OPERATION(Activation, ActivationOperationCreate);
REGISTER_OPERATION(Gather, GatherOperationCreate);
REGISTER_OPERATION(Split, SplitOperationCreate);
REGISTER_OPERATION(Concat, ConcatOperationCreate);
REGISTER_OPERATION(Slice, SliceOperationCreate);
REGISTER_OPERATION(Transpose, TransposeOperationCreate);
REGISTER_OPERATION(Elewise, ElewiseOperationCreate);
REGISTER_OPERATION(ReshapeAndCache, ReshapeAndCacheOperationCreate);
REGISTER_OPERATION(LayerNorm, LayerNormOperationCreate);
REGISTER_OPERATION(RmsNorm, RmsNormOperationCreate);
REGISTER_OPERATION(AllGather, AllGatherOperationCreate);
REGISTER_OPERATION(AllReduce, AllReduceOperationCreate);
REGISTER_OPERATION(Linear, LinearOperationCreate);
REGISTER_OPERATION(LinearParallel, LinearParallelOperationCreate);
REGISTER_OPERATION(LinearSparse, LinearSparseOperationCreate);
REGISTER_OPERATION(Rope, RopeOperationCreate);
REGISTER_OPERATION(SelfAttention, SelfAttentionOperationCreate);
REGISTER_OPERATION(PagedAttention, PagedAttentionOperationCreate);
REGISTER_OPERATION(Transdata, TransdataOperationCreate);
REGISTER_OPERATION(W8A16MatMul, W8A16OperationCreate);
REGISTER_OPERATION(W4A16MatMul, W4A16OperationCreate);
REGISTER_OPERATION(Indexput, IndexputOperationCreate);
REGISTER_OPERATION(IndexSelect, IndexSelectOperationCreate);
REGISTER_OPERATION(Softmax, SoftmaxOperationCreate);
REGISTER_OPERATION(Sort, SortOperationCreate);
REGISTER_OPERATION(Reduce, ReduceOperationCreate);
} // namespace atb_torch