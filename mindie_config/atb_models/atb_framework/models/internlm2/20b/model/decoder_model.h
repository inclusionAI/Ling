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
#ifndef ATB_SPEED_MODELS_INTERNLM2_20B_PAGE_ATTENTION_MODEL_H
#define ATB_SPEED_MODELS_INTERNLM2_20B_PAGE_ATTENTION_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "models/internlm2/20b/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace internlm2_20b {

enum DecoderModelTensorIdx : uint32_t {
    // define inTensor
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT_IDS = 0,
    // idx: 1, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    IN_TENSOR_INPUT_EMBEDDING,
    // idx: 2, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 4, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 5, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 6, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 7, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 8, shape: [1]; FA所需入参
    IN_TENSOR_KV_CACHE_IDX,
    // idx: 9, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 10, shape: [1]
    IN_TENSOR_PLACE_HOLDER,
    // idx: 11, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LEN,
    // idx: 12, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,
    // idx: 13, shape: [seqLen, 1]; Lora所需参数，首token推理只关注图像部分
    IN_TENSOR_LORA_IM_MASK,
};

enum DecoderModelInternalTensorIdx : uint32_t {
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    INTERNAL_TENSOR_COS_EMB,
    INTERNAL_TENSOR_SIN_EMB
};

class DecoderModel : public Model {
public:
    struct Param {
        // skipWordEmbedding为true会跳过Word Embedding阶段，直接使用入参中的IN_TENSOR_INPUT_EMBEDDING
        bool skipWordEmbedding = false;
        // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isFA = true;
        // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isPrefill = false;
        // isBF16为true时采用BF16精度; 反之，则采用FP16精度
        bool isBF16 = false;
        // isEmbeddingParallel为true时，embedding的权重在hiddenSize维度进行切分; 反之，则不对权重进行切分; 测试表明embedding切分并不会带来性能提升
        bool isEmbeddingParallel = false;
        // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
        bool isLmHeadParallel = true;
        int lmHeadTransposeType = -1;
        // MLP是否使用SwiGLU，若为true时，则使用；反之，使用swish
        bool supportSwiGLU = false;
        // 是否支持通信计算掩盖
        bool supportLcoc = false;
        // 是否支持lora
        bool supportLora = false;
        // 是否使用image_mask, 仅在supportLora为true时生效
        bool useImMask = false;
        // 是否使用kv cache int8 量化
        bool kvQuant = false;
        float rmsNormEps = 0;
        uint32_t numAttentionHeadsPerRank = 0;
        uint32_t hiddenSizePerAttentionHead = 0;
        uint32_t numHiddenLayers = 0;
        uint32_t numKeyValueHeadsPerRank = 0;
        int rank = 0;
        int worldSize = 1;
        bool splitWithStride = true;
        std::string backend = "hccl";
        std::string rankTableFile = "";
        std::string positionEmbeddingType = "ROPE";
        std::vector<int> tokenOffset = {};
        std::vector<int> seqLen = {};
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> linearQuantType = {};
        std::vector<std::vector<int>> linearTransposeType = {};
        void FromString(const std::string &param);
        void PrintParam();
        void ParseBasicParams(const nlohmann::json &paramJson);
    };

    explicit DecoderModel(const std::string &param);
    ~DecoderModel() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddPositionalEmbedding();
    int64_t AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmhead();
    void SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId);
    void AddLoraLayerIntensor(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    Param param_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    std::vector<int> qLen_;
    uint32_t layerId_ = 0;
    uint32_t weightCountPerLayer = 50;
};

REGISTER_MODEL(internlm2_20b, DecoderModel);

}  // namespace internlm2_20b
}  // namespace atb_speed
#endif
