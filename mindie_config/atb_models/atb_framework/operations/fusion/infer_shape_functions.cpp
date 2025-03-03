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
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace common {

void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape = oldShape;
    if (oldShape.dimNum >= 2) {  // 2: 对输入tensor的后两维进行合并，若维度小于2则不做修改
        newShape.dimNum = oldShape.dimNum - 1;
        newShape.dims[newShape.dimNum - 1] = \
            CheckIntMulOverFlow(oldShape.dims[oldShape.dimNum - 2], oldShape.dims[oldShape.dimNum - 1]);  // 2: index
    }
}

void UnsqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape, int32_t headNum, int32_t headDim)
{
    newShape = oldShape;
    if (oldShape.dimNum == 0) {
        return;
    }
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[newShape.dimNum - 2] = headNum;  // -2: headNum
    newShape.dims[newShape.dimNum - 1] =  headDim;  // -1: headDim
}

void UnsqueezeAxis(const atb::Dims &oldShape, atb::Dims &newShape, int32_t axis)
{
    newShape = oldShape;
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[axis] = 1;
    for (uint64_t i = axis + 1; i < std::min(newShape.dimNum, static_cast<uint64_t>(8)); i++) {  // 8: tensor维度上限
        newShape.dims[i] = oldShape.dims[i - 1];
    }
}

void SqueezeBatchAndHiddenSize(const atb::Dims& oldShape, atb::Dims& newShape)
{
    if (oldShape.dimNum == 4) {  // 4: 若输入是[B,S,N,D]，则合并为[BS,ND]
        newShape.dimNum = 2;  // 2: [BS,ND]
        newShape.dims[0] = CheckIntMulOverFlow(oldShape.dims[0], oldShape.dims[1]);  // 0,0,1: [B,S] => [BS]
        newShape.dims[1] = CheckIntMulOverFlow(oldShape.dims[2], oldShape.dims[3]);  // 1,2,3: [N,D] => [ND]
    } else {
        newShape = oldShape;
    }
}

void InternlmV2QKVSplit(
    const atb::Dims& oldShape, atb::Dims& newShape, int32_t headNum, int32_t kvHeadNum, int32_t headDim)
{
    if (kvHeadNum == 0 || headDim == 0) {
        ATB_SPEED_LOG_ERROR("kvHeadNum or headDim is 0 in InternlmV2QKVSplit, "
                       << "reshape failed, newShape remains the same as oldShape");
        newShape = oldShape;
        return;
    }
    newShape.dimNum = 4;  // 4: 新的shape维度为4
    size_t newShapeDimIndex = 0;
    size_t oldShapeDimIndex = 0;
    newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    newShape.dims[newShapeDimIndex++] = \
        oldShape.dims[oldShapeDimIndex++] / (CheckIntMulOverFlow(
            (2 + headNum / kvHeadNum), headDim)  // 2: k + v linear
    );
    if ((2 + headNum / kvHeadNum)  // 2: k + v linear
        > std::numeric_limits<int32_t>::max()) {
        newShape = oldShape;
        return;
    }
    newShape.dims[newShapeDimIndex++] = 2 + headNum / kvHeadNum; // 2: k + v linear
    newShape.dims[newShapeDimIndex++] = headDim;
}

} // namespace common
} // namespace atb_speed