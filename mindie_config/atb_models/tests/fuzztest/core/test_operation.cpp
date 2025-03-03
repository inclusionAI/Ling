/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <atb/atb_infer.h>
#include <algorithm>
#include <random>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(OperationDTFuzz, TopkToppSampling)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "TopkToppSamplingDTFuzzTopkToppSampling";
    
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["topk"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 10);
        parameter["randSeed"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 100000);
        int seedSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 100);
        std::vector<int> seeds(seedSize);
        for (int i = 0; i < seedSize; ++i) {
            seeds[i] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 100000);
        }
        parameter["randSeeds"] = seeds;
        parameter["topkToppSamplingType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, -1, 3);

        std::string parameter_string = parameter.dump();

        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("TopkToppSamplingOperation");
            opFuzzIns.attr("set_name")("TopkToppSampling");
            opFuzzIns.attr("set_param")(parameter_string);
            opFuzzIns.attr("execute")();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, Broadcast)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "BroadcastDTFuzzBroadcast";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        parameter["rankRoot"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        parameter["rankSize"] = worldSize;
        std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
        parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(
            &g_Element[fuzzIndex++], 0, 0, 1)];

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("BroadcastOperation");
            opFuzzIns.attr("set_name")("Broadcast");
            opFuzzIns.attr("execute_with_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, AllReduce)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "AllReduceDTFuzzAllReduce";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        parameter["rankRoot"] = parameter["rank"];
        parameter["rankSize"] = worldSize;
        std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
        parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
        std::vector<std::string> allReduceTypeEnumTable = {"sum", "prod", "max", "min"};
        parameter["allReduceType"] = allReduceTypeEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 3)];

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("AllReduceOperation");
            opFuzzIns.attr("set_name")("AllReduce");
            opFuzzIns.attr("execute_out_with_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, AllGather)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "AllGatherDTFuzzAllGather";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        parameter["rankRoot"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        parameter["rankSize"] = worldSize;

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("AllGatherOperation");
            opFuzzIns.attr("set_name")("AllGather");
            opFuzzIns.attr("set_param")(parameter_string);
            opFuzzIns.attr("execute_out")();
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, Transdata)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "TransdataDTFuzzTransdata";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["transdataType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 2);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("TransdataOperation");
            opFuzzIns.attr("set_name")("Transdata");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, Transpose)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "TransposeDTFuzzTranspose";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    int index = 0;
    DT_FUZZ_START(0, 2, const_cast<char*>(fuzzName.c_str()), 0) {
        index++;
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        int permSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        std::vector<int> perm(permSize);
        for (int i = 0; i < permSize; ++i) {
            perm[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(perm.begin(), perm.end(), g);

        parameter["perm"] = perm;

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("TransposeOperation");
            opFuzzIns.attr("set_name")("Transpose");
            if (index % 2 == 0) {
                opFuzzIns.attr("set_param")(parameter_string);
            } else {
                opFuzzIns.attr("execute_out_with_param")("notajsonstring");
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, W8A16Operation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "W8A16OperationDTFuzzW8A16Operation";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["transposeB"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["groupSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 32, 32, 128);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("W8A16Operation");
            opFuzzIns.attr("set_name")("W8A16Operation");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, W4A16Operation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "W8A16OperationDTFuzzW8A16Operation";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["transposeB"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["groupSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 32, 32, 128);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("W4A16Operation");
            opFuzzIns.attr("set_name")("W4A16Operation");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, W8A8Operation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "W8A8OperationDTFuzzW8A8Operation";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["transposeB"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("W8A8Operation");
            opFuzzIns.attr("set_name")("W8A8Operation");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, Activation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "ActivationDTFuzzActivation";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["activationType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 8);
        parameter["scale"] = float(std::rand()) / RAND_MAX;

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("ActivationOperation");
            opFuzzIns.attr("set_name")("Activation");
            opFuzzIns.attr("set_param")("notajsonstring");
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, Linear)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "LinearOpDTFuzzLinear";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    int index = 0;
    DT_FUZZ_START(0, 2, const_cast<char*>(fuzzName.c_str()), 0) {
        index++;
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["transposeA"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["transposeB"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["outDataType"] = FuzzUtil::GetRandomAclDataType(std::rand());

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("LinearOperation");
            opFuzzIns.attr("set_name")("Linear");
            if (index % 2 == 0) {
                opFuzzIns.attr("execute")();
            } else {
                opFuzzIns.attr("execute_with_param")("notajsonstring");
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, AclNNAttention)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "AclNNAttentionDTFuzzAclNNAttention";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    int index = 0;
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        index++;
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["headNum"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 32, 32, 64);
        parameter["kvHeadNum"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 32, 32, 64);
        parameter["headDim"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 128, 128, 512);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("LinearOperation");
            opFuzzIns.attr("set_name")("Linear");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, GroupedMatmul)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "GroupedMatmulOpDTFuzzLinear";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    int index = 0;
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        index++;
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["transposeB"] = FuzzUtil::GetRandomBool(fuzzIndex);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("GroupedMatmulOperationCreate");
            opFuzzIns.attr("set_name")("GroupedMatmulOperation");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, LinearWithLora)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "LinearWithLoraOpDTFuzzLinear";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    int index = 0;
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        index++;
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["supportLora"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["loraEnableGMM"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["transposeType"] = std::rand() % 3 - 1;

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("LinearWithLoraOperationCreate");
            opFuzzIns.attr("set_name")("LinearWithLoraOperation");
            opFuzzIns.attr("set_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, GeluOperation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "GeluOperationDTFuzzGeluOperation";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["geluApproximate"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], -1, 1, 1);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("GeluOperation");
            opFuzzIns.attr("set_name")("GeluOperation");
            opFuzzIns.attr("set_param")(parameter_string);
            opFuzzIns.attr("execute_out")();
            opFuzzIns.attr("execute_out_with_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, LayerNormOperation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "LayerNormOperationDTFuzzLayerNormOperation";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["layerNormEps"] = float(std::rand()) / RAND_MAX;
        parameter["layerNormImplMode"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 2);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("LayerNormOperation");
            opFuzzIns.attr("set_name")("LayerNormOperation");
            opFuzzIns.attr("set_param")(parameter_string);
            opFuzzIns.attr("execute_out")();
            opFuzzIns.attr("execute_out_with_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, IndexSelectOperation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "IndexSelectOperationDTFuzz";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["dim"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, -8, 7);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("IndexSelectOperation");
            opFuzzIns.attr("set_name")("IndexSelectOperation");
            opFuzzIns.attr("set_param")(parameter_string);
            opFuzzIns.attr("execute_out")();
            opFuzzIns.attr("execute_out_with_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(OperationDTFuzz, IndexputOperation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module opModule = pybind11::module_::import("operation_fuzz");
    std::string fuzzName = "IndexputOperationDTFuzz";
    DT_FUZZ_START(0, 1, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter;
        parameter["accumulate"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["unsafe"] = FuzzUtil::GetRandomBool(fuzzIndex);

        std::string parameter_string = parameter.dump();
        try {
            pybind11::object opFuzz = opModule.attr("OperationFuzz");
            pybind11::object opFuzzIns = opFuzz("IndexputOperation");
            opFuzzIns.attr("set_name")("IndexputOperation");
            opFuzzIns.attr("set_param")(parameter_string);
            opFuzzIns.attr("execute_out")();
            opFuzzIns.attr("execute_out_with_param")(parameter_string);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}