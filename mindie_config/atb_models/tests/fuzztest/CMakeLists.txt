# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendTransformerBoost is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# 

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
set(pybind11_DIR ${pybind11_DIR})
find_package(pybind11 CONFIG)

add_executable(speed_fuzztest ${SOURCE_FILES})
target_compile_options(speed_fuzztest PRIVATE "-Wno-unused-function" "-Wno-unused-variable" "-Wno-unused-parameter" "-Wno-attributes")
target_include_directories(speed_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/googletest/include)
target_link_directories(speed_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/googletest/lib)
target_include_directories(speed_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/secodefuzz/include)
target_link_directories(speed_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/secodefuzz/lib)
target_link_libraries(speed_fuzztest PRIVATE atb_speed_torch atb_speed_core atb_speed_models atb_speed_operations pybind11::embed ascendcl -lSecodefuzz -lgtest -lgtest_main -lpthread -lm)
install(TARGETS speed_fuzztest DESTINATION bin)