cmake_minimum_required(VERSION 3.8)
project("atb_models")
set(CMAKE_CXX_STANDARD 17)

execute_process(
    COMMAND sudo sh -c "echo 2 > /proc/sys/kernel/randomize_va_space"
    RESULT_VARIABLE result
)
if(NOT result EQUAL 0)
    message(WARNING "Failed to set ASLR level.")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atb_framework
    ${PROJECT_SOURCE_DIR}/atb_framework/core/include
    ${PROJECT_SOURCE_DIR}/atb_framework/models
    ${PROJECT_SOURCE_DIR}/atb_framework/operations
    ${PROJECT_SOURCE_DIR}/third_party/nlohmannJson/include
    ${PROJECT_SOURCE_DIR}/third_party/spdlog/include
    ${PROJECT_SOURCE_DIR}/../../third_party/nlohmann/include
    ${PROJECT_SOURCE_DIR}/../../third_party/spdlog/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_TOOLKIT_HOME}/lib64)

#options, these can be pass in by cmake argument
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
option(USE_OPTIMIZE "USE_OPTIMIZE" ON)
option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
option(USE_PYTHON_TEST "USE_PYTHON_TEST" OFF)
option(USE_FUZZ_TEST "USE_FUZZ_TEST" OFF)

message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")
message(STATUS "USE_PYTHON_TEST:${USE_PYTHON_TEST}")
message(STATUS "USE_FUZZ_TEST:${USE_FUZZ_TEST}")

# compile flag
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output/atb"
        CACHE PATH "..." FORCE)
endif()
message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 -O2")

if(USE_OPTIMIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fpie -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")

IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_LINK_OPTIONS(-s)
ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -g")
ENDIF()

set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
    ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
if(TORCH_GET_TENSOR_NPU_FORMAT_OLD)
    message(STATUS "define TORCH_GET_TENSOR_NPU_FORMAT_OLD")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_GET_TENSOR_NPU_FORMAT_OLD")
endif()
if(TORCH_SETCUSTOMHANDLER)
    message(STATUS "define TORCH_SETCUSTOMHANDLER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_SETCUSTOMHANDLER")
endif()
if(TORCH_HIGHER_THAN_PTA6)
    message(STATUS "define TORCH_HIGHER_THAN_PTA6")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_HIGHER_THAN_PTA6")
endif()
if(USE_UNIT_TEST OR USE_PYTHON_TEST OR USE_FUZZ_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O3")
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
    add_subdirectory(tests)
endif()

add_subdirectory(atb_framework/core)
add_subdirectory(atb_framework/models)
add_subdirectory(atb_framework/pytorch)
add_subdirectory(atb_framework/operations)

install(FILES ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./)