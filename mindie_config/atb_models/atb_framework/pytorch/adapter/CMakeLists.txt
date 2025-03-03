file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

include_directories(
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
)

link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
)

add_library(atb_speed_torch SHARED ${SOURCE_FILES})
target_compile_options(atb_speed_torch PRIVATE "-Wno-unused-function" "-Wno-unused-variable" "-Wno-unused-parameter" "-Wno-attributes")
target_link_libraries(atb_speed_torch PRIVATE
    atb atb_speed_core atb_speed_models atb_speed_operations torch c10
    torch_cpu torch_python torch_npu)
install(TARGETS atb_speed_torch DESTINATION lib)