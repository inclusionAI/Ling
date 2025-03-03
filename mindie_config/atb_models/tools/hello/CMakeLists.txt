file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(acltransformer_hello ${SOURCE_FILES})

if(USE_TORCH_RUNNER)
    target_link_libraries(acltransformer_hello PRIVATE acltransformer dl
        torch c10 torch_cpu torch_python torch_npu python3 pthread)
else()
    target_link_libraries(acltransformer_hello PRIVATE acltransformer dl pthread)
endif()

if(USE_LCCL_RUNNER)
    target_link_libraries(acltransformer_hello PRIVATE lccl ascendcl)
endif()

if(USE_HCCL_RUNNER)
    target_link_libraries(acltransformer_hello PRIVATE hccl pthread)
endif()

install(TARGETS acltransformer_hello DESTINATION bin)