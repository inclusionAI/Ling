file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_library(atb_speed_operations SHARED ${SOURCE_FILES})
target_link_libraries(atb_speed_operations PRIVATE atb opapi)
install(TARGETS atb_speed_operations DESTINATION lib)