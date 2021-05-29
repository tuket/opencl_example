cmake_minimum_required (VERSION 3.2)

project (cl_example)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")

find_package(OpenCL REQUIRED)
find_package(Vulkan REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

#add_compile_definitions(TRACY_ENABLE)

add_executable(cl_example
    main.cpp
    tracy/TracyClient.cpp
    tracy/TracyC.h
    tracy/Tracy.hpp
)
target_link_libraries(cl_example
    ${OpenCL_LIBRARY}
    ${Vulkan_LIBRARY}
)