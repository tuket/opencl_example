cmake_minimum_required (VERSION 3.2)

project (cl_example)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(cl_example
    main.cpp
)
target_link_libraries(cl_example
    ${OpenCL_LIBRARY}
)