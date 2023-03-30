cmake_minimum_required(VERSION 3.16)

project(render)

set (SOURCES
    render.cpp
)

set (HEADERS
    include/core.h
)

find_package(OpenGL REQUIRED)
include_directories(
    include
)
link_directories(
    lib
)

add_library(render SHARED 
    ${SOURCES} ${HEADERS}
)
target_link_libraries(render PUBLIC ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib)

add_custom_command(TARGET render POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/render.dll"
    "../../../../libs/render.dll"
)

target_compile_features(render PRIVATE cxx_std_17)