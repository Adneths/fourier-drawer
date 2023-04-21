cmake_minimum_required(VERSION 3.16)

project(render)

set (SOURCES
    render.cpp
    src/Shader.cpp
    src/VideoEncoder.cpp
    
    src/LineStrip.cpp
    src/Lines.cpp
    src/CyclicalLineStrip.cpp
    
    src/NpFourierSeries.cpp
)

set (HEADERS
    include/core.h
    include/Shader.h
    include/VideoEncoder.h
    
    include/Lines.h
    include/LineStrip.h
    include/CyclicalLineStrip.h

    include/FourierSeries.h
    include/NpFourierSeries.h
)

find_package(OpenGL REQUIRED)
find_package(NumCpp 2.10.1 REQUIRED)
include_directories(
    include
    $ENV{FFMPEG_INCLUDE_PATH}
)
link_directories(
    lib
    $ENV{FFMPEG_LIB_PATH}
)

add_library(render SHARED ${SOURCES} ${HEADERS})
target_link_libraries(render PUBLIC ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib NumCpp::NumCpp avcodec.lib avutil.lib avformat.lib swscale.lib)

add_custom_command(TARGET render POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/render.dll"
    "../../../../libs/render.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"../../../../libs/shaders"
)

target_compile_features(render PRIVATE cxx_std_17)