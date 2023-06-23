cmake_minimum_required(VERSION 3.16)
project(render LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit)


set (LIB_FILES
    src/Lines.cpp
    src/LineStrip.cpp
    
    include/Lines.h
    include/LineStrip.h
)

set (SOURCES
    render.cpp
    src/Shader.cpp
    src/VideoEncoder.cpp
    
    src/NpFourierSeries.cpp

    src/MultiBuffer.cpp
)

set (HEADERS
    include/constant.h

    include/core.h
    include/Shader.h
    include/VideoEncoder.h

    include/FourierSeries.h
    include/NpFourierSeries.h

    include/MultiBuffer.h
)

set (CUDA_FILES    
    include/CudaFourierSeries.cuh
    src/CudaFourierSeries.cu
)

find_package(OpenGL REQUIRED)
find_package(NumCpp 2.10.1 REQUIRED)
include_directories(
    include
    $ENV{FFMPEG_INCLUDE_PATH}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
link_directories(
    lib
    $ENV{FFMPEG_LIB_PATH}
)

add_library(lines ${LIB_FILES})
target_link_libraries(lines PUBLIC ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib)

add_library(render SHARED ${SOURCES} ${HEADERS})
target_link_libraries(render PUBLIC ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib NumCpp::NumCpp avcodec.lib avutil.lib avformat.lib swscale.lib lines)

add_library(fourier_cuda SHARED ${HEADERS} ${CUDA_FILES})
#set_target_properties(fourier_cuda PROPERTIES CUDA_ARCHITECTURES "50;61;75")
#set_target_properties(fourier_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(fourier_cuda PUBLIC ${OPENGL_LIBRARIES} CUDA::cudart)
target_link_libraries(fourier_cuda PUBLIC lines ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib)

add_custom_command(TARGET render POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/render.dll"
    "../../../../libs/render.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"../../../../libs/shaders"
)
add_custom_command(TARGET fourier_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/fourier_cuda.dll"
    "../../../../libs/fourier_cuda.dll"
)

target_compile_features(render PRIVATE cxx_std_17)
target_compile_features(fourier_cuda PRIVATE cxx_std_17)

add_executable(debug debug.cpp)
add_custom_command(TARGET debug POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"${CMAKE_BINARY_DIR}/libs/shaders"
)