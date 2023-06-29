cmake_minimum_required(VERSION 3.16)
project(render LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit)


set (SOURCES
    render.cpp
    src/Shader.cpp
    src/VideoEncoder.cpp

    src/MultiBuffer.cpp
    src/RenderInstance.cpp
    
    src/Lines.cpp
    src/LineStrip.cpp
)

set (HEADERS
    include/constant.h

    include/core.h
    include/Shader.h
    include/VideoEncoder.h

    include/FourierSeries.h

    include/RenderParam.h
    include/MultiBuffer.h
    include/RenderInstance.h

    include/Lines.h
    include/LineStrip.h
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

add_library(render SHARED ${SOURCES} ${HEADERS} include/NpFourierSeries.h src/NpFourierSeries.cpp)
target_link_libraries(render PUBLIC ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib NumCpp::NumCpp avcodec.lib avutil.lib avformat.lib swscale.lib)
target_compile_features(render PRIVATE cxx_std_17)

add_library(cuda_render SHARED ${SOURCES} ${HEADERS} include/CudaFourierSeries.cuh src/CudaFourierSeries.cu)
#set_target_properties(fourier_cuda PROPERTIES CUDA_ARCHITECTURES "50;60;61;75")
target_link_libraries(cuda_render PUBLIC CUDA::cudart ${OPENGL_LIBRARIES} glew32s.lib glfw3.lib NumCpp::NumCpp avcodec.lib avutil.lib avformat.lib swscale.lib)
target_compile_features(cuda_render PRIVATE cxx_std_17)
target_compile_definitions(cuda_render PRIVATE COMPILE_CUDA)

add_library(cuda_info SHARED include/constant.h src/cudaInfo.cu)
target_link_libraries(cuda_info PUBLIC CUDA::cudart)
target_compile_features(cuda_info PRIVATE cxx_std_17)
target_compile_definitions(cuda_info PRIVATE COMPILE_CUDA)

add_custom_command(TARGET render POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/render.dll"
    "../../../../libs/render.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"../../../../libs/shaders"
)
add_custom_command(TARGET cuda_render POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/cuda_render.dll"
    "../../../../libs/cuda_render.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"../../../../libs/shaders"
)
add_custom_command(TARGET cuda_info POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/cuda_info.dll"
    "../../../../libs/cuda_info.dll"
)

add_executable(debug debug.cpp)
add_custom_command(TARGET debug POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"${CMAKE_BINARY_DIR}/libs/shaders"
)