cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(doc2vecc LANGUAGES CXX CUDA)

include(FindCUDAToolkit)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()


set(CMAKE_CUDA_ARCHITECTURES 72)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(
    doc2vecc_lib STATIC
    model.cu
    utils/arg_parser.cpp
    components/vocab.cpp
    libs/huffman.cu
    libs/unigram.cu
)

add_executable(doc2vecc doc2vecc.cpp)
target_link_libraries(doc2vecc doc2vecc_lib)
