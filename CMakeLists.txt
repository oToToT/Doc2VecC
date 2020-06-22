cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(doc2vecc LANGUAGES CXX CUDA)

include(FindCUDA)

cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
 
add_library(doc2vecc_lib STATIC
    model.cu
    arg_parser.cpp
    vocab.cpp
    huffman.cu
    unigram.cu
)
 
# Request that particles be built with -std=c++14
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++14
target_compile_features(doc2vecc_lib PUBLIC cxx_std_14)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(doc2vecc_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(doc2vecc doc2vecc.cu)
 
#set_property(TARGET doc2vecc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(doc2vecc PRIVATE doc2vecc_lib)
 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET doc2vecc PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

