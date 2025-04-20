cmake_minimum_required(VERSION 3.18)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ✅ Pybind11 소스 직접 추가 (submodule로 추가한 경우)
add_subdirectory(extern/pybind11)

# 라이브러리 생성
add_library(my_cuda_module MODULE
    src/vector_add.cu
    src/matmul.cu
    src/softmax.cu
    src/relu.cu
    src/loss.cu
    src/two_layers_mlp.cu
    src/optimizer.cu
    src/wrapper.cpp
)

# ✅ pybind11::module 연결
target_link_libraries(my_cuda_module PRIVATE pybind11::module)

# ✅ 확장자 제거 (Windows용)
set_target_properties(my_cuda_module PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
)
