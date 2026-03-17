cmake_minimum_required(VERSION 3.16)
project(OpenFHE_Toeplitz)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenFHE REQUIRED)

add_executable(openfhe_toeplitz openfhe_toeplitz.cpp)

target_include_directories(openfhe_toeplitz PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)

target_compile_options(openfhe_toeplitz PRIVATE -fopenmp)
target_link_libraries(openfhe_toeplitz OPENFHEpke OPENFHEcore OPENFHEbinfhe -fopenmp)
