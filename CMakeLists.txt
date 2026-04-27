cmake_minimum_required(VERSION 3.20)

project(rbp_cpd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(rbp_cpd
    src/main.cpp
)

target_include_directories(rbp_cpd PRIVATE include)
target_link_libraries(rbp_cpd PRIVATE Eigen3::Eigen)