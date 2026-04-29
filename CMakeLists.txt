cmake_minimum_required(VERSION 3.16)

project(rbp_cpd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(rbp_cpd
    src/main.cpp
    src/csv_loader.cpp
    src/rbp.cpp
)

target_include_directories(rbp_cpd PRIVATE include)
target_link_libraries(rbp_cpd PRIVATE Eigen3::Eigen)