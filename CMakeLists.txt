cmake_minimum_required(VERSION 3.15)
project(quartet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_TIMING "Enable per-window timing instrumentation" OFF)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(quartet
    src/main.cpp
    src/DataLoader.cpp
    src/GeometryTools.cpp
    src/Optimizer.cpp
    src/FileUtils.cpp
)

target_include_directories(quartet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(quartet PUBLIC
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
)

if(ENABLE_TIMING)
    target_compile_definitions(quartet PRIVATE ENABLE_TIMING)
endif()
