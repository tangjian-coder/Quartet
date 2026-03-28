cmake_minimum_required(VERSION 3.15)
project(QuaternionOptimizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -static")
endif()

# Use vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(CMAKE_PREFIX_PATH "C:/vcpkg/installed/x64-mingw-dynamic" ${CMAKE_PREFIX_PATH})

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)

# Add executable
add_executable(quaternion_optimizer src/main.cpp src/DataLoader.cpp src/GeometryTools.cpp src/Optimizer.cpp src/FileUtils.cpp)

# Include directories for the target
target_include_directories(quaternion_optimizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "C:/Users/24923/Desktop/yaml-cpp/include"
)

# Add compile definition for static yaml-cpp
target_compile_definitions(quaternion_optimizer PUBLIC YAML_CPP_STATIC_DEFINE)

# Link libraries
target_link_libraries(quaternion_optimizer PUBLIC
    ${NLOPT_LIBRARIES}
    "C:/Users/24923/Desktop/yaml-cpp/build/libyaml-cpp.a"
    Eigen3::Eigen
)
