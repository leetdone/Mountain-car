cmake_minimum_required(VERSION 3.21)

project(cuda_bindings
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Skip LTO to avoid issue with nvcc passthrough
set(PYBIND11_LTO_CXX_FLAGS "")

# Find dependencies
set(PYTHON_EXECUTABLE ${CMAKE_SOURCE_DIR}/../python-3.10.0.amd64/python.exe)
set(PYTHONINTERP_FOUND TRUE)

find_package(PythonLibs)

find_package(CUDA REQUIRED)

# Download pybind11 and set it up
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.8.1
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# Build CUDA
add_library(cuda STATIC
  lib/agent.cu
)
target_include_directories(cuda PUBLIC include)
set_property(TARGET cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(cuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")

# Build CUDA bindings
add_library(cuda_bindings MODULE lib/bindings.cpp)
target_include_directories(cuda_bindings PRIVATE include ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_bindings PRIVATE cuda pybind11::module pybind11::lto pybind11::windows_extras ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})
pybind11_extension(cuda_bindings)
pybind11_strip(cuda_bindings)
set_target_properties(cuda_bindings PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                               CUDA_VISIBILITY_PRESET "hidden")