cmake_minimum_required(VERSION 3.16)
project(mosaic_state_kernel LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mosaic_demo src/demo.cpp)
target_include_directories(mosaic_demo PRIVATE include)

add_executable(mosaic_kernel_test tests/state_kernel_test.cpp)
target_include_directories(mosaic_kernel_test PRIVATE include)

if(MSVC)
  target_compile_options(mosaic_demo PRIVATE /O2 /W4)
  target_compile_options(mosaic_kernel_test PRIVATE /O2 /W4)
else()
  target_compile_options(mosaic_demo PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_kernel_test PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME mosaic_state_kernel COMMAND mosaic_kernel_test)
