cmake_minimum_required(VERSION 3.16)
project(mosaic_natural_chat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mosaic_dialogue STATIC src/dialogue_model.cpp)
target_include_directories(mosaic_dialogue PUBLIC include)

add_executable(mosaic_natural_chat src/main.cpp)
target_link_libraries(mosaic_natural_chat PRIVATE mosaic_dialogue)

add_executable(mosaic_natural_bench src/benchmark.cpp)
target_link_libraries(mosaic_natural_bench PRIVATE mosaic_dialogue)

add_executable(mosaic_dialogue_test tests/dialogue_model_test.cpp)
target_link_libraries(mosaic_dialogue_test PRIVATE mosaic_dialogue)

if(MSVC)
  target_compile_options(mosaic_dialogue PRIVATE /O2 /W4)
  target_compile_options(mosaic_natural_chat PRIVATE /O2 /W4)
  target_compile_options(mosaic_natural_bench PRIVATE /O2 /W4)
  target_compile_options(mosaic_dialogue_test PRIVATE /O2 /W4)
else()
  target_compile_options(mosaic_dialogue PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_natural_chat PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_natural_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_dialogue_test PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME mosaic_native_dialogue_model COMMAND mosaic_dialogue_test)
