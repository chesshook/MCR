cmake_minimum_required(VERSION 3.16)
project(mosaic_native_chatbot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mosaic_agent STATIC src/agent.cpp)
target_include_directories(mosaic_agent PUBLIC include)

add_executable(mosaic_chat src/main.cpp)
target_link_libraries(mosaic_chat PRIVATE mosaic_agent)

add_executable(mosaic_demo src/demo.cpp)
target_include_directories(mosaic_demo PRIVATE include)

add_executable(mosaic_kernel_test tests/state_kernel_test.cpp)
target_include_directories(mosaic_kernel_test PRIVATE include)

add_executable(mosaic_agent_test tests/agent_test.cpp)
target_link_libraries(mosaic_agent_test PRIVATE mosaic_agent)

if(MSVC)
  target_compile_options(mosaic_agent PRIVATE /O2 /W4)
  target_compile_options(mosaic_chat PRIVATE /O2 /W4)
  target_compile_options(mosaic_demo PRIVATE /O2 /W4)
  target_compile_options(mosaic_kernel_test PRIVATE /O2 /W4)
  target_compile_options(mosaic_agent_test PRIVATE /O2 /W4)
else()
  target_compile_options(mosaic_agent PRIVATE -O2 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_chat PRIVATE -O2 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_demo PRIVATE -O2 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_kernel_test PRIVATE -O2 -Wall -Wextra -Wpedantic)
  target_compile_options(mosaic_agent_test PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME mosaic_state_kernel COMMAND mosaic_kernel_test)
add_test(NAME mosaic_dialogue_agent COMMAND mosaic_agent_test)
