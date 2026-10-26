cmake_minimum_required(VERSION 3.16)
project(MiniTPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # clang-tidy -p build

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Globbed with CONFIGURE_DEPENDS, so a new file needs no edit here.
file(GLOB_RECURSE CORE_SRC CONFIGURE_DEPENDS src/*.cpp)

# tpu_core: the whole machine. Header-only until the first .cpp arrives.
if(CORE_SRC)
  add_library(tpu_core STATIC ${CORE_SRC})
  target_include_directories(tpu_core PUBLIC src)
else()
  add_library(tpu_core INTERFACE)
  target_include_directories(tpu_core INTERFACE src)
endif()

# One executable per tools/*.cpp that exists so far.
foreach(tool tpu_sim gen_programs roofline check_headers)
  if(EXISTS ${CMAKE_SOURCE_DIR}/tools/${tool}.cpp)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE tpu_core)
  endif()
endforeach()

file(GLOB TEST_SRC CONFIGURE_DEPENDS tests/*.cpp)
add_executable(tpu_tests ${TEST_SRC})
target_include_directories(tpu_tests PRIVATE tests)
target_link_libraries(tpu_tests PRIVATE tpu_core)

# `cmake --build build --target lint`: the /// header rule on every file, then pinned clang-tidy over every .cpp.
file(GLOB_RECURSE LINT_SRC CONFIGURE_DEPENDS src/*.cpp tests/*.cpp tools/*.cpp)
file(GLOB_RECURSE LINT_HDR CONFIGURE_DEPENDS src/*.h tests/*.h)
set(LINT_HEADER_FILTER "${CMAKE_SOURCE_DIR}/(src|tests|tools)/")   # headers clang-tidy reports on
set(LINT_ARGS)
if(APPLE)
  # The pip-built clang-tidy does not know where the macOS SDK headers are.
  execute_process(COMMAND xcrun --show-sdk-path
                  OUTPUT_VARIABLE SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND LINT_ARGS --extra-arg=-isysroot${SDK_PATH})
endif()
add_custom_target(lint
  COMMAND check_headers ${LINT_HDR} ${LINT_SRC}
  COMMAND uvx clang-tidy@19.1.0.1 --quiet -p ${CMAKE_BINARY_DIR} ${LINT_ARGS}
          --header-filter=${LINT_HEADER_FILTER} ${LINT_SRC}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  VERBATIM)
add_dependencies(lint check_headers)
