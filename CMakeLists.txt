cmake_minimum_required(VERSION 3.20)
project(cpp_template_gpt2 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gpt2 main.cpp)
target_include_directories(gpt2 PRIVATE include)
# Deep template recursion headroom (the EXP LUT is a constexpr data array, but
# reductions/attention still nest); bump well past the default.
target_compile_options(gpt2 PRIVATE -ftemplate-depth=4096 -Wall -Wextra)

# -DSHOW makes the compiler print the generated token ids in a diagnostic
# (see main.cpp). Off by default so the build succeeds on the static_assert gate.
option(SHOW "Emit generated ids via a deliberate compile-time diagnostic" OFF)
if(SHOW)
  target_compile_definitions(gpt2 PRIVATE SHOW=1)
endif()

# Build-time prompt / generation length. Empty => use the defaults baked into
# main.cpp + model.hpp (which keep the golden static_assert gate active).
set(PROMPT "" CACHE STRING "Compile-time prompt (in-vocab: space a-z . , ' ! ?)")
set(GEN_N "" CACHE STRING "Number of tokens to greedily generate")
set(EXPECT "" CACHE STRING "Expected full output text; static_asserts Generated==EXPECT")
# NOTE: test for non-empty with STREQUAL, not if(VAR) -- CMake treats strings
# like "no"/"off"/"0" as boolean false, which would silently drop the prompt.
if(NOT PROMPT STREQUAL "")
  target_compile_definitions(gpt2 PRIVATE PROMPT="${PROMPT}")
endif()
if(NOT GEN_N STREQUAL "")
  target_compile_definitions(gpt2 PRIVATE GEN_N=${GEN_N})
endif()
if(NOT EXPECT STREQUAL "")
  target_compile_definitions(gpt2 PRIVATE EXPECT="${EXPECT}")
endif()
