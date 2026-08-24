cmake_minimum_required(VERSION 3.20)
project(cpp_template_gpt2 CXX)

set(CMAKE_CXX_STANDARD 23)
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
