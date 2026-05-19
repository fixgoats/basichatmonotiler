cmake_minimum_required(VERSION 3.24)

project(basichatmonotile)

set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD 26)

add_compile_options(-w "$<$<CONFIG:release>:-O3;-march=native>")

include_directories(include external/eigen-5.0.1 external/kdtree
                    external/cxxopts)

find_package(OpenMP)
if(OpenMP_Found)
  message("Found OpenMP")
endif()
find_package(BLAS COMPONENTS openmp)
message("Found BLAS libraries ${BLAS_LIBRARIES}")
if(BLAS_FOUND)
  set(EIGEN_USE_BLAS)
endif()

add_executable(makemonotile src/makemonotile.cpp)

option(MONOTILE_VISUAL "Compile with Raylib" OFF)
if(MONOTILE_VISUAL)
  target_compile_definitions(makemonotile PRIVATE MONOTILE_VISUAL)
endif()
include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT output)

if(ipoSupported AND CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Using LTO")
  set_property(TARGET makemonotile PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Build type is not release, LTO not enabled.")
else()
  message(STATUS "LTO is not supported: ${output}")
endif()

set_property(
  TARGET makemonotile PROPERTY COMPILE_OPTIONS -Wall -Wextra -Wpedantic
                               "$<$<CONFIG:debug>:-fsanitize=address;-ggdb3>")
set_property(TARGET makemonotile
             PROPERTY LINK_OPTIONS "$<$<CONFIG:debug>:-fsanitize=address>")

if(OpenMP_Found)
  target_link_libraries(makemonotile PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MONOTILE_VISUAL)
  include(FetchContent)
  set(FETCHCONTENT_QUIET FALSE)
  set(BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE) # don't build the supplied examples
  set(BUILD_GAMES
      OFF
      CACHE BOOL "" FORCE) # don't build the supplied example games

  FetchContent_Declare(
    raylib
    GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
    GIT_TAG "5.5"
    GIT_PROGRESS TRUE)

  FetchContent_MakeAvailable(raylib)
  target_link_libraries(makemonotile PRIVATE raylib)
endif()
