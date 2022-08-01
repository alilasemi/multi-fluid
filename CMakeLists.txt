cmake_minimum_required(VERSION 3.10)
# Default build type
set(CMAKE_BUILD_TYPE "Release")

# run path for installed executables
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set the project info
project(level_set VERSION 1.0)

# specify default cmake options
# TODO keep this or no?
option(USE_DOUBLES      "Use double precision" ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#TODO this vs next line?
option(USE_OMP          "Use OpenMP" ON)
find_package(OpenMP REQUIRED)

# Specify compiler information
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(warning_flags "\
-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
-Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op \
-Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual \
-Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wswitch-default -Wundef \
")

set(CMAKE_CXX_FLAGS_DEBUG    "-g  -DDEBUG ${warning_flags}")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 ${warning_flags}")

# Set preprocessor definitions
# TODO
if(USE_DOUBLES)
    add_compile_definitions(USE_DOUBLES)
endif()

# Pybind includes - get them from the Python module command and clean up the
# string to get the paths
execute_process (
    COMMAND python -m pybind11 --includes
    OUTPUT_VARIABLE pybind11_includes
)
STRING(REGEX REPLACE "^-I" "" pybind11_includes ${pybind11_includes})
STRING(REGEX REPLACE " -I" ";" pybind11_includes ${pybind11_includes})

# Global include
include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${CMAKE_SOURCE_DIR}/src")
# Making Eigen and Pybind SYSTEM includes makes warnings not apply to those headers
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/eigen")
include_directories(SYSTEM ${pybind11_includes})

# Project subdirectories
add_subdirectory(src)

# Print summary
message(STATUS "-----------------------------------------------------------")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "CMAKE_CXX_COMPILER:      ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:   ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Pybind11 includes:       ${pybind11_includes}")
message(STATUS "")
message(STATUS "-----------------------------------------------------------")
