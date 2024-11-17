cmake_minimum_required(VERSION 3.1...3.28)

option(Demo "Demo" ON)

project(ft_cpp LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)

find_package(OpenCV REQUIRED)

add_subdirectory(src)

if (Demo)
    add_subdirectory(demo)
endif()