cmake_minimum_required(VERSION 3.8)

project(ft_cpp LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)
include(CTest)

find_package(OpenCV REQUIRED)

add_subdirectory(src)

add_subdirectory(test)