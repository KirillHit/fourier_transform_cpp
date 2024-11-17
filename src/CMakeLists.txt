option(BUILD_SHARED_LIBS "Build simple_socket shared library" OFF)

add_library(ft_lib)

target_compile_features(ft_lib PUBLIC cxx_std_20)

target_link_libraries(
    ft_lib
  PUBLIC
    ${OpenCV_LIBS})

target_include_directories(
    ft_lib
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>"
)

target_sources(
    ft_lib
  PRIVATE 
    dtf.cpp
    utils.cpp
)