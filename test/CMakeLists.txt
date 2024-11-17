ADD_DEFINITIONS(-DSOURCE_DIR="${PROJECT_SOURCE_DIR}/")

add_executable(dft_test dft_test.cpp)
target_link_libraries(dft_test
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)
target_include_directories(
  dft_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
add_test(NAME DFTTes
         COMMAND dft_test)