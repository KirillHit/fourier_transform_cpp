ADD_DEFINITIONS(-DSOURCE_DIR="${PROJECT_SOURCE_DIR}/")

add_executable(cft_test cft_test.cpp)
target_link_libraries(cft_test
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)
target_include_directories(
  cft_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
add_test(NAME CustomFourierTransformTest
         COMMAND cft_test)