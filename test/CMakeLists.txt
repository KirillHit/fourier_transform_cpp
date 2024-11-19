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

add_executable(freq_cut_test freq_cut_test.cpp)
target_link_libraries(freq_cut_test
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)
target_include_directories(
  freq_cut_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
add_test(NAME FrequencyCuttingTest
         COMMAND freq_cut_test)

add_executable(filter_test filter_test.cpp)
target_link_libraries(filter_test
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)
target_include_directories(
  filter_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
add_test(NAME FilterTest
         COMMAND filter_test)

add_executable(template_test template_test.cpp)
target_link_libraries(template_test
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)
target_include_directories(
  template_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
add_test(NAME TemplateTest
COMMAND template_test)

add_executable(deconvolution_test deconvolution_test.cpp)
target_link_libraries(deconvolution_test
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)
target_include_directories(
  deconvolution_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
add_test(NAME DeconvolutionTest
        COMMAND deconvolution_test)