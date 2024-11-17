ADD_DEFINITIONS(-DSOURCE_DIR="${PROJECT_SOURCE_DIR}/")

add_executable(task1 task1.cpp)

target_link_libraries(task1
  PUBLIC
    ${OpenCV_LIBS}
    ft_lib
)

target_include_directories(
  task1
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
