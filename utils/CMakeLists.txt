project(utils)
add_executable(utils utils.cpp)
target_link_libraries(utils ${OpenCV_LIBS})