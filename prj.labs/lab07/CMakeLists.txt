project(lab07)
add_executable(lab07 lab07.cpp lab07.report.md.in.txt)
target_link_libraries(lab07 ${OpenCV_LIBS})

file(READ lab07.cpp cpp_source)
# prepare report md file
configure_file(lab07.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/lab07.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/lab07.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")