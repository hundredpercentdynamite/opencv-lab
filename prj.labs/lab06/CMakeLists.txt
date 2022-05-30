project(lab06)
add_executable(lab06 lab06.cpp lab06.report.md.in.txt)
target_link_libraries(lab06 ${OpenCV_LIBS})

file(READ lab06.cpp cpp_source)
# prepare report md file
configure_file(lab06.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/lab06.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/lab06.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")