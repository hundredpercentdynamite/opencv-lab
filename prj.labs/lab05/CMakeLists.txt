project(lab05)
add_executable(lab05 lab05.cpp lab05.report.md.in.txt)
target_link_libraries(lab05 ${OpenCV_LIBS})

file(READ lab05.cpp cpp_source)
# prepare report md file
configure_file(lab05.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/lab05.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/lab05.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")