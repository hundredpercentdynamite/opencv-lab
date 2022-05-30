project(lab0k)
add_executable(lab0k lab0k.cpp lab0k.report.md.in.txt)
target_link_libraries(lab0k ${OpenCV_LIBS})

# prepare report md file
file(READ lab0k.cpp cpp_source)
configure_file(lab0k.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/lab0k.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/lab0k.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")