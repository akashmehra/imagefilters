cmake_minimum_required (VERSION 2.8)
project (imagefilters)
file(COPY images DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(src)
