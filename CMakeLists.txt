cmake_minimum_required(VERSION 2.8)
project(creative_face)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( creative_face src/creative_face.cpp )
target_link_libraries( creative_face ${OpenCV_LIBS} )
# Copy static files
file(COPY src/face_cascade.xml DESTINATION ${CMAKE_BINARY_DIR})
file(COPY src/logo.png DESTINATION ${CMAKE_BINARY_DIR})
