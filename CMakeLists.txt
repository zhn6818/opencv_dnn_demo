cmake_minimum_required(VERSION 3.19)
project(predict)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(yolo main_yolo.cpp yolo.h )
target_link_libraries(yolo ${OpenCV_LIBS})