cmake_minimum_required(VERSION 3.15)
project(Task2Demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(Task2Demo combined_task1.cpp)
target_link_libraries(Task2Demo PRIVATE ${OpenCV_LIBS})