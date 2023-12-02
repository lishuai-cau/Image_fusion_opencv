cmake_minimum_required(VERSION 3.25)
project(imagefusion)
set(OpenCV_DIR "D:\\code\\CLionProjects\\extrapkg\\opencv\\build")
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(imagefusion main.cpp)

target_link_libraries(imagefusion ${OpenCV_LIBS})