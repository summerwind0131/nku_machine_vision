cmake_minimum_required(VERSION 3.10)

project(GesturePart1Traditional)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(train_gesture src/train.cpp include/gesture_features.hpp)
add_executable(test_gesture src/test.cpp include/gesture_features.hpp)

target_link_libraries(train_gesture ${OpenCV_LIBS})
target_link_libraries(test_gesture ${OpenCV_LIBS})