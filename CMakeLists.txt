cmake_minimum_required(VERSION 3.10)

project(machine_vision)

set(CMAKE_CXX_STANDARD 17)

# OpenCV路径
#set(OpenCV_DIR "E:/opencv/build")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# 编译UTF-8，避免乱码
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 自动寻找 src 里的所有 cpp
file(GLOB SRC_FILES "src/*.cpp")

# 为每个 cpp 生成一个可执行程序
foreach(src ${SRC_FILES})

    # 取文件名（不含扩展名）
    get_filename_component(name ${src} NAME_WE)

    add_executable(${name} ${src})

    target_link_libraries(${name} ${OpenCV_LIBS})

endforeach()