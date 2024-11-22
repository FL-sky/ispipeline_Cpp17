cmake_minimum_required(VERSION 3.0)
project(ispipelineCpp17)

set(CMAKE_CXX_STANDARD 17)

include_directories(thirdparty/easylog)
include_directories(thirdparty/easybmp)
include_directories(thirdparty/json/nlohmann)
include_directories(srcs/include)

add_subdirectory(thirdparty/easylog)
add_subdirectory(thirdparty/easybmp)
add_subdirectory(srcs)

add_executable(ispipelineCpp17 main.cpp
        parse.cpp)

target_link_libraries(${PROJECT_NAME}  easylog easybmp modules)