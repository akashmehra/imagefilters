cmake_minimum_required(VERSION 2.8)
ENABLE_LANGUAGE(C)
ENABLE_LANGUAGE(CXX)

file(GLOB HEADERS "*.h")
add_library(CpuImageFilters STATIC cpuProcessImage.cpp ${HEADERS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -L/usr/X11R6/lib -lm -lpthread -lX11")
add_executable(pimage cpuProcessImage.cpp)

