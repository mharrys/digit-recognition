cmake_minimum_required(VERSION 2.8)

project(gui)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets)
find_package(Lua)

add_subdirectory(gui)
