set(gui_cpp
    main.cpp
    digit_widget.cpp
    canvas_widget.cpp
    lua_classifier.cpp
)

set(gui_ui
    digit_widget.ui
)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

qt5_wrap_ui(gui_ui ${gui_ui})

add_executable(${PROJECT_NAME} ${gui_cpp} ${gui_ui})

target_link_libraries(${PROJECT_NAME} Qt5::Widgets lua)
