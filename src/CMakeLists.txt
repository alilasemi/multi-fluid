add_library(pybind_bindings SHARED pybind_bindings.cpp)
target_link_libraries(pybind_bindings
    PRIVATE face_residual)

add_library(face_residual STATIC face_residual.cpp)
target_link_libraries(face_residual
    PRIVATE roe
    PRIVATE forced_interface)

add_library(roe STATIC roe.cpp)

add_library(forced_interface STATIC forced_interface.cpp)
