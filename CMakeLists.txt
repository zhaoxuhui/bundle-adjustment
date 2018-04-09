cmake_minimum_required(VERSION 3.10)
project(bundle)

set(CMAKE_CXX_STANDARD 11)

# 添加cmake模块以使用ceres库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# 寻找G2O
include_directories(
        ${G2O_INCLUDE_DIRS}
        "/usr/include/eigen3"
        "/usr/include/suitesparse"
)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(bundle main.cpp ORBMatching.cpp ORBMatching.h sphere2500.cpp)

# 与G2O和OpenCV链接
target_link_libraries(bundle
        g2o_core g2o_stuff g2o_cli
        GL GLU cholmod
        g2o_incremental g2o_interactive g2o_interface g2o_parser
        g2o_solver_cholmod g2o_solver_csparse g2o_solver_dense g2o_solver_pcg
        g2o_types_icp g2o_types_sba g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d
        ${OpenCV_LIBS}
        ${QT_LIBRARIES}
        ${QT_QTOPENGL_LIBRARY}
        ${GLUT_LIBRARY}
        ${OPENGL_LIBRARY}
        )