include_guard(GLOBAL)
include(FetchContent)

# Reference solvers for the comparisons, pinned so that results do not move with their default branches. LexLS and
# HierarchicalQP are header-only and used as such; DAQP is built, and its Eigen interface needs an installed Eigen.
FetchContent_Declare(hlq_lexls
    GIT_REPOSITORY https://github.com/jrl-umi3218/lexls.git
    GIT_TAG d5562f9c707060710ef8c9a4d83deed4e14c8b66
    SOURCE_SUBDIR no-cmake-project)
FetchContent_Declare(hlq_hqp
    GIT_REPOSITORY https://github.com/gianluca-garofalo/Weighted-Hierarchical-QP.git
    GIT_TAG 384670c376d2dc74f30e08197f98feb816e21bbd
    SOURCE_SUBDIR no-cmake-project)
FetchContent_Declare(hlq_daqp
    GIT_REPOSITORY https://github.com/darnstrom/daqp.git
    GIT_TAG ac404cbc1cced268b4742ec40a01169ec41d0c08
    SYSTEM)

set(EIGEN ON CACHE BOOL "Build DAQP's Eigen interface")
set(hlq_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(hlq_lexls hlq_hqp hlq_daqp)
set(BUILD_TESTING ${hlq_build_testing})

add_library(hlq_lexls INTERFACE)
target_include_directories(hlq_lexls SYSTEM INTERFACE "${hlq_lexls_SOURCE_DIR}/include")
target_link_libraries(hlq_lexls INTERFACE Eigen3::Eigen)

add_library(hlq_hqp INTERFACE)
target_include_directories(hlq_hqp SYSTEM INTERFACE "${hlq_hqp_SOURCE_DIR}/include")
target_link_libraries(hlq_hqp INTERFACE Eigen3::Eigen)
target_compile_features(hlq_hqp INTERFACE cxx_std_20)
