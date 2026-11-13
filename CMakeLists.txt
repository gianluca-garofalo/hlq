cmake_minimum_required(VERSION 3.21)

# The version lives in include/hlq/version.hpp.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/hlq/version.hpp" hlq_version_header)
foreach(part MAJOR MINOR PATCH)
    string(REGEX MATCH "#define HLQ_VERSION_${part} ([0-9]+)" _ "${hlq_version_header}")
    set(hlq_version_${part} "${CMAKE_MATCH_1}")
endforeach()

project(hlq
    VERSION ${hlq_version_MAJOR}.${hlq_version_MINOR}.${hlq_version_PATCH}
    DESCRIPTION "Hierarchical least-squares with inequality constraints on one priority-ordered LQ factorization"
    HOMEPAGE_URL "https://github.com/gianluca-garofalo/hlq"
    LANGUAGES CXX)

option(HLQ_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(HLQ_FETCH_DEPENDENCIES "Download the dependencies that are not found" ON)
option(HLQ_INSTALL "Generate the install rules" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(hlqDependencies)
hlq_find_eigen()

add_library(hlq INTERFACE)
add_library(hlq::hlq ALIAS hlq)
target_include_directories(hlq INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hlq INTERFACE Eigen3::Eigen)
target_compile_features(hlq INTERFACE cxx_std_20)

if(HLQ_BUILD_TESTS)
    include(hlqWarnings)
endif()

if(HLQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HLQ_INSTALL)
    include(hlqInstall)
endif()
