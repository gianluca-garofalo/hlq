include_guard(GLOBAL)
include(FetchContent)

# Eigen 3.4: the one dependency of the library. An installed Eigen is preferred; otherwise its release tarball is
# downloaded and used header-only, without configuring Eigen's own project.
function(hlq_find_eigen)
    if(TARGET Eigen3::Eigen)
        return()
    endif()
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(Eigen3_FOUND)
        message(STATUS "hlq: Eigen ${Eigen3_VERSION} from ${Eigen3_DIR}")
        return()
    endif()
    if(NOT HLQ_FETCH_DEPENDENCIES)
        message(FATAL_ERROR "hlq: Eigen 3.4 not found (set Eigen3_DIR, or HLQ_FETCH_DEPENDENCIES=ON)")
    endif()
    FetchContent_Declare(hlq_eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        SOURCE_SUBDIR no-cmake-project)
    FetchContent_MakeAvailable(hlq_eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${hlq_eigen_SOURCE_DIR}")
    message(STATUS "hlq: Eigen 3.4.0 downloaded to ${hlq_eigen_SOURCE_DIR}")
endfunction()

# GoogleTest for the tests.
function(hlq_find_gtest)
    if(TARGET GTest::gtest_main)
        return()
    endif()
    find_package(GTest 1.11 QUIET CONFIG)
    if(GTest_FOUND)
        return()
    endif()
    if(NOT HLQ_FETCH_DEPENDENCIES)
        message(FATAL_ERROR "hlq: GoogleTest not found (or set HLQ_FETCH_DEPENDENCIES=ON)")
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
        SYSTEM)
    FetchContent_MakeAvailable(googletest)
endfunction()

# Google Benchmark for the benchmarks.
function(hlq_find_benchmark)
    if(TARGET benchmark::benchmark_main)
        return()
    endif()
    find_package(benchmark 1.6 QUIET CONFIG)
    if(benchmark_FOUND)
        return()
    endif()
    if(NOT HLQ_FETCH_DEPENDENCIES)
        message(FATAL_ERROR "hlq: Google Benchmark not found (or set HLQ_FETCH_DEPENDENCIES=ON)")
    endif()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
        GIT_SHALLOW TRUE
        SYSTEM)
    FetchContent_MakeAvailable(benchmark)
endfunction()
