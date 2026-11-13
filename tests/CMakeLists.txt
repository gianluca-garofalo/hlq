hlq_find_gtest()
include(GoogleTest)

add_executable(hlq_tests
    test_factorization.cpp
    test_solver.cpp
    test_random.cpp
    test_capacity.cpp)
target_link_libraries(hlq_tests PRIVATE hlq::hlq hlq_warnings GTest::gtest_main)
target_include_directories(hlq_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(hlq_tests DISCOVERY_TIMEOUT 60)

# solve() must not allocate. With EIGEN_RUNTIME_NO_MALLOC, Eigen reports every heap allocation through
# eigen_assert, which this test counts (in every build type).
add_executable(hlq_no_malloc_tests test_no_malloc.cpp)
target_compile_definitions(hlq_no_malloc_tests PRIVATE EIGEN_RUNTIME_NO_MALLOC)
target_link_libraries(hlq_no_malloc_tests PRIVATE hlq::hlq hlq_warnings GTest::gtest_main)
target_include_directories(hlq_no_malloc_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(hlq_no_malloc_tests DISCOVERY_TIMEOUT 60)
