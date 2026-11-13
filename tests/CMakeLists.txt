hlq_find_gtest()
include(GoogleTest)

add_executable(hlq_tests test_factorization.cpp)
target_link_libraries(hlq_tests PRIVATE hlq::hlq hlq_warnings GTest::gtest_main)
target_include_directories(hlq_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(hlq_tests DISCOVERY_TIMEOUT 60)
