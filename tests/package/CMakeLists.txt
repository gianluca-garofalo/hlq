# A project that uses an installed hlq, as a downstream project would. CI builds it against `cmake --install`.
cmake_minimum_required(VERSION 3.21)
project(hlq_package_test LANGUAGES CXX)

find_package(hlq 0.1 REQUIRED)

add_executable(hlq_package_test main.cpp)
target_link_libraries(hlq_package_test PRIVATE hlq::hlq)

enable_testing()
add_test(NAME hlq_package_test COMMAND hlq_package_test)
