# Timings of hlq (Google Benchmark), and comparisons with LexLS, DAQP and HierarchicalQP.

if(HLQ_BUILD_BENCHMARKS)
    hlq_find_benchmark()
    add_executable(hlq_benchmarks bench_solver.cpp)
    target_link_libraries(hlq_benchmarks PRIVATE hlq::hlq hlq_warnings benchmark::benchmark_main)
    target_include_directories(hlq_benchmarks PRIVATE "${PROJECT_SOURCE_DIR}/tests")
endif()

if(HLQ_BUILD_COMPARISONS)
    include(hlqCompetitors)

    # Lexicographic verdicts against the other solvers; exits with 1 when hlq loses on a well-conditioned family.
    add_executable(hlq_compare compare.cpp)
    target_link_libraries(hlq_compare PRIVATE hlq::hlq hlq_lexls hlq_hqp daqp_eigen)
    target_include_directories(hlq_compare PRIVATE "${PROJECT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_SOURCE_DIR}")
    if(HLQ_BUILD_TESTS)
        add_test(NAME hlq_compare COMMAND hlq_compare 1000)
    endif()

    if(HLQ_BUILD_BENCHMARKS)
        add_executable(hlq_compare_benchmarks bench_compare.cpp)
        target_link_libraries(hlq_compare_benchmarks PRIVATE
            hlq::hlq hlq_lexls hlq_hqp daqp_eigen benchmark::benchmark_main)
        target_include_directories(hlq_compare_benchmarks PRIVATE
            "${PROJECT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()
