find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind ships its CMake package inside the Python module.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE nanobind_found)
if(NOT nanobind_found EQUAL 0)
    message(FATAL_ERROR "hlq: the Python module needs nanobind (pip install nanobind)")
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_hlq NB_STATIC STABLE_ABI src/module.cpp)
target_link_libraries(_hlq PRIVATE hlq::hlq)
target_compile_definitions(_hlq PRIVATE HLQ_VERSION_INFO="${PROJECT_VERSION}")
install(TARGETS _hlq LIBRARY DESTINATION hlq)
