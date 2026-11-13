include_guard(GLOBAL)

# Warnings for hlq's own tests, benchmarks and bindings (not propagated to users of the library).
option(HLQ_WARNINGS_AS_ERRORS "Treat warnings as errors in hlq's own targets" OFF)

add_library(hlq_warnings INTERFACE)
if(MSVC)
    target_compile_options(hlq_warnings INTERFACE /W4 /permissive- $<$<BOOL:${HLQ_WARNINGS_AS_ERRORS}>:/WX>)
else()
    target_compile_options(hlq_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast -Wnon-virtual-dtor
        $<$<BOOL:${HLQ_WARNINGS_AS_ERRORS}>:-Werror>)
endif()
