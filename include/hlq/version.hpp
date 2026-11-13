#pragma once

/// @file
/// Version of hlq. CMake reads the project version from these macros.

#define HLQ_VERSION_MAJOR 0
#define HLQ_VERSION_MINOR 1
#define HLQ_VERSION_PATCH 0

#define HLQ_STRINGIFY_DETAIL(x) #x
#define HLQ_STRINGIFY(x) HLQ_STRINGIFY_DETAIL(x)
#define HLQ_VERSION_STRING           \
    HLQ_STRINGIFY(HLQ_VERSION_MAJOR) \
    "." HLQ_STRINGIFY(HLQ_VERSION_MINOR) "." HLQ_STRINGIFY(HLQ_VERSION_PATCH)

namespace hlq {

/// Version as "major.minor.patch".
inline constexpr const char* version = HLQ_VERSION_STRING;

}  // namespace hlq
