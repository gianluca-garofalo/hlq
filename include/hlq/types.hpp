#pragma once

/// @file
/// The index type, the marker of a capacity known only at run time, and the assertion macro.

#include <cstddef>

#ifndef HLQ_ASSERT
#include <cassert>
/// Checks a precondition of the interface (a capacity, an index). Define it before including hlq to change
/// what a failure does; it compiles to nothing with NDEBUG.
#define HLQ_ASSERT(condition) assert(condition)
#endif

namespace hlq {

/// Signed index of rows, columns and levels.
using Index = std::ptrdiff_t;

/// A capacity known only at run time, as a template argument of `Solver`.
inline constexpr int dynamic = -1;

}  // namespace hlq
