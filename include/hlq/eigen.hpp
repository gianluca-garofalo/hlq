#pragma once

/// @file
/// Optional Eigen interoperability. The solver reads Eigen matrices and vectors as they are; this header maps
/// its results (`std::span`s) as Eigen vectors, without copying. Include it only where Eigen is available.

#include <span>
#include <type_traits>

#include <Eigen/Core>

#include <hlq/types.hpp>

namespace hlq {

/// A span of the solver's results as an Eigen column vector, sharing its memory.
template <typename T, std::size_t Extent>
[[nodiscard]] auto as_eigen(std::span<T, Extent> values) noexcept {
    using Vector = Eigen::Matrix<std::remove_const_t<T>, Eigen::Dynamic, 1>;
    using Map = Eigen::Map<std::conditional_t<std::is_const_v<T>, const Vector, Vector>>;
    return Map(values.data(), static_cast<Eigen::Index>(values.size()));
}

}  // namespace hlq
