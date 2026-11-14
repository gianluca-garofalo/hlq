#pragma once

/// @file
/// What the solver reads its data from: any matrix with `rows()`, `cols()` and `a(i, j)`, and any vector with
/// `size()` and `v[i]`. Eigen matrices and vectors, `std::vector`, `std::array` and `std::span` all qualify,
/// and `MatrixView` wraps raw memory.

#include <concepts>

#include <hlq/types.hpp>

namespace hlq {

/// A matrix read elementwise, such as an Eigen matrix or expression, or a `MatrixView`.
template <typename T>
concept MatrixInput = requires(const T& a, Index i) {
    { a.rows() } -> std::convertible_to<Index>;
    { a.cols() } -> std::convertible_to<Index>;
    { a(i, i) } -> std::convertible_to<double>;
};

/// A vector read elementwise, such as a `std::vector`, `std::array`, `std::span` or Eigen vector.
template <typename T>
concept VectorInput = requires(const T& v) {
    { v.size() } -> std::convertible_to<Index>;
    { v[0] } -> std::convertible_to<double>;
};

/// A read-only view of a matrix in memory, with any strides (counted in elements).
template <typename T>
class MatrixView {
public:
    /// A row-major matrix: element (i, j) at `data[i * cols + j]`.
    constexpr MatrixView(const T* data, Index rows, Index cols) noexcept
        : MatrixView(data, rows, cols, cols, 1) {}

    /// Element (i, j) at `data[i * row_stride + j * col_stride]`.
    constexpr MatrixView(const T* data, Index rows, Index cols, Index row_stride, Index col_stride) noexcept
        : data_(data), rows_(rows), cols_(cols), row_stride_(row_stride), col_stride_(col_stride) {}

    /// A column-major matrix: element (i, j) at `data[i + j * rows]`.
    [[nodiscard]] static constexpr MatrixView column_major(const T* data, Index rows, Index cols) noexcept {
        return MatrixView(data, rows, cols, 1, rows);
    }

    [[nodiscard]] constexpr Index rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr Index cols() const noexcept { return cols_; }
    constexpr const T& operator()(Index i, Index j) const noexcept {
        return data_[i * row_stride_ + j * col_stride_];
    }

private:
    const T* data_;
    Index rows_;
    Index cols_;
    Index row_stride_;
    Index col_stride_;
};

namespace detail {

/// Element i of a `VectorInput`, whatever the type of its index.
template <typename V>
[[nodiscard]] decltype(auto) element(const V& v, Index i) {
    if constexpr (requires { typename V::size_type; }) {
        return v[static_cast<typename V::size_type>(i)];
    } else {
        return v[i];
    }
}

/// Size of a `VectorInput`.
template <typename V>
[[nodiscard]] Index length(const V& v) {
    return static_cast<Index>(v.size());
}

}  // namespace detail

}  // namespace hlq
