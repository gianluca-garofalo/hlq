#pragma once

/// @file
/// The dense linear algebra hlq needs, without dependencies: row-major matrices and vectors of compile-time
/// or run-time capacity, views of their blocks, and the kernels of the factorization (dot products, axpys,
/// triangular solves, Givens rotations and Householder reflectors) on raw pointers.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <hlq/types.hpp>

/// GCC and Clang vector types make the kernels use SIMD instructions without reassociation flags. Define
/// HLQ_NO_VECTOR_EXTENSIONS to use plain loops instead, as other compilers do.
#if !defined(HLQ_NO_VECTOR_EXTENSIONS) && (defined(__GNUC__) || defined(__clang__))
#define HLQ_VECTOR_EXTENSIONS 1
#else
#define HLQ_VECTOR_EXTENSIONS 0
#endif

namespace hlq::detail {

#if HLQ_VECTOR_EXTENSIONS
#ifndef HLQ_SIMD_LENGTH
/// Vectors shorter than this use plain loops, which the compiler vectorizes itself where it can: for them,
/// setting up and reducing SIMD registers costs more than it saves.
#define HLQ_SIMD_LENGTH 16
#endif
inline constexpr Index simd_length = HLQ_SIMD_LENGTH;

/// Bytes of a SIMD register of the target.
#if defined(__AVX512F__)
inline constexpr std::size_t pack_bytes = 64;
#elif defined(__AVX__)
inline constexpr std::size_t pack_bytes = 32;
#else
inline constexpr std::size_t pack_bytes = 16;
#endif

template <typename T>
struct PackOf;
template <>
struct PackOf<float> {
    typedef float type __attribute__((vector_size(pack_bytes)));
};
template <>
struct PackOf<double> {
    typedef double type __attribute__((vector_size(pack_bytes)));
};

/// A SIMD register of scalars, loaded and stored without alignment requirements.
template <typename T>
struct Simd {
    using Pack = typename PackOf<T>::type;
    static constexpr Index width = static_cast<Index>(pack_bytes / sizeof(T));

    static Pack load(const T* p) noexcept {
        Pack v;
        std::memcpy(&v, p, sizeof v);
        return v;
    }
    static void store(T* p, Pack v) noexcept { std::memcpy(p, &v, sizeof v); }
    static Pack splat(T x) noexcept {
        Pack v;
        for (Index k = 0; k < width; ++k) {
            v[k] = x;
        }
        return v;
    }
    static T sum(Pack v) noexcept {
        T s = 0;
        for (Index k = 0; k < width; ++k) {
            s += v[k];
        }
        return s;
    }
};
#endif

/// A compile-time capacity plus `extra`, or `dynamic`.
constexpr int grow(int capacity, int extra) noexcept {
    return capacity == dynamic ? dynamic : capacity + extra;
}

/// The sum of two compile-time capacities, or `dynamic`.
constexpr int sum(int a, int b) noexcept {
    return a == dynamic || b == dynamic ? dynamic : a + b;
}

/// The product of two compile-time capacities, or `dynamic`.
constexpr int product(int a, int b) noexcept {
    return a == dynamic || b == dynamic ? dynamic : a * b;
}

/// Contiguous storage of `Capacity` elements, in place.
template <typename T, int Capacity>
class Storage {
public:
    void reserve(Index size) noexcept {
        HLQ_ASSERT(size <= Capacity);
        static_cast<void>(size);
    }
    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

private:
    std::array<T, static_cast<std::size_t>(Capacity)> data_{};
};

/// Contiguous storage on the heap, allocated only when it grows.
template <typename T>
class Storage<T, dynamic> {
public:
    Storage() = default;
    Storage(const Storage& other) { copy(other); }
    Storage(Storage&&) noexcept = default;
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            copy(other);
        }
        return *this;
    }
    Storage& operator=(Storage&&) noexcept = default;
    ~Storage() = default;

    /// Makes room for `size` elements. Allocates (keeping nothing, and leaving the elements uninitialized, as
    /// every user writes before it reads) only when that is more than before.
    void reserve(Index size) {
        if (size > capacity_) {
            data_.reset(new T[static_cast<std::size_t>(size)]);
            capacity_ = size;
        }
    }
    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }

private:
    void copy(const Storage& other) {
        data_.reset();
        capacity_ = 0;
        reserve(other.capacity_);
        std::copy_n(other.data_.get(), other.capacity_, data_.get());
    }

    std::unique_ptr<T[]> data_;
    Index capacity_ = 0;
};

/// A view of `size` contiguous elements.
template <typename T>
struct VectorRef {
    T* data = nullptr;
    Index size = 0;

    T& operator()(Index i) const noexcept { return data[i]; }
    [[nodiscard]] VectorRef segment(Index start, Index count) const noexcept { return {data + start, count}; }
    [[nodiscard]] VectorRef head(Index count) const noexcept { return {data, count}; }
    [[nodiscard]] VectorRef tail(Index count) const noexcept { return {data + size - count, count}; }
};

/// A view of a row-major block: `rows` rows of `cols` elements, `stride` apart.
template <typename T>
struct MatrixRef {
    T* data = nullptr;
    Index rows = 0;
    Index cols = 0;
    Index stride = 0;

    T& operator()(Index i, Index j) const noexcept { return data[i * stride + j]; }
    [[nodiscard]] T* row(Index i) const noexcept { return data + i * stride; }
    [[nodiscard]] MatrixRef block(Index row, Index col, Index count, Index width) const noexcept {
        return {data + row * stride + col, count, width, stride};
    }
    [[nodiscard]] MatrixRef top_rows(Index count) const noexcept { return {data, count, cols, stride}; }
    [[nodiscard]] MatrixRef middle_rows(Index row, Index count) const noexcept {
        return {data + row * stride, count, cols, stride};
    }
    operator MatrixRef<const T>() const noexcept { return {data, rows, cols, stride}; }
};

/// A vector of up to `Capacity` elements (or any number, with `dynamic`).
template <typename T, int Capacity>
class Vector {
public:
    Vector() = default;
    explicit Vector(Index size) { resize(size); }

    /// Sets the size; the elements are unspecified after growing past the storage.
    void resize(Index size) {
        storage_.reserve(size);
        size_ = size;
    }
    [[nodiscard]] Index size() const noexcept { return size_; }
    [[nodiscard]] T* data() noexcept { return storage_.data(); }
    [[nodiscard]] const T* data() const noexcept { return storage_.data(); }
    T& operator()(Index i) noexcept { return data()[i]; }
    const T& operator()(Index i) const noexcept { return data()[i]; }
    void fill(const T& value) noexcept { std::fill_n(data(), size_, value); }
    [[nodiscard]] VectorRef<T> segment(Index start, Index count) noexcept { return {data() + start, count}; }
    [[nodiscard]] VectorRef<const T> segment(Index start, Index count) const noexcept {
        return {data() + start, count};
    }
    [[nodiscard]] VectorRef<T> head(Index count) noexcept { return {data(), count}; }
    [[nodiscard]] VectorRef<const T> head(Index count) const noexcept { return {data(), count}; }

private:
    Storage<T, Capacity> storage_;
    Index size_ = 0;
};

/// A row-major matrix of up to `MaxRows` by `MaxCols` elements (or any number, with `dynamic`).
template <typename T, int MaxRows, int MaxCols>
class Matrix {
public:
    Matrix() = default;
    Matrix(Index rows, Index cols) { resize(rows, cols); }

    /// Sets the shape; the elements are unspecified after growing past the storage.
    void resize(Index rows, Index cols) {
        HLQ_ASSERT(MaxRows == dynamic || rows <= MaxRows);
        HLQ_ASSERT(MaxCols == dynamic || cols <= MaxCols);
        storage_.reserve(rows * cols);
        rows_ = rows;
        cols_ = cols;
    }
    [[nodiscard]] Index rows() const noexcept { return rows_; }
    [[nodiscard]] Index cols() const noexcept { return cols_; }
    T& operator()(Index i, Index j) noexcept { return storage_.data()[i * cols_ + j]; }
    const T& operator()(Index i, Index j) const noexcept { return storage_.data()[i * cols_ + j]; }
    [[nodiscard]] T* row(Index i) noexcept { return storage_.data() + i * cols_; }
    [[nodiscard]] const T* row(Index i) const noexcept { return storage_.data() + i * cols_; }
    [[nodiscard]] MatrixRef<T> view() noexcept { return {storage_.data(), rows_, cols_, cols_}; }
    [[nodiscard]] MatrixRef<const T> view() const noexcept { return {storage_.data(), rows_, cols_, cols_}; }
    [[nodiscard]] MatrixRef<T> block(Index row, Index col, Index count, Index width) noexcept {
        return view().block(row, col, count, width);
    }
    [[nodiscard]] MatrixRef<const T> block(Index row, Index col, Index count, Index width) const noexcept {
        return view().block(row, col, count, width);
    }
    void fill(const T& value) noexcept { std::fill_n(storage_.data(), rows_ * cols_, value); }

private:
    Storage<T, product(MaxRows, MaxCols)> storage_;
    Index rows_ = 0;
    Index cols_ = 0;
};

/// A read-only view as a kernel parameter: not deduced, so that a mutable view converts.
template <typename T>
using ConstRef = std::type_identity_t<MatrixRef<const T>>;

// Kernels. Sizes may be zero, and outputs never overlap inputs.
//
// Each kernel is a plain loop, which the compiler inlines and vectorizes itself where it can. With vector
// extensions, vectors of at least `simd_length` elements go to an out-of-line version on SIMD registers
// instead: below that length setting up and reducing the registers costs more than it saves, and inlining the
// SIMD code would keep the compiler from inlining the plain loops.

/// A plane rotation [c s; -s c].
template <typename T>
struct Rotation {
    T c;
    T s;
};

#if HLQ_VECTOR_EXTENSIONS
namespace simd {

template <typename T>
[[gnu::noinline]] T dot(const T* a, const T* b, Index n) noexcept {
    using S = Simd<T>;
    constexpr Index w = S::width;
    typename S::Pack s0{};
    typename S::Pack s1{};
    Index i = 0;
    for (; i + 2 * w <= n; i += 2 * w) {
        s0 += S::load(a + i) * S::load(b + i);
        s1 += S::load(a + i + w) * S::load(b + i + w);
    }
    if (i + w <= n) {
        s0 += S::load(a + i) * S::load(b + i);
        i += w;
    }
    T s = S::sum(s0 + s1);
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

template <typename T>
[[gnu::noinline]] void axpy(T alpha, const T* x, T* y, Index n) noexcept {
    using S = Simd<T>;
    const auto a = S::splat(alpha);
    Index i = 0;
    for (; i + S::width <= n; i += S::width) {
        S::store(y + i, S::load(y + i) + a * S::load(x + i));
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/// y <- y + alpha A x (or y <- A x, without `Add`), four rows of A at a time.
template <bool Add, typename T>
[[gnu::noinline]] void multiply(MatrixRef<const T> a, const T* x, T* y, T alpha) noexcept {
    using S = Simd<T>;
    constexpr Index w = S::width;
    const Index n = a.cols;
    const Index packed = n - n % w;
    const auto put = [&](Index i, T value) {
        if constexpr (Add) {
            y[i] += alpha * value;
        } else {
            y[i] = value;
        }
    };
    Index i = 0;
    for (; i + 4 <= a.rows; i += 4) {
        const T* r0 = a.row(i);
        const T* r1 = a.row(i + 1);
        const T* r2 = a.row(i + 2);
        const T* r3 = a.row(i + 3);
        typename S::Pack s0{};
        typename S::Pack s1{};
        typename S::Pack s2{};
        typename S::Pack s3{};
        for (Index j = 0; j < packed; j += w) {
            const auto v = S::load(x + j);
            s0 += S::load(r0 + j) * v;
            s1 += S::load(r1 + j) * v;
            s2 += S::load(r2 + j) * v;
            s3 += S::load(r3 + j) * v;
        }
        T t0 = S::sum(s0);
        T t1 = S::sum(s1);
        T t2 = S::sum(s2);
        T t3 = S::sum(s3);
        for (Index j = packed; j < n; ++j) {
            t0 += r0[j] * x[j];
            t1 += r1[j] * x[j];
            t2 += r2[j] * x[j];
            t3 += r3[j] * x[j];
        }
        put(i, t0);
        put(i + 1, t1);
        put(i + 2, t2);
        put(i + 3, t3);
    }
    for (; i < a.rows; ++i) {
        put(i, dot(a.row(i), x, n));
    }
}

/// y[i] <- A.row(i) x for the rows i of a list, four at a time.
template <typename T>
[[gnu::noinline]] void multiply_rows(MatrixRef<const T> a, const Index* rows, Index count, const T* x,
                                     T* y) noexcept {
    using S = Simd<T>;
    constexpr Index w = S::width;
    const Index n = a.cols;
    const Index packed = n - n % w;
    Index k = 0;
    for (; k + 4 <= count; k += 4) {
        const T* r0 = a.row(rows[k]);
        const T* r1 = a.row(rows[k + 1]);
        const T* r2 = a.row(rows[k + 2]);
        const T* r3 = a.row(rows[k + 3]);
        typename S::Pack s0{};
        typename S::Pack s1{};
        typename S::Pack s2{};
        typename S::Pack s3{};
        for (Index j = 0; j < packed; j += w) {
            const auto v = S::load(x + j);
            s0 += S::load(r0 + j) * v;
            s1 += S::load(r1 + j) * v;
            s2 += S::load(r2 + j) * v;
            s3 += S::load(r3 + j) * v;
        }
        T t0 = S::sum(s0);
        T t1 = S::sum(s1);
        T t2 = S::sum(s2);
        T t3 = S::sum(s3);
        for (Index j = packed; j < n; ++j) {
            t0 += r0[j] * x[j];
            t1 += r1[j] * x[j];
            t2 += r2[j] * x[j];
            t3 += r3[j] * x[j];
        }
        y[rows[k]] = t0;
        y[rows[k + 1]] = t1;
        y[rows[k + 2]] = t2;
        y[rows[k + 3]] = t3;
    }
    for (; k < count; ++k) {
        y[rows[k]] = dot(a.row(rows[k]), x, n);
    }
}

/// y <- y + alpha A^T x, four rows of A at a time.
template <typename T>
[[gnu::noinline]] void multiply_transposed_add(T alpha, MatrixRef<const T> a, const T* x, T* y) noexcept {
    using S = Simd<T>;
    constexpr Index w = S::width;
    const Index n = a.cols;
    const Index packed = n - n % w;
    Index i = 0;
    for (; i + 4 <= a.rows; i += 4) {
        const T* r0 = a.row(i);
        const T* r1 = a.row(i + 1);
        const T* r2 = a.row(i + 2);
        const T* r3 = a.row(i + 3);
        const T c0 = alpha * x[i];
        const T c1 = alpha * x[i + 1];
        const T c2 = alpha * x[i + 2];
        const T c3 = alpha * x[i + 3];
        const auto p0 = S::splat(c0);
        const auto p1 = S::splat(c1);
        const auto p2 = S::splat(c2);
        const auto p3 = S::splat(c3);
        for (Index j = 0; j < packed; j += w) {
            S::store(y + j, S::load(y + j)
                                + ((p0 * S::load(r0 + j) + p1 * S::load(r1 + j))
                                   + (p2 * S::load(r2 + j) + p3 * S::load(r3 + j))));
        }
        for (Index j = packed; j < n; ++j) {
            y[j] += (c0 * r0[j] + c1 * r1[j]) + (c2 * r2[j] + c3 * r3[j]);
        }
    }
    for (; i < a.rows; ++i) {
        axpy(alpha * x[i], a.row(i), y, n);
    }
}

template <typename T>
[[gnu::noinline]] void rotate(Rotation<T> g, T* x, T* y, Index n) noexcept {
    using S = Simd<T>;
    const auto c = S::splat(g.c);
    const auto s = S::splat(g.s);
    Index i = 0;
    for (; i + S::width <= n; i += S::width) {
        const auto a = S::load(x + i);
        const auto b = S::load(y + i);
        S::store(x + i, c * a + s * b);
        S::store(y + i, c * b - s * a);
    }
    for (; i < n; ++i) {
        const T a = x[i];
        const T b = y[i];
        x[i] = g.c * a + g.s * b;
        y[i] = g.c * b - g.s * a;
    }
}

}  // namespace simd
#endif

template <typename T>
[[nodiscard]] T dot(const T* a, const T* b, Index n) noexcept {
#if HLQ_VECTOR_EXTENSIONS
    if (n >= simd_length) {
        return simd::dot(a, b, n);
    }
#endif
    // Four partial sums pipeline (and vectorize) without reassociation by the compiler.
    T s0 = 0;
    T s1 = 0;
    T s2 = 0;
    T s3 = 0;
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
}

template <typename T>
[[nodiscard]] T squared_norm(const T* x, Index n) noexcept {
    return dot(x, x, n);
}

/// y <- y + alpha x.
template <typename T>
void axpy(T alpha, const T* x, T* y, Index n) noexcept {
#if HLQ_VECTOR_EXTENSIONS
    if (n >= simd_length) {
        simd::axpy(alpha, x, y, n);
        return;
    }
#endif
    for (Index i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

template <typename T>
void scale(T alpha, T* x, Index n) noexcept {
    for (Index i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

/// Index of the largest of `n >= 1` values, the first one on ties.
template <typename T>
[[nodiscard]] Index argmax(const T* x, Index n) noexcept {
    Index best = 0;
    for (Index i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

/// The largest of `n >= 1` values.
template <typename T>
[[nodiscard]] T max_of(const T* x, Index n) noexcept {
    return x[argmax(x, n)];
}

/// y <- y + alpha A x (or y <- A x, without `Add`).
template <bool Add = false, typename T>
void multiply(ConstRef<T> a, const T* x, T* y, T alpha = T(1)) noexcept {
#if HLQ_VECTOR_EXTENSIONS
    if (a.cols >= simd_length) {
        simd::multiply<Add>(a, x, y, alpha);
        return;
    }
#endif
    for (Index i = 0; i < a.rows; ++i) {
        const T value = dot(a.row(i), x, a.cols);
        if constexpr (Add) {
            y[i] += alpha * value;
        } else {
            y[i] = value;
        }
    }
}

/// y[i] <- A.row(i) x for the rows i of a list.
template <typename T>
void multiply_rows(ConstRef<T> a, const Index* rows, Index count, const T* x, T* y) noexcept {
#if HLQ_VECTOR_EXTENSIONS
    if (a.cols >= simd_length) {
        simd::multiply_rows(a, rows, count, x, y);
        return;
    }
#endif
    for (Index k = 0; k < count; ++k) {
        y[rows[k]] = dot(a.row(rows[k]), x, a.cols);
    }
}

/// y <- y + alpha A^T x.
template <typename T>
void multiply_transposed_add(T alpha, ConstRef<T> a, const T* x, T* y) noexcept {
#if HLQ_VECTOR_EXTENSIONS
    if (a.cols >= simd_length) {
        simd::multiply_transposed_add(alpha, a, x, y);
        return;
    }
#endif
    for (Index i = 0; i < a.rows; ++i) {
        axpy(alpha * x[i], a.row(i), y, a.cols);
    }
}

/// Solves L x = b in place, with L lower triangular.
template <typename T>
void solve_lower(ConstRef<T> l, T* x) noexcept {
    for (Index i = 0; i < l.rows; ++i) {
        x[i] = (x[i] - dot(l.row(i), x, i)) / l(i, i);
    }
}

/// Solves L^T x = b in place, with L lower triangular.
template <typename T>
void solve_lower_transposed(ConstRef<T> l, T* x) noexcept {
    for (Index i = l.rows - 1; i >= 0; --i) {
        x[i] /= l(i, i);
        axpy(-x[i], l.row(i), x, i);
    }
}

/// The rotation that maps (a, b) to (r, 0), with r = c a + s b, without overflow.
template <typename T>
[[nodiscard]] Rotation<T> make_rotation(T a, T b) noexcept {
    if (b == T(0)) {
        return {T(1), T(0)};
    }
    if (a == T(0)) {
        return {T(0), T(1)};
    }
    const T big = std::max(std::abs(a), std::abs(b));
    const T x = a / big;
    const T y = b / big;
    const T r = big * std::sqrt(x * x + y * y);
    return {a / r, b / r};
}

/// (x, y) <- (c x + s y, c y - s x), elementwise over `n` contiguous pairs.
template <typename T>
void rotate(Rotation<T> g, T* x, T* y, Index n) noexcept {
#if HLQ_VECTOR_EXTENSIONS
    if (n >= simd_length) {
        simd::rotate(g, x, y, n);
        return;
    }
#endif
    for (Index i = 0; i < n; ++i) {
        const T a = x[i];
        const T b = y[i];
        x[i] = g.c * a + g.s * b;
        y[i] = g.c * b - g.s * a;
    }
}

/// The same as `rotate`, for pairs `stride` apart.
template <typename T>
void rotate_strided(Rotation<T> g, T* x, T* y, Index n, Index stride) noexcept {
    for (Index i = 0; i < n; ++i) {
        const T a = x[i * stride];
        const T b = y[i * stride];
        x[i * stride] = g.c * a + g.s * b;
        y[i * stride] = g.c * b - g.s * a;
    }
}

/**
 * Householder reflector H = I - tau v v^T, v = (1, w), that maps (alpha, x) to (beta, 0), in place: `alpha`
 * becomes beta and `x` (n entries) becomes w. Returns tau, zero when x already is.
 */
template <typename T>
[[nodiscard]] T make_householder(T& alpha, T* x, Index n) noexcept {
    const T rest = squared_norm(x, n);
    if (rest <= std::numeric_limits<T>::min()) {
        std::fill_n(x, n, T(0));
        return T(0);
    }
    T beta = std::sqrt(alpha * alpha + rest);
    if (alpha >= T(0)) {
        beta = -beta;
    }
    const T denominator = alpha - beta;
    for (Index i = 0; i < n; ++i) {
        x[i] /= denominator;
    }
    const T tau = (beta - alpha) / beta;
    alpha = beta;
    return tau;
}

/// (y0, y) <- H (y0, y), with H from `make_householder` (its w in `w`, n entries).
template <typename T>
void apply_householder(const T* w, T tau, T& y0, T* y, Index n) noexcept {
    if (tau == T(0)) {
        return;
    }
    const T s = tau * (y0 + dot(w, y, n));
    y0 -= s;
    axpy(-s, w, y, n);
}

/**
 * Householder QR of a tall matrix F (dim x rank, dim >= rank), in place, given by its transpose: row j of
 * `ft` is column j of F. Afterwards R^T is the lower triangle of `ft` and the reflectors' w lie after the
 * diagonal, with their tau in `tau`.
 */
template <typename T>
void householder_qr(MatrixRef<T> ft, T* tau) noexcept {
    const Index rank = ft.rows;
    const Index dim = ft.cols;
    for (Index j = 0; j < rank; ++j) {
        T* column = ft.row(j);
        tau[j] = make_householder(column[j], column + j + 1, dim - j - 1);
        for (Index l = j + 1; l < rank; ++l) {
            T* other = ft.row(l);
            apply_householder(column + j + 1, tau[j], other[j], other + j + 1, dim - j - 1);
        }
    }
}

/// v <- Q^T v, with Q from `householder_qr`.
template <typename T>
void apply_qt(ConstRef<T> ft, const T* tau, T* v) noexcept {
    for (Index j = 0; j < ft.rows; ++j) {
        apply_householder(ft.row(j) + j + 1, tau[j], v[j], v + j + 1, ft.cols - j - 1);
    }
}

/// v <- Q v, with Q from `householder_qr`.
template <typename T>
void apply_q(ConstRef<T> ft, const T* tau, T* v) noexcept {
    for (Index j = ft.rows - 1; j >= 0; --j) {
        apply_householder(ft.row(j) + j + 1, tau[j], v[j], v + j + 1, ft.cols - j - 1);
    }
}

/**
 * Cholesky factor of a symmetric matrix, W = L L^T, in place in its lower triangle (the upper one is left as
 * is). Returns false when W is not positive definite.
 */
template <typename T>
[[nodiscard]] bool cholesky(MatrixRef<T> w) noexcept {
    for (Index j = 0; j < w.rows; ++j) {
        const T pivot = w(j, j) - squared_norm(w.row(j), j);
        if (!(pivot > T(0)) || !std::isfinite(pivot)) {
            return false;
        }
        w(j, j) = std::sqrt(pivot);
        for (Index i = j + 1; i < w.rows; ++i) {
            w(i, j) = (w(i, j) - dot(w.row(i), w.row(j), j)) / w(j, j);
        }
    }
    return true;
}

}  // namespace hlq::detail
