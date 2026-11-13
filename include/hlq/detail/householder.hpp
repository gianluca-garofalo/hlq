#pragma once

/// @file
/// In-place Householder QR of a tall block and the application of its Q, without allocation.

#include <type_traits>

#include <Eigen/Dense>

namespace hlq::detail {

/// Householder QR of `a` (rows >= cols) in place: R in the upper triangle, the reflectors below it and their
/// coefficients in `tau`. `workspace` holds at least `a.cols()` scalars.
template <typename Block, typename Coefficients, typename Scalar>
void householder_qr(Block&& a, Coefficients&& tau, Scalar* workspace) {
    const Eigen::Index rows = a.rows();
    const Eigen::Index cols = a.cols();
    for (Eigen::Index j = 0; j < cols; ++j) {
        Scalar beta;
        a.col(j).tail(rows - j).makeHouseholderInPlace(tau.coeffRef(j), beta);
        a.coeffRef(j, j) = beta;
        if (j + 1 < cols) {
            a.bottomRightCorner(rows - j, cols - j - 1)
                .applyHouseholderOnTheLeft(a.col(j).tail(rows - j - 1), tau.coeff(j), workspace);
        }
    }
}

/// v <- Q^T v, with Q from `householder_qr`.
template <typename Block, typename Coefficients, typename Vector>
void apply_qt(const Block& a, const Coefficients& tau, Vector&& v) {
    const Eigen::Index rows = a.rows();
    typename std::decay_t<Vector>::Scalar workspace;
    for (Eigen::Index j = 0; j < a.cols(); ++j) {
        v.tail(rows - j).applyHouseholderOnTheLeft(a.col(j).tail(rows - j - 1), tau.coeff(j), &workspace);
    }
}

/// v <- Q v, with Q from `householder_qr`.
template <typename Block, typename Coefficients, typename Vector>
void apply_q(const Block& a, const Coefficients& tau, Vector&& v) {
    const Eigen::Index rows = a.rows();
    typename std::decay_t<Vector>::Scalar workspace;
    for (Eigen::Index j = a.cols() - 1; j >= 0; --j) {
        v.tail(rows - j).applyHouseholderOnTheLeft(a.col(j).tail(rows - j - 1), tau.coeff(j), &workspace);
    }
}

}  // namespace hlq::detail
