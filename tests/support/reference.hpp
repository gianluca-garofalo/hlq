#pragma once

// Reference solutions of equality hierarchies, by SVD on nested null spaces: slow, simple and independent of
// the factorization under test.

#include <vector>

#include <Eigen/Dense>

namespace hlq::test {

struct ReferenceLevel {
    Eigen::MatrixXd null_space;  // basis of what the levels above leave free
    Eigen::MatrixXd projected;   // rows of the level times null_space
    Eigen::Index rank = 0;
};

struct ReferenceSolution {
    Eigen::VectorXd x;
    std::vector<ReferenceLevel> levels;
};

/**
 * Minimum-norm hierarchical least-squares solution of the equalities `rows_k x = targets_k`, with an absolute
 * rank threshold of `tolerance` times the longest row of each level.
 */
inline ReferenceSolution reference_solve(const std::vector<Eigen::MatrixXd>& rows,
                                         const std::vector<Eigen::VectorXd>& targets,
                                         double tolerance = 1e-9) {
    const Eigen::Index n = rows.front().cols();
    ReferenceSolution s{Eigen::VectorXd::Zero(n), {}};
    Eigen::MatrixXd z = Eigen::MatrixXd::Identity(n, n);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        ReferenceLevel level{z, rows[k] * z, 0};
        if (rows[k].rows() > 0 && z.cols() > 0) {
            const double noise = tolerance * rows[k].rowwise().norm().maxCoeff();
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(level.projected, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::VectorXd& sigma = svd.singularValues();
            while (level.rank < sigma.size() && sigma(level.rank) > noise) {
                ++level.rank;
            }
            const Eigen::Index r = level.rank;
            const Eigen::VectorXd goal = targets[k] - rows[k] * s.x;
            const Eigen::VectorXd eta =
                svd.matrixV().leftCols(r)
                * (svd.matrixU().leftCols(r).transpose() * goal).cwiseQuotient(sigma.head(r));
            s.x += z * eta;
            z = z * svd.matrixV().rightCols(z.cols() - r);
        }
        s.levels.push_back(level);
    }
    return s;
}

/// Multipliers of the subproblem of level h, with the minimum-norm choice within each level above.
inline std::vector<Eigen::VectorXd> reference_multipliers(const ReferenceSolution& s,
                                                          const std::vector<Eigen::MatrixXd>& rows,
                                                          const std::vector<Eigen::VectorXd>& targets,
                                                          std::size_t h) {
    std::vector<Eigen::VectorXd> lambda(h + 1);
    lambda[h] = rows[h] * s.x - targets[h];
    Eigen::VectorXd tau = rows[h].transpose() * lambda[h];
    for (std::size_t k = h; k-- > 0;) {
        const ReferenceLevel& level = s.levels[k];
        lambda[k] = Eigen::VectorXd::Zero(rows[k].rows());
        if (level.rank > 0) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(level.projected, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Eigen::Index r = level.rank;
            const Eigen::VectorXd projected = level.null_space.transpose() * tau;
            lambda[k] = -svd.matrixU().leftCols(r)
                        * (svd.matrixV().leftCols(r).transpose() * projected)
                              .cwiseQuotient(svd.singularValues().head(r));
        }
        tau += rows[k].transpose() * lambda[k];
    }
    return lambda;
}

}  // namespace hlq::test
