// The optimality certificate on random problems: cold, warm, perturbed and with a metric.

#include <random>

#include <gtest/gtest.h>

#include <hlq/eigen.hpp>
#include <hlq/hlq.hpp>

#include "support/certificate.hpp"

namespace {

using hlq::test::Family;
using Index = Eigen::Index;

struct Sample {
    Family family;
    unsigned count;
};

class RandomProblems : public ::testing::TestWithParam<Sample> {};

std::string name(const ::testing::TestParamInfo<Sample>& info) {
    return hlq::test::to_string(info.param.family);
}

}  // namespace

TEST_P(RandomProblems, AreSolvedColdAndWarm) {
    const auto [family, count] = GetParam();
    for (unsigned seed = 0; seed < count; ++seed) {
        const auto p = hlq::test::make_problem(family, seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved) << "seed " << seed;
        const auto cert = hlq::test::certify(p, hlq::as_eigen(solver.solution()));
        EXPECT_TRUE(cert.optimal) << "seed " << seed << ": level " << cert.level << " unbalanced by "
                                  << cert.residual;
        const Eigen::VectorXd cold = hlq::as_eigen(solver.solution());

        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved);
        EXPECT_EQ(solver.changes(), 0) << "seed " << seed;
        EXPECT_LT((hlq::as_eigen(solver.solution()) - cold).norm(), 1e-12 * (1 + cold.norm()))
            << "seed " << seed;
    }
}

TEST_P(RandomProblems, AreSolvedFromAnotherProblemsActiveSet) {
    const auto [family, count] = GetParam();
    for (unsigned seed = 0; seed < count; ++seed) {
        auto p = hlq::test::make_problem(family, seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved);
        // Move every bound, keeping the rows: the old active set is a warm start for a different optimum.
        std::mt19937 gen(seed + 99991u);
        std::uniform_real_distribution<double> shift(-0.3, 0.3);
        for (Index i = 0; i < p.rows() - p.cols(); ++i) {
            const double dl = shift(gen), du = shift(gen);
            if (p.lower(i) == p.upper(i)) {
                p.upper(i) = p.lower(i) += dl;
            } else {
                p.lower(i) += dl;
                p.upper(i) += du;
                if (p.lower(i) > p.upper(i)) {
                    std::swap(p.lower(i), p.upper(i));
                }
            }
        }
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved) << "seed " << seed;
        const auto cert = hlq::test::certify(p, hlq::as_eigen(solver.solution()));
        EXPECT_TRUE(cert.optimal) << "seed " << seed << ": level " << cert.level << " unbalanced by "
                                  << cert.residual;
    }
}

TEST_P(RandomProblems, AreSolvedInAMetric) {
    const auto [family, count] = GetParam();
    for (unsigned seed = 0; seed < count / 4; ++seed) {
        const auto p = hlq::test::make_problem(family, seed);
        const Eigen::MatrixXd metric = hlq::test::make_metric(p.cols(), seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.set_metric(metric), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved) << "seed " << seed;
        const auto cert = hlq::test::certify(p, hlq::as_eigen(solver.solution()), metric);
        EXPECT_TRUE(cert.optimal) << "seed " << seed << ": level " << cert.level << " unbalanced by "
                                  << cert.residual;
    }
}

INSTANTIATE_TEST_SUITE_P(Families, RandomProblems,
                         ::testing::Values(Sample{Family::Random, 2000}, Sample{Family::Integer, 2000},
                                           Sample{Family::Degenerate, 2000}, Sample{Family::Large, 100}),
                         name);

TEST(RandomProblems, OptimaWithoutAFinalRegularization) {
    // Without the final level x == 0 the optima form a set: the certificate also checks that the solution is
    // the point of it closest to the reference.
    for (unsigned seed = 0; seed < 1000; ++seed) {
        auto p = hlq::test::make_problem(Family::Random, seed);
        const Index rows = p.rows() - p.cols();
        hlq::test::Problem q{p.matrix.topRows(rows), p.lower.head(rows), p.upper.head(rows),
                             p.level_ends.head(p.levels() - 1)};
        const Eigen::VectorXd reference = Eigen::VectorXd::LinSpaced(p.cols(), -1, 1);
        hlq::Solver<> solver(q.rows(), q.cols());
        ASSERT_EQ(solver.set_problem(q.matrix, q.lower, q.upper, q.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.set_reference(reference), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved);
        const auto cert = hlq::test::certify(q, hlq::as_eigen(solver.solution()), {}, reference);
        EXPECT_TRUE(cert.optimal) << "seed " << seed << ": level " << cert.level << " unbalanced by "
                                  << cert.residual;
    }
}

TEST(RandomProblems, CertificateRejectsPerturbedSolutions) {
    // The certificate must not pass everything: moving the solution off the optimum is caught.
    int rejected = 0;
    for (unsigned seed = 0; seed < 500; ++seed) {
        const auto p = hlq::test::make_problem(Family::Random, seed);
        hlq::Solver<> solver(p.rows(), p.cols());
        ASSERT_EQ(solver.set_problem(p.matrix, p.lower, p.upper, p.level_ends), hlq::Status::Ready);
        ASSERT_EQ(solver.solve(), hlq::Status::Solved);
        const Eigen::VectorXd moved =
            hlq::as_eigen(solver.solution()) + 1e-3 * Eigen::VectorXd::Ones(p.cols());
        rejected += hlq::test::certify(p, moved).optimal ? 0 : 1;
    }
    EXPECT_EQ(rejected, 500);
}
