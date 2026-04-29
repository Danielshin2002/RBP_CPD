#include "rbp.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

StandardizedData standardize_train_and_task(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXd& x_task
) {
    if (X_train.cols() != x_task.size()) {
        throw std::runtime_error("X_train columns must match x_task size.");
    }

    const int n = static_cast<int>(X_train.rows());
    const int k = static_cast<int>(X_train.cols());

    if (n < 2) {
        throw std::runtime_error("Need at least two training rows to standardize.");
    }

    Eigen::VectorXd mean(k);
    Eigen::VectorXd stddev(k);

    for (int j = 0; j < k; ++j) {
        mean(j) = X_train.col(j).mean();

        double variance = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = X_train(i, j) - mean(j);
            variance += diff * diff;
        }
        variance /= static_cast<double>(n - 1);
        stddev(j) = std::sqrt(variance);

        if (stddev(j) == 0.0) {
            throw std::runtime_error("Zero standard deviation in feature column.");
        }
    }

    Eigen::MatrixXd X_train_std(n, k);
    Eigen::VectorXd x_task_std(k);

    for (int j = 0; j < k; ++j) {
        X_train_std.col(j) = (X_train.col(j).array() - mean(j)) / stddev(j);
        x_task_std(j) = (x_task(j) - mean(j)) / stddev(j);
    }
    return {X_train_std, x_task_std, mean, stddev};
}
Eigen::MatrixXd covariance_matrix(
    const Eigen::MatrixXd& X
) {
    const int n = static_cast<int>(X.rows());

    if (n < 2) {
        throw std::runtime_error("Need at least two rows to compute covariance.");
    }

    Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();

    return (centered.transpose() * centered) / static_cast<double>(n - 1);
}


double mahalanobis_quadratic(
    const Eigen::VectorXd& v,
    const Eigen::LDLT<Eigen::MatrixXd>& ldlt
) {
    return v.dot(ldlt.solve(v));
}

Eigen::VectorXd compute_relevance(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXd& x_task,
    const Eigen::LDLT<Eigen::MatrixXd>& ldlt
) {
    const int n = static_cast<int>(X_train.rows());

    Eigen::VectorXd relevance(n);
    Eigen::VectorXd x_bar = X_train.colwise().mean();

    Eigen::VectorXd task_diff = x_task - x_bar;
    double info_task = mahalanobis_quadratic(task_diff, ldlt);

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd xi = X_train.row(i).transpose();

        Eigen::VectorXd sim_diff = xi - x_task;
        double similarity = -0.5 * mahalanobis_quadratic(sim_diff, ldlt);

        Eigen::VectorXd info_diff = xi - x_bar;
        double info_i = mahalanobis_quadratic(info_diff, ldlt);

        relevance(i) = similarity + 0.5 * (info_i + info_task);
    }

    return relevance;
}

double percentile_threshold(
    const Eigen::VectorXd& values,
    double percentile
) {
    if (values.size() == 0) {
        throw std::runtime_error("Cannot compute percentile of empty vector.");
    }

    if (percentile < 0.0 || percentile > 1.0) {
        throw std::runtime_error("Percentile must be between 0 and 1.");
    }

    std::vector<double> v(values.data(), values.data() + values.size());
    std::sort(v.begin(), v.end());

    int idx = static_cast<int>(
        percentile * static_cast<double>(v.size() - 1)
    );

    return v[idx];
}






Eigen::VectorXd compute_rbp_weights_censored(
    const Eigen::VectorXd& relevance,
    double r_star
) {
    const int N = static_cast<int>(relevance.size());

    if (N < 2) {
        throw std::runtime_error("Need at least two observations.");
    }

    Eigen::VectorXd delta(N);
    int n = 0;

    for (int i = 0; i < N; ++i) {
        if (relevance(i) >= r_star) {
            delta(i) = 1.0;
            ++n;
        } else {
            delta(i) = 0.0;
        }
    }

    if (n < 2) {
        throw std::runtime_error("Too few retained observations after censoring.");
    }

    double phi = static_cast<double>(n) / static_cast<double>(N);

    double retained_relevance_sum = 0.0;
    double retained_relevance_sq_sum = 0.0;
    double full_relevance_sq_sum = relevance.squaredNorm();

    for (int i = 0; i < N; ++i) {
        if (delta(i) == 1.0) {
            retained_relevance_sum += relevance(i);
            retained_relevance_sq_sum += relevance(i) * relevance(i);
        }
    }

    double r_sub_bar = retained_relevance_sum / static_cast<double>(n);

    double full_second_moment =
        full_relevance_sq_sum / static_cast<double>(N - 1);

    double retained_second_moment =
        retained_relevance_sq_sum / static_cast<double>(n - 1);

    if (retained_second_moment == 0.0) {
        throw std::runtime_error("Retained relevance second moment is zero.");
    }

    double lambda2 = full_second_moment / retained_second_moment;

    Eigen::VectorXd weights(N);

    for (int i = 0; i < N; ++i) {
        weights(i) =
            1.0 / static_cast<double>(N)
            + (lambda2 / static_cast<double>(n - 1))
                * (delta(i) * relevance(i) - phi * r_sub_bar);
    }

    return weights;
}


double correlation(
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Correlation vectors must have same size.");
    }

    const int n = static_cast<int>(a.size());

    if (n < 2) {
        throw std::runtime_error("Need at least two observations for correlation.");
    }

    Eigen::VectorXd a_centered = a.array() - a.mean();
    Eigen::VectorXd b_centered = b.array() - b.mean();

    double denom = std::sqrt(
        a_centered.squaredNorm() * b_centered.squaredNorm()
    );

    if (denom == 0.0) {
        return 0.0;
    }

    return a_centered.dot(b_centered) / denom;
}


RBPPrediction predict_rbp_single(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXd& y_train,
    const Eigen::VectorXd& x_task,
    double censor_percentile
) {
    if (X_train.rows() != y_train.size()) {
        throw std::runtime_error("X_train rows must match y_train size.");
    }

    if (censor_percentile < 0.0 || censor_percentile > 1.0) {
        throw std::runtime_error("Censor percentile must be between 0 and 1.");
    }

    StandardizedData std_data = standardize_train_and_task(
        X_train,
        x_task
    );

    Eigen::MatrixXd cov = covariance_matrix(std_data.X_train_std);

    // Tiny ridge term for numerical stability.
    const double ridge = 1e-8;
    cov += ridge * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());

    Eigen::LDLT<Eigen::MatrixXd> ldlt(cov);

    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("LDLT decomposition failed.");
    }

    Eigen::VectorXd relevance = compute_relevance(
        std_data.X_train_std,
        std_data.x_task_std,
        ldlt
    );

    double r_star = percentile_threshold(
        relevance,
        censor_percentile
    );

    Eigen::VectorXd weights = compute_rbp_weights_censored(
        relevance,
        r_star
    );

    double prediction = weights.dot(y_train);

    double rho = correlation(weights, y_train);
    double fit = rho * rho;

    return {prediction, fit, weights, relevance};
}
