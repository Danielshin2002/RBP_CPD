#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>


struct StandardizedData {
    Eigen::MatrixXd X_train_std;
    Eigen::VectorXd x_task_std;
    Eigen::VectorXd mean;
    Eigen::VectorXd stddev;
};


struct RBPPrediction {
    double prediction;
    double fit;
    Eigen::VectorXd weights;
    Eigen::VectorXd relevance;
};


StandardizedData standardize_train_and_task(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXd& x_task
);

Eigen::MatrixXd covariance_matrix(
    const Eigen::MatrixXd& X
);

double mahalanobis_quadratic(
    const Eigen::VectorXd& v,
    const Eigen::LDLT<Eigen::MatrixXd>& ldlt
);

Eigen::VectorXd compute_relevance(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXd& x_task,
    const Eigen::LDLT<Eigen::MatrixXd>& ldlt
);

double percentile_threshold(
    const Eigen::VectorXd& values,
    double percentile
);

Eigen::VectorXd compute_rbp_weights_censored(
    const Eigen::VectorXd& relevance,
    double r_star
);

double correlation(
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b
);

RBPPrediction predict_rbp_single(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXd& y_train,
    const Eigen::VectorXd& x_task,
    double censor_percentile
);