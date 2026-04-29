#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

struct DataSet{
    std::vector<std::string> dates;
    std::vector<std::string> feature_names;
    Eigen::MatrixXd X; // dynamic size X
    Eigen::VectorXd Y; // dynamic size Y   
};

DataSet load_csv(
    const std::string& path,
    const std::string& target_col
);