#include "csv_loader.hpp"
#include "rbp.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        const std::string csv_path = "data/processed.csv";
        const std::string target_col = "SPX_21d_vol_fw";

        DataSet data = load_csv(csv_path, target_col);

        std::cout << "Rows: " << data.X.rows() << "\n";
        std::cout << "Features: " << data.X.cols() << "\n";

        const int train_size = 1000;
        const double censor_percentile = 0.25;

        std::ofstream out("outputs/predictions_censor_25.csv");

        if (!out.is_open()) {
            throw std::runtime_error("Could not open outputs/predictions.csv");
        }

        out << "date,prediction,actual,fit,weight_sum,min_weight,max_weight\n";

        for (int task_idx = train_size; task_idx < data.X.rows(); ++task_idx) {
            Eigen::MatrixXd X_train = data.X.middleRows(task_idx - train_size, train_size);
            Eigen::VectorXd y_train = data.Y.segment(task_idx - train_size, train_size);
            Eigen::VectorXd x_task = data.X.row(task_idx).transpose();

            RBPPrediction pred = predict_rbp_single(
                X_train,
                y_train,
                x_task,
                censor_percentile
            );

            out << data.dates[task_idx] << ","
                << pred.prediction << ","
                << data.Y(task_idx) << ","
                << pred.fit << ","
                << pred.weights.sum() << ","
                << pred.weights.minCoeff() << ","
                << pred.weights.maxCoeff()
                << "\n";

            if (task_idx % 250 == 0) {
                std::cout << "Processed: " << data.dates[task_idx] << "\n";
            }
        }

        std::cout << "Saved outputs/predictions.csv\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}