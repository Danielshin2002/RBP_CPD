#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd X(2, 2);
    X << 1, 2,
         3, 4;

    std::cout << "Eigen is working.\n";
    std::cout << X << "\n";

    return 0;
}
