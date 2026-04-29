#include <csv_loader.hpp>
#include <fstream> //io to files
#include <sstream> //io to strings in files
#include <string>
#include <vector>

// split each line into cells , delimiter is ,
static std::vector<std::string> split_csv_line(const std::string& line){ 
    std::vector<std::string> result;
    std::stringstream ss(line); // create stringstream object with the line
    std::string cell;

    while(std::getline(ss, cell, ',')){
        result.push_back(cell);
    }
    return result;
}

DataSet load_csv(
    const std::string& path,
    const std::string& target_col
) {
    std::ifstream file(path);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;
    if(!std::getline(file, line)){
        throw std::runtime_error("CSV file is empty:" + path);     
    }

    std::vector<std::string> header = split_csv_line(line); //headers are the csv
    int date_col = -1;
    int target_idx = -1;

    for(int i = 0; i <static_cast<int>(header.size()); ++i){//
        if(header[i] == "date"){
            date_col = i;
        }
        if(header[i] == target_col){
            target_idx = i;
        }
    } 

    if(date_col == -1){
        throw std::runtime_error("Could not find the date column.");
    }
    if(target_idx == -1){
        throw std::runtime_error("Could not find the target column: " + target_col);
    }

    std::vector<int> feature_idx;
    std::vector<std::string> feature_names;

    for(int i = 0; i < static_cast<int>(header.size()); ++i){ 
        // read the features, ignore date and target index of header
        if(i != date_col && i != target_idx){
            feature_idx.push_back(i);
            feature_names.push_back(header[i]);
        }
    }

    std::vector<std::string> dates;
    std::vector<std::vector<double>> X_rows;
    std::vector<double> Y_values;

    while (std::getline(file, line)){
        if(line.empty()){
            continue;
        }
        std::vector<std::string> cells = split_csv_line(line); //split the line, push into cell

        if(cells.size() != header.size()){
            throw std::runtime_error("Row has wrong number of elements.");
        }
        dates.push_back(cells[date_col]);

        std::vector<double> x_row;
        for(int idx: feature_idx){
            x_row.push_back(std::stod(cells[idx]));
        }
        X_rows.push_back(x_row);
        Y_values.push_back(std::stod(cells[target_idx]));
    }

    int n_rows = static_cast<int>(X_rows.size());
    int n_cols = static_cast<int>(feature_idx.size());

    Eigen::MatrixXd X(n_rows, n_cols); //initialize matrix with same row/col
    Eigen::VectorXd y(n_rows);

    for (int i = 0; i < n_rows; ++i){
        for(int j = 0; j < n_cols; ++j){
            X(i,j) = X_rows[i][j];
        }
        y(i) = Y_values[i];
    }

    DataSet data;
    data.dates = dates;
    data.feature_names = feature_names;
    data.X = X;
    data.Y = y;
    return data;

}