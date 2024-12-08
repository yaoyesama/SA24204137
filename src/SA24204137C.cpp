#include <Rcpp.h>
using namespace Rcpp;

//' @title Calculate Variance Inflation Factor
//' @description Calculate Variance Inflation Factor (VIF) for each predictor variable in a dataset.
//' @param x A numeric matrix of predictor variables (independent variables).
//' @return A numeric vector of VIF values for each predictor variable in the dataset.
//' @examples
//' \dontrun{
//' data(mtcars)
//' vif(as.matrix(mtcars[, -1]))
//' }
//' @export
// [[Rcpp::export]]
NumericVector vif(NumericMatrix x) {
  int n = x.ncol();
  NumericVector vif_values(n);

  // Loop over each predictor variable to compute its VIF
  for (int i = 0; i < n; i++) {
    // Extract the i-th column as the dependent variable
    NumericVector y = x(_, i);

    // Create a matrix excluding the i-th column
    NumericMatrix x_others(x.nrow(), x.ncol() - 1);
    int col_idx = 0;
    for (int j = 0; j < n; j++) {
      if (j != i) {
        x_others(_, col_idx) = x(_, j);
        col_idx++;
      }
    }

    // Helper function to compute the correlation coefficient
    auto compute_correlation = [](NumericVector x, NumericVector y) {
      int n = x.size();
      double mean_x = mean(x);
      double mean_y = mean(y);

      double numerator = 0.0;
      double sum_sq_x = 0.0;
      double sum_sq_y = 0.0;

      for (int i = 0; i < n; i++) {
        double diff_x = x[i] - mean_x;
        double diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
      }

      return numerator / std::sqrt(sum_sq_x * sum_sq_y);
    };

    // Compute R^2 for regressing y on x_others
    NumericVector r_squared_values(x_others.ncol());
    for (int j = 0; j < x_others.ncol(); j++) {
      r_squared_values[j] = std::pow(compute_correlation(y, x_others(_, j)), 2);
    }
    double r_squared = mean(r_squared_values);

    // Calculate VIF: VIF = 1 / (1 - R^2)
    vif_values[i] = 1 / (1 - r_squared);
  }

  return vif_values;
}