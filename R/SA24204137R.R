#' @title Ridge Regression for optimal lambda selection
#' @description Ridge Regression for optimal lambda selection
#' @param X A numeric matrix or data frame of predictor variables (independent variables).
#' @param y A numeric vector of the response variable (dependent variable).
#' @param lambdas A numeric vector of lambda values to consider in the model.
#' @param k The number of folds for cross-validation. Default is 5.
#' @return The best lambda value that minimizes the Mean Squared Error (MSE) from cross-validation.
#' @examples
#' \dontrun{
#' data(mtcars)
#' # Perform cross-validation for ridge regression on the mtcars dataset
#' X <- mtcars[, -1]  # Predictor variables (excluding the first column)
#' y <- mtcars[, 1]   # Response variable (the first column)
#' best_lambda <- ridge_cv(X, y, lambdas = seq(0, 10, by = 0.1), k = 5)
#' print(best_lambda)
#' }
#' @export
ridge_cv <- function(X, y, lambdas = seq(0, 10, by = 0.1), k = 5) {
  # Check if X and y are valid
  if (nrow(X) != length(y)) {
    stop("Number of rows in X must match the length of y.")
  }
  
  # Number of observations
  n <- nrow(X)
  
  # Create k-folds for cross-validation
  set.seed(123)  # For reproducibility
  fold_ids <- sample(1:k, n, replace = TRUE)
  
  # Initialize a vector to store MSE values for each lambda
  mse_values <- numeric(length(lambdas))
  
  # Perform cross-validation for each lambda
  for (lambda_idx in 1:length(lambdas)) {
    lambda <- lambdas[lambda_idx]
    
    mse_fold <- numeric(k)  # MSE for each fold
    
    # Perform k-fold cross-validation
    for (fold in 1:k) {
      # Split data into training and validation sets
      train_X <- X[fold_ids != fold, , drop = FALSE]
      train_y <- y[fold_ids != fold]
      valid_X <- X[fold_ids == fold, , drop = FALSE]
      valid_y <- y[fold_ids == fold]
      
      # Standardize the predictor variables (important for ridge regression)
      mean_train_X <- colMeans(train_X)
      sd_train_X <- apply(train_X, 2, sd)
      train_X <- scale(train_X, center = mean_train_X, scale = sd_train_X)
      valid_X <- scale(valid_X, center = mean_train_X, scale = sd_train_X)
      
      # Ridge regression: solve for beta using the formula
      # beta = (X'X + lambda * I)^(-1) X'y
      I <- diag(ncol(train_X))  # Identity matrix
      beta <- solve(t(train_X) %*% train_X + lambda * I) %*% t(train_X) %*% train_y
      
      # Predict on validation set
      pred_y <- valid_X %*% beta
      
      # Compute MSE for this fold
      mse_fold[fold] <- mean((valid_y - pred_y)^2)
    }
    
    # Store the average MSE for the current lambda
    mse_values[lambda_idx] <- mean(mse_fold)
  }
  
  # Return the lambda with the minimum MSE
  best_lambda <- lambdas[which.min(mse_values)]
  return(best_lambda)
}

#' @title Variance Inflation Factor (VIF) Wrapper
#' @description This function serves as an R wrapper to the C++ implementation of VIF calculation.
#' @param x A numeric matrix or data frame of predictor variables (independent variables).
#' @return A numeric vector of VIF values for each predictor variable.
#' @examples
#' \dontrun{
#' data(mtcars)
#' vif(as.matrix(mtcars[, -1]))
#' }
#' @importFrom stats sd
#' @useDynLib SA24204137
#' @export
vif <- function(x) {
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  .Call('_SA24204137_vif', x, PACKAGE = "SA24204137")
}