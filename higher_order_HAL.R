# This script includes an implementation of of the higher order HAL
# 
# Todo:
#   Add general loss function parameter to penalized lm
# 
# 
# 
# 
# ==============================================================================


library(hal9001)
library(data.table)
library(glmnet)
library(ggplot2)

# ----------------------
# Step 1. Simulate data (non-linear)
# ----------------------
set.seed(123)
n <- 10000
p <- 2
X <- matrix(ncol = 2, c(runif(n,0,1),rbinom(n,size = 1, prob = 0.5)))  # uniform covariates
colnames(X) <- paste0("X", 1:p)

# Non-linear outcome
Y <- exp(2*X[,1])^X[,2]*(10*exp(-3*(sin(10*(X[,1]-0.5))^2)))^(1-X[,2]) + rnorm(n, sd = 0.2)

# ----------------------
# Step 2. Fit HAL
# ----------------------
fit_hal <- fit_hal(
  X = X, Y = Y,
  family = "gaussian",
  yolo = FALSE,
  return_lasso = TRUE,
  max_degree = p,
  smoothness_orders = 0
)

# ----------------------
# Step 3. Extract basis functions with non-zero coefficients
#   and convert to higher order basis functions
# ----------------------
hal_basis <- data.table(phi_0 = summary(fit_hal,
                     include_redundant_terms = TRUE)$table$term)
hal_basis[phi_0 == "(Intercept)", phi_0 := "1"]
hal_basis[,phi_0 := gsub("\\[|\\]","",phi_0)]
hal_basis[,phi_0 := gsub("I","as.numeric",phi_0)]

# "Integrate" single factor
mu_factor <- function(factor_str) {
  # match exponent
  m <- regmatches(factor_str, regexpr("\\^\\d+", factor_str))
  if (length(m) == 0){
    k <- 0
    factor_str <- gsub("as.numeric\\(([^ ]+) >= ([^\\)]+)\\)","as.numeric(\\1 >= \\2)*(\\1 - \\2)^0",factor_str)
  }else{
    k <- as.integer(sub("\\^", "", m))
  }
  new_exp <- k + 1
  new_fact <- new_exp
  
  # replace old exponent with new exponent/factorial
  factor_str <- sub("\\^\\d+", paste0("^", new_exp, "/", new_fact), factor_str)
  factor_str
}

# Promote an entire spline string (product of factors)
mu <- function(spline_str) {
  factors <- strsplit(spline_str, "\\*")[[1]]
  factors <- trimws(factors)
  factors <- sapply(factors, mu_factor)
  paste(factors, collapse = " * ")
}

# Promote k times and create columns
mu_k <- function(dt, colname = "phi_0", k = 3) {
  exprs <- dt[[colname]]
  for (i in 1:k) {
    exprs <- sapply(exprs, mu)
    newcol <- paste0("phi_", i)
    dt[[newcol]] <- exprs
  }
  dt
}

hal_spline_basis <- mu_k(hal_basis, k=3)

# ----------------------
# Step 4. Map basis functions to raw covariates at point
# ----------------------
evaluate_spline_at_point <- function(k = 3, points) {
  dt <- as.data.table(points)
  for(i in 0:k){
    expan <- paste0("phi_",i)
    expr_list <- hal_spline_basis[,get(expan)]
    for (expr in expr_list) {
      dt[, (expr) := eval(parse(text = expr), envir = .SD)]
    }
  }
  dt[,(paste0("X", 1:p)):=NULL]
}
basis_expansion <- evaluate_spline_at_point(points = X, k = 3)
d <- cbind(Y,basis_expansion)

# ----------------------
# Step 5a. Refit a lasso penalized LM for different penalty hyperparameters
# ----------------------
lambda_seq <- seq(0,0.1,by = 0.01)

# Prediction grid
n_grid <- 500
X1_vals <- seq(0, 1, length.out = n_grid)

# Grids for X2 = 0 and X2 = 1
grid0 <- data.table(X1 = X1_vals, X2 = 0)
grid1 <- data.table(X1 = X1_vals, X2 = 1)

# Evaluate HAL basis at grid points
nd0 <- evaluate_spline_at_point(points = grid0)
nd1 <- evaluate_spline_at_point(points = grid1)

# Store predictions for each lambda
plot_list <- list()

for (lam in lambda_seq) {
  # Fit glmnet with chosen lambda (lasso example)
  fit <- glmnet(
    x = as.matrix(basis_expansion),
    y = Y,
    alpha = 1,           # 1 = lasso, 0 = ridge
    lambda = lam
  )
  
  # Predict on the grids
  pen_pred0 <- predict(fit, newx = as.matrix(nd0))
  pen_pred1 <- predict(fit, newx = as.matrix(nd1))
  
  # Collect in data.table
  plot_list[[as.character(lam)]] <- data.table(
    X1 = rep(X1_vals, 2),
    X2 = rep(c(0, 1), each = n_grid),
    pred = c(pen_pred0, pen_pred1),
    lambda = lam
  )
}

# Combine results
plot_dt <- rbindlist(plot_list)

# Plot penalized fits
ggplot(plot_dt, aes(x = X1, y = pred, color = factor(X2), group = interaction(X2, lambda))) +
  geom_line(alpha = 0.7) +
  labs(
    x = "X1",
    y = "Predicted Y",
    color = "X2"
  ) +
  theme_minimal() +
  stat_function(fun = function(x) exp(2*x), color = "red", size = 1) +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "red", size = 1) +
  facet_wrap(~lambda, ncol = 3)   # one panel per lambda

# ----------------------
# Step 5b. Refit a ridge penalized LM for different penalty hyperparameters
# ----------------------
lambda_seq <- seq(0,0.1,by = 0.01)

# Prediction grid
n_grid <- 500
X1_vals <- seq(0, 1, length.out = n_grid)

# Grids for X2 = 0 and X2 = 1
grid0 <- data.table(X1 = X1_vals, X2 = 0)
grid1 <- data.table(X1 = X1_vals, X2 = 1)

# Evaluate HAL basis at grid points
nd0 <- evaluate_spline_at_point(points = grid0)
nd1 <- evaluate_spline_at_point(points = grid1)

# Store predictions for each lambda
plot_list <- list()

for (lam in lambda_seq) {
  # Fit glmnet with chosen lambda (lasso example)
  fit <- glmnet(
    x = as.matrix(basis_expansion),
    y = Y,
    alpha = 0,           # 1 = lasso, 0 = ridge
    lambda = lam
  )
  
  # Predict on the grids
  pen_pred0 <- predict(fit, newx = as.matrix(nd0))
  pen_pred1 <- predict(fit, newx = as.matrix(nd1))
  
  # Collect in data.table
  plot_list[[as.character(lam)]] <- data.table(
    X1 = rep(X1_vals, 2),
    X2 = rep(c(0, 1), each = n_grid),
    pred = c(pen_pred0, pen_pred1),
    lambda = lam
  )
}

# Combine results
plot_dt <- rbindlist(plot_list)

# Plot penalized fits
ggplot(plot_dt, aes(x = X1, y = pred, color = factor(X2), group = interaction(X2, lambda))) +
  geom_line(alpha = 0.7) +
  labs(
    x = "X1",
    y = "Predicted Y",
    color = "X2"
  ) +
  theme_minimal() +
  stat_function(fun = function(x) exp(2*x), color = "red", size = 1) +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "red", size = 1) +
  facet_wrap(~lambda, ncol = 3)   # one panel per lambda

# ----------------------
# Step 5c1. Refit a penalized LM using selected cross-validation using lambda from
#   a "raw" lasso fit
# ----------------------
## Add penalizing (OLS) with both ridge and lasso penalizing
##  and general loss function

# ===OUTDATED===================================================================
# A penalized linear model (todo: general loss) 
# penalized_lm <- function(X, y, lambda = 1, penalty = c("ridge", "lasso")) {
#   penalty <- match.arg(penalty)
#   
#   # Add intercept
#   X <- as.matrix(X)
#   n <- nrow(X)
#   p <- ncol(X)
#   
#   # Objective function
#   obj <- function(beta) {
#     residuals <- y - X %*% beta
#     rss <- sum(residuals^2)
#     
#     if (penalty == "ridge") {
#       penalty_val <- lambda * sum(beta[-1]^2)  # exclude intercept
#     } else if (penalty == "lasso") {
#       penalty_val <- lambda * sum(abs(beta[-1]))
#     }
#     
#     return(rss + penalty_val)
#   }
#   
#   # Optimize using optim
#   fit <- optim(rep(0, p), obj, method = "BFGS")
#   
#   coefficients <- fit$par
#   names(coefficients) <- c("Intercept", colnames(X)[-1])
#   
#   list(coefficients = coefficients,
#        lambda = lambda,
#        penalty = penalty,
#        value = fit$value,
#        convergence = fit$convergence)
# }
#===============================================================================

# Add a cross validation step to the penalized lm
cv_penalized_lm_lasso <- function(X, y, alpha,
                         K = 5, seed = 123) {
  set.seed(seed)
  
  # Define hyperparameters
  lambda_seq <- glmnet(
    x = as.matrix(basis_expansion),
    y = Y,
    alpha = alpha         # 1 = lasso, 0 = ridge
  )$lambda
  
  # define folds
  n <- nrow(X)
  folds <- sample(rep(1:K, length.out = n))
  
  cv_errors <- matrix(NA, nrow = K, ncol = length(lambda_seq))
  
  for(k in 1:K){
    
    train_idx <- which(folds != k)
    valid_idx <- which(folds == k)
    
    X_train <- X[train_idx, drop = FALSE]
    y_train <- y[train_idx]
    X_valid <- X[valid_idx, drop = FALSE]
    y_valid <- y[valid_idx]
    for (j in seq_along(lambda_seq)){ 
    
      fit <- glmnet(x = X_train, y = y_train,
                          lambda = lambda_seq[j])
    
      # Prediction
      Xmat_valid <- as.matrix(X_valid)
      yhat <- predict(fit, newx = Xmat_valid)
      
      cv_errors[k, j] <- mean((y_valid - yhat)^2)
    }
  }
  
  mean_cv_error <- colMeans(cv_errors)
  best_index <- which.min(mean_cv_error)
  best_lambda <- lambda_seq[best_index]
  # Refit on full data at best lambda
  best_fit <- glmnet(X, y, lambda = best_lambda, alpha = alpha)
  
  return(best_fit)
  
  # list(cv_errors = cv_errors,
  #      mean_cv_error = mean_cv_error,
  #      lambda_seq = lambda_seq,
  #      best_lambda = best_lambda,
  #      best_index = best_index)
}

# lambda_seq <- seq(0.1, 1, length.out = 20)

cv_higher_order_hal_lasso <- cv_penalized_lm_lasso(X = basis_expansion, y = Y,alpha = 1)
cv_higher_order_hal_ridge <- cv_penalized_lm_lasso(X = basis_expansion, y = Y,alpha = 0)

higher_order_hal_MLE_unpen <- lm(Y~.-1,data = d)

# ----------------------
# Step 6. Compute regression function along grid
# ----------------------
n_grid <- 500
X1_vals <- seq(0, 1, length.out = n_grid)

# Create grids for X2 = 0 and X2 = 1
grid0 <- data.table(X1 = X1_vals, X2 = 0)
grid1 <- data.table(X1 = X1_vals, X2 = 1)

# Step 2: Evaluate HAL basis at these new points
nd0 <- evaluate_spline_at_point(points = grid0)
nd1 <- evaluate_spline_at_point(points = grid1)

# Step 3: Predict using the refitted penalized LM
pen_pred0_lasso <- predict(cv_higher_order_hal_lasso, newx = as.matrix(nd0))
pen_pred1_lasso <- predict(cv_higher_order_hal_lasso, newx = as.matrix(nd1))

pen_pred0_ridge <- predict(cv_higher_order_hal_ridge, newx = as.matrix(nd0))
pen_pred1_ridge <- predict(cv_higher_order_hal_ridge, newx = as.matrix(nd1))

pred0 <- predict(higher_order_hal_MLE_unpen, newdata = nd0)
pred1 <- predict(higher_order_hal_MLE_unpen, newdata = nd1)

# Step 4: Combine for plotting
plot_dt_lasso <- data.table(
  X1 = rep(X1_vals, 2),
  X2 = rep(c(0, 1), each = n_grid),
  pred = c(pen_pred0_lasso, pen_pred1_lasso)
)
plot_dt_ridge <- data.table(
  X1 = rep(X1_vals, 2),
  X2 = rep(c(0, 1), each = n_grid),
  pred = c(pen_pred0_ridge, pen_pred1_ridge)
)
plot_dt_unpen <- data.table(
  X1 = rep(X1_vals, 2),
  X2 = rep(c(0, 1), each = n_grid),
  pred = c(pred0, pred1)
)

# Step 5: Plot
library(ggplot2)
ggplot(plot_dt_lasso, aes(x = X1, y = pred, color = factor(X2))) +
  # geom_smooth() +
  geom_line(size = 1) +
  labs(x = "X1", y = "Predicted Y", color = "X2", title = "Lasso penalization") +
  theme_minimal() + 
  stat_function(fun = function(x) exp(2*x), color = "blue", size = 1, linetype = "dashed") +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "red", size = 1, linetype = "dashed")
ggplot(plot_dt_ridge, aes(x = X1, y = pred, color = factor(X2))) +
  # geom_smooth() +
  geom_line(size = 1) +
  labs(x = "X1", y = "Predicted Y", color = "X2", title = "Ridge penalization") +
  theme_minimal() + 
  stat_function(fun = function(x) exp(2*x), color = "blue", size = 1, linetype = "dashed") +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "red", size = 1, linetype = "dashed")
# plot the unpenalized model
ggplot(plot_dt_unpen, aes(x = X1, y = pred, color = factor(X2))) +
  # geom_smooth() +
  geom_line(size = 1) +
  labs(x = "X1", y = "Predicted Y", color = "X2", title = "No penalization") +
  theme_minimal() + 
  stat_function(fun = function(x) exp(2*x), color = "blue", size = 1, linetype = "dashed") +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "red", size = 1, linetype = "dashed")

# =================================
# Compare to hal9001 with smoothness_orders = 0 (ordinary HAL)
# =================================

# ----------------------
# Step 1. Fit HAL
# ----------------------
library(hal9001)
fit_hal_0 <- fit_hal(
  X = X, Y = Y,
  family = "gaussian",
  max_degree = p,
  smoothness_orders = 0
)

# ----------------------
# Step 2. Define prediction grids
# ----------------------
n_grid <- 500
X1_vals <- seq(0, 1, length.out = n_grid)

grid0 <- data.frame(X1 = X1_vals, X2 = 0)
grid1 <- data.frame(X1 = X1_vals, X2 = 1)

# ----------------------
# Step 3. Predict with HAL
# ----------------------
pred0_hal9001_0 <- predict(fit_hal_0, new_data = grid0)
pred1_hal9001_0 <- predict(fit_hal_0, new_data = grid1)

# ----------------------
# Step 4. Combine for plotting
# ----------------------
library(data.table)
plot_dt_hal9001_0 <- data.table(
  X1 = rep(X1_vals, 2),
  X2 = rep(c(0, 1), each = n_grid),
  pred = c(pred0_hal9001_0, pred1_hal9001_0)
)

# ----------------------
# Step 5. Plot
# ----------------------
library(ggplot2)
ggplot(plot_dt_hal9001_0, aes(x = X1, y = pred, color = factor(X2))) +
  geom_line(size = 1) +
  labs(x = "X1", y = "Predicted Y", color = "X2") +
  theme_minimal() + 
  stat_function(fun = function(x) exp(2*x), color = "green", size = 1) +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "orange", size = 1)



# =================================
# Compare to hal9001 with smoothness_orders = 3
# =================================

# ----------------------
# Step 1. Fit HAL
# ----------------------
library(hal9001)
fit_hal_3 <- fit_hal(
  X = X, Y = Y,
  family = "gaussian",
  max_degree = p,
  smoothness_orders = 3
)

# ----------------------
# Step 2. Define prediction grids
# ----------------------
n_grid <- 500
X1_vals <- seq(0, 1, length.out = n_grid)

grid0 <- data.frame(X1 = X1_vals, X2 = 0)
grid1 <- data.frame(X1 = X1_vals, X2 = 1)

# ----------------------
# Step 3. Predict with HAL
# ----------------------
pred0_hal9001 <- predict(fit_hal_3, new_data = grid0)
pred1_hal9001 <- predict(fit_hal_3, new_data = grid1)

# ----------------------
# Step 4. Combine for plotting
# ----------------------
library(data.table)
plot_dt_hal9001 <- data.table(
  X1 = rep(X1_vals, 2),
  X2 = rep(c(0, 1), each = n_grid),
  pred = c(pred0_hal9001, pred1_hal9001)
)

# ----------------------
# Step 5. Plot
# ----------------------
library(ggplot2)
ggplot(plot_dt_hal9001, aes(x = X1, y = pred, color = factor(X2))) +
  geom_line(size = 1) +
  labs(x = "X1", y = "Predicted Y", color = "X2") +
  theme_minimal() + 
  stat_function(fun = function(x) exp(2*x), color = "green", size = 1) +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "orange", size = 1)

