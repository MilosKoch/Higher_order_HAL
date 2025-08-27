library(hal9001)
library(data.table)

# ----------------------
# Step 1. Simulate data (non-linear)
# ----------------------
set.seed(123)
n <- 2000
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
# Step 5. Refit LM using selected covariates
# ----------------------
## Add penalizing (OLS)
higher_order_hal_MLE <- lm(Y~.-1,data = d)

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

# Step 3: Predict using the refitted LM
pred0 <- predict(higher_order_hal_MLE, newdata = nd0)
pred1 <- predict(higher_order_hal_MLE, newdata = nd1)

# Step 4: Combine for plotting
plot_dt <- data.table(
  X1 = rep(X1_vals, 2),
  X2 = rep(c(0, 1), each = n_grid),
  pred = c(pred0, pred1)
)

# Step 5: Plot
library(ggplot2)
ggplot(plot_dt, aes(x = X1, y = pred, color = factor(X2))) +
  geom_smooth() +
  # geom_line(size = 1) +
  labs(x = "X1", y = "Predicted Y", color = "X2") +
  theme_minimal() + 
  # stat_function(fun = function(x) exp(2*x), color = "red", size = 1) +
  stat_function(fun = function(x) 10*exp(-3*(sin(10*(x-0.5))^2)), color = "red", size = 1)

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
# 10*exp(-3*(sin(10*(X[,1]-0.5))^2))
# grid0 <- matrix(ncol = 2,c(seq(0,1,0.001),rep(0,length(seq(0,1,0.001)))))
# colnames(grid0) <- paste0("X", 1:p)
# nd0 <- evaluate_spline_at_point(points = grid0)
# pred0 <- predict(higher_order_hal_MLE, newdata = nd0)
# 
# grid1 <- matrix(ncol = 2,c(seq(0,1,0.001),rep(1,length(seq(0,1,0.001)))))
# colnames(grid1) <- paste0("X", 1:p)
# nd1 <- evaluate_spline_at_point(points = grid1)
# pred1 <- predict(higher_order_hal_MLE, newdata = nd1)
# 
# n_grid <- 500
# X1_vals <- seq(0, 1, length.out = n_grid)
# 
# grid0 <- data.table(X1 = X1_vals, X2 = 0)
# colnames(grid0) <- paste0("X", 1:p)
# nd0 <- evaluate_spline_at_point(points = grid0)
# 
# grid1 <- data.table(X1 = X1_vals, X2 = 1)
# colnames(grid1) <- paste0("X", 1:p)
# nd1 <- evaluate_spline_at_point(points = grid1)
# 
# pred0 <- predict(higher_order_hal_MLE, newdata = nd0)
# pred1 <- predict(higher_order_hal_MLE, newdata = nd1)
# 
# plot_dt <- data.table(
#   X1 = rep(X1_vals, 2),
#   X2 = rep(c(0,1), each = n_grid),
#   pred = c(pred0, pred1)
# )
# ggplot(plot_dt, aes(x = X1, y = pred, color = factor(X2))) +
#   geom_line(size = 1) +
#   labs(x = "X1", y = "HAL estimate", color = "X2") +
#   theme_minimal()
# 
