# This script simulates 
# 
# 
# 
# 
# 
# 
# 
# ==============================================================================

# Load libraries
library(hal9001)
library(data.table)
library(riskRegression)
library(ggplot2)
library(survival)
library(glmnet)

#==========================
# Step 1: Simulate data
#==========================
set.seed(1234)
n <- 200
hal_grid <- TRUE
if(!hal_grid){
  grid_size <- n/2 # Temporary. Should be nrow("fit_hal$basis")(?) 
}else{
  grid_size <- n
}
follow_up <- 365.25/10


T_star <- rexp(n, rate = 0.5)
C <- rexp(n, rate = 0.2)

E <- pmin(T_star, C)

d <- data.table(time = E,
                 delta = as.numeric(T_star<=C))

setorder(d,time)

if(!hal_grid){
grid <- data.table(t = seq(0,follow_up,length.out = grid_size + 1))[-1]
}else{
  grid <- data.table(t = d[,time])
}
grid_size <- nrow(grid)
grid[grid_size, t_next := follow_up]
grid[1:(grid_size-1), t_next := grid[2:grid_size, t]]
Rmat_tmp <- outer(grid[,t],d[,time],function(g,t) pmin(t,g))
B <- data.table(outer(grid[,t],d[,time],function(g,t) as.integer(g >= t)))
D <- diff(c(0,as.matrix(B)%*%c(d[,delta])))

R <- diff(c(0,d[,time]))*n:1

dt <- data.table(time = d[,round(time,digits = 8)], D = D, R = R)
#==========================
# Step 2: Define basis functions
#==========================
dt[,phi_0 := paste0("as.numeric(time >= ",round(time,digits = 8),")")]

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

splines <- mu_k(dt, k=3)

#==========================
# Step 1: Simulate data
#==========================
evaluate_spline_at_point <- function(k = 3, points, splines_dt) {
  library(data.table)
  
  d <- data.table(time = points)
  
  for(i in 0:k){
    colname <- paste0("phi_", i)
    expr_list <- splines_dt[[colname]]  # expressions stored as strings
    
    # Make safe column names
    safe_names <- paste0("phi_", i, "_", seq_along(expr_list))
    
    for(j in seq_along(expr_list)){
      expr <- expr_list[j]
      name <- safe_names[j]
      
      # Use lapply to evaluate the expression for each time individually
      d[, (name) := eval(parse(text = expr))]
    }
  }
  
  d
}


# Use it:
basis_expansion <- evaluate_spline_at_point(points = splines[,time], splines_dt = splines, k = 3)
dt_splines <- cbind(dt[,.(R,D)], basis_expansion)

# Step 1: Build design matrix
X <- as.data.table(model.matrix(D ~ ., data = dt_splines))
X[, "(Intercept)" := NULL]   # remove intercept
R <- X[, R]                  # save R separately if needed
X[, R := NULL]               # drop R column
X[, time := NULL]            # drop time column

# Step 2: Keep only independent columns
# lic <- qr(as.matrix(X))$pivot[1:qr(as.matrix(X))$rank]
# X <- X[, ..lic]

# Step 3: Combine response + offset + predictors
dt_clean <- cbind(time = d[,time],D = dt_splines$D, R = R, X)

# # Step 4: Build formula dynamically
predictors <- setdiff(names(dt_clean), c("D","R","time"))


# Build model matrix
X <- model.matrix(~., dt_clean[, c(predictors), with = FALSE])[,2:801]

# Fit Poisson model
fit1 <- cv.glmnet(X, dt_clean$D, family = "poisson", offset = log(dt_clean$R), alpha = 1)

# pred1_link <- predict(fit1, newx = X, s = "lambda.min", type = "link", newoffset = log(dt_clean$R))
pred1 <- predict(fit1, newx = X, s = "lambda.min", type = "response", newoffset = log(dt_clean$R))

mu_hat <- pred1[,1]/n:1
dt_clean$mu_hat <- cumsum(mu_hat)

# Step 5b: Fit coxph
fit_coxph <- coxph(Surv(time, delta) ~ 1, data = d)

# --- Baseline cumulative hazard from Cox
bh_cox <- basehaz(fit_coxph, centered = FALSE)

# --- Baseline cumulative hazard from Poisson-GLMnet
# cumulative hazard estimate = sum of hazard increments
bh_pois <- data.table(
  time = dt_clean$time,
  cumhaz = dt_clean[,mu_hat]# hazard ~ exp(eta)/risk
)

# --- Combine for plotting
bh_cox_dt <- data.table(time = bh_cox$time, cumhaz = bh_cox$hazard, model = "CoxPH")
bh_pois_dt <- data.table(time = bh_pois$time, cumhaz = bh_pois$cumhaz, model = "Poisson")

plot_dt <- rbind(bh_cox_dt, bh_pois_dt)

# --- Plot
ggplot(plot_dt, aes(x = time, y = cumhaz, color = model)) +
  geom_line(size = 1) +
  labs(x = "Time", y = "Cumulative hazard",
       title = "Baseline cumulative hazard: CoxPH vs Poisson-GLMnet") +
  theme_minimal()

