## GP model criticism with the KSD V test
##
## Fits a Gaussian-process regression model to the solar data, checks the
## predictive residuals with KSD V, and saves the fitted curve.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

library(ggplot2)

set.seed(20240629)

## ------------------------------------------------------------------
## Data and GP helpers
## ------------------------------------------------------------------

load_solar_data <- function() {
  data_path <- file.path(replication_dir, "data", "ksd_v_gp_model_criticism", "02-solar.csv")
  if (!file.exists(data_path)) {
    stop("Missing solar data file: ", data_path, call. = FALSE)
  }
  dat <- read.csv(data_path)
  if (!all(c("x", "y") %in% names(dat))) {
    stop("Solar data must contain x and y columns.", call. = FALSE)
  }
  dat[, c("x", "y")]
}

split_and_standardize <- function(dat, train_frac = 0.9) {
  n <- nrow(dat)
  train_n <- floor(train_frac * n)
  train_id <- sample(seq_len(n), train_n)
  test_id <- setdiff(seq_len(n), train_id)

  train <- dat[train_id, , drop = FALSE]
  test <- dat[test_id, , drop = FALSE]

  x_mean <- mean(train$x)
  x_sd <- sd(train$x)
  y_mean <- mean(train$y)
  y_sd <- sd(train$y)

  standardize <- function(df) {
    data.frame(
      x = (df$x - x_mean) / x_sd,
      y = (df$y - y_mean) / y_sd
    )
  }

  list(
    train = standardize(train),
    test = standardize(test),
    x_mean = x_mean,
    x_sd = x_sd,
    y_mean = y_mean,
    y_sd = y_sd
  )
}

rbf_cov <- function(x1, x2, signal_sd, lengthscale) {
  sq <- outer(as.numeric(x1), as.numeric(x2), "-")^2
  signal_sd^2 * exp(-0.5 * sq / lengthscale^2)
}

safe_chol <- function(mat, jitter = 1e-8, max_tries = 8) {
  for (i in seq_len(max_tries)) {
    attempt <- try(chol(mat + diag(jitter, nrow(mat))), silent = TRUE)
    if (!inherits(attempt, "try-error")) {
      return(attempt)
    }
    jitter <- jitter * 10
  }
  stop("Unable to Cholesky factorize GP covariance matrix.", call. = FALSE)
}

fit_gp_rbf <- function(x, y) {
  negative_log_marginal <- function(log_par) {
    lengthscale <- exp(log_par[1])
    signal_sd <- exp(log_par[2])
    noise_sd <- exp(log_par[3])
    k <- rbf_cov(x, x, signal_sd, lengthscale) + diag(noise_sd^2, length(x))
    r <- safe_chol(k)
    alpha <- backsolve(r, forwardsolve(t(r), y))
    0.5 * sum(y * alpha) + sum(log(diag(r))) + 0.5 * length(y) * log(2 * pi)
  }

  opt <- optim(
    par = log(c(lengthscale = 0.35, signal_sd = 1, noise_sd = 0.35)),
    fn = negative_log_marginal,
    method = "L-BFGS-B",
    lower = log(c(0.03, 0.05, 0.03)),
    upper = log(c(3, 5, 2))
  )

  lengthscale <- exp(opt$par[1])
  signal_sd <- exp(opt$par[2])
  noise_sd <- exp(opt$par[3])
  k <- rbf_cov(x, x, signal_sd, lengthscale) + diag(noise_sd^2, length(x))
  r <- safe_chol(k)
  alpha <- backsolve(r, forwardsolve(t(r), y))

  list(
    x_train = x,
    y_train = y,
    lengthscale = lengthscale,
    signal_sd = signal_sd,
    noise_sd = noise_sd,
    chol = r,
    alpha = alpha,
    convergence = opt$convergence,
    nll = opt$value
  )
}

predict_gp <- function(model, x_new, include_noise = TRUE) {
  k_star <- rbf_cov(model$x_train, x_new, model$signal_sd, model$lengthscale)
  mean <- as.numeric(t(k_star) %*% model$alpha)
  v <- forwardsolve(t(model$chol), k_star)
  latent_var <- pmax(model$signal_sd^2 - colSums(v^2), 1e-10)
  pred_var <- latent_var
  if (include_noise) pred_var <- pred_var + model$noise_sd^2
  data.frame(
    x = as.numeric(x_new),
    mean = mean,
    variance = pmax(pred_var, 1e-10),
    sd = sqrt(pmax(pred_var, 1e-10))
  )
}

make_gp_score <- function(pred_mean, pred_variance) {
  force(pred_mean)
  force(pred_variance)
  function(y_mat) {
    y <- as.numeric(y_mat[, 1])
    if (length(y) != length(pred_mean)) {
      stop("Score function expects the original GP test-set ordering.", call. = FALSE)
    }
    matrix(-(y - pred_mean) / pred_variance, ncol = 1)
  }
}

## ------------------------------------------------------------------
## Run the model check
## ------------------------------------------------------------------

solar <- load_solar_data()
prepared <- split_and_standardize(solar)
train <- prepared$train
test <- prepared$test[order(prepared$test$x), , drop = FALSE]

gp <- fit_gp_rbf(train$x, train$y)
grid_x <- seq(min(c(train$x, test$x)), max(c(train$x, test$x)), length.out = 300)
grid_pred <- predict_gp(gp, grid_x)
test_pred <- predict_gp(gp, test$x)

score_gp_predictive <- make_gp_score(test_pred$mean, test_pred$variance)
ksd_res <- ksd_v_test(
  X = matrix(test$y, ncol = 1),
  score_function = score_gp_predictive,
  boot_method = "rademacher",
  nboot = 10000,
  return_raw_boot = TRUE
)

## ------------------------------------------------------------------
## Plot and report
## ------------------------------------------------------------------

fit_plot <- ggplot() +
  geom_line(data = grid_pred, aes(x = x, y = mean), color = "blue", linewidth = 0.45) +
  geom_line(data = grid_pred, aes(x = x, y = mean + 2 * variance), color = "blue", linewidth = 0.45, linetype = "dashed") +
  geom_line(data = grid_pred, aes(x = x, y = mean - 2 * variance), color = "blue", linewidth = 0.45, linetype = "dashed") +
  geom_point(data = train, aes(x = x, y = y), color = "blue", size = 0.75) +
  geom_point(data = test, aes(x = x, y = y), color = "red", size = 1.4) +
  coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 4)) +
  labs(x = "X", y = "y") +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray82", linewidth = 0.35),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black")
  )

fit_path <- file.path(replication_dir, "ksd_v_gp_model_criticism.pdf")
ggsave(fit_path, plot = fit_plot, width = 6, height = 4.2, dpi = 150, device = grDevices::pdf)

cat(sprintf("GP train/test sizes: %d/%d\n", nrow(train), nrow(test)))
cat(sprintf(
  "GP hyperparameters: lengthscale = %.4f, signal_sd = %.4f, noise_sd = %.4f\n",
  gp$lengthscale, gp$signal_sd, gp$noise_sd
))
cat(sprintf(
  "KSD V statistic = %.6f, p-value = %.6f\n",
  as.numeric(ksd_res$statistic), as.numeric(ksd_res$p.value)
))
cat(sprintf("Saved GP model criticism fit plot: %s\n", normalizePath(fit_path, mustWork = FALSE)))
