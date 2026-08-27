#' ---
#' title: "Replication material for the steinsampling manuscript"
#' author: "Junhao Gao and Ery Arias-Castro"
#' ---
#'
#' 1. **Sections 3-4** -- every code example shown in the text.
#' 2. **Section 5.1** -- the Chicago crime goodness-of-fit study
#'    (Figure 1, Table 3).
#' 3. **Section 5.2** -- the IGARCH sampling and compression study
#'    (Figure 2, Table 4).
#'
###################################################
### code chunk number 1: preliminaries
###################################################

options(
  prompt = "R> ", continue = "+  ", width = 70,
  useFancyQuotes = FALSE
)
replication_dir <- normalizePath(".")

# Run this script from the replication-material directory.
# Needs reticulate (install.packages("reticulate")) to run
# replication_chicago.py.

# Install the article package if needed.
if (!requireNamespace("steinsampling", quietly = TRUE) ||
    packageVersion("steinsampling") < "0.1.1") {
  install.packages(
    file.path(replication_dir, "steinsampling_0.1.1.tar.gz"),
    type = "source", repos = NULL
  )
}

###################################################
### code chunk number 2: article.Rnw:467-473
###################################################
library("steinsampling")
target <- gmm(nComp = 1, mu = c(0, 0), sigma = diag(2), d = 2)
score_p <- get_score_evaluator(target)
set.seed(2026)
X <- rgmm(target, n = 400)

###################################################
### code chunk number 3: article.Rnw:507-509
###################################################
gof_kernel <- stein_kernel("gaussian_rbf", h = 1)

###################################################
### code chunk number 4: article.Rnw:619-623
###################################################
set.seed(1101)
fit_u <- ksd_u_test(X, score_p, kernel = gof_kernel, nboot = 9999)
fit_u

###################################################
### code chunk number 5: article.Rnw:708-713
###################################################
set.seed(1102)
fit_v <- ksd_v_test(X, score_p, boot_method = "rademacher",
  kernel = gof_kernel, nboot = 9999)
fit_v

###################################################
### code chunk number 6: article.Rnw:743-756
###################################################
set.seed(1201)
phi <- 0.8
Z <- matrix(0, 400, 2)
Z[1, ] <- rnorm(2)
for (t in 2:400)
  Z[t, ] <- phi * Z[t - 1, ] + sqrt(1 - phi^2) * rnorm(2)
set.seed(1202)
ksd_v_test(Z, score_p, boot_method = "markov", change_prob = 0.02,
  kernel = gof_kernel, nboot = 9999)
set.seed(1203)
ksd_v_test(Z, score_p, boot_method = "rademacher",
  kernel = gof_kernel, nboot = 9999)

###################################################
### code chunk number 7: article.Rnw:907-911
###################################################
fit_r <- fssd_test(X, score_p, variant = "rand", J = 2,
  kernel = gof_kernel, n_simulations = 9999, seed = 1103)
fit_r

###################################################
### code chunk number 8: article.Rnw:969-974
###################################################
fit_o <- fssd_opt_test(X, score_p, J = 2,
  kernel = gof_kernel, train_ratio = 0.2, maxit = 50,
  n_simulations = 9999, seed = 1108)
fit_o

###################################################
### code chunk number 9: article.Rnw:1070-1078
###################################################
mu <- matrix(c(-2, 0, 2, 0), nrow = 2)
sigma <- array(diag(2), c(2, 2, 2))
target <- gmm(nComp = 2, mu = mu, sigma = sigma,
  weights = c(0.5, 0.5), d = 2)
score_mix <- get_score_evaluator(target)
log_p_mix <- function(X) log(densitygmm(target, X))
kern <- stein_kernel("imq", c = 1, beta = -0.5)

###################################################
### code chunk number 10: article.Rnw:1203-1211
###################################################
set.seed(2101)
x0 <- matrix(rnorm(100), ncol = 2)
particles <- svgd(x0, score_mix,
  n_iter = 500, step_size = 0.1)$X
round(c(left_share = mean(particles[, 1] < 0),
  left_mean = mean(particles[particles[, 1] < 0, 1]),
  right_mean = mean(particles[particles[, 1] >= 0, 1])), 3)

###################################################
### code chunk number 11: article.Rnw:1308-1314
###################################################
set.seed(2102)
opt <- fmin_nm(lb = c(-6, -4), ub = c(6, 4))
sp <- stein_points(score_mix, kern, n_points = 50, d = 2,
  optimizer = opt, log_p = log_p_mix)
round(sp$ksd[c(1, 10, 50)], 3)

###################################################
### code chunk number 12: article.Rnw:1413-1419
###################################################
set.seed(2103)
spm <- sp_mcmc(score_mix, log_p_mix, kern, n_points = 50, d = 2,
  mcmc = "mala", criterion = "last", m_seq = 10, h = 0.5,
  x_init = c(-2, 0))
round(spm$ksd[c(1, 10, 50)], 3)

###################################################
### code chunk number 13: article.Rnw:1426-1430
###################################################
c(sp_evals = sp$cum_n_eval[50],
  spm_evals = spm$cum_n_eval[50],
  accept = round(mean(spm$accept_rate, na.rm = TRUE), 3))

###################################################
### code chunk number 14: article.Rnw:1510-1515
###################################################
set.seed(2104)
chain <- rwm(log_p_mix, x0 = c(-8, -6), h = 0.3, Sigma = diag(2),
  m_iter = 2000)
idx <- stein_thinning(chain$X, score_function = score_mix, m = 50)

###################################################
### code chunk number 15: article.Rnw:1516-1524
###################################################
root_ksd <- function(X, S, kernel) {
  sqrt(ksd_v_statistic(stein_kernel_matrix(kernel, X, S)) / nrow(X))
}
round(c(ksd_chain = root_ksd(chain$X, score_mix(chain$X), kern),
  ksd_thin = root_ksd(chain$X[idx, ], score_mix(chain$X[idx, ]), kern),
  left_chain = mean(chain$X[, 1] < 0),
  left_thin = mean(chain$X[idx, 1] < 0)), 3)

###################################################
### code chunk number 16: chicago_data_and_fitted_mixtures
###################################################

chicago_csv <- file.path(replication_dir, "chicago_robbery_2016.csv")
python_dir <- tempfile("steinsampling-chicago-")
dir.create(python_dir)
reticulate::py_require(c(
  "numpy==2.4.5", "scikit-learn==1.8.0", "pandas==3.0.5"
))
status <- system2(
  reticulate::py_config()$python,
  c(
    shQuote(file.path(replication_dir, "replication_chicago.py")),
    "--data", shQuote(chicago_csv),
    "--output-dir", shQuote(python_dir)
  )
)
if (status != 0) stop("replication_chicago.py failed.", call. = FALSE)

read_locations <- function(filename) {
  frame <- read.csv(file.path(python_dir, filename))
  as.matrix(frame[c("longitude", "latitude")])
}
X_holdout <- read_locations("held_rows.csv")
X_test <- read_locations("fssd_test_rows.csv")
X_plot <- read_locations("surface_rows.csv")

training_frame <- read.csv(file.path(python_dir, "fssd_training_rows.csv"))
X_tune <- as.matrix(training_frame[, c("longitude", "latitude")])
initial_location <- as.logical(training_frame$initial_location)
L_start <- matrix(
  as.numeric(training_frame[initial_location, c(
    "longitude", "latitude"
  )]),
  nrow = 1
)
gmm_parameters <- read.csv(file.path(
  python_dir, "spherical_gmm_parameters.csv"
))
unlink(python_dir, recursive = TRUE)

# Reconstruct a fitted spherical GMM and optionally reweight it.
make_spherical_gmm <- function(n_components, source_notebook = FALSE) {
  parameters <- gmm_parameters[
    gmm_parameters$n_components == n_components, , drop = FALSE
  ]
  parameters <- parameters[order(parameters$component), , drop = FALSE]
  mu <- rbind(
    longitude = parameters$longitude_mean,
    latitude = parameters$latitude_mean
  )
  sigma <- array(0, c(2, 2, n_components))
  sigma[1, 1, ] <- sigma[2, 2, ] <- parameters$variance
  weights <- parameters$fit_weight
  if (source_notebook) {
    weights <- weights * sqrt(parameters$variance)
    weights <- weights / sum(weights)
  }
  gmm(nComp = n_components, mu = mu, sigma = sigma,
    weights = weights, d = 2
  )
}

###################################################
### code chunk number 17: chicago_targets
###################################################

# Use normalized mixtures for the formal tests.
model2 <- make_spherical_gmm(2)
model10 <- make_spherical_gmm(10)
score2 <- get_score_evaluator(model2)
score10 <- get_score_evaluator(model10)

# Use reweighted mixtures for the source-study figure.
figure_model2 <- make_spherical_gmm(2, source_notebook = TRUE)
figure_model10 <- make_spherical_gmm(10, source_notebook = TRUE)
figure_score2 <- get_score_evaluator(figure_model2)
figure_score10 <- get_score_evaluator(figure_model10)

###################################################
### code chunk number 18: chicago_fssd_optimization
###################################################

# Compute the source-study FSSD power criterion.
source_power_objective <- function(X, score_fn, L, h2, reg) {
  kernel_obj <- stein_kernel("gaussian_rbf", h = sqrt(h2))
  tau <- compute_tau(X, score_fn(X), L, kernel_obj)
  mu <- colMeans(tau)
  sigma_h1_sq <- 4 * mean(as.numeric(tau %*% mu)^2) - 4 * sum(mu^2)^2
  (fssd_statistic(tau) / nrow(tau)) /
    sqrt(max(sigma_h1_sq + reg, .Machine$double.eps))
}

# Estimate h^2 from a seeded subsample of at most 1000 rows.
median_h2_subsample <- function(X, size = 1000, seed = 9827) {
  set.seed(seed)
  rows <- sample.int(nrow(X), min(size, nrow(X)))
  find_median_distance(X[rows, , drop = FALSE])
}

initial_h2 <- median_h2_subsample(X_holdout)

# Optimize the FSSD location and h^2 on the tuning sample.
fit_source_fssd <- function(score_fn) {
  location_sd <- sqrt(colMeans(
    sweep(X_tune, 2, colMeans(X_tune), "-")^2
  ))
  lower_location <- apply(X_tune, 2, min) - 100 * location_sd
  upper_location <- apply(X_tune, 2, max) + 100 * location_sd

  objective <- function(par) {
    L <- matrix(par[-1], nrow = 1)
    -source_power_objective(
      X_tune, score_fn, L, h2 = par[1]^2, reg = 1e-1
    )
  }
  h2_start <- min(max(initial_h2, 7e-4), 1e-3)
  opt <- optim(
    c(sqrt(h2_start), as.numeric(L_start)), objective,
    method = "L-BFGS-B",
    lower = c(sqrt(7e-4), lower_location),
    upper = c(sqrt(1e-3), upper_location),
    control = list(
      maxit = 50, factr = 1e-7 / .Machine$double.eps, pgtol = 1e-5
    )
  )

  L <- matrix(opt$par[-1], nrow = 1)
  h2 <- opt$par[1]^2
  list(L = L, h2 = h2, objective = -opt$value)
}

fssd2_fit <- fit_source_fssd(score2)
figure2_fit <- fit_source_fssd(figure_score2)
figure10_fit <- fit_source_fssd(figure_score10)

fssd2_kernel <- stein_kernel("gaussian_rbf", h = sqrt(fssd2_fit$h2))
fssd2_tau <- compute_tau(X_test, score2(X_test), fssd2_fit$L, fssd2_kernel)
fssd2_fit$statistic <- fssd_statistic(fssd2_tau)
set.seed(10)
fssd2_null <- fssd_null_pvalue(
  fssd2_tau, fssd2_fit$statistic, n_simulations = 1000
)
fssd2_fit$p.value <- fssd2_null$p_value

###################################################
### code chunk number 19: article.Rnw:1586-1594
###################################################
fssd10_fit <- fit_source_fssd(score10)
tau10 <- compute_tau(X_test, score10(X_test), fssd10_fit$L,
  stein_kernel("gaussian_rbf", h = sqrt(fssd10_fit$h2)))
statistic10 <- fssd_statistic(tau10)
set.seed(10)
fssd10_null <- fssd_null_pvalue(
  tau10, statistic10, n_simulations = 1000)

###################################################
### code chunk number 20: chicago_fssd_results
###################################################
fssd10_fit$statistic <- statistic10
fssd10_fit$p.value <- fssd10_null$p_value

###################################################
### code chunk number 21: article.Rnw:1627-1637
###################################################
ksd_h2 <- median_h2_subsample(X_tune, size = 1000,
  seed = 9827)
ksd_kernel <- stein_kernel("gaussian_rbf", h = sqrt(ksd_h2))
set.seed(2201)
ksd_u <- ksd_u_test(X_test, score10, kernel = ksd_kernel,
  nboot = 9999)
set.seed(2202)
ksd_v <- ksd_v_test(X_test, score10, kernel = ksd_kernel,
  boot_method = "rademacher", nboot = 9999)

###################################################
### code chunk number 22: chicago_gof_table
###################################################
set.seed(2101)
ksd_u2 <- ksd_u_test(X_test, score2, kernel = ksd_kernel,
                     nboot = 9999)
set.seed(2102)
ksd_v2 <- ksd_v_test(X_test, score2, kernel = ksd_kernel,
                     boot_method = "rademacher", nboot = 9999)

table3 <- data.frame(
  test = c("FSSD-opt", "KSD-U", "KSD-V"),
  K2_statistic = c(
    fssd2_fit$statistic, unname(ksd_u2$statistic),
    unname(ksd_v2$statistic)
  ),
  K2_p_value = c(fssd2_fit$p.value, ksd_u2$p.value, ksd_v2$p.value),
  K10_statistic = c(
    fssd10_fit$statistic, unname(ksd_u$statistic),
    unname(ksd_v$statistic)
  ),
  K10_p_value = c(fssd10_fit$p.value, ksd_u$p.value, ksd_v$p.value),
  check.names = FALSE
)
write.csv(
  table3, file.path(replication_dir, "chicago-gof-table.csv"),
  row.names = FALSE
)
print(table3)

###################################################
### code chunk number 23: chicago_fssd_surface
###################################################

padded_grid <- function(X, length_out = 50, pad_fraction = 0.1) {
  ranges <- apply(X, 2, range)
  pad <- pad_fraction * (ranges[2, ] - ranges[1, ])
  axes <- lapply(seq_len(ncol(X)), function(j) {
    seq(
      ranges[1, j] - pad[j], ranges[2, j] + pad[j],
      length.out = length_out
    )
  })
  list(axes = axes, points = as.matrix(expand.grid(axes)))
}

surface_grid_info <- padded_grid(X_plot)
gx_fssd <- surface_grid_info$axes[[1]]
gy_fssd <- surface_grid_info$axes[[2]]
surface_grid <- surface_grid_info$points

# Compute the fixed-bandwidth FSSD criterion on the plotting grid.
fssd_surface <- function(score_fn, h2) {
  values <- vapply(seq_len(nrow(surface_grid)), function(i) {
    source_power_objective(
      X_plot, score_fn, matrix(surface_grid[i, ], nrow = 1),
      h2 = h2, reg = 1e-2
    )
  }, numeric(1))
  matrix(
    values,
    nrow = length(gx_fssd), ncol = length(gy_fssd)
  )
}

surface2 <- fssd_surface(figure_score2, figure2_fit$h2)
surface10 <- fssd_surface(figure_score10, figure10_fit$h2)

density_grid <- function(model) {
  matrix(
    densitygmm(model, as.matrix(expand.grid(gx_fssd, gy_fssd))),
    nrow = length(gx_fssd), ncol = length(gy_fssd)
  )
}

surface_limits <- c(-0.08, 0.20)
surface_levels <- seq(surface_limits[1], surface_limits[2], length.out = 31)
surface_colours <- grey(seq(0.97, 0.15, length.out = 30))

fssd_panel <- function(surface, model, label) {
  clipped <- pmin(pmax(surface, surface_limits[1]), surface_limits[2])
  plot.new()
  plot.window(
    xlim = range(gx_fssd), ylim = range(gy_fssd), asp = 1,
    xaxs = "i", yaxs = "i"
  )
  .filled.contour(
    gx_fssd, gy_fssd, clipped,
    levels = surface_levels, col = surface_colours
  )
  points(
    X_plot[, 1], X_plot[, 2], pch = 16, cex = 0.22,
    col = rgb(0.86, 0, 0.86, 0.72)
  )
  contour(
    gx_fssd, gy_fssd, density_grid(model)^(1 / 2.7),
    add = TRUE, drawlabels = FALSE, nlevels = 7,
    col = hcl.colors(7, "Viridis"), lwd = 1.05
  )
  maximum <- surface_grid[which.max(surface), ]
  points(maximum[1], maximum[2], pch = 8, cex = 3, col = "black", lwd = 2.5)
  points(maximum[1], maximum[2], pch = 8, cex = 2.75, col = "red", lwd = 2.2)
  mtext(label, side = 1, line = 0.35, cex = 1)
}

pdf(
  file.path(replication_dir, "chicago-gof.pdf"),
  width = 7.2, height = 3, pointsize = 12
)
layout(matrix(c(1, 2, 3), nrow = 1), widths = c(1, 1, 0.2))
par(mar = c(2, 0.2, 0.2, 0.2))
fssd_panel(surface2, figure_model2, "(a) two-component GMM")
fssd_panel(surface10, figure_model10, "(b) ten-component GMM")
par(mar = c(1.3, 0.2, 0.5, 2.8))
plot.new()
plot.window(xlim = c(0, 1), ylim = surface_limits, xaxs = "i", yaxs = "i")
rect(
  0, surface_levels[-length(surface_levels)], 1, surface_levels[-1],
  col = surface_colours, border = NA
)
axis(4, at = seq(-0.08, 0.20, by = 0.04), las = 1, tck = -0.25)
dev.off()

###################################################
### code chunk number 24: igarch_setup
###################################################

N_REF_KEEP <- 20000L
N_THIN <- 5L
M_SPMCMC <- 5L

LB <- c(0.002, 0.05)
UB <- c(0.04, 0.2)
X0 <- (LB + UB) / 2
V_MCMC <- diag(5e-4, 2)

ADAPT_EPOCH <- c(rep(500L, 15L), 1000L, 5000L, 100000L)

###################################################
### code chunk number 25: igarch_likelihood_and_score
###################################################

# Compute the IGARCH log posterior for each parameter row.
igarch_logp <- function(theta, y, h1) {
  theta <- as.matrix(theta)
  n <- nrow(theta)
  theta1 <- theta[, 1L]
  theta2 <- theta[, 2L]
  valid <- is.finite(theta1) & is.finite(theta2) &
    theta1 > 0 & theta2 > 0 & theta2 < 1

  y_sq <- y^2
  h <- rep(h1, n)
  logp <- numeric(n)

  for (i in 2:length(y)) {
    h <- theta1 + theta2 * y_sq[i - 1L] + (1 - theta2) * h
    valid <- valid & is.finite(h) & h > 0
    idx <- which(valid)
    if (length(idx) > 0L) {
      logp[idx] <- logp[idx] - 0.5 * log(2 * pi) - 0.5 * log(h[idx]) -
        y_sq[i] / (2 * h[idx])
    }
  }

  logp[!valid] <- -Inf
  logp
}

# Compute the IGARCH score from the variance recursion.
igarch_score <- function(theta, y, h1) {
  theta <- as.matrix(theta)
  n <- nrow(theta)
  theta1 <- theta[, 1L]
  theta2 <- theta[, 2L]
  valid <- is.finite(theta1) & is.finite(theta2) &
    theta1 > 0 & theta2 > 0 & theta2 < 1

  dh1 <- numeric(n)
  dh2 <- numeric(n)
  h <- rep(h1, n)
  score1 <- numeric(n)
  score2 <- numeric(n)
  y_sq <- y^2
  beta <- 1 - theta2

  for (i in 2:length(y)) {
    dh1 <- 1 + beta * dh1
    dh2 <- y_sq[i - 1L] - h + beta * dh2
    h <- theta1 + theta2 * y_sq[i - 1L] + beta * h
    valid <- valid & is.finite(h) & h > 0

    idx <- which(valid)
    if (length(idx) > 0L) {
      score_h <- -1 / (2 * h[idx]) + y_sq[i] / (2 * h[idx]^2)
      score1[idx] <- score1[idx] + score_h * dh1[idx]
      score2[idx] <- score2[idx] + score_h * dh2[idx]
    }
  }

  out <- cbind(score1, score2)
  out[!valid, ] <- 0
  out
}

###################################################
### code chunk number 26: adaptive_mala
###################################################

logistic <- function(x, k) 1 / (1 + exp(-k * x))

nearest_spd <- function(S, eps = 1e-10) {
  S <- (S + t(S)) / 2
  eig <- eigen(S, symmetric = TRUE)
  vals <- pmax(eig$values, eps)
  out <- eig$vectors %*% (vals * t(eig$vectors))
  (out + t(out)) / 2
}

rmvn <- function(n, mu, Sigma) {
  mu <- as.numeric(mu)
  U <- chol(nearest_spd(Sigma))
  sweep(matrix(stats::rnorm(n * length(mu)), nrow = n) %*% U, 2L, mu, "+")
}

# Generate the reference chain with adaptive MALA.
mala_adapt <- function(log_p, score_function, x0, h0, C0, alpha, epoch) {
  h <- h0

  fit <- mala(log_p, score_function, x0, h, nearest_spd(C0), epoch[1L])

  for (i in 2:length(epoch)) {
    C <- if (alpha[i] == 1) {
      C0
    } else {
      alpha[i] * C0 + (1 - alpha[i]) * stats::cov(fit$X)
    }
    h <- h * exp(2 * (mean(fit$accept) - 0.57))
    fit <- mala(
      log_p, score_function, fit$X[nrow(fit$X), ],
      h, nearest_spd(C), epoch[i]
    )
  }

  fit$X
}

###################################################
### code chunk number 27: med_and_energy_distance
###################################################

# Construct the minimum energy design path.
med_greedy_path <- function(log_p, optimizer, n_points, d) {
  X <- matrix(NA_real_, n_points, d)
  logp <- numeric(n_points)
  n_eval <- integer(n_points)

  first <- optimizer(
    function(X_new) list(objective_values = -log_p(X_new), scores = NULL),
    matrix(0, 0, d)
  )
  X[1L, ] <- first$x_min
  logp[1L] <- log_p(matrix(first$x_min, 1L, d))
  n_eval[1L] <- first$n_eval

  for (j in 2:n_points) {
    obj_med <- function(X_new) {
      logp_new <- log_p(X_new)
      vals <- vapply(seq_len(nrow(X_new)), function(ii) {
        dst <- sqrt(rowSums(sweep(
          X[seq_len(j - 1L), , drop = FALSE],
          2L, X_new[ii, ], "-"
        )^2))
        min(logp_new[ii] + logp[seq_len(j - 1L)] +
          2 * d * log(pmax(dst, 1e-300)))
      }, numeric(1))
      list(objective_values = -vals, scores = NULL)
    }
    res <- optimizer(obj_med, X[seq_len(j - 1L), , drop = FALSE])
    X[j, ] <- res$x_min
    logp[j] <- log_p(matrix(res$x_min, 1L, d))
    n_eval[j] <- res$n_eval + 1L
  }

  list(X = X, n_eval = n_eval, cum_n_eval = cumsum(n_eval))
}

sum_distance_to_reference <- function(x, reference, block = 5000L) {
  total <- 0
  for (lo in seq(1L, nrow(reference), by = block)) {
    hi <- min(nrow(reference), lo + block - 1L)
    total <- total + sum(sqrt(rowSums(sweep(
      reference[lo:hi, , drop = FALSE],
      2L, x, "-"
    )^2)))
  }
  total
}

sum_pairwise_distances <- function(X, block = 1200L) {
  total <- 0
  for (lo in seq(1L, nrow(X), by = block)) {
    hi <- min(nrow(X), lo + block - 1L)
    A <- X[lo:hi, , drop = FALSE]
    for (jlo in seq(1L, nrow(X), by = block)) {
      jhi <- min(nrow(X), jlo + block - 1L)
      B <- X[jlo:jhi, , drop = FALSE]
      d2 <- outer(rowSums(A^2), rowSums(B^2), "+") - 2 * tcrossprod(A, B)
      total <- total + sum(sqrt(pmax(d2, 0)))
    }
  }
  total
}

# Compute energy distance for each prefix.
energy_path <- function(X, reference, reference_self_distance) {
  X <- as.matrix(X)
  cross_distance <- vapply(seq_len(nrow(X)), function(i) {
    sum_distance_to_reference(X[i, ], reference)
  }, numeric(1))
  within_distance <- numeric(nrow(X))
  for (n in seq_len(nrow(X))) {
    if (n > 1L) {
      d <- sqrt(rowSums(sweep(X[seq_len(n - 1L), , drop = FALSE],
        2L, X[n, ], "-")^2))
      within_distance[n] <- within_distance[n - 1L] + 2 * sum(d)
    }
  }
  n <- seq_len(nrow(X))
  pmax(2 * cumsum(cross_distance) / (nrow(reference) * n) -
      reference_self_distance - within_distance / n^2,
    .Machine$double.eps)
}

# Compute one method's energy-distance trajectory.
method_energy <- function(result, reference_chain, reference_self_distance) {
  if (!is.null(result$trace)) {
    vapply(result$trace, function(X) {
      tail(energy_path(X, reference_chain, reference_self_distance), 1L)
    }, numeric(1))
  } else {
    energy_path(result$X, reference_chain, reference_self_distance)
  }
}

spx <- read.csv(file.path(replication_dir, "spx_returns.csv"))
rows <- spx$date >= "2005-12-06" & spx$date <= "2013-11-14"
returns <- spx$return_percent[rows]
initial_variance <- stats::var(returns)

###################################################
### code chunk number 28: reference_chain
###################################################

log_p <- function(X) igarch_logp(X, returns, initial_variance)
score_fn <- function(X) igarch_score(X, returns, initial_variance)

set.seed(2026)
alpha_adapt <- 1 - logistic(
  seq(-0.1, 1.9, length.out = length(ADAPT_EPOCH)), 20
)
reference_states <- mala_adapt(
  log_p, score_fn, X0, 0.004^2, diag(2),
  alpha_adapt, ADAPT_EPOCH
)
reference_chain <- reference_states[
  round(seq(1L, nrow(reference_states), length.out = N_REF_KEEP)), ,
  drop = FALSE
]

reference_covariance <- stats::cov(reference_chain)

###################################################
### code chunk number 29: sampling_methods
###################################################

results <- list()

set.seed(2101)
n_states <- 1000 * N_THIN
mala_chain <- mala(log_p, score_fn, X0, 0.13^2, V_MCMC, n_states)
thin_idx <- seq.int(N_THIN, n_states, by = N_THIN)
results$MALA <- list(X = mala_chain$X[thin_idx, , drop = FALSE],
  n_eval = rep(2L * N_THIN, 1000))

set.seed(2102)
rwm_chain <- rwm(log_p, X0, h = 1, Sigma = nearest_spd(0.06 * V_MCMC),
  m_iter = n_states)
results$RWM <- list(X = rwm_chain$X[thin_idx, , drop = FALSE],
  n_eval = rep(N_THIN, 1000))

set.seed(2103)
particles0 <- cbind(
  stats::runif(1000, LB[1], UB[1]),
  stats::runif(1000, LB[2], UB[2])
)

mc_alpha <- 1 - logistic(seq(-0.15, 1.85, length.out = 1000), 50)
mc_alpha[1L] <- 1
# Generate Monte Carlo candidates for Stein Points and MED.
source_optimizer <- function(f, X_curr, t = nrow(X_curr) + 1L) {
  X_mc <- if (nrow(X_curr) == 0L || runif(1L) <= mc_alpha[t]) {
    rmvn(M_SPMCMC, X0, 0.5 * V_MCMC)
  } else {
    rows <- sample.int(nrow(X_curr), M_SPMCMC, replace = TRUE)
    X_curr[rows, , drop = FALSE] + rmvn(M_SPMCMC, c(0, 0), 5e-6 * diag(2))
  }
  obj <- f(X_mc)
  valid <- X_mc[, 1L] > 0 & X_mc[, 2L] > 0 & X_mc[, 2L] < 1
  obj$objective_values[!valid] <- Inf
  i <- which.min(obj$objective_values)
  list(x_min = X_mc[i, ],
    d_min = if (is.null(obj$scores)) NA_real_ else obj$scores[i, ],
    f_min = obj$objective_values[i], n_eval = M_SPMCMC)
}

sp_alpha <- 1 - logistic(seq(-0.1, 1.9, length.out = 1000), 100)
sp_alpha[1:10] <- 1

###################################################
### code chunk number 30: article.Rnw:1780-1782
###################################################
kernel <- stein_kernel("imq", c = 1, beta = -0.5,
  precon = solve(reference_covariance))

###################################################
### code chunk number 31: sp_mcmc_helpers
###################################################
# Run one source-study SP-MCMC configuration.
run_source_sp_mcmc <- function(mcmc, criterion, h, initial_covariance, seed,
                               covariance_scale = 1) {
  mcmc <- match.arg(mcmc, c("mala", "rwm"))
  initial_covariance <- nearest_spd(initial_covariance)

  # Adapt the proposal covariance along the source schedule.
  proposal_fn <- function(j, X_curr, ...) {
    Sigma <- if (nrow(X_curr) < 2L || sp_alpha[j] == 1) {
      initial_covariance
    } else {
      empirical_covariance <- covariance_scale * stats::cov(X_curr)
      nearest_spd(
        sp_alpha[j] * initial_covariance +
          (1 - sp_alpha[j]) * empirical_covariance
      )
    }
    list(Sigma = Sigma)
  }

  set.seed(seed)
  initial_chain <- switch(mcmc,
    mala = mala(log_p, score_fn, X0, h = h,
      Sigma = initial_covariance, m_iter = M_SPMCMC),
    rwm = rwm(log_p, X0, h = h,
      Sigma = initial_covariance, m_iter = M_SPMCMC)
  )
  keep <- !duplicated(as.data.frame(initial_chain$X))
  candidates <- initial_chain$X[keep, , drop = FALSE]
  candidate_scores <- if (is.null(initial_chain$D)) {
    score_fn(candidates)
  } else {
    initial_chain$D[keep, , drop = FALSE]
  }
  x_init <- candidates[
    which.min(diag(stein_kernel_matrix(kernel, candidates, candidate_scores))),
  ]

  fit <- sp_mcmc(
    score_fn, log_p, kernel, n_points = 1000, d = 2,
    mcmc = mcmc, criterion = criterion, m_seq = M_SPMCMC,
    h = h, Sigma = initial_covariance, x_init = x_init,
    proposal_fn = proposal_fn
  )

  candidate_score_evals <- if (is.null(initial_chain$D)) {
    nrow(candidates)
  } else {
    0L
  }
  initial_counts <- c(
    log_p = initial_chain$counts$log_p,
    score = initial_chain$counts$score,
    candidate_score = candidate_score_evals,
    transition_total = initial_chain$counts$total,
    total = initial_chain$counts$total + candidate_score_evals
  )
  # Select the first point from the initial candidate chain.
  fit$counts[1L, ] <- initial_counts
  fit$n_eval[1L] <- initial_counts[["total"]]
  fit$cum_n_eval <- cumsum(fit$n_eval)
  fit
}

set.seed(2105)
sp_first <- source_optimizer(function(X) {
  D <- score_fn(X)
  list(
    objective_values = diag(stein_kernel_matrix(kernel, X, D)),
    scores = D
  )
}, matrix(0, 0, 2), t = 1L)
sp_x1 <- sp_first$x_min

###################################################
### code chunk number 32: article.Rnw:1783-1790
###################################################
svgd_fit <- svgd(particles0, score_fn, kernel = kernel,
  n_iter = 199, step_size = 0.001, trace_iters = seq_len(199))
sp <- stein_points(score_fn, kernel, n_points = 1000,
  d = 2, optimizer = source_optimizer, x_init = sp_x1)
sp_mala_last <- run_source_sp_mcmc(
  "mala", "last", h = 0.5^2,
  initial_covariance = 0.02 * V_MCMC / 0.5^2, seed = 2106)

###################################################
### code chunk number 33: remaining_sampling_methods
###################################################
sp_mala_infl <- run_source_sp_mcmc(
  "mala", "infl", h = 0.8^2,
  initial_covariance = 0.02 * V_MCMC / 0.8^2, seed = 2107
)
sp_rwm_last <- run_source_sp_mcmc(
  "rwm", "last", h = 1,
  initial_covariance = 0.2 * V_MCMC, seed = 2108,
  covariance_scale = 2.38^2 / 2
)
sp_rwm_infl <- run_source_sp_mcmc(
  "rwm", "infl", h = 1,
  initial_covariance = 0.2 * V_MCMC, seed = 2109,
  covariance_scale = 2.38^2 / 2
)

sp$n_eval[1L] <- sp_first$n_eval
sp$cum_n_eval <- cumsum(sp$n_eval)

results$SVGD <- list(X = svgd_fit$X,
  trace = c(list(particles0), unname(svgd_fit$trace)),
  n_eval = c(0L, svgd_fit$n_eval),
  cum_n_eval = c(0L, svgd_fit$cum_n_eval))

set.seed(2014)
results$MED <- med_greedy_path(log_p, source_optimizer, 1000, 2)

results$SP <- sp
results$`SP-MALA LAST` <- sp_mala_last
results$`SP-MALA INFL` <- sp_mala_infl
results$`SP-RWM LAST` <- sp_rwm_last
results$`SP-RWM INFL` <- sp_rwm_infl

results <- results[c(
  "MALA", "RWM", "SVGD", "MED", "SP",
  "SP-MALA LAST", "SP-MALA INFL", "SP-RWM LAST", "SP-RWM INFL"
)]

###################################################
### code chunk number 34: energy_distance_figure
###################################################

reference_self_distance <-
  sum_pairwise_distances(reference_chain) / nrow(reference_chain)^2

energy_curves <- lapply(results, function(result) {
  if (is.null(result$cum_n_eval)) result$cum_n_eval <- cumsum(result$n_eval)
  data.frame(
    log_neval = log(pmax(result$cum_n_eval, 1)),
    log_ep = log(method_energy(
      result, reference_chain, reference_self_distance
    ))
  )
})

labels <- names(energy_curves)
cols <- c(
  "#0072BD", "#7E2F8E", "#D95319", "#EDB120", "#77AC30",
  "#0072BD", "#4DBEEE", "#7E2F8E", "#EE82EE"
)
ltys <- c(3, 3, rep(1, 7))
lwds <- c(1.7, 1.7, rep(1.0, 7))

all_x <- unlist(lapply(energy_curves, `[[`, "log_neval"), use.names = FALSE)
all_y <- unlist(lapply(energy_curves, `[[`, "log_ep"), use.names = FALSE)
pdf(file.path(replication_dir, "igarch-energy.pdf"),
  width = 7.2, height = 3.4, pointsize = 12, bg = "white")
par(mar = c(3.2, 3.4, 0.5, 0.5), mgp = c(1.8, 0.55, 0), tcl = -0.25)
plot(NULL,
  xlim = range(all_x[is.finite(all_x)]),
  ylim = range(all_y[is.finite(all_y)]),
  xlab = expression(log(n[eval])),
  ylab = expression(log(widehat(E)))
)
for (i in seq_along(labels)) {
  lines(energy_curves[[i]]$log_neval, energy_curves[[i]]$log_ep,
    col = cols[i], lty = ltys[i], lwd = lwds[i]
  )
}
legend("topright",
  legend = labels, col = cols, lty = ltys, lwd = lwds,
  bty = "n", cex = 0.85
)
box()
dev.off()

###################################################
### code chunk number 35: compression_setup
###################################################

# Compute the final IGARCH variance for each parameter row.
igarch_final_variance <- function(theta, y, h1) {
  theta <- as.matrix(theta)
  h <- rep(h1, nrow(theta))
  y_sq <- y^2
  for (i in 2:length(y)) {
    h <- theta[, 1L] + theta[, 2L] * y_sq[i - 1L] + (1 - theta[, 2L]) * h
  }
  h
}

# Support points by the sp.ccp algorithm of Mak and Joseph (2018).
sp_ccp <- function(Y, n, iter_max = 250L, iter_min = 50L, tol = 1e-10) {
  N <- nrow(Y)
  p <- ncol(Y)
  n0 <- n * p
  center <- colMeans(Y)
  sdev <- sqrt(apply(Y, 2, stats::var))
  Y <- sweep(sweep(Y, 2, center, "-"), 2, sdev, "/")
  bound <- apply(Y, 2, range)
  clamp <- function(X) pmin(pmax(X, rep(bound[1L, ], each = nrow(X))),
    rep(bound[2L, ], each = nrow(X)))
  if (any(duplicated(Y))) Y <- clamp(jitter(Y))
  X <- clamp(matrix(jitter(Y[sample.int(N, n), ]), ncol = p))
  run_weight <- numeric(n)
  for (iter in seq_len(iter_max)) {
    X_prev <- X
    gamma <- n0 / (iter - 1L + n0)
    W <- 1 / as.matrix(stats::dist(X_prev))
    diag(W) <- 0
    sq_dist <- 0
    for (j in seq_len(p)) sq_dist <- sq_dist + outer(X_prev[, j], Y[, j], "-")^2
    inv_dist <- 1 / sqrt(sq_dist)
    weight <- rowSums(inv_dist)
    repulsion <- (rowSums(W) * X_prev - W %*% X_prev) * (N / n)
    X <- clamp(((1 - gamma) * run_weight * X_prev +
      gamma * (repulsion + inv_dist %*% Y)) /
      ((1 - gamma) * run_weight + gamma * weight))
    run_weight <- (1 - gamma) * run_weight + gamma * weight
    if (max(rowSums((X - X_prev)^2)) < tol && iter >= iter_min) break
  }
  sweep(sweep(X, 2, sdev, "*"), 2, center, "+")
}

###################################################
### code chunk number 36: article.Rnw:1840-1849
###################################################
reference_scores <- score_fn(reference_chain)
idx_stein <- stein_thinning(reference_chain, S = reference_scores,
  m = 100, pre = "smpcov")
idx_regular <- round(seq(1, nrow(reference_chain), length.out = 100))
set.seed(4104)
idx_random <- sort(sample.int(nrow(reference_chain), 100))
set.seed(4105)
support_points <- sp_ccp(reference_chain, n = 100)

###################################################
### code chunk number 37: compression_summaries
###################################################
ref_mean <- colMeans(reference_chain)
ref_vol_q90 <- stats::quantile(
  sqrt(igarch_final_variance(reference_chain, returns, initial_variance)), 0.9
)

colnames(support_points) <- colnames(reference_chain)
compressions <- list(
  "Stein thinning" = list(X = reference_chain[idx_stein, ],
    D = reference_scores[idx_stein, ]),
  "regular thinning" = list(X = reference_chain[idx_regular, ],
    D = reference_scores[idx_regular, ]),
  "random subsample" = list(X = reference_chain[idx_random, ],
    D = reference_scores[idx_random, ]),
  "support points" = list(X = support_points,
    D = score_fn(support_points))
)

###################################################
### code chunk number 38: igarch_table4
###################################################

med_rows <- reference_chain[seq_len(1000L), , drop = FALSE]
med_sq <- stats::median(stats::dist(med_rows))^2
kernel_med <- stein_kernel(
  type = "imq", c = 1, beta = -0.5, precon = diag(1 / med_sq, 2)
)

thinning_summary <- do.call(rbind, lapply(names(compressions), function(nm) {
  X <- compressions[[nm]]$X
  D <- compressions[[nm]]$D
  vol <- sqrt(igarch_final_variance(X, returns, initial_variance))
  data.frame(
    method = nm,
    ksd_med = root_ksd(X, D, kernel_med),
    ksd_smpcov = root_ksd(X, D, kernel),
    energy = tail(energy_path(
      X, reference_chain, reference_self_distance
    ), 1L),
    mean_err = max(abs(colMeans(X) - ref_mean) / abs(ref_mean)),
    cov_err = norm(stats::cov(X) - reference_covariance, "F") /
      norm(reference_covariance, "F"),
    vol_q90_err = abs(stats::quantile(vol, 0.9) - ref_vol_q90) / ref_vol_q90,
    row.names = NULL
  )
}))
utils::write.csv(
  thinning_summary,
  file.path(replication_dir, "igarch-thinning-table.csv"),
  row.names = FALSE
)
print(thinning_summary, digits = 4, row.names = FALSE)

###################################################
### code chunk number 39: session_information
###################################################

sessionInfo()
