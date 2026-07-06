## SP-MCMC Gaussian mixture example
##
## Runs MCMC and three SP-MCMC variants on a two-component Gaussian mixture,
## then saves the four-panel comparison figure.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))


## ------------------------------------------------------------------
## Target distribution
## ------------------------------------------------------------------
n_dim <- 2L
n_pts <- 1000L
scale <- 0.5
Mu <- rbind(rep(-1, n_dim), rep(1, n_dim))
weights <- c(0.5, 0.5)
Sigma_comp <- array(0, dim = c(n_dim, n_dim, 2L))
Sigma_comp[, , 1L] <- scale * diag(n_dim)
Sigma_comp[, , 2L] <- scale * diag(n_dim)

logsumexp <- function(A) {
  m <- apply(A, 1L, max)
  m + log(rowSums(exp(A - m)))
}

component_log_density <- function(X) {
  X <- as.matrix(X)
  out <- matrix(NA_real_, nrow(X), 2L)
  for (k in 1:2) {
    Y <- sweep(X, 2L, Mu[k, ], "-")
    quad <- rowSums(Y^2) / scale
    out[, k] <- log(weights[k]) - n_dim * log(2 * pi) / 2 -
      determinant(Sigma_comp[, , k], logarithm = TRUE)$modulus / 2 -
      quad / 2
  }
  out
}

log_p <- function(X) logsumexp(component_log_density(X))

score_fn <- function(X) {
  X <- as.matrix(X)
  log_comp <- component_log_density(X)
  lse <- logsumexp(log_comp)
  post <- exp(log_comp - lse)
  score <- matrix(0, nrow(X), n_dim)
  for (k in 1:2) {
    score <- score + post[, k] * (-(sweep(X, 2L, Mu[k, ], "-")) / scale)
  }
  score
}

rgmm_ref <- function(n) {
  comp <- sample.int(2L, size = n, replace = TRUE, prob = weights)
  X <- matrix(0, n, n_dim)
  for (k in 1:2) {
    idx <- which(comp == k)
    if (length(idx) > 0L) {
      X[idx, ] <- sweep(matrix(stats::rnorm(length(idx) * n_dim), length(idx), n_dim) %*%
        chol(Sigma_comp[, , k]), 2L, Mu[k, ], "+")
    }
  }
  X
}


## ------------------------------------------------------------------
## Kernel setup
## ------------------------------------------------------------------
X_kernel <- rgmm_ref(n_pts)
ll <- median(stats::dist(X_kernel))^2 / log(n_pts)
LInv <- diag(1 / ll, n_dim)
kernel <- stein_kernel(type = "imq", c = 1, beta = -0.5, precon = LInv)


## ------------------------------------------------------------------
## Generate point sets
## ------------------------------------------------------------------
x0 <- rep(0, n_dim)
n_iter <- 5L
S0 <- diag(0.1, n_dim)

cat("Running MCMC baseline (MALA h = 1.1, C = I)...\n")
X_mcmc <- mala(log_p, score_fn,
  x0 = x0, h = 1.1,
  Sigma = diag(n_dim), m_iter = n_pts
)$X

cat("Running SP-MCMC LAST (GRW, nIter = 5, S0 = 0.1 I)...\n")
res_last <- sp_mcmc(score_fn, log_p, kernel,
  n_points = n_pts, d = n_dim,
  mcmc = "grw", criterion = "last",
  m_seq = n_iter, Sigma = S0, x_init = x0
)

cat("Running SP-MCMC RAND (GRW, nIter = 5, S0 = 0.1 I)...\n")
res_rand <- sp_mcmc(score_fn, log_p, kernel,
  n_points = n_pts, d = n_dim,
  mcmc = "grw", criterion = "rand",
  m_seq = n_iter, Sigma = S0, x_init = x0
)

cat("Running SP-MCMC INFL (GRW, nIter = 5, S0 = 0.1 I)...\n")
res_infl <- sp_mcmc(score_fn, log_p, kernel,
  n_points = n_pts, d = n_dim,
  mcmc = "grw", criterion = "infl",
  m_seq = n_iter, Sigma = S0, x_init = x0
)


## ------------------------------------------------------------------
## Plot helpers
## ------------------------------------------------------------------
ksd_trajectory <- function(X) {
  X <- as.matrix(X)
  D <- score_fn(X)
  K <- stein_kernel_matrix(kernel, X, D)
  inc <- vapply(seq_len(nrow(X)), function(j) {
    2 * sum(K[seq_len(j), j]) - K[j, j]
  }, numeric(1))
  sqrt(pmax(cumsum(inc), 0)) / seq_len(nrow(X))
}

logistic <- function(x, k) 1 / (1 + exp(-k * x))

point_cols <- function(X) {
  r <- logistic(rowSums(X), 5)
  grDevices::rgb(r, 0, 1 - r)
}

trace_cols <- function(X) {
  r <- logistic(X[, 1L], 10)
  grDevices::rgb(r, 0, 1 - r)
}

jump2 <- function(X) rowSums(diff(as.matrix(X))^2)

safe_density <- function(x, from = NULL, to = NULL) {
  x <- x[is.finite(x)]
  if (length(unique(x)) < 2L) x <- x + stats::rnorm(length(x), sd = 1e-8)
  args <- list(x = x)
  if (!is.null(from)) args$from <- from
  if (!is.null(to)) args$to <- to
  do.call(stats::density, args)
}

grid_1 <- seq(-3.5, 3.5, length.out = 100L)
grid_2 <- seq(-3.5, 3.5, length.out = 100L)
T_grid <- cbind(
  rep(grid_1, each = length(grid_2)),
  rep(grid_2, times = length(grid_1))
)
Z <- matrix(exp(log_p(T_grid)), nrow = length(grid_1), ncol = length(grid_2))

X_list <- list(X_mcmc, res_last$X, res_rand$X, res_infl$X)
labels <- c("MCMC", "LAST", "RAND", "INFL")
ksd_list <- lapply(X_list, ksd_trajectory)
jump_list <- lapply(X_list, jump2)
chain_jump_last <- list(
  NULL, res_last$chain_d2_last, res_rand$chain_d2_last,
  res_infl$chain_d2_last
)

sp_col <- grDevices::rgb(44, 160, 44, maxColorValue = 255)
mc_col <- grDevices::rgb(255, 127, 14, maxColorValue = 255)
contour_col <- grDevices::rgb(0.7, 0.7, 0.7)


## ------------------------------------------------------------------
## Plot the comparison figure
## ------------------------------------------------------------------
pdf("sp_mcmc_figure2.pdf", width = 25 / 2.54, height = 25 / 2.54, bg = "white")
op <- par(
  mfrow = c(4, 4),
  mar = c(3.2, 3.4, 2.2, 0.8),
  oma = c(0.4, 0.3, 0.2, 0.2),
  mgp = c(2.0, 0.6, 0),
  cex.axis = 0.85,
  cex.lab = 0.95
)

for (i in seq_along(X_list)) {
  X <- X_list[[i]]
  contour(grid_1, grid_2, Z,
    levels = seq(0.01, max(Z), by = 0.01),
    drawlabels = FALSE, col = contour_col, lwd = 1,
    xlim = c(-3.5, 3.5), ylim = c(-3.5, 3.5),
    xlab = "", ylab = "", axes = FALSE, asp = 1
  )
  title(labels[i])
  points(X[, 1L], X[, 2L], pch = 16, cex = 0.35, col = point_cols(X))
  box()
}

for (i in seq_along(X_list)) {
  y <- log(ksd_list[[i]])
  plot(seq_len(n_pts), y,
    type = "l", col = sp_col, lwd = 1,
    xlim = c(1, n_pts), ylim = c(-4.2, 1),
    xlab = if (i == 1L) "j" else "", ylab = if (i == 1L) "log KSD" else ""
  )
  box()
}

for (i in seq_along(X_list)) {
  X <- X_list[[i]]
  plot(seq_len(n_pts), X[, 1L],
    pch = 16, cex = 0.25, col = trace_cols(X),
    xlim = c(1, n_pts), ylim = c(-3.5, 3.5),
    xlab = if (i == 1L) "j" else "", ylab = ""
  )
  box()
}

for (i in seq_along(X_list)) {
  if (i == 1L) {
    dens <- safe_density(jump_list[[i]])
    plot(dens,
      col = mc_col, lwd = 1,
      xlim = c(-0.2, 6), ylim = c(0, 1.5),
      xlab = "Jump^2", ylab = "Density", main = ""
    )
  } else {
    dens_sp <- safe_density(jump_list[[i]])
    dens_mc <- safe_density(chain_jump_last[[i]])
    plot(dens_sp,
      col = sp_col, lwd = 1,
      xlim = c(-0.2, 6), ylim = c(0, 1.5),
      xlab = "", ylab = "", main = ""
    )
    lines(dens_mc, col = mc_col, lwd = 1)
    if (i == 2L) {
      legend("topright",
        legend = c("SP-MCMC", "MCMC"),
        col = c(sp_col, mc_col), lwd = 1, bty = "n", cex = 0.7
      )
    }
  }
  box()
}

par(op)
dev.off()

cat("Saved: stein_point_mcmc_Gaussian_mixture.pdf\n")
