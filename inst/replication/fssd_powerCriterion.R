## FSSD power-criterion example
##
## Reproduces the one-dimensional Gaussian-vs-Laplace experiment from the
## FSSD paper. The script scans test locations and plots the best location.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

library(ggplot2)

## ------------------------------------------------------------------
## Target and sampler helpers
## ------------------------------------------------------------------

score_gaussian <- function(X) {
  X_mat <- if (is.null(dim(X))) matrix(as.numeric(X), ncol = 1) else as.matrix(X)
  -X_mat
}

score_laplace <- function(X, b) {
  X_mat <- if (is.null(dim(X))) matrix(as.numeric(X), ncol = 1) else as.matrix(X)
  -sign(X_mat) / b
}

rlaplace <- function(n, mu = 0, b = 1) {
  u <- stats::runif(n) - 0.5
  mu - b * sign(u) * log1p(-2 * abs(u))
}

## ------------------------------------------------------------------
## Experiment setup
## ------------------------------------------------------------------

set.seed(0)
n_samples <- 2000
b_laplace <- 1 / sqrt(2)
v_grid <- seq(-5, 5, length.out = 401)

X_q <- matrix(rlaplace(n_samples, mu = 0, b = b_laplace), ncol = 1)
grads_p <- score_gaussian(X_q)

med_sq <- find_median_distance(X_q)
sigma_k <- sqrt(med_sq)
kernel_obj <- stein_kernel(type = "gaussian_rbf", h = sigma_k)

cat(sprintf(
  "Median-heuristic bandwidth: sigma_k = %.4f (sigma_k^2 = %.4f)\n",
  sigma_k, sigma_k^2
))

## ------------------------------------------------------------------
## Sweep test locations
## ------------------------------------------------------------------

n <- nrow(X_q)

fssd2 <- numeric(length(v_grid))
sigH1 <- numeric(length(v_grid))
ratio <- numeric(length(v_grid))

for (i in seq_along(v_grid)) {
  V <- matrix(v_grid[i], nrow = 1, ncol = 1)
  tau <- compute_tau(X = X_q, grads = grads_p, V = V, kernel_obj = kernel_obj)

  # Estimate the curve value at this test location.
  fssd2[i] <- sum(colMeans(tau)^2)

  # Estimate the scaling term used in the paper's criterion.
  mu_tau <- mean(tau[, 1])
  var_tau <- mean(tau[, 1]^2) - mu_tau^2
  s_sq <- 4 * mu_tau^2 * var_tau
  sigH1[i] <- sqrt(max(s_sq, 0))
}

denom <- sigH1
denom[denom < 1e-12] <- NA_real_
ratio <- fssd2 / denom

v_star_idx <- which.max(ratio)
v_star <- v_grid[v_star_idx]
cat(sprintf("Argmax of FSSD^2 / sigma_H1: v* = %.4f\n", v_star))

## ------------------------------------------------------------------
## Plot the densities and criterion
## ------------------------------------------------------------------

v_plot_range <- c(-4, 4)
keep_idx <- which(v_grid >= v_plot_range[1] & v_grid <= v_plot_range[2])
v_plot <- v_grid[keep_idx]
ratio_plot <- ratio[keep_idx]

ratio_plot[!is.finite(ratio_plot)] <- 0

p_dens <- stats::dnorm(v_plot, 0, 1)
q_dens <- (1 / (2 * b_laplace)) * exp(-abs(v_plot) / b_laplace)

# Put the criterion on the same panel without changing its axis label.
y_max_dens <- max(c(p_dens, q_dens), na.rm = TRUE)
y_max_ratio <- max(ratio_plot, na.rm = TRUE)
scale_fac <- if (y_max_ratio > 1e-12) y_max_dens / y_max_ratio else 1
ratio_scaled <- ratio_plot * scale_fac

df_curves <- data.frame(
  v = rep(v_plot, 3),
  value = c(p_dens, q_dens, ratio_scaled),
  curve = factor(rep(c("p", "q", "criterion"), each = length(v_plot)),
    levels = c("p", "q", "criterion")
  )
)

curve_labels <- c(
  p         = "p = N(0, 1)",
  q         = expression(q == Laplace(0, 1 / sqrt(2))),
  criterion = expression(FSSD^2 / sigma[H[1]])
)
curve_colors <- c(p = "#1f78b4", q = "#e31a1c", criterion = "black")
curve_ltys <- c(p = "solid", q = "dashed", criterion = "solid")
curve_widths <- c(p = 0.9, q = 0.9, criterion = 1.1)

p_fig1 <- ggplot(df_curves, aes(
  x = v, y = value, color = curve,
  linetype = curve, linewidth = curve
)) +
  geom_line(na.rm = TRUE) +
  geom_vline(
    xintercept = c(v_star, -v_star),
    color = "grey30", linetype = "dotted", linewidth = 0.5
  ) +
  annotate("text",
    x = v_star, y = y_max_dens * 1.05,
    label = "v*", color = "grey30", vjust = 0
  ) +
  annotate("text",
    x = -v_star, y = y_max_dens * 1.05,
    label = "v*", color = "grey30", vjust = 0
  ) +
  scale_color_manual(values = curve_colors, labels = curve_labels) +
  scale_linetype_manual(values = curve_ltys, labels = curve_labels) +
  scale_linewidth_manual(values = curve_widths, labels = curve_labels) +
  scale_y_continuous(
    name = "density",
    sec.axis = sec_axis(~ . / scale_fac,
      name = expression(FSSD^2 / sigma[H[1]])
    )
  ) +
  coord_cartesian(
    xlim = v_plot_range,
    ylim = c(0, y_max_dens * 1.15)
  ) +
  theme_bw(base_size = 13) +
  labs(
    x = "test location v", color = NULL, linetype = NULL, linewidth = NULL,
    title = "FSSD power criterion vs test location (paper Fig. 1)"
  ) +
  theme(
    legend.position = "top",
    legend.text.align = 0
  )

ggsave("fssd__powerCriterion.pdf",
  plot = p_fig1,
  width = 7, height = 4.5
)

cat("Saved fssd_powerCriterion.pdf\n")
