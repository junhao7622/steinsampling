## SVGD one-dimensional Gaussian mixture example
##
## Moves particles from a poor initial distribution toward a two-component
## target, then compares particle estimates with Monte Carlo estimates.

library(ggplot2)
library(tidyr)
library(dplyr)

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

## ------------------------------------------------------------------
## Target helpers
## ------------------------------------------------------------------

grad_log_p <- function(x) {
  x_vec <- as.numeric(x)
  p1 <- (1 / 3) * dnorm(x_vec, -2, 1)
  p2 <- (2 / 3) * dnorm(x_vec, 2, 1)
  p_x <- p1 + p2
  dp_x <- p1 * (-(x_vec + 2)) + p2 * (-(x_vec - 2))
  matrix(dp_x / p_x, ncol = 1)
}

target_pdf <- function(x) (1 / 3) * dnorm(x, -2, 1) + (2 / 3) * dnorm(x, 2, 1)

sample_target <- function(n) {
  z <- rbinom(n, 1, 2 / 3)
  mu <- ifelse(z == 1, 2, -2)
  rnorm(n, mu, 1)
}

true_E_x <- 2 / 3
true_E_x2 <- 5

## ------------------------------------------------------------------
## Figure 1: particle movement over iterations
## ------------------------------------------------------------------
fig1_n_iter <- 500
fig1_stepsize <- 0.2
fig1_alpha <- 0.9

set.seed(42)
n_particles <- 100
x_current <- matrix(rnorm(n_particles, -10, 1), ncol = 1)

milestones <- c(0, 50, 75, 100, 150, 500)
history_states <- list()
history_states[["iter_0"]] <- x_current

my_svgd <- svgd(kernel = stein_kernel(type = "gaussian_rbf"))
historical_grad <- 0
fudge_factor <- 1e-6

for (iter in 1:fig1_n_iter) {
  lnpgrad <- grad_log_p(x_current)

  adjusted_h <- svgd_median_bandwidth(x_current)
  current_kernel <- stein_kernel(type = "gaussian_rbf", h = adjusted_h)

  kernel_out <- my_svgd$svgd_kernel(x_current, current_kernel)
  kxy <- kernel_out$Kxy
  dxkxy <- kernel_out$dxkxy

  grad_theta <- (kxy %*% lnpgrad + dxkxy) / n_particles
  if (iter == 1) {
    historical_grad <- historical_grad + grad_theta^2
  } else {
    historical_grad <- fig1_alpha * historical_grad +
      (1 - fig1_alpha) * (grad_theta^2)
  }
  adj_grad <- grad_theta / (fudge_factor + sqrt(historical_grad))
  x_current <- x_current + fig1_stepsize * adj_grad

  if (iter %in% milestones) {
    history_states[[paste0("iter_", iter)]] <- x_current
  }
}

## ------------------------------------------------------------------
## Figure 2: estimation error by particle count
## ------------------------------------------------------------------
n_list <- c(10, 20, 50, 100, 250)
n_trials <- 100
fig2_n_iter <- fig1_n_iter
fig2_stepsize <- fig1_stepsize

set.seed(123)
n_cos_funcs <- 20
omega_draws <- rnorm(n_cos_funcs, 0, 1)
b_draws <- runif(n_cos_funcs, 0, 2 * pi)

# Fixed cosine probes for the comparison panel.
true_cos <- (1 / 3) * cos(-2 * omega_draws + b_draws) * exp(-omega_draws^2 / 2) +
  (2 / 3) * cos(2 * omega_draws + b_draws) * exp(-omega_draws^2 / 2)

run_svgd <- function(n) {
  x0 <- matrix(rnorm(n, -10, 1), ncol = 1)
  as.numeric(my_svgd$update(
    x0, grad_log_p,
    n_iter = fig2_n_iter,
    stepsize = fig2_stepsize,
    alpha = fig1_alpha
  ))
}

avg_squared_cos_error <- function(samples) {
  est <- vapply(seq_len(n_cos_funcs), function(i) {
    mean(cos(omega_draws[i] * samples + b_draws[i]))
  }, numeric(1))
  mean((est - true_cos)^2)
}

mse_results_list <- list()

set.seed(2024)
for (n in n_list) {
  se_x_svgd <- numeric(n_trials)
  se_x2_svgd <- numeric(n_trials)
  se_cos_svgd <- numeric(n_trials)

  se_x_mc <- numeric(n_trials)
  se_x2_mc <- numeric(n_trials)
  se_cos_mc <- numeric(n_trials)

  for (trial in seq_len(n_trials)) {
    x_svgd <- run_svgd(n)
    se_x_svgd[trial] <- (mean(x_svgd) - true_E_x)^2
    se_x2_svgd[trial] <- (mean(x_svgd^2) - true_E_x2)^2
    se_cos_svgd[trial] <- avg_squared_cos_error(x_svgd)

    x_mc <- sample_target(n)
    se_x_mc[trial] <- (mean(x_mc) - true_E_x)^2
    se_x2_mc[trial] <- (mean(x_mc^2) - true_E_x2)^2
    se_cos_mc[trial] <- avg_squared_cos_error(x_mc)
  }

  mse_results_list[[length(mse_results_list) + 1]] <- data.frame(
    n = n, Method = "Stein Variational Gradient Descent",
    MSE_x = log10(mean(se_x_svgd)),
    MSE_x2 = log10(mean(se_x2_svgd)),
    MSE_cos = log10(mean(se_cos_svgd))
  )

  mse_results_list[[length(mse_results_list) + 1]] <- data.frame(
    n = n, Method = "Monte Carlo",
    MSE_x = log10(mean(se_x_mc)),
    MSE_x2 = log10(mean(se_x2_mc)),
    MSE_cos = log10(mean(se_cos_mc))
  )
}

mse_results <- bind_rows(mse_results_list)

## ------------------------------------------------------------------
## Plot Figure 1
## ------------------------------------------------------------------
df_fig1 <- bind_rows(lapply(names(history_states), function(iter_name) {
  data.frame(
    Iteration = factor(iter_name,
      levels = c("iter_0", "iter_50", "iter_75", "iter_100", "iter_150", "iter_500"),
      labels = c(
        "0th Iteration", "50th Iteration", "75th Iteration",
        "100th Iteration", "150th Iteration", "500th Iteration"
      )
    ),
    x = as.numeric(history_states[[iter_name]])
  )
}))

p1 <- ggplot() +
  stat_function(
    fun = target_pdf, geom = "line",
    color = "red", linetype = "dashed", linewidth = 0.8
  ) +
  geom_density(data = df_fig1, aes(x = x), color = "green4", linewidth = 1) +
  geom_rug(data = df_fig1, aes(x = x), alpha = 0.3) +
  xlim(-15, 6) +
  theme_bw() +
  labs(
    title = "Figure 1: Toy example with 1D Gaussian mixture",
    x = "x", y = "Density"
  ) +
  facet_wrap(~Iteration, ncol = 3)

ggsave("svgd_density_evolution.pdf", plot = p1, width = 12, height = 8, dpi = 150)

## ------------------------------------------------------------------
## Plot Figure 2
## ------------------------------------------------------------------
df_fig2 <- mse_results %>%
  pivot_longer(
    cols = starts_with("MSE"),
    names_to = "Estimator",
    values_to = "Log10_MSE"
  ) %>%
  mutate(
    Estimator = recode(Estimator,
      "MSE_x"   = "(a) Estimating E(x)",
      "MSE_x2"  = "(b) Estimating E(x^2)",
      "MSE_cos" = "(c) Estimating E(cos(omega*x + b))"
    ),
    Method = factor(Method,
      levels = c(
        "Monte Carlo",
        "Stein Variational Gradient Descent"
      )
    )
  )

p2 <- ggplot(df_fig2, aes(
  x = n, y = Log10_MSE,
  color = Method, shape = Method
)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3.5, stroke = 1.1, fill = NA) +
  scale_x_log10(breaks = n_list, labels = n_list) +
  facet_wrap(~Estimator, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(
    "Monte Carlo" = "red",
    "Stein Variational Gradient Descent" = "green4"
  )) +
  scale_shape_manual(values = c(
    "Monte Carlo" = 0,
    "Stein Variational Gradient Descent" = 1
  )) +
  theme_bw() +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    strip.text = element_text(size = 11, face = "bold")
  ) +
  labs(
    title = "Figure 2: Mean Square Error Comparison",
    x = "Sample Size (n)", y = "Log10 MSE"
  )

ggsave("svgd_mse_comparison.pdf", plot = p2, width = 12, height = 4.5, dpi = 300)
