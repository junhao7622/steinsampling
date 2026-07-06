## KSD-U RBM goodness-of-fit experiments
##
## Compares KSD, MMD, and likelihood-ratio checks on Gaussian-Bernoulli RBM
## samples, then saves the Figure 2-style error and discrepancy panels.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

set.seed(123)

alpha <- 0.05
# Full replication: n_default <- 100L
n_default <- 30L
# Full replication: N_trials <- 1000L
N_trials <- 3L
# Full replication: nboot <- 1000L
nboot <- 30L
# Full replication: mmd_boot <- 1000L
mmd_boot <- 30L

## Full replication: dx <- 50L
dx <- 10L
## Full replication: dh <- 10L
dh <- 4L
rbm_scale <- 0.5
## Full replication: burnin <- 1000L
burnin <- 50L
## Full replication: ais_chains <- 100L
ais_chains <- 20L
## Full replication: ais_temps <- 1000L
ais_temps <- 50L

## Full replication: sigma_grid <- c(0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1)
sigma_grid <- c(0.01, 0.1, 0.5)
## Full replication: mmd_n_grid <- c(10L, 20L, 50L, 100L, 200L, 500L, 1000L)
mmd_n_grid <- c(10L, 30L, 100L)
mmd_sigma <- 0.1

output_dir <- "."
output_pdf <- "ksd_u_RBM_figure2.pdf"

hidden_states <- as.matrix(expand.grid(rep(list(c(-1, 1)), dh)))

METHOD_NAMES <- c(
  "ksd_u", "ksd_linear", "mmd_mc_100", "mmd_mc_1000",
  "mmd_mcmc_1000", "lr_exact"
)
METHOD_LABELS <- c(
  ksd_u          = "KSD-U",
  ksd_linear     = "KSD-Linear",
  mmd_mc_100     = "MMD-MC(100)",
  mmd_mc_1000    = "MMD-MC(1000)",
  mmd_mcmc_1000  = "MMD-MCMC(1000)",
  lr_exact       = "LR(Simple vs. Simple)"
)
METHOD_COL <- c(
  ksd_u          = "red",
  ksd_linear     = "magenta",
  mmd_mc_100     = "forestgreen",
  mmd_mc_1000    = "deepskyblue3",
  mmd_mcmc_1000  = "blue",
  lr_exact       = "black"
)
METHOD_PCH <- c(
  ksd_u          = 1,
  ksd_linear     = 2,
  mmd_mc_100     = 4,
  mmd_mc_1000    = 3,
  mmd_mcmc_1000  = 0,
  lr_exact       = 5
)

DISC_NAMES <- c(
  "ksd_u", "ksd_linear", "mmd_mc_1000",
  "mmd_mcmc_1000", "lr_exact", "lr_ais"
)

require_or_stop <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required but not installed.", pkg))
  }
}

require_or_stop("foreach")
require_or_stop("doParallel")
require_or_stop("doRNG")
require(foreach)
require(doParallel)
require(doRNG)

## Full replication: n_cores <- max(1L, parallel::detectCores() - 1L)
n_cores <- 1L
cl <- NULL
if (n_cores > 1L) {
  cl <- tryCatch(
    parallel::makeCluster(n_cores),
    error = function(e) {
      warning(
        sprintf(
          "Could not start parallel cluster (%s); using sequential backend.",
          conditionMessage(e)
        ),
        call. = FALSE
      )
      NULL
    }
  )
}
if (is.null(cl)) {
  foreach::registerDoSEQ()
  n_cores <- 1L
} else {
  doParallel::registerDoParallel(cl)
}
doRNG::registerDoRNG(123)

## ------------------------------------------------------------------
## Numeric helpers
## ------------------------------------------------------------------

logsumexp <- function(x) {
  mx <- max(x)
  if (!is.finite(mx)) {
    return(mx)
  }
  mx + log(sum(exp(x - mx)))
}

logmeanexp <- function(x) {
  logsumexp(x) - log(length(x))
}

log_2cosh <- function(x) {
  ax <- abs(x)
  ax + log1p(exp(-2 * ax))
}

sigmoid <- function(x) {
  ifelse(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))
}

signed_sqrt <- function(x) {
  sign(x) * sqrt(abs(x))
}

## ------------------------------------------------------------------
## Gaussian-Bernoulli RBM helpers
## ------------------------------------------------------------------

rbm <- function(B, b, c, scale = rbm_scale) {
  B <- as.matrix(B)
  b <- as.numeric(b)
  c <- as.numeric(c)

  means <- sweep(scale * hidden_states %*% t(B), 2, b, "+")
  log_w <- as.numeric(hidden_states %*% c) + 0.5 * rowSums(means^2)
  log_z_hidden <- logsumexp(log_w)

  list(
    B = B,
    b = b,
    c = c,
    dx = nrow(B),
    dh = ncol(B),
    scale = scale,
    means = means,
    hidden_prob = exp(log_w - log_z_hidden),
    logZ = 0.5 * nrow(B) * log(2 * pi) + log_z_hidden
  )
}

init_model_p <- function() {
  B <- matrix(sample(c(-1, 1), dx * dh, replace = TRUE), nrow = dx, ncol = dh)
  b <- rnorm(dx)
  c <- rnorm(dh)
  rbm(B, b, c)
}

perturb_rbm <- function(model, sigma_per) {
  B_q <- model$B + matrix(rnorm(length(model$B), sd = sigma_per),
    nrow = model$dx, ncol = model$dh
  )
  rbm(B_q, model$b, model$c, scale = model$scale)
}

rbm_sample_exact <- function(model, n) {
  idx <- sample.int(nrow(hidden_states), n, replace = TRUE, prob = model$hidden_prob)
  model$means[idx, , drop = FALSE] +
    matrix(rnorm(n * model$dx), nrow = n, ncol = model$dx)
}

rbm_gibbs_step <- function(X, H, model, beta = 1) {
  logits_h <- beta * sweep(model$scale * (X %*% model$B), 2, model$c, "+")
  probs_h <- sigmoid(2 * logits_h)
  H <- matrix(as.numeric(runif(length(probs_h)) <= as.vector(probs_h)),
    nrow = nrow(X), ncol = model$dh
  )
  H <- 2 * H - 1

  mean_x <- beta * sweep(model$scale * (H %*% t(model$B)), 2, model$b, "+")
  X <- mean_x + matrix(rnorm(length(mean_x)), nrow = nrow(mean_x), ncol = ncol(mean_x))
  list(X = X, H = H)
}

rbm_sample_mcmc <- function(model, n, burnin = 1000L) {
  # Match the paper's MMD-MCMC sample: one Gibbs chain after burn-in.
  X <- matrix(rnorm(model$dx), nrow = 1L, ncol = model$dx)
  H <- matrix(sample(c(-1, 1), model$dh, replace = TRUE), nrow = 1L, ncol = model$dh)

  for (i in seq_len(burnin)) {
    step <- rbm_gibbs_step(X, H, model)
    X <- step$X
    H <- step$H
  }

  out <- matrix(NA_real_, nrow = n, ncol = model$dx)
  for (i in seq_len(n)) {
    step <- rbm_gibbs_step(X, H, model)
    X <- step$X
    H <- step$H
    out[i, ] <- X
  }
  out
}

rbm_score <- function(X, model) {
  X <- as.matrix(X)
  logits_h <- sweep(model$scale * (X %*% model$B), 2, model$c, "+")
  sweep(model$scale * (tanh(logits_h) %*% t(model$B)), 2, model$b, "+") - X
}

rbm_log_unnormalized <- function(X, model) {
  X <- as.matrix(X)
  logits_h <- sweep(model$scale * (X %*% model$B), 2, model$c, "+")
  as.numeric(X %*% model$b) - 0.5 * rowSums(X^2) + rowSums(log_2cosh(logits_h))
}

rbm_log_density <- function(X, model) {
  rbm_log_unnormalized(X, model) - model$logZ
}

make_score_function <- function(model) {
  force(model)
  function(Z) rbm_score(Z, model)
}

## ------------------------------------------------------------------
## Exact LR and AIS
## ------------------------------------------------------------------

rbm_joint_linear_part <- function(X, H, model) {
  cross <- model$scale * rowSums((X %*% model$B) * H)
  as.numeric(X %*% model$b) + as.numeric(H %*% model$c) + cross
}

rbm_ais_logZ <- function(model, n_chains = ais_chains, n_temps = ais_temps) {
  betas <- seq(0, 1, length.out = n_temps + 1L)
  X <- matrix(rnorm(n_chains * model$dx), nrow = n_chains, ncol = model$dx)
  H <- matrix(sample(c(-1, 1), n_chains * model$dh, replace = TRUE),
    nrow = n_chains, ncol = model$dh
  )
  log_w <- numeric(n_chains)

  for (k in 2:length(betas)) {
    log_w <- log_w + (betas[k] - betas[k - 1L]) * rbm_joint_linear_part(X, H, model)
    step <- rbm_gibbs_step(X, H, model, beta = betas[k])
    X <- step$X
    H <- step$H
  }

  0.5 * model$dx * log(2 * pi) + model$dh * log(2) + logmeanexp(log_w)
}

lr_exact_score <- function(X, model_p, model_q) {
  mean(rbm_log_density(X, model_p) - rbm_log_density(X, model_q))
}

lr_ais_score <- function(X, model_p, model_q) {
  logZ_p <- rbm_ais_logZ(model_p)
  logZ_q <- rbm_ais_logZ(model_q)
  mean((rbm_log_unnormalized(X, model_p) - logZ_p) -
    (rbm_log_unnormalized(X, model_q) - logZ_q))
}

## ------------------------------------------------------------------
## KSD and MMD tests
## ------------------------------------------------------------------

ksd_u_gof <- function(X, model_q) {
  ksd_u_test(
    X = X,
    score_function = make_score_function(model_q),
    nboot = nboot,
    kernel = "gaussian_rbf"
  )
}

ksd_u_score <- function(X, model_q) {
  U <- ksd_uq_matrix(
    X = X,
    score_function = make_score_function(model_q),
    kernel = "gaussian_rbf"
  )
  ksd_u_statistic(U)
}

ksd_linear_values <- function(X, model_q) {
  prep <- prepare_ksd_u_inputs(
    X = X,
    score_function = make_score_function(model_q),
    kernel = "gaussian_rbf"
  )

  m <- nrow(prep$X) %/% 2L
  first <- seq_len(m)
  second <- m + seq_len(m)
  K <- stein_kernel_matrix(
    prep$kernel_obj,
    prep$X[first, , drop = FALSE],  prep$grads[first, , drop = FALSE],
    prep$X[second, , drop = FALSE], prep$grads[second, , drop = FALSE]
  )
  diag(K)
}

ksd_linear_gof <- function(X, model_q) {
  h <- ksd_linear_values(X, model_q)
  v <- mean(h^2)
  if (!is.finite(v) || v <= 0) {
    return(list(p.value = 1, statistic = 0))
  }
  stat <- sqrt(length(h)) * mean(h) / sqrt(v)
  list(p.value = pnorm(stat, lower.tail = FALSE), statistic = stat)
}

mmd_test <- function(X, Y, nboot = mmd_boot) {
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  m <- nrow(X)
  n <- nrow(Y)
  Z <- rbind(X, Y)
  scale <- find_median_distance(Z, max_samples = 2000, use_sampling = TRUE)
  K <- eval_kernel(stein_kernel(type = "gaussian_rbf", h = sqrt(scale)), Z)

  calc_mmd <- function(idx_x, idx_y) {
    sum(K[idx_x, idx_x]) / (m^2) +
      sum(K[idx_y, idx_y]) / (n^2) -
      2 * sum(K[idx_x, idx_y]) / (m * n)
  }

  idx_x <- seq_len(m)
  idx_y <- m + seq_len(n)
  stat <- calc_mmd(idx_x, idx_y)

  boot_stats <- vapply(seq_len(nboot), function(b) {
    perm <- sample.int(m + n)
    calc_mmd(perm[seq_len(m)], perm[m + seq_len(n)])
  }, numeric(1))

  list(statistic = stat, p.value = bootstrap_pvalue_right_tail(boot_stats, stat))
}

mmd_mc_gof <- function(X, model_q, n_prime) {
  Y <- rbm_sample_exact(model_q, n_prime)
  mmd_test(X, Y)
}

mmd_mcmc_gof <- function(X, model_q, n_prime) {
  Y <- rbm_sample_mcmc(model_q, n_prime, burnin = burnin)
  mmd_test(X, Y)
}

mmd_mc_score <- function(X, model_q, n_prime) {
  Y <- rbm_sample_exact(model_q, n_prime)
  mmd_test(X, Y, nboot = 1L)$statistic
}

mmd_mcmc_score <- function(X, model_q, n_prime) {
  Y <- rbm_sample_mcmc(model_q, n_prime, burnin = burnin)
  mmd_test(X, Y, nboot = 1L)$statistic
}

## ------------------------------------------------------------------
## Trial runners
## ------------------------------------------------------------------

run_gof_trial <- function(sigma_per) {
  model_p <- init_model_p()
  X <- rbm_sample_exact(model_p, n_default)
  is_null <- rbinom(1, 1, 0.5) == 1
  model_q <- if (is_null) model_p else perturb_rbm(model_p, sigma_per)

  out <- list()

  res <- ksd_u_gof(X, model_q)
  out[["ksd_u"]] <- isTRUE(res$p.value < alpha)

  res <- ksd_linear_gof(X, model_q)
  out[["ksd_linear"]] <- isTRUE(res$p.value < alpha)

  res <- mmd_mc_gof(X, model_q, n_prime = 100L)
  out[["mmd_mc_100"]] <- isTRUE(res$p.value < alpha)

  res <- mmd_mc_gof(X, model_q, n_prime = 1000L)
  out[["mmd_mc_1000"]] <- isTRUE(res$p.value < alpha)

  res <- mmd_mcmc_gof(X, model_q, n_prime = 1000L)
  out[["mmd_mcmc_1000"]] <- isTRUE(res$p.value < alpha)

  out[["lr_exact"]] <- isTRUE(sum(rbm_log_density(X, model_p) -
    rbm_log_density(X, model_q)) > 0)

  data.frame(
    method = names(out),
    rejected = unlist(out, use.names = FALSE),
    is_null = is_null,
    stringsAsFactors = FALSE
  )
}

run_mmd_size_trial <- function(n_prime) {
  model_p <- init_model_p()
  X <- rbm_sample_exact(model_p, n_default)
  is_null <- rbinom(1, 1, 0.5) == 1
  model_q <- if (is_null) model_p else perturb_rbm(model_p, mmd_sigma)

  ksd_res <- ksd_u_gof(X, model_q)
  mmd_res <- mmd_mc_gof(X, model_q, n_prime = n_prime)

  data.frame(
    method = c("ksd_u", "mmd_mc"),
    rejected = c(ksd_res$p.value < alpha, mmd_res$p.value < alpha),
    is_null = is_null,
    stringsAsFactors = FALSE
  )
}

run_discrepancy_trial <- function(sigma_per) {
  model_p <- init_model_p()
  model_q <- perturb_rbm(model_p, sigma_per)
  X <- rbm_sample_exact(model_p, n_default)

  score_for <- function(target, condition) {
    data.frame(
      metric = DISC_NAMES,
      condition = condition,
      value = c(
        signed_sqrt(ksd_u_score(X, target)),
        signed_sqrt(mean(ksd_linear_values(X, target))),
        sqrt(max(mmd_mc_score(X, target, n_prime = 1000L), 0)),
        sqrt(max(mmd_mcmc_score(X, target, n_prime = 1000L), 0)),
        signed_sqrt(lr_exact_score(X, model_p, target)),
        signed_sqrt(lr_ais_score(X, model_p, target))
      ),
      stringsAsFactors = FALSE
    )
  }

  rbind(
    score_for(model_p, "null"),
    score_for(model_q, "alternative")
  )
}

filter_valid_trials <- function(trial_out, label = "") {
  ok <- vapply(trial_out, is.data.frame, logical(1))
  if (!all(ok)) {
    warning(sprintf(
      "[%s] dropped %d/%d failed trials",
      label, sum(!ok), length(trial_out)
    ), call. = FALSE)
  }
  trial_out[ok]
}

summarise_error_rates <- function(rows, x_name) {
  rows$error <- ifelse(rows$is_null, rows$rejected, !rows$rejected)
  form <- stats::as.formula(paste("error ~", x_name, "+ method"))
  out <- aggregate(form, rows, mean, na.rm = TRUE)
  names(out)[names(out) == x_name] <- "x_value"
  names(out)[names(out) == "error"] <- "error_rate"
  out
}

summarise_discrepancies <- function(rows) {
  parts <- split(rows, list(rows$sigma_per, rows$metric, rows$condition), drop = TRUE)
  out <- lapply(parts, function(x) {
    data.frame(
      sigma_per = x$sigma_per[1],
      metric = x$metric[1],
      condition = x$condition[1],
      mean = mean(x$value, na.rm = TRUE),
      sd = stats::sd(x$value, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, out)
  out$sd[!is.finite(out$sd)] <- 0
  out
}

run_sigma_experiment <- function() {
  all_rows <- list()

  for (i in seq_along(sigma_grid)) {
    sigma_per <- sigma_grid[i]
    message(sprintf("[panel a] sigma_per = %.3g", sigma_per))

    trial_out <- foreach(
      trial = seq_len(N_trials),
      .inorder = TRUE,
      .export = ls(globalenv()),
      .errorhandling = "pass"
    ) %dorng% {
      rows <- run_gof_trial(sigma_per)
      rows$sigma_per <- sigma_per
      rows
    }

    trial_out <- filter_valid_trials(trial_out, sprintf("panel a %.3g", sigma_per))
    all_rows[[i]] <- do.call(rbind, trial_out)
  }

  summarise_error_rates(do.call(rbind, all_rows), "sigma_per")
}

run_mmd_size_experiment <- function() {
  all_rows <- list()

  for (i in seq_along(mmd_n_grid)) {
    n_prime <- mmd_n_grid[i]
    message(sprintf("[panel b] n_prime = %d", n_prime))

    trial_out <- foreach(
      trial = seq_len(N_trials),
      .inorder = TRUE,
      .export = ls(globalenv()),
      .errorhandling = "pass"
    ) %dorng% {
      rows <- run_mmd_size_trial(n_prime)
      rows$n_prime <- n_prime
      rows
    }

    trial_out <- filter_valid_trials(trial_out, sprintf("panel b %d", n_prime))
    all_rows[[i]] <- do.call(rbind, trial_out)
  }

  summarise_error_rates(do.call(rbind, all_rows), "n_prime")
}

run_discrepancy_experiment <- function() {
  all_rows <- list()

  for (i in seq_along(sigma_grid)) {
    sigma_per <- sigma_grid[i]
    message(sprintf("[panels c-h] sigma_per = %.3g", sigma_per))

    trial_out <- foreach(
      trial = seq_len(N_trials),
      .inorder = TRUE,
      .export = ls(globalenv()),
      .errorhandling = "pass"
    ) %dorng% {
      rows <- run_discrepancy_trial(sigma_per)
      rows$sigma_per <- sigma_per
      rows
    }

    trial_out <- filter_valid_trials(trial_out, sprintf("panels c-h %.3g", sigma_per))
    all_rows[[i]] <- do.call(rbind, trial_out)
  }

  summarise_discrepancies(do.call(rbind, all_rows))
}

## ------------------------------------------------------------------
## Plotting helpers
## ------------------------------------------------------------------

plot_error_panel <- function(df, methods, main, xlab, ylim = c(0, 0.6),
                             legend_pos = NULL) {
  plot(range(df$x_value), ylim,
    type = "n", log = "x", xaxt = "n",
    xlab = xlab, ylab = "Error Rate", main = main
  )
  axis(1, at = c(0.01, 0.1, 1), labels = c("0.01", "0.1", "1"))
  grid(col = "grey90")
  box()

  for (method in methods) {
    sub <- df[df$method == method, ]
    sub <- sub[order(sub$x_value), ]
    lines(sub$x_value, sub$error_rate,
      type = "b", lwd = 2,
      col = METHOD_COL[[method]], pch = METHOD_PCH[[method]]
    )
  }

  if (!is.null(legend_pos)) {
    legend(legend_pos,
      legend = unname(METHOD_LABELS[methods]),
      col = unname(METHOD_COL[methods]), pch = unname(METHOD_PCH[methods]),
      lwd = 2, bty = "n", cex = 0.75
    )
  }
}

plot_mmd_size_panel <- function(df) {
  cols <- c(ksd_u = "red", mmd_mc = "forestgreen")
  pchs <- c(ksd_u = 1, mmd_mc = 4)

  plot(range(df$x_value), c(0, max(0.25, 1.05 * max(df$error_rate))),
    type = "n", log = "x", xaxt = "n",
    xlab = "Monte Carlo Sample Size n' in MMD",
    ylab = "Error Rate", main = "(b)"
  )
  axis(1, at = c(10, 100, 1000), labels = c("10", "100", "1000"))
  grid(col = "grey90")
  box()

  for (method in c("ksd_u", "mmd_mc")) {
    sub <- df[df$method == method, ]
    sub <- sub[order(sub$x_value), ]
    lines(sub$x_value, sub$error_rate,
      type = "b", lwd = 2,
      col = cols[[method]], pch = pchs[[method]]
    )
  }
  legend("topright",
    legend = c("KSD-U", "MMD-MC"),
    col = cols, pch = pchs, lwd = 2, bty = "n", cex = 0.85
  )
}

draw_error_bars <- function(x, y, sd, col) {
  lower <- y - sd
  upper <- y + sd
  keep <- is.finite(lower) & is.finite(upper) & upper > lower
  if (any(keep)) {
    arrows(x[keep], lower[keep], x[keep], upper[keep],
      angle = 90, code = 3, length = 0.035, col = col
    )
  }
}

plot_discrepancy_panel <- function(df, metric, main, legend = FALSE) {
  sub <- df[df$metric == metric, ]
  y_min <- min(0, sub$mean - sub$sd, na.rm = TRUE)
  y_max <- max(sub$mean + sub$sd, na.rm = TRUE)
  if (!is.finite(y_max) || y_max <= y_min) y_max <- y_min + 1

  plot(range(sigma_grid), c(y_min, 1.05 * y_max),
    type = "n",
    log = "x", xaxt = "n", yaxt = "n",
    xlab = expression(Perturbation ~ Magnitude ~ sigma[per]),
    ylab = "Discrepancy", main = main
  )
  axis(1, at = c(0.01, 0.1, 1), labels = c("0.01", "0.1", "1"))
  axis(2, labels = FALSE)
  grid(col = "grey90")
  box()

  for (condition in c("null", "alternative")) {
    s <- sub[sub$condition == condition, ]
    s <- s[order(s$sigma_per), ]
    col <- if (condition == "null") "blue" else "red"
    lty <- if (condition == "null") 1 else 2
    lines(s$sigma_per, s$mean, col = col, lty = lty, lwd = 2)
    draw_error_bars(s$sigma_per, s$mean, s$sd, col)
  }

  if (legend) {
    legend("topleft",
      legend = c("p = q", "p != q"),
      col = c("blue", "red"), lty = c(1, 2), lwd = 2,
      bty = "n", cex = 0.85
    )
  }
}

write_figure2 <- function(panel_a, panel_b, panel_disc) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  pdf(file.path(output_dir, output_pdf), width = 13, height = 7.2)
  layout(matrix(c(
    1, 3, 5, 7,
    2, 4, 6, 8
  ), nrow = 2, byrow = TRUE))
  par(mar = c(4.2, 4.2, 2.2, 1.2), mgp = c(2.4, 0.75, 0))

  plot_error_panel(
    panel_a, METHOD_NAMES, "(a)",
    expression(Perturbation ~ Magnitude ~ sigma[per]),
    ylim = c(0, max(0.6, 1.05 * max(panel_a$error_rate))),
    legend_pos = "bottomleft"
  )
  plot_mmd_size_panel(panel_b)
  plot_discrepancy_panel(panel_disc, "ksd_u", "KSD-U\n(c)", legend = TRUE)
  plot_discrepancy_panel(panel_disc, "ksd_linear", "KSD-Linear\n(d)")
  plot_discrepancy_panel(panel_disc, "mmd_mc_1000", "MMD-MC(1000)\n(e)")
  plot_discrepancy_panel(panel_disc, "mmd_mcmc_1000", "MMD-MCMC(1000)\n(f)")
  plot_discrepancy_panel(panel_disc, "lr_exact", "Loglikelihood Ratio (Exact)\n(g)")
  plot_discrepancy_panel(panel_disc, "lr_ais", "Loglikelihood Ratio (AIS)\n(h)")

  dev.off()
}

## ------------------------------------------------------------------
## Run experiments
## ------------------------------------------------------------------

message("Running RBM KSD-U Figure 2 experiment")
message(sprintf(
  "N_trials = %d, n = %d, dx = %d, dh = %d, KSD boot = %d, MMD boot = %d",
  N_trials, n_default, dx, dh, nboot, mmd_boot
))

panel_a <- run_sigma_experiment()
panel_b <- run_mmd_size_experiment()
panel_disc <- run_discrepancy_experiment()

write_figure2(panel_a, panel_b, panel_disc)
if (!is.null(cl)) {
  parallel::stopCluster(cl)
}

message("Saved ", file.path(output_dir, output_pdf))
