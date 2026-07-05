## SVGD logistic regression on Covertype
##
## Compares SVGD with several particle and stochastic-gradient baselines on a
## binary Covertype classification task.

## ------------------------------------------------------------------
## Data preparation
## ------------------------------------------------------------------

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
replication_dir <- if (length(file_arg)) {
  normalizePath(dirname(sub("^--file=", "", file_arg[[1L]])), mustWork = FALSE)
} else if (dir.exists(file.path("inst", "replication"))) {
  normalizePath(file.path("inst", "replication"), mustWork = FALSE)
} else {
  normalizePath(".", mustWork = FALSE)
}
repo_root <- normalizePath(file.path(replication_dir, "..", ".."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "R"))) {
  setwd(repo_root)
}
suppressPackageStartupMessages(library(steinsampling))

load_covtype <- function(path) {
  dat <- readRDS(path)
  X <- as.matrix(dat$X)
  y <- as.numeric(dat$y)
  y[y == 0 | y == 2] <- -1
  X <- cbind(X, intercept = 1)
  storage.mode(X) <- "double"
  list(X = X, y = y)
}

init_particles <- function(n, d, a0, b0, weight_scale = 1) {
  alpha <- pmax(rgamma(n, shape = a0, scale = b0), 1e-12)
  w <- matrix(rnorm(n * d), n, d) * sqrt(1 / alpha) * weight_scale
  cbind(w, log_alpha = log(alpha))
}

## ------------------------------------------------------------------
## Model helpers
## ------------------------------------------------------------------

sigmoid <- function(z) {
  out <- z
  pos <- z >= 0
  out[pos] <- 1 / (1 + exp(-z[pos]))
  ez <- exp(z[!pos])
  out[!pos] <- ez / (1 + ez)
  out
}

log_sigmoid <- function(z) -log1p(exp(-abs(z))) + pmin(z, 0)

logsumexp <- function(x) {
  m <- max(x)
  if (!is.finite(m)) {
    return(m)
  }
  m + log(sum(exp(x - m)))
}

lr_score <- function(theta, X, y, batch_size, a0, b0) {
  theta <- as.matrix(theta)
  n <- nrow(X)
  if (batch_size < n) {
    idx <- sample.int(n, batch_size)
    Xb <- X[idx, , drop = FALSE]
    yb <- y[idx]
  } else {
    Xb <- X
    yb <- y
  }

  m <- nrow(Xb)
  d <- ncol(X)
  w <- theta[, seq_len(d), drop = FALSE]
  log_alpha <- pmax(pmin(theta[, d + 1L], 30), -30)
  alpha <- exp(log_alpha)

  p1 <- sigmoid(Xb %*% t(w))
  y01 <- (yb + 1) / 2
  dw <- t(matrix(y01, m, nrow(theta)) - p1) %*% Xb
  dw <- dw * n / m - w * alpha
  dlog_alpha <- d / 2 - (alpha / 2) * rowSums(w * w) + (a0 - 1) - b0 * alpha + 1
  cbind(dw, dlog_alpha)
}

score_particles <- function(theta, X, y, weights = NULL, chunk = 20000L) {
  theta <- as.matrix(theta)
  d <- ncol(X)
  w <- if (ncol(theta) == d + 1L) theta[, seq_len(d), drop = FALSE] else theta
  weights <- if (is.null(weights)) rep(1 / nrow(w), nrow(w)) else weights / sum(weights)

  prob <- numeric(nrow(X))
  start <- 1L
  while (start <= nrow(X)) {
    end <- min(nrow(X), start + chunk - 1L)
    idx <- start:end
    prob[idx] <- as.numeric(sigmoid((X[idx, , drop = FALSE] %*% t(w)) * y[idx]) %*% weights)
    start <- end + 1L
  }
  prob <- pmin(pmax(prob, 1e-12), 1 - 1e-12)
  c(accuracy = mean(prob > 0.5), loglik = mean(log(prob)))
}

result_row <- function(trial, panel, method, x, n_particles, score) {
  data.frame(
    trial = trial,
    panel = panel,
    method = method,
    x = x,
    particle_size = n_particles,
    accuracy = unname(score["accuracy"]),
    loglik = unname(score["loglik"])
  )
}

run_svgd <- function(theta0, Xtr, ytr, Xts, yts, iters, xvals, n_particles,
                     trial, panel, batch_size, stepsize, alpha, a0, b0, chunk) {
  fit <- svgd(kernel = stein_kernel(type = "gaussian_rbf"))$update(
    x0 = theta0,
    lnprob = function(theta) lr_score(theta, Xtr, ytr, batch_size, a0, b0),
    n_iter = max(iters),
    stepsize = stepsize,
    alpha = alpha,
    trace_iters = iters
  )
  out <- vector("list", length(iters))
  for (j in seq_along(iters)) {
    theta <- fit$trace[[as.character(iters[[j]])]]
    out[[j]] <- result_row(
      trial, panel, "svgd", xvals[[j]], n_particles,
      score_particles(theta, Xts, yts, chunk = chunk)
    )
  }
  out
}

sgld_eps <- function(iter, a, decay) a / ((iter + 1)^decay)

run_parallel_sgld <- function(theta0, Xtr, ytr, Xts, yts, iters, xvals,
                              n_particles, trial, panel, batch_size, a, decay,
                              grad_scale, a0, b0, chunk) {
  theta <- theta0
  out <- list()
  trace_pos <- match(seq_len(max(iters)), iters, nomatch = 0L)
  for (iter in seq_len(max(iters))) {
    eps <- sgld_eps(iter, a, decay)
    theta <- theta +
      0.5 * eps * grad_scale * lr_score(theta, Xtr, ytr, batch_size, a0, b0) +
      matrix(rnorm(length(theta), sd = sqrt(eps)), nrow(theta), ncol(theta))
    pos <- trace_pos[[iter]]
    if (pos > 0L) {
      out[[length(out) + 1L]] <- result_row(
        trial, panel, "parallel_sgld", xvals[[pos]], n_particles,
        score_particles(theta, Xts, yts, chunk = chunk)
      )
    }
  }
  out
}

run_sequential_sgld <- function(theta0, Xtr, ytr, Xts, yts, iters, xvals,
                                n_particles, trial, panel, batch_size, a, decay,
                                grad_scale, average_last, a0, b0, chunk) {
  theta <- theta0
  buffer <- matrix(NA_real_, average_last, ncol(theta0))
  out <- list()
  trace_pos <- match(seq_len(max(iters)), iters, nomatch = 0L)
  for (iter in seq_len(max(iters))) {
    eps <- sgld_eps(iter, a, decay)
    theta <- theta +
      0.5 * eps * grad_scale * lr_score(theta, Xtr, ytr, batch_size, a0, b0) +
      matrix(rnorm(length(theta), sd = sqrt(eps)), nrow(theta), ncol(theta))
    buffer[((iter - 1L) %% average_last) + 1L, ] <- theta
    pos <- trace_pos[[iter]]
    if (pos > 0L) {
      keep <- buffer[stats::complete.cases(buffer), , drop = FALSE]
      out[[length(out) + 1L]] <- result_row(
        trial, panel, "sequential_sgld", xvals[[pos]], n_particles,
        score_particles(keep, Xts, yts, chunk = chunk)
      )
    }
  }
  out
}

median_sqdist <- function(theta) {
  if (nrow(theta) < 2L) {
    return(1)
  }
  out <- stats::median(as.numeric(stats::dist(theta))^2)
  if (!is.finite(out) || out <= 0) 1e-5 else out
}

pmd_log_prior <- function(theta, d, a0, b0) {
  w <- theta[, seq_len(d), drop = FALSE]
  log_alpha <- pmax(pmin(theta[, d + 1L], 30), -30)
  alpha <- exp(log_alpha)
  d / 2 * log_alpha - alpha / 2 * rowSums(w * w) + (a0 - 1) * log_alpha - b0 * alpha + log_alpha
}

pmd_log_lik <- function(theta, X, y) {
  d <- ncol(X)
  colSums(log_sigmoid((X %*% t(theta[, seq_len(d), drop = FALSE])) * y))
}

pmd_log_kde <- function(theta, centers, weights, h) {
  sq <- compute_cross_squared_distance(theta, centers)
  log_w <- log(pmax(weights, 1e-300))
  apply(-sq / h, 1L, function(v) logsumexp(log_w + v))
}

run_pmd <- function(theta0, Xtr, ytr, Xts, yts, iters, xvals, n_particles,
                    trial, panel, batch_size, a, bandwidth_scale, a0, b0, chunk) {
  centers <- theta0
  weights <- rep(1 / nrow(theta0), nrow(theta0))
  n_train <- nrow(Xtr)
  d <- ncol(Xtr)
  out <- list()
  trace_pos <- match(seq_len(max(iters)), iters, nomatch = 0L)
  for (iter in seq_len(max(iters))) {
    h <- max(bandwidth_scale * median_sqdist(centers), 1e-8)
    idx <- sample.int(nrow(centers), nrow(centers), replace = TRUE, prob = weights)
    theta <- centers[idx, , drop = FALSE] + matrix(rnorm(length(centers), sd = sqrt(h / 2)), nrow(centers), ncol(centers))
    theta[, d + 1L] <- pmax(pmin(theta[, d + 1L], 30), -30)

    bid <- if (batch_size < n_train) sample.int(n_train, batch_size) else seq_len(n_train)
    Xb <- Xtr[bid, , drop = FALSE]
    yb <- ytr[bid]
    log_post <- pmd_log_prior(theta, d, a0, b0) + n_train / nrow(Xb) * pmd_log_lik(theta, Xb, yb)
    gamma <- a / n_train / (100 + sqrt(iter))
    log_weights <- gamma * (log_post - pmd_log_kde(theta, centers, weights, h))
    log_weights <- log_weights - logsumexp(log_weights)
    centers <- theta
    weights <- exp(log_weights)

    pos <- trace_pos[[iter]]
    if (pos > 0L) {
      out[[length(out) + 1L]] <- result_row(
        trial, panel, "pmd", xvals[[pos]], n_particles,
        score_particles(centers, Xts, yts, weights = weights, chunk = chunk)
      )
    }
  }
  out
}

run_dsvi <- function(Xtr, ytr, Xts, yts, iters, xvals, n_particles, trial,
                     panel, batch_size, lr, decay, stage_iter, chunk) {
  n <- nrow(Xtr)
  d <- ncol(Xtr)
  mu <- numeric(d)
  C <- rep(0.1, d)
  y01_all <- (ytr + 1) / 2
  out <- list()
  trace_pos <- match(seq_len(max(iters)), iters, nomatch = 0L)
  for (iter in seq_len(max(iters))) {
    if (stage_iter > 0L && iter > 1L && (iter - 1L) %% stage_iter == 0L) lr <- decay * lr
    idx <- if (batch_size < n) sample.int(n, batch_size) else seq_len(n)
    Xb <- Xtr[idx, , drop = FALSE]
    z <- rnorm(d)
    theta <- C * z + mu
    dg <- as.numeric(n / length(idx) * crossprod(Xb, y01_all[idx] - sigmoid(drop(Xb %*% theta))))
    C2 <- C * C
    Cmu <- C2 + mu * mu
    mu <- mu + lr * (dg - mu / Cmu)
    C <- pmax(C + 0.1 * lr * (dg * z + 1 / C - C / Cmu), 1e-4)
    pos <- trace_pos[[iter]]
    if (pos > 0L) {
      out[[length(out) + 1L]] <- result_row(
        trial, panel, "dsvi", xvals[[pos]], n_particles,
        score_particles(matrix(mu, nrow = 1), Xts, yts, chunk = chunk)
      )
    }
  }
  out
}

summarise_results <- function(raw) {
  aggregate(cbind(accuracy, loglik) ~ panel + method + x + particle_size, raw, mean)
}

plot_figure3 <- function(summary_df, output_pdf, epoch_particles, panel_b_iter, batch_size, n_train) {
  labels <- c(
    svgd = "Stein Variational Gradient Descent (Our Method)",
    parallel_sgld = "Stochastic Langevin (Parallel SGLD)",
    pmd = "Particle Mirror Descent (PMD)",
    dsvi = "Doubly Stochastic (DSVI)",
    sequential_sgld = "Stochastic Langevin (Sequential SGLD)"
  )
  colors <- c(
    svgd = "red", parallel_sgld = "blue", pmd = "forestgreen",
    dsvi = "gray35", sequential_sgld = "purple"
  )
  ltys <- c(svgd = 1, parallel_sgld = 1, pmd = 1, dsvi = 3, sequential_sgld = 2)
  pchs <- c(svgd = 1, parallel_sgld = 3, pmd = 2, dsvi = 8, sequential_sgld = 8)
  methods <- names(labels)
  present <- methods[methods %in% unique(summary_df$method)]
  ylim <- c(0.62, 0.755)

  pdf(output_pdf, width = 12, height = 4.5, useDingbats = FALSE)
  on.exit(dev.off(), add = TRUE)
  layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE), heights = c(4, 1.05))

  epoch_df <- summary_df[summary_df$panel == "epochs", , drop = FALSE]
  par(mar = c(4.2, 4.4, 2.2, 0.8), xpd = FALSE)
  plot(NA, NA,
    xlim = range(epoch_df$x), ylim = ylim, xlab = "Number of Epoches",
    ylab = "Testing Accuracy", main = sprintf("(a) Particle size n = %d", epoch_particles)
  )
  grid(col = "gray88", lty = 1)
  for (m in present) {
    d <- epoch_df[epoch_df$method == m, , drop = FALSE]
    d <- d[order(d$x), , drop = FALSE]
    lines(d$x, d$accuracy, col = colors[[m]], lty = ltys[[m]], lwd = 2.2)
    points(d$x, d$accuracy, col = colors[[m]], pch = pchs[[m]], cex = 0.65)
  }

  size_df <- summary_df[summary_df$panel == "particles", , drop = FALSE]
  x_vals <- sort(unique(size_df$particle_size))
  par(mar = c(4.2, 4.4, 2.2, 0.8), xpd = FALSE)
  plot(NA, NA,
    xlim = range(x_vals), ylim = ylim, log = "x", xaxt = "n",
    xlab = "Particle Size (n)", ylab = "Testing Accuracy",
    main = sprintf(
      "(b) Results at %d iteration (~%.2f epoches)",
      panel_b_iter, panel_b_iter * batch_size / n_train
    )
  )
  axis(1, at = x_vals, labels = x_vals)
  grid(col = "gray88", lty = 1)
  for (m in present) {
    d <- size_df[size_df$method == m, , drop = FALSE]
    if (nrow(d) == 0L) next
    d <- d[order(d$particle_size), , drop = FALSE]
    lines(d$particle_size, d$accuracy, col = colors[[m]], lty = ltys[[m]], lwd = 2.2)
    points(d$particle_size, d$accuracy, col = colors[[m]], pch = pchs[[m]], cex = 0.65)
  }

  par(mar = c(0, 0, 0, 0), xpd = NA)
  plot.new()
  legend("center",
    legend = labels[present], col = colors[present], lty = ltys[present],
    pch = pchs[present], lwd = 2.2, bty = "n", ncol = 2, cex = 0.9,
    x.intersp = 0.7, y.intersp = 0.9
  )
}

## ------------------------------------------------------------------
## Run experiments
## ------------------------------------------------------------------

cfg <- list(
  seed = 20240630L,
  data_path = file.path(replication_dir, "data", "covtype", "covtype.libsvm.binary.scale.bz2.dense.rds"),
  output_pdf = file.path(replication_dir, "svgd_logistic_covtype.pdf"),
  # Full replication: trials = 50L,
  trials = 1L,
  train_ratio = 0.8,
  batch_size = 50L,
  # Full replication: epoch_grid = seq(0.1, 2, by = 0.1),
  epoch_grid = c(0.05, 0.1),
  # Full replication: epoch_particles = 100L,
  epoch_particles = 10L,
  # Full replication: particle_grid = c(1L, 10L, 50L, 250L),
  particle_grid = c(1L, 5L),
  # Full replication: panel_b_iter = 3000L,
  panel_b_iter = 5L,
  a0 = 1,
  b0 = 0.01,
  svgd_stepsize = 0.05,
  svgd_alpha = 0.9,
  parallel_sgld_a = 0.1,
  sequential_sgld_a = 2e-4,
  sgld_decay = 0.55,
  sequential_sgld_decay = 0.75,
  sequential_average_last = 100L,
  sequential_scale_particles = 1L,
  pmd_a = 5000,
  pmd_bandwidth_scale = 0.002,
  dsvi_lr = 0.05,
  dsvi_decay = 0.95,
  dsvi_stage_iter = 5000L,
  chunk = 20000L,
  # Full replication: cores = max(1L, min(4L, parallel::detectCores() - 1L))
  cores = 1L
)

dat <- load_covtype(cfg$data_path)
message(sprintf("Using %d worker(s)", cfg$cores))

trial_results <- parallel::mclapply(seq_len(cfg$trials), function(trial) {
  raw <- list()
  set.seed(cfg$seed + trial - 1L)
  n_total <- nrow(dat$X)
  train_idx <- sample.int(n_total, round(cfg$train_ratio * n_total))
  test_idx <- setdiff(seq_len(n_total), train_idx)
  Xtr <- dat$X[train_idx, , drop = FALSE]
  ytr <- dat$y[train_idx]
  Xts <- dat$X[test_idx, , drop = FALSE]
  yts <- dat$y[test_idx]
  n_train <- nrow(Xtr)
  epoch_iters <- pmax(1L, unique(as.integer(round(cfg$epoch_grid * n_train / cfg$batch_size))))
  epoch_x <- epoch_iters * cfg$batch_size / n_train
  fixed_iter <- cfg$panel_b_iter
  fixed_x <- fixed_iter * cfg$batch_size / n_train

  message(sprintf("Trial %d/%d", trial, cfg$trials))
  theta_epoch <- init_particles(cfg$epoch_particles, ncol(Xtr), cfg$a0, cfg$b0)
  theta_epoch_pmd <- init_particles(cfg$epoch_particles, ncol(Xtr), cfg$a0, cfg$b0, weight_scale = 0.1)
  theta_seq <- init_particles(1L, ncol(Xtr), cfg$a0, cfg$b0)

  raw <- c(raw, run_svgd(
    theta_epoch, Xtr, ytr, Xts, yts, epoch_iters, epoch_x,
    cfg$epoch_particles, trial, "epochs", cfg$batch_size,
    cfg$svgd_stepsize, cfg$svgd_alpha, cfg$a0, cfg$b0, cfg$chunk
  ))
  raw <- c(raw, run_parallel_sgld(
    theta_epoch, Xtr, ytr, Xts, yts, epoch_iters,
    epoch_x, cfg$epoch_particles, trial, "epochs",
    cfg$batch_size, cfg$parallel_sgld_a, cfg$sgld_decay,
    1 / cfg$epoch_particles, cfg$a0, cfg$b0, cfg$chunk
  ))
  raw <- c(raw, run_pmd(
    theta_epoch_pmd, Xtr, ytr, Xts, yts, epoch_iters, epoch_x,
    cfg$epoch_particles, trial, "epochs", cfg$batch_size,
    cfg$pmd_a, cfg$pmd_bandwidth_scale, cfg$a0, cfg$b0, cfg$chunk
  ))
  raw <- c(raw, run_dsvi(
    Xtr, ytr, Xts, yts, epoch_iters, epoch_x, cfg$epoch_particles,
    trial, "epochs", cfg$batch_size, cfg$dsvi_lr / n_train,
    cfg$dsvi_decay, cfg$dsvi_stage_iter, cfg$chunk
  ))
  raw <- c(raw, run_sequential_sgld(
    theta_seq, Xtr, ytr, Xts, yts, epoch_iters,
    epoch_x, cfg$epoch_particles, trial, "epochs",
    cfg$batch_size, cfg$sequential_sgld_a, cfg$sequential_sgld_decay,
    1 / cfg$sequential_scale_particles,
    cfg$sequential_average_last, cfg$a0, cfg$b0, cfg$chunk
  ))

  for (n_particles in cfg$particle_grid) {
    theta0 <- init_particles(n_particles, ncol(Xtr), cfg$a0, cfg$b0)
    theta0_pmd <- init_particles(n_particles, ncol(Xtr), cfg$a0, cfg$b0, weight_scale = 0.1)
    raw <- c(raw, run_svgd(
      theta0, Xtr, ytr, Xts, yts, fixed_iter, n_particles,
      n_particles, trial, "particles", cfg$batch_size,
      cfg$svgd_stepsize, cfg$svgd_alpha, cfg$a0, cfg$b0, cfg$chunk
    ))
    raw <- c(raw, run_parallel_sgld(
      theta0, Xtr, ytr, Xts, yts, fixed_iter, n_particles,
      n_particles, trial, "particles", cfg$batch_size,
      cfg$parallel_sgld_a, cfg$sgld_decay, 1 / n_particles,
      cfg$a0, cfg$b0, cfg$chunk
    ))
    raw <- c(raw, run_pmd(
      theta0_pmd, Xtr, ytr, Xts, yts, fixed_iter, n_particles,
      n_particles, trial, "particles", cfg$batch_size,
      cfg$pmd_a, cfg$pmd_bandwidth_scale, cfg$a0, cfg$b0, cfg$chunk
    ))
  }
  raw
}, mc.cores = cfg$cores, mc.preschedule = FALSE)

raw <- do.call(c, trial_results)
raw <- do.call(rbind, raw)
plot_figure3(
  summarise_results(raw), cfg$output_pdf, cfg$epoch_particles,
  cfg$panel_b_iter, cfg$batch_size, round(cfg$train_ratio * nrow(dat$X))
)
message("Wrote figure: ", cfg$output_pdf)
