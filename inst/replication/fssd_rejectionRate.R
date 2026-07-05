## FSSD rejection-rate comparison
##
## Runs the Figure 2-style settings for Gaussian-Laplace and RBM examples.
## The plotted methods are FSSD-opt, FSSD-rand, KSD, and LKS.

suppressPackageStartupMessages({
  library(stats)
})

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

## ------------------------------------------------------------------
## Experiment settings
## ------------------------------------------------------------------

# Full replication: N_trials <- 200L
N_trials <- 3L
alpha <- 0.05
# Full replication: J <- 5L
J <- 2L
# Full replication: nboot <- 2000L
nboot <- 30L
# Full replication: ksd_nboot <- 1000L
ksd_nboot <- 30L
train_ratio <- 0.2
# Full replication: opt_maxit_ex2 <- 40L
opt_maxit_ex2 <- 5L
# Full replication: opt_locs_bounds_frac_ex2 <- 10
opt_locs_bounds_frac_ex2 <- 5
# Full replication: opt_maxit_ex1 <- 30L
opt_maxit_ex1 <- 5L
# Full replication: opt_locs_bounds_frac_ex1 <- 30
opt_locs_bounds_frac_ex1 <- 5
# Full replication: use_parallel <- TRUE
use_parallel <- FALSE

# Full replication: glaplace_n <- 1000L
glaplace_n <- 100L
# Full replication: glaplace_dims <- c(1L, 5L, 10L, 15L)
glaplace_dims <- c(1L, 5L)

# Full replication: rbm_n_fixed <- 1000L
rbm_n_fixed <- 100L
# Full replication: rbm_dims_x <- 50L
rbm_dims_x <- 10L
# Full replication: rbm_dims_h <- 40L
rbm_dims_h <- 4L
# Full replication: rbm_burnin <- 2000L
rbm_burnin <- 50L
# Full replication: rbm_all_stds <- c(0, 0.01, 0.02, 0.04, 0.06)
rbm_all_stds <- c(0, 0.04)
# Full replication: rbm_ns <- c(1000L, 2000L, 3000L, 4000L)
rbm_ns <- c(100L, 200L)
rbm_single_var <- 0.1

## ------------------------------------------------------------------
## Dependency and plotting labels
## ------------------------------------------------------------------

resolve_numpy_python <- function() {
  configured <- Sys.getenv("FSSD_NUMPY_PYTHON", "")
  candidates <- if (nzchar(configured)) configured else c("python3", "python")
  for (candidate in candidates) {
    available <- if (grepl("/", candidate, fixed = TRUE)) {
      file.exists(candidate) && file.access(candidate, mode = 1) == 0
    } else {
      nzchar(Sys.which(candidate))
    }
    if (!available) {
      next
    }
    probe <- tryCatch(
      system2(candidate, c("-c", shQuote("import numpy")), stdout = TRUE, stderr = TRUE),
      error = function(e) e
    )
    status <- attr(probe, "status")
    if (!inherits(probe, "error") && (is.null(status) || status == 0)) {
      return(candidate)
    }
  }
  stop(
    "Could not find a Python executable with numpy. ",
    "Install numpy or set FSSD_NUMPY_PYTHON to the real Python path, e.g. ",
    "FSSD_NUMPY_PYTHON=/opt/miniconda3/bin/python3",
    call. = FALSE
  )
}

numpy_python <- resolve_numpy_python()

method_order <- c("fssd_opt", "fssd_rand", "ksd", "lks")
method_labels <- c(
  fssd_opt = "FSSD-opt",
  fssd_rand = "FSSD-rand",
  ksd = "KSD",
  lks = "LKS"
)
method_col <- c(
  fssd_opt = "red",
  fssd_rand = "red",
  ksd = "darkgreen",
  lks = "darkgreen"
)
method_lty <- c(
  fssd_opt = 1,
  fssd_rand = 2,
  ksd = 1,
  lks = 4
)
method_pch <- c(
  fssd_opt = 15,
  fssd_rand = 17,
  ksd = 16,
  lks = 18
)

## ------------------------------------------------------------------
## Problem helpers
## ------------------------------------------------------------------

rlaplace <- function(n) {
  u <- runif(n) - 0.5
  -sign(u) * log1p(-2 * abs(u)) / sqrt(2)
}

score_std_normal <- function(X) {
  -X
}

make_glaplace_problem <- function(d) {
  list(type = "glaplace", d = d)
}

rbm_sigmoid <- function(x) {
  ifelse(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))
}

rbm_gibbs_step <- function(X, H, B, b, c) {
  logits_h <- sweep(X %*% B, 2, 2 * c, "+")
  probs_h <- rbm_sigmoid(logits_h)
  H <- matrix(as.numeric(runif(length(probs_h)) <= as.vector(probs_h)),
    nrow = nrow(X), ncol = ncol(B)
  )
  H <- 2 * H - 1

  mean_x <- sweep(H %*% t(B) / 2, 2, b, "+")
  X <- matrix(rnorm(length(mean_x)), nrow = nrow(mean_x), ncol = ncol(mean_x)) + mean_x
  list(X = X, H = H)
}

rbm_sample <- function(n, B, b, c, burnin, seed) {
  with_local_seed(seed, {
    dx <- nrow(B)
    dh <- ncol(B)
    X <- matrix(rnorm(n * dx), nrow = n, ncol = dx)
    H <- matrix(1, nrow = n, ncol = dh)
    if (burnin > 0) {
      for (i in seq_len(burnin)) {
        step <- rbm_gibbs_step(X, H, B, b, c)
        X <- step$X
        H <- step$H
      }
    }
    rbm_gibbs_step(X, H, B, b, c)$X
  })
}

rbm_score <- function(X, B, b, c) {
  hidden_mean <- tanh(sweep(0.5 * (X %*% B), 2, c, "+"))
  sweep(hidden_mean %*% (0.5 * t(B)), 2, b, "+") - X
}

run_numpy_rbm_generator <- function(py_code, dx, dh) {
  out <- tryCatch(
    system2(
      numpy_python,
      args = "-",
      input = py_code,
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    stop(
      "Could not run Python executable: ", numpy_python, "\n",
      conditionMessage(out), "\n",
      "Set FSSD_NUMPY_PYTHON to the real Python path, e.g. ",
      "FSSD_NUMPY_PYTHON=/opt/miniconda3/bin/python3",
      call. = FALSE
    )
  }
  status <- attr(out, "status")
  if (!is.null(status) && status != 0) {
    stop(
      "Could not generate source-compatible RBM parameters with NumPy using ",
      numpy_python, ".\n",
      paste(out, collapse = "\n"),
      call. = FALSE
    )
  }

  vals <- scan(text = paste(out, collapse = " "), quiet = TRUE)
  expected <- 2L * dx * dh + dx + dh
  if (length(vals) != expected) {
    stop(sprintf("NumPy RBM generator returned %d values; expected %d", length(vals), expected))
  }

  i1 <- dx * dh
  i2 <- 2L * dx * dh
  list(
    B_p = matrix(vals[seq_len(i1)], nrow = dx, ncol = dh, byrow = TRUE),
    B_q = matrix(vals[(i1 + 1L):i2], nrow = dx, ncol = dh, byrow = TRUE),
    b = vals[(i2 + 1L):(i2 + dx)],
    c = vals[(i2 + dx + 1L):expected]
  )
}

make_rbm_all_problem <- function(std, index, dx, dh, burnin) {
  # Perturb all entries in the RBM interaction matrix.
  seed <- 1000L + index - 1L
  py <- sprintf("
import numpy as np
dx = %d
dh = %d
std = %.17g
np.random.seed(%d)
B = np.random.randint(0, 2, (dx, dh))*2 - 1.0
b = np.random.randn(dx)
c = np.random.randn(dh)
if std <= 1e-8:
    Bq = B
else:
    Bq = B + np.random.randn(dx, dh)*std
vals = np.concatenate([B.reshape(-1), Bq.reshape(-1), b, c])
print(' '.join(format(float(x), '.17g') for x in vals))
", dx, dh, std, seed)
  params <- run_numpy_rbm_generator(py, dx, dh)
  list(type = "rbm", B_p = params$B_p, B_q = params$B_q, b = params$b, c = params$c, burnin = burnin)
}

make_rbm_single_problem <- function(var_perturb_B, dx, dh, burnin) {
  # Perturb one RBM interaction entry.
  py <- sprintf("
import numpy as np
dx = %d
dh = %d
var_perturb_B = %.17g
np.random.seed(10)
B = np.random.randint(0, 2, (dx, dh))*2 - 1.0
b = np.random.randn(dx)
c = np.random.randn(dh)
Bq = B.copy()
if var_perturb_B > 1e-7:
    Bq[0, 0] = Bq[0, 0] + np.random.randn(1)[0]*np.sqrt(var_perturb_B)
vals = np.concatenate([B.reshape(-1), Bq.reshape(-1), b, c])
print(' '.join(format(float(x), '.17g') for x in vals))
", dx, dh, var_perturb_B)
  params <- run_numpy_rbm_generator(py, dx, dh)
  list(type = "rbm", B_p = params$B_p, B_q = params$B_q, b = params$b, c = params$c, burnin = burnin)
}

sample_problem <- function(problem, n, seed) {
  if (identical(problem$type, "glaplace")) {
    return(with_local_seed(seed, matrix(rlaplace(n * problem$d), nrow = n, ncol = problem$d)))
  }
  if (identical(problem$type, "rbm")) {
    return(rbm_sample(n, problem$B_q, problem$b, problem$c, problem$burnin, seed))
  }
  stop("Unknown problem type: ", problem$type)
}

score_problem <- function(problem, X) {
  if (identical(problem$type, "glaplace")) {
    return(score_std_normal(X))
  }
  if (identical(problem$type, "rbm")) {
    return(rbm_score(X, problem$B_p, problem$b, problem$c))
  }
  stop("Unknown problem type: ", problem$type)
}

make_score_function <- function(problem) {
  if (identical(problem$type, "glaplace")) {
    return(function(Z) -Z)
  }
  if (identical(problem$type, "rbm")) {
    B <- problem$B_p
    b <- problem$b
    c <- problem$c
    force(B)
    force(b)
    force(c)
    return(function(Z) {
      hidden_mean <- tanh(sweep(0.5 * (Z %*% B), 2, c, "+"))
      sweep(hidden_mean %*% (0.5 * t(B)), 2, b, "+") - Z
    })
  }
  stop("Unknown problem type: ", problem$type)
}

## ------------------------------------------------------------------
## Test wrappers
## ------------------------------------------------------------------

# Compute paired Stein-kernel values in blocks.
stein_pair_values <- function(kernel_obj, A, gA, B, gB, block = 512L) {
  m <- nrow(A)
  out <- numeric(m)
  for (s in seq.int(1L, m, by = block)) {
    e <- min(s + block - 1L, m)
    idx <- s:e
    mat <- stein_kernel_matrix(
      kernel_obj,
      A[idx, , drop = FALSE], gA[idx, , drop = FALSE],
      B[idx, , drop = FALSE], gB[idx, , drop = FALSE]
    )
    out[idx] <- diag(mat)
  }
  out
}

# Linear-time KSD test using paired sample halves.
lks_test <- function(X, score_function, alpha = 0.05, kernel = "gaussian_rbf") {
  prep <- prepare_ksd_u_inputs(X, score_function, kernel = kernel)
  Xm <- prep$X
  grads <- prep$grads
  kernel_obj <- prep$kernel_obj
  n <- nrow(Xm)
  if (n < 4) {
    stop("LKS requires at least four observations.")
  }

  m <- n %/% 2L
  first <- seq_len(m)
  second <- m + seq_len(m)
  h <- stein_pair_values(
    kernel_obj,
    Xm[first, , drop = FALSE], grads[first, , drop = FALSE],
    Xm[second, , drop = FALSE], grads[second, , drop = FALSE]
  )

  ssl <- mean(h)
  h_bar <- mean(h)
  v_h <- mean(h^2)
  if (!is.finite(v_h) || v_h <= 0) {
    stat <- 0
    p_val <- 1
  } else {
    stat <- sqrt(m) * h_bar / sqrt(v_h)
    p_val <- pnorm(stat, lower.tail = FALSE)
  }

  list(
    statistic = c(lks = stat),
    ssl = ssl,
    p.value = p_val,
    reject = isTRUE(p_val < alpha),
    method = "lks",
    alpha = alpha
  )
}

## ------------------------------------------------------------------
## Trial runners
## ------------------------------------------------------------------

run_htest_method <- function(expr, alpha) {
  elapsed <- system.time({
    result <- tryCatch(
      suppressWarnings(expr),
      error = function(e) e
    )
  })[["elapsed"]]

  if (inherits(result, "error")) {
    return(list(
      rejected = NA,
      p_value = NA_real_,
      time_secs = elapsed,
      error_message = conditionMessage(result)
    ))
  }

  p_value <- result$p.value
  list(
    rejected = isTRUE(is.finite(p_value) && p_value < alpha),
    p_value = p_value,
    time_secs = elapsed,
    error_message = NA_character_
  )
}

run_test_methods <- function(problem, n, trial, x_value, panel, alpha, J,
                             fssd_nboot, ksd_nboot, train_ratio, opt_maxit,
                             opt_locs_bounds_frac) {
  rep_id <- as.integer(trial) - 1L
  X <- sample_problem(problem, n, seed = rep_id)
  score_function <- make_score_function(problem)
  fssd_rand_seed <- rep_id + if (identical(panel, "rbm_B11_n")) 1L else 3L

  tests <- list(
    fssd_opt = run_htest_method(
      fssd_opt_test(
        X,
        score_function,
        J = J,
        nboot = fssd_nboot,
        train_ratio = train_ratio,
        eval_on = "test",
        kernel = "gaussian_rbf",
        maxit = opt_maxit,
        locs_bounds_frac = opt_locs_bounds_frac,
        seed = rep_id + 21L
      ),
      alpha
    ),
    fssd_rand = run_htest_method(
      fssd_rand_test(
        X,
        score_function,
        J = J,
        nboot = fssd_nboot,
        kernel = "gaussian_rbf",
        seed = fssd_rand_seed
      ),
      alpha
    ),
    ksd = run_htest_method(
      ksd_v_test(
        X,
        score_function,
        nboot = ksd_nboot,
        kernel = "gaussian_rbf",
        boot_method = "rademacher"
      ),
      alpha
    ),
    lks = run_htest_method(
      lks_test(
        X,
        score_function,
        alpha = alpha,
        kernel = "gaussian_rbf"
      ),
      alpha
    )
  )

  do.call(rbind, lapply(names(tests), function(method) {
    data.frame(
      panel = panel,
      x_value = x_value,
      n = n,
      trial = trial,
      method = method,
      rejected = tests[[method]]$rejected,
      p_value = tests[[method]]$p_value,
      time_secs = tests[[method]]$time_secs,
      error_message = tests[[method]]$error_message,
      stringsAsFactors = FALSE
    )
  }))
}

summarise_trials <- function(rows) {
  errors <- rows[!is.na(rows$error_message), c("method", "error_message"), drop = FALSE]
  if (nrow(errors) > 0) {
    errors <- unique(errors)
    warning(
      paste(
        sprintf("%s: %s", errors$method, errors$error_message),
        collapse = "\n"
      ),
      call. = FALSE
    )
  }

  by_method <- split(rows, rows$method)
  out <- lapply(method_order, function(method) {
    x <- by_method[[method]]
    if (is.null(x)) {
      return(NULL)
    }
    data.frame(
      panel = x$panel[1],
      x_value = x$x_value[1],
      n = x$n[1],
      method = method,
      rejection_rate = mean(x$rejected, na.rm = TRUE),
      mean_time_secs = mean(x$time_secs, na.rm = TRUE),
      n_success = sum(!is.na(x$rejected)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

run_trials_for_problem <- function(problem, n, x_value, panel,
                                   opt_maxit, opt_locs_bounds_frac) {
  message(sprintf("[%s] x = %s, n = %d", panel, x_value, n))

  if (use_parallel && requireNamespace("foreach", quietly = TRUE) &&
    requireNamespace("doParallel", quietly = TRUE) &&
    requireNamespace("doRNG", quietly = TRUE)) {
    detected_cores <- parallel::detectCores()
    if (!is.finite(detected_cores) || detected_cores < 2L) {
      ncores <- 1L
    } else {
      ncores <- max(1L, min(detected_cores - 1L, N_trials))
    }
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    doParallel::registerDoParallel(cl)
    doRNG::registerDoRNG(20240614)

    alpha_i <- alpha
    J_i <- J
    fssd_nboot_i <- nboot
    ksd_nboot_i <- ksd_nboot
    train_ratio_i <- train_ratio
    opt_maxit_i <- opt_maxit
    opt_locs_bounds_frac_i <- opt_locs_bounds_frac

    exports <- c(
      "run_test_methods", "run_htest_method", "sample_problem", "score_problem",
      "score_std_normal", "rlaplace", "rbm_sample", "rbm_gibbs_step",
      "rbm_sigmoid", "rbm_score", "make_score_function", "lks_test",
      "stein_pair_values"
    )

    `%dorng%` <- doRNG::`%dorng%`
    rows <- foreach::foreach(
      trial = seq_len(N_trials),
      .inorder = TRUE,
      .errorhandling = "pass",
      .export = exports
    ) %dorng% {
      suppressPackageStartupMessages(library(steinsampling))
      run_test_methods(
        problem, n, trial, x_value, panel,
        alpha_i, J_i, fssd_nboot_i, ksd_nboot_i, train_ratio_i,
        opt_maxit_i, opt_locs_bounds_frac_i
      )
    }
  } else {
    set.seed(20240614)
    rows <- lapply(seq_len(N_trials), function(trial) {
      run_test_methods(
        problem, n, trial, x_value, panel,
        alpha, J, nboot, ksd_nboot, train_ratio, opt_maxit,
        opt_locs_bounds_frac
      )
    })
  }

  ok <- vapply(rows, is.data.frame, logical(1))
  if (!all(ok)) {
    warning(sprintf(
      "Dropped %d failed trial jobs for panel %s, x = %s",
      sum(!ok), panel, x_value
    ))
  }
  summarise_trials(do.call(rbind, rows[ok]))
}

run_panel <- function(panel, x_values, make_problem, sample_size_for_x,
                      opt_maxit, opt_locs_bounds_frac) {
  results <- vector("list", length(x_values))
  for (i in seq_along(x_values)) {
    x <- x_values[[i]]
    problem <- make_problem(x, i)
    n <- sample_size_for_x(x)
    results[[i]] <- run_trials_for_problem(
      problem, n, x, panel,
      opt_maxit = opt_maxit,
      opt_locs_bounds_frac = opt_locs_bounds_frac
    )
  }
  do.call(rbind, results)
}

## ------------------------------------------------------------------
## Plot helpers
## ------------------------------------------------------------------

plot_metric_panel <- function(df, y_col, xlab, ylab, main, ylim = NULL) {
  df$method <- factor(df$method, levels = method_order)
  x_values <- sort(unique(df$x_value))
  if (is.null(ylim)) {
    y_max <- max(df[[y_col]], na.rm = TRUE)
    ylim <- c(0, if (is.finite(y_max) && y_max > 0) 1.08 * y_max else 1)
  }

  plot(
    range(x_values), ylim,
    type = "n",
    xlab = xlab,
    ylab = ylab,
    main = main,
    xaxt = "n"
  )
  axis(1, at = x_values, labels = x_values)
  grid(col = "grey90")
  box()

  for (method in method_order) {
    method_df <- df[df$method == method, , drop = FALSE]
    method_df <- method_df[order(method_df$x_value), , drop = FALSE]
    lines(
      method_df$x_value,
      method_df[[y_col]],
      type = "b",
      col = method_col[[method]],
      lty = method_lty[[method]],
      pch = method_pch[[method]],
      lwd = 2
    )
  }
}

write_figure <- function(panel_a, panel_b, panel_c, path) {
  pdf(path, width = 13, height = 4.2)
  on.exit(dev.off(), add = TRUE)
  layout(matrix(c(1, 1, 1, 1, 2, 3, 4, 5), nrow = 2, byrow = TRUE),
    heights = c(0.22, 1)
  )

  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend(
    "center",
    legend = unname(method_labels[method_order]),
    col = unname(method_col[method_order]),
    lty = unname(method_lty[method_order]),
    pch = unname(method_pch[method_order]),
    lwd = 2,
    horiz = TRUE,
    bty = "n",
    cex = 1.1
  )

  par(mar = c(4, 4, 2, 1), mgp = c(2.2, 0.7, 0))
  plot_metric_panel(
    panel_a,
    "rejection_rate",
    xlab = expression(dimension ~ d),
    ylab = "Rejection rate",
    main = "(a) Gaussian vs. Laplace",
    ylim = c(0, 1.02)
  )
  plot_metric_panel(
    panel_b,
    "rejection_rate",
    xlab = expression(Perturbation ~ SD ~ sigma[per]),
    ylab = "Rejection rate",
    main = "(b) RBM, perturb all B",
    ylim = c(0, 1.02)
  )
  plot_metric_panel(
    panel_c,
    "rejection_rate",
    xlab = expression(Sample ~ size ~ n),
    ylab = "Rejection rate",
    main = "(c) RBM, perturb B[1,1]",
    ylim = c(0, 1.02)
  )
  plot_metric_panel(
    panel_c,
    "mean_time_secs",
    xlab = expression(Sample ~ size ~ n),
    ylab = "Time (s)",
    main = "(d) Runtime (RBM)"
  )
}

## ------------------------------------------------------------------
## Run experiments
## ------------------------------------------------------------------

message("Running FSSD Figure 2 rejection-rate demo")
message(sprintf(
  "N_trials = %d, J = %d, FSSD nboot = %d, KSD nboot = %d, alpha = %.3f, train_ratio = %.2f",
  N_trials, J, nboot, ksd_nboot, alpha, train_ratio
))

panel_a <- run_panel(
  "gaussian_laplace",
  glaplace_dims,
  make_problem = function(x, i) make_glaplace_problem(as.integer(x)),
  sample_size_for_x = function(x) glaplace_n,
  opt_maxit = opt_maxit_ex2,
  opt_locs_bounds_frac = opt_locs_bounds_frac_ex2
)

panel_b <- run_panel(
  "rbm_all_B",
  rbm_all_stds,
  make_problem = function(x, i) {
    make_rbm_all_problem(as.numeric(x), i, rbm_dims_x, rbm_dims_h, rbm_burnin)
  },
  sample_size_for_x = function(x) rbm_n_fixed,
  opt_maxit = opt_maxit_ex2,
  opt_locs_bounds_frac = opt_locs_bounds_frac_ex2
)

rbm_single_problem <- make_rbm_single_problem(
  rbm_single_var, rbm_dims_x, rbm_dims_h, rbm_burnin
)
panel_c <- run_panel(
  "rbm_B11_n",
  rbm_ns,
  make_problem = function(x, i) rbm_single_problem,
  sample_size_for_x = function(x) as.integer(x),
  opt_maxit = opt_maxit_ex1,
  opt_locs_bounds_frac = opt_locs_bounds_frac_ex1
)

results <- rbind(panel_a, panel_b, panel_c)
print(results)

write_figure(panel_a, panel_b, panel_c, "fssd_rejectionRate.pdf")
message("Saved fssd_rejectionRate.pdf")
