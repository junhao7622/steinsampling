## KSD-U one-dimensional Gaussian mixture experiments
##
## Compares several goodness-of-fit tests on perturbed Gaussian mixtures and
## saves error-rate and ROC plots.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

set.seed(123)

alpha <- 0.05
# Full replication: n_default <- 100
n_default <- 50
# Full replication: N_trials <- 1000
N_trials <- 10
# Full replication: nboot <- 1000
nboot <- 30
# Full replication: mmd_boot <- 1000
mmd_boot <- 30

METHOD_NAMES <- c("ksd_u", "ks", "lr", "chisq", "cvm", "mmd")

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

# Full replication: n_cores <- max(1L, parallel::detectCores() - 1L)
n_cores <- 1L
cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)
doRNG::registerDoRNG(123)

## ------------------------------------------------------------------
## GMM helpers
## ------------------------------------------------------------------

init_model_p <- function(nComp = 5, d = 1) {
  mu <- runif(nComp, min = 0, max = 10)
  sigma <- array(1, dim = c(d, d, nComp))
  weights <- rep(1 / nComp, nComp)
  gmm(nComp = nComp, mu = mu, sigma = sigma, weights = weights, d = d)
}

custom_perturb_gmm <- function(model,
                               type = c("mean", "variance", "weights"),
                               sigma_per = 1) {
  type <- match.arg(type)
  k <- model$nComp
  d <- model$d
  mu <- model$mu
  sigma <- model$sigma
  weights <- model$weights

  if (type == "mean") {
    if (d == 1) {
      mu <- as.numeric(mu) + rnorm(k, sd = sigma_per)
    } else {
      mu <- mu + matrix(rnorm(d * k, sd = sigma_per), nrow = d)
    }
  } else if (type == "variance") {
    noise <- rnorm(1, sd = sigma_per)
    for (i in seq_len(k)) {
      var_i <- as.numeric(sigma[, , i])
      var_new <- exp(log(var_i) + noise)
      sigma[, , i] <- diag(var_new, d)
    }
  } else { # weights
    # Paper:  noise added to each log w_k independently  ->  k draws.
    logw <- log(as.numeric(weights)) + rnorm(k, sd = sigma_per)
    w_new <- exp(logw)
    weights <- w_new / sum(w_new)
  }

  gmm(nComp = k, mu = mu, sigma = sigma, weights = weights, d = d)
}

## ------------------------------------------------------------------
## CDF / quantile helpers for GMM (needed by chi-sq and CvM tests)
## ------------------------------------------------------------------

pgmm <- function(x, model) {
  x_vec <- as.numeric(x)
  k <- model$nComp
  mu <- as.numeric(model$mu)
  wts <- as.numeric(model$weights)
  stdev <- vapply(seq_len(k), function(i) sqrt(model$sigma[, , i]), numeric(1))

  cdf_mat <- vapply(
    seq_len(k), function(i) {
      stats::pnorm(x_vec, mean = mu[i], sd = stdev[i])
    },
    numeric(length(x_vec))
  )

  as.numeric(cdf_mat %*% wts)
}

gmm_support_range <- function(model, expand = 8) {
  mu <- as.numeric(model$mu)
  stdev <- vapply(seq_len(model$nComp), function(i) sqrt(model$sigma[, , i]), numeric(1))
  c(min(mu - expand * stdev), max(mu + expand * stdev))
}

gmm_quantile <- function(p, model) {
  if (p <= 0) {
    return(-Inf)
  }
  if (p >= 1) {
    return(Inf)
  }
  bounds <- gmm_support_range(model, expand = 10)
  f <- function(x) pgmm(x, model) - p
  out <- tryCatch(stats::uniroot(f, lower = bounds[1], upper = bounds[2])$root,
    error = function(e) NA_real_
  )
  if (!is.finite(out)) {
    out <- stats::qnorm(p, mean = mean(bounds), sd = diff(bounds) / 6)
  }
  out
}

make_equal_prob_bins <- function(model, n_bins) {
  probs <- seq(0, 1, length.out = n_bins + 1)
  bounds <- vapply(probs, gmm_quantile, model = model, numeric(1))
  bounds[1] <- -Inf
  bounds[length(bounds)] <- Inf
  bounds
}

## ------------------------------------------------------------------
## Individual GOF tests
## ------------------------------------------------------------------

chisq_gof <- function(X, model, n_bins = NULL) {
  n <- length(X)
  if (is.null(n_bins)) n_bins <- max(10, floor(sqrt(n)))
  breaks <- make_equal_prob_bins(model, n_bins)
  obs <- hist(X, breaks = breaks, plot = FALSE)$counts
  probs <- diff(pgmm(breaks, model))
  probs <- pmax(probs, 1e-10)
  probs <- probs / sum(probs)
  stats::chisq.test(x = obs, p = probs, rescale.p = TRUE)
}

cvm_gof <- function(X, model) {
  if (requireNamespace("goftest", quietly = TRUE)) {
    return(goftest::cvm.test(X, null = function(z) pgmm(z, model)))
  }
  if (requireNamespace("dgof", quietly = TRUE)) {
    return(dgof::cvm.test(X, null = function(z) pgmm(z, model)))
  }
  stop("Package 'goftest' or 'dgof' is required for CvM test.")
}

## ------------------------------------------------------------------
## MMD goodness-of-fit test
## ------------------------------------------------------------------

mmd_test <- function(x, y, nboot = 200) {
  if (is.null(dim(x))) x <- matrix(as.numeric(x), ncol = 1)
  if (is.null(dim(y))) y <- matrix(as.numeric(y), ncol = 1)

  m <- nrow(x)
  n <- nrow(y)
  combined <- rbind(x, y)

  sq_med <- find_median_distance(combined, max_samples = 2000, use_sampling = TRUE)

  kernel_obj <- stein_kernel(type = "gaussian_rbf", h = sqrt(sq_med))
  Kz <- eval_kernel(kernel_obj, combined)

  calc_mmd_v <- function(idx_x, idx_y) {
    sum(Kz[idx_x, idx_x]) / (m^2) +
      sum(Kz[idx_y, idx_y]) / (n^2) -
      2 * sum(Kz[idx_x, idx_y]) / (m * n)
  }

  idx_x_true <- seq_len(m)
  idx_y_true <- (m + 1):(m + n)
  testStat <- calc_mmd_v(idx_x_true, idx_y_true)

  boot_stats <- vapply(seq_len(nboot), function(b) {
    indShuff <- sample.int(m + n, replace = FALSE)
    calc_mmd_v(indShuff[seq_len(m)], indShuff[(m + 1):(m + n)])
  }, numeric(1))

  list(statistic = testStat, p.value = mean(boot_stats >= testStat))
}

## ------------------------------------------------------------------
## Run one goodness-of-fit trial
## ------------------------------------------------------------------

run_gof_tests <- function(X, model_p, model_q, alpha, nboot, mmd_boot) {
  empty <- setNames(rep(NA_real_, length(METHOD_NAMES)), METHOD_NAMES)
  decisions <- setNames(rep(NA, length(METHOD_NAMES)), METHOD_NAMES)
  scores <- empty
  pvalues <- empty

  decisions["ksd_u"] <- tryCatch(
    {
      grad_log_prob <- get_score_evaluator(model_q)
      res <- ksd_u_test(
        X              = X,
        score_function = grad_log_prob,
        nboot          = nboot
      )
      pvalues["ksd_u"] <- res$p.value
      scores["ksd_u"] <- res$statistic
      isTRUE(res$p.value < alpha)
    },
    error = function(e) NA
  )

  ## KS test
  decisions["ks"] <- tryCatch(
    {
      res <- stats::ks.test(X, function(z) pgmm(z, model_q))
      pvalues["ks"] <- res$p.value
      scores["ks"] <- res$statistic
      isTRUE(res$p.value < alpha)
    },
    error = function(e) NA
  )

  ## Likelihood ratio
  decisions["lr"] <- tryCatch(
    {
      loglik_p <- sum(log(likelihoodgmm(model_p, X)))
      loglik_q <- sum(log(likelihoodgmm(model_q, X)))
      scores["lr"] <- 2 * (loglik_p - loglik_q)
      isTRUE(scores["lr"] > 0)
    },
    error = function(e) NA
  )

  ## Chi-squared
  decisions["chisq"] <- tryCatch(
    {
      res <- chisq_gof(X, model_q)
      pvalues["chisq"] <- res$p.value
      scores["chisq"] <- res$statistic
      isTRUE(res$p.value < alpha)
    },
    error = function(e) NA
  )

  ## CvM
  decisions["cvm"] <- tryCatch(
    {
      res <- cvm_gof(X, model_q)
      pvalues["cvm"] <- res$p.value
      scores["cvm"] <- res$statistic
      isTRUE(res$p.value < alpha)
    },
    error = function(e) NA
  )

  ## MMD Monte Carlo test
  decisions["mmd"] <- tryCatch(
    {
      Y <- rgmm(model_q, n = 1000)
      res <- mmd_test(X, Y, nboot = mmd_boot)
      pvalues["mmd"] <- res$p.value
      scores["mmd"] <- res$statistic
      isTRUE(res$p.value < alpha)
    },
    error = function(e) NA
  )

  list(decisions = decisions, scores = scores, pvalues = pvalues)
}

## ------------------------------------------------------------------
## Collect error rates
## ------------------------------------------------------------------

collect_error_rates <- function(trial_results, is_null,
                                method_names = METHOD_NAMES) {
  type1 <- setNames(rep(0L, length(method_names)), method_names)
  type2 <- setNames(rep(0L, length(method_names)), method_names)
  n1 <- setNames(rep(0L, length(method_names)), method_names)
  n2 <- setNames(rep(0L, length(method_names)), method_names)

  for (i in seq_along(is_null)) {
    dec_i <- trial_results$decisions[[i]]

    if (is.null(dec_i) || !is.vector(dec_i) || is.null(names(dec_i))) next

    for (m in method_names) {
      dec <- dec_i[m]
      if (is.na(dec)) next
      if (is_null[i]) {
        type1[m] <- type1[m] + as.integer(isTRUE(dec))
        n1[m] <- n1[m] + 1L
      } else {
        type2[m] <- type2[m] + as.integer(!isTRUE(dec))
        n2[m] <- n2[m] + 1L
      }
    }
  }

  data.frame(
    method = method_names,
    type1 = ifelse(n1 > 0, type1 / n1, NA_real_),
    type2 = ifelse(n2 > 0, type2 / n2, NA_real_),
    error = 0.5 * (ifelse(n1 > 0, type1 / n1, NA_real_) +
      ifelse(n2 > 0, type2 / n2, NA_real_)),
    stringsAsFactors = FALSE
  )
}

## ------------------------------------------------------------------
## Filter failed trial jobs
## ------------------------------------------------------------------

filter_valid_trials <- function(trial_out, label = "") {
  ok <- vapply(
    trial_out,
    function(x) is.list(x) && length(x) > 0,
    logical(1)
  )
  n_fail <- sum(!ok)
  if (n_fail > 0) {
    message(sprintf(
      "[%s] %d/%d trials failed and were dropped.",
      label, n_fail, length(trial_out)
    ))
  }
  trial_out[ok]
}

## ------------------------------------------------------------------
## Plotting helper
## ------------------------------------------------------------------

plot_error_curves <- function(results, x_var, title_text, xlab_text) {
  methods <- unique(results$method)
  cols <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d")
  pchs <- c(16, 17, 15, 18, 19, 8)
  x_ticks <- NULL
  x_labels <- NULL
  if (identical(x_var, "sigma_per")) {
    x_ticks <- c(0.1, 10^(-0.5), 1, 10^(0.5), 10)
    x_labels <- c("0.1", "0.3", "1", "3", "10")
  } else if (identical(x_var, "n")) {
    x_ticks <- c(50, 100, 500, 1000)
    x_labels <- c("50", "100", "500", "1000")
  }

  plot(results[[x_var]], results$error,
    type = "n", xlab = xlab_text,
    log = "x", xaxt = "n",
    ylab = "Total error rate", main = title_text
  )
  if (!is.null(x_ticks)) {
    axis(1, at = x_ticks, labels = x_labels)
  }

  for (i in seq_along(methods)) {
    sub <- results[results$method == methods[i], ]
    lines(sub[[x_var]], sub$error,
      type = "b", pch = pchs[i], col = cols[i], lwd = 2
    )
  }
  legend_labels <- ifelse(methods == "mmd", "MMD-MC(1000)", methods)
  legend("topright",
    legend = legend_labels,
    col = cols[seq_along(methods)],
    pch = pchs[seq_along(methods)], lwd = 2, bty = "n"
  )
}

## ------------------------------------------------------------------
## Experiment 1: error vs perturbation magnitude
## ------------------------------------------------------------------

run_sigma_experiment <- function(type, sigma_grid, n, N_trials,
                                 alpha, nboot, mmd_boot) {
  all_rows <- list()

  for (sigma_per in sigma_grid) {
    trial_out <- foreach(
      trial          = seq_len(N_trials),
      .inorder       = TRUE,
      .export        = ls(globalenv()),
      .errorhandling = "pass"
    ) %dorng% {
      model_p <- init_model_p(nComp = 5, d = 1)
      X <- rgmm(model_p, n = n)
      is_null <- rbinom(1, 1, 0.5) == 1
      model_q <- if (is_null) {
        model_p
      } else {
        custom_perturb_gmm(model_p, type = type, sigma_per = sigma_per)
      }

      res <- run_gof_tests(X, model_p, model_q,
        alpha = alpha, nboot = nboot, mmd_boot = mmd_boot
      )
      list(decisions = res$decisions, is_null = is_null)
    }

    label <- sprintf("sigma=%.3f, type=%s", sigma_per, type)
    trial_out <- filter_valid_trials(trial_out, label)
    if (length(trial_out) == 0) {
      warning(sprintf("All trials failed for %s - skipping.", label))
      next
    }

    decisions_list <- lapply(trial_out, `[[`, "decisions")
    is_null_vec <- vapply(trial_out, `[[`, logical(1), "is_null")

    rates <- collect_error_rates(
      list(decisions = decisions_list),
      is_null_vec
    )
    rates$type <- type
    rates$sigma_per <- sigma_per
    all_rows[[length(all_rows) + 1]] <- rates
  }

  do.call(rbind, all_rows)
}

## ------------------------------------------------------------------
## Experiment 2: error vs sample size
## ------------------------------------------------------------------

run_sample_size_experiment <- function(type, n_grid, sigma_per, N_trials,
                                       alpha, nboot, mmd_boot) {
  all_rows <- list()

  for (n in n_grid) {
    trial_out <- foreach(
      trial          = seq_len(N_trials),
      .inorder       = TRUE,
      .export        = ls(globalenv()),
      .errorhandling = "pass"
    ) %dorng% {
      model_p <- init_model_p(nComp = 5, d = 1)
      X <- rgmm(model_p, n = n)
      is_null <- rbinom(1, 1, 0.5) == 1
      model_q <- if (is_null) {
        model_p
      } else {
        custom_perturb_gmm(model_p, type = type, sigma_per = sigma_per)
      }

      res <- run_gof_tests(X, model_p, model_q,
        alpha = alpha, nboot = nboot, mmd_boot = mmd_boot
      )
      list(decisions = res$decisions, is_null = is_null)
    }

    label <- sprintf("n=%d, type=%s", n, type)
    trial_out <- filter_valid_trials(trial_out, label)
    if (length(trial_out) == 0) {
      warning(sprintf("All trials failed for %s - skipping.", label))
      next
    }

    decisions_list <- lapply(trial_out, `[[`, "decisions")
    is_null_vec <- vapply(trial_out, `[[`, logical(1), "is_null")

    rates <- collect_error_rates(
      list(decisions = decisions_list),
      is_null_vec
    )
    rates$type <- type
    rates$n <- n
    all_rows[[length(all_rows) + 1]] <- rates
  }

  do.call(rbind, all_rows)
}

## ------------------------------------------------------------------
## Experiment 3: ROC curves
## ------------------------------------------------------------------

run_roc_experiment <- function(type, n, sigma_per, N_trials,
                               alpha, nboot, mmd_boot) {
  require_or_stop("pROC")

  method_names <- METHOD_NAMES

  trial_out <- foreach(
    trial          = seq_len(N_trials),
    .inorder       = TRUE,
    .export        = ls(globalenv()),
    .errorhandling = "pass"
  ) %dorng% {
    model_p <- init_model_p(nComp = 5, d = 1)
    X <- rgmm(model_p, n = n)
    is_null <- rbinom(1, 1, 0.5) == 1
    model_q <- if (is_null) {
      model_p
    } else {
      custom_perturb_gmm(model_p, type = type, sigma_per = sigma_per)
    }

    res <- run_gof_tests(X, model_p, model_q,
      alpha = alpha, nboot = nboot, mmd_boot = mmd_boot
    )
    list(label = if (is_null) 0L else 1L, scores = res$scores)
  }

  label <- sprintf("ROC type=%s", type)
  trial_out <- filter_valid_trials(trial_out, label)
  if (length(trial_out) == 0) stop("All ROC trials failed.")

  labels <- vapply(trial_out, `[[`, integer(1), "label")
  scores <- lapply(method_names, function(m) {
    vapply(trial_out, function(z) {
      v <- z$scores[m]
      if (is.null(v) || length(v) == 0) NA_real_ else as.numeric(v)
    }, numeric(1))
  })
  names(scores) <- method_names

  cols <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d")
  roc_list <- list()
  for (m in method_names) {
    valid <- is.finite(scores[[m]])
    if (sum(valid) < 2) next
    roc_list[[m]] <- pROC::roc(labels[valid], scores[[m]][valid],
      direction = "<", quiet = TRUE
    )
  }

  if (length(roc_list) == 0) stop("No valid ROC curves to plot.")
  plot(roc_list[[1]],
    col = cols[1], lwd = 2,
    legacy.axes = TRUE,
    xlab = "False Positive Rate", ylab = "True Positive Rate",
    main = paste0("ROC curves (1D GMM, ", type, " perturbation)")
  )
  if (length(roc_list) > 1) {
    for (i in 2:length(roc_list)) {
      plot(roc_list[[i]], col = cols[i], lwd = 2, add = TRUE, legacy.axes = TRUE)
    }
  }

  auc_text <- vapply(
    roc_list,
    function(r) sprintf("%.3f", pROC::auc(r)), character(1)
  )
  legend("bottomright",
    legend = paste0(names(roc_list), " (AUC=", auc_text, ")"),
    col = cols[seq_along(roc_list)], lwd = 2, bty = "n"
  )
}

## ------------------------------------------------------------------
## Run experiments
## ------------------------------------------------------------------

pdf("ksd_u_1dGaussianMixture.pdf", width = 7, height = 5, onefile = TRUE)

# Full replication: sigma_grid <- exp(seq(log(0.1), log(10), length.out = 10))
sigma_grid <- exp(seq(log(0.1), log(10), length.out = 3))
perturb_types <- c("mean", "variance", "weights")

## Experiment 1: error vs perturbation size
results_sigma <- list()
for (ptype in perturb_types) {
  results_sigma[[ptype]] <- run_sigma_experiment(
    type = ptype,
    sigma_grid = sigma_grid,
    n = n_default,
    N_trials = N_trials,
    alpha = alpha,
    nboot = nboot,
    mmd_boot = mmd_boot
  )
  plot_error_curves(
    results_sigma[[ptype]],
    x_var      = "sigma_per",
    title_text = paste0("Error vs sigma_per (", ptype, " perturbation)"),
    xlab_text  = "Perturbation magnitude (sigma_per)"
  )
}

## Experiment 2: error vs sample size
# Full replication: n_grid <- c(50, 100, 500, 1000)
n_grid <- c(50, 100)
sigma_per_fixed <- 1

results_n <- list()
for (ptype in perturb_types) {
  results_n[[ptype]] <- run_sample_size_experiment(
    type      = ptype,
    n_grid    = n_grid,
    sigma_per = sigma_per_fixed,
    N_trials  = N_trials,
    alpha     = alpha,
    nboot     = nboot,
    mmd_boot  = mmd_boot
  )
  plot_error_curves(
    results_n[[ptype]],
    x_var      = "n",
    title_text = paste0("Error vs n (", ptype, " perturbation)"),
    xlab_text  = "Sample size (n)"
  )
}

## Experiment 3: ROC curves
run_roc_experiment(
  type      = "mean",
  n         = n_default,
  sigma_per = 1,
  N_trials  = N_trials,
  alpha     = alpha,
  nboot     = nboot,
  mmd_boot  = mmd_boot
)

dev.off()
parallel::stopCluster(cl)
