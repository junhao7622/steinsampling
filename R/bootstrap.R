# Internal bootstrap support for the KSD tests.

#
# `ksd_u_test()` and `ksd_u_bootstrap()` use centered multinomial weights.
# `ksd_v_test()` and `ksd_v_bootstrap()` use independent Rademacher or
# dependent Markov sign sequences. The test wrappers also use the common
# right-tail p-value helper below.

# Return the corrected KSD right-tail p-value
# (1 + sum(T_b >= T_obs)) / (B + 1). FSSD uses a separate uncorrected null
# simulation and does not call this helper.
.bootstrap_pvalue_right_tail <- function(boot_stats, stat) {
  boot_stats <- as.numeric(boot_stats)
  stat <- as.numeric(stat)[1L]
  nboot <- length(boot_stats)
  (1 + sum(boot_stats >= stat)) / (1 + nboot)
}

.resolve_markov_change_prob <- function(change_prob) {
  if (is.null(change_prob))
    stop("`change_prob` must be supplied explicitly for Markov bootstrap", call. = FALSE)
  if (!is.numeric(change_prob) || length(change_prob) != 1L ||
    !is.finite(change_prob) || change_prob <= 0 || change_prob >= 1) {
    stop("`change_prob` must be numeric and lie in (0, 1)", call. = FALSE)
  }

  as.numeric(change_prob)
}

# Columns are independent bootstrap replicates. For Markov weights, cumprod()
# turns adjacent flip indicators into a sign sequence starting at one.
.generate_bootstrap_weights <- function(n, nboot, boot_method,
                                        change_prob = NULL) {
  n <- validate_integer(n, "n")
  nboot <- validate_integer(nboot, "nboot")
  if (identical(boot_method, "markov")) {
    change_prob <- .resolve_markov_change_prob(change_prob)
  }

  switch(boot_method,
    multinomial_centered =
      stats::rmultinom(nboot, size = n, prob = rep(1 / n, n)) / n - (1 / n),
    rademacher =
      matrix(sample(c(-1, 1), n * nboot, replace = TRUE), nrow = n, ncol = nboot),
    markov =
      replicate(nboot, cumprod(c(1, 1 - 2 * (stats::runif(n - 1) < change_prob)))),
    stop("unknown `boot_method`", call. = FALSE)
  )
}

# Supplied multipliers may come from families other than the built-in methods.
.validate_bootstrap_weights <- function(W_mat, n) {
  W_mat <- as.matrix(W_mat)
  if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1 ||
    any(!is.finite(W_mat))) {
    stop("W_mat must be a finite numeric matrix with nrow(K0) rows and at least one column",
         call. = FALSE)
  }
  W_mat
}
