# Internal bootstrap support for the KSD tests.
#
# `ksd_u_test()` and `ksd_u_bootstrap()` use centered multinomial weights.
# `ksd_v_test()` and `ksd_v_bootstrap()` use independent Rademacher or
# dependent Markov sign sequences. The test wrappers also use the common
# right-tail p-value helper below. This file has no exported functions.

# Return the corrected KSD right-tail p-value
# (1 + sum(T_b >= T_obs)) / (B + 1). FSSD uses a separate uncorrected null
# simulation and does not call this helper.
.bootstrap_pvalue_right_tail <- function(boot_stats, stat) {
  boot_stats <- as.numeric(boot_stats)
  stat <- as.numeric(stat)[1L]
  nboot <- length(boot_stats)
  (1 + sum(boot_stats >= stat)) / (1 + nboot)
}

# Centered multinomial bootstrap for KSD-U: N_i / n - 1 / n.
.generate_centered_multinomial_weights <- function(n, nboot) {
  stats::rmultinom(
    nboot,
    size = n,
    prob = rep(1 / n, n)
  ) / n - (1 / n)
}

# Wild-bootstrap signs for KSD-V.
# Draw independent Rademacher signs, one replicate per column.
.generate_rademacher_weights <- function(n, nboot) {
  matrix(
    sample(c(-1, 1), n * nboot, replace = TRUE),
    nrow = n,
    ncol = nboot
  )
}

# Simulate one sign sequence that starts at 1 and flips with probability
# `change_prob` between adjacent observations. Its caller validates both inputs.
.simulate_markov_signs <- function(n, change_prob) {
  out <- numeric(n)
  out[1] <- 1

  flips <- stats::runif(n - 1) < change_prob
  for (i in 2:n) {
    out[i] <- if (flips[i - 1]) -out[i - 1] else out[i - 1]
  }
  out
}

.generate_markov_weights <- function(n, nboot, change_prob) {
  replicate(nboot, .simulate_markov_signs(n, change_prob))
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

.generate_bootstrap_weights <- function(n, nboot, boot_method = NULL,
                                        change_prob = NULL) {
  n <- validate_integer(n, "n")
  nboot <- validate_integer(nboot, "nboot")
  if (is.null(boot_method) || !is.character(boot_method) ||
      length(boot_method) != 1L) {
    stop("`boot_method` must be a single character string", call. = FALSE)
  }

  change_prob <- if (identical(boot_method, "markov")) {
    .resolve_markov_change_prob(change_prob)
  } else {
    NA_real_
  }

  switch(boot_method,
    multinomial_centered = .generate_centered_multinomial_weights(
      n, nboot
    ),
    rademacher = .generate_rademacher_weights(n, nboot),
    markov = .generate_markov_weights(n, nboot, change_prob),
    stop("unknown `boot_method`", call. = FALSE)
  )
}
