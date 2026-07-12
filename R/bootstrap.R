# Bootstrap weight generators for KSD U/V tests.

#' Check bootstrap dimensions
#'
#' Converts the sample size and bootstrap count to positive integers before
#' weight matrices are generated.
#'
#' @param n Number of sample rows.
#' @param nboot Number of bootstrap replicates.
#'
#' @return A list with integer entries `n` and `nboot`.
#' @noRd
validate_bootstrap_size <- function(n, nboot) {
  if (exists("validate_positive_integer", mode = "function", inherits = TRUE)) {
    n <- validate_positive_integer(n, "n")
    nboot <- validate_positive_integer(nboot, "nboot")
  } else {
    if (!is.numeric(n) || length(n) != 1 || !is.finite(n) || n < 1) {
      stop("n must be a positive scalar")
    }
    if (!is.numeric(nboot) || length(nboot) != 1 || !is.finite(nboot) || nboot <= 0) {
      stop("nboot must be a positive scalar")
    }
    n <- as.integer(n)
    nboot <- as.integer(nboot)
  }
  list(n = n, nboot = nboot)
}

#' Compute a right-tailed bootstrap p-value
#'
#' Compares an observed statistic with bootstrap statistics from the null
#' distribution. Larger statistic values are treated as more extreme.
#'
#' @details
#' The returned p-value uses the common finite-sample correction from Davison
#' and Hinkley (1997):
#' \deqn{p = \frac{1 + \#\{b: T_b >= T_{\mathrm{obs}}\}}{1 + B},}
#' where `B` is the number of bootstrap statistics.
#'
#' This is a package-wide finite-bootstrap convention, not the literal decision
#' rule in either KSD source paper. Liu et al. Algorithm 1 uses the uncorrected
#' strict exceedance proportion, while Chwialkowski et al. compare the observed
#' statistic with an empirical bootstrap quantile. Their bootstrap statistics
#' are retained by the package; only their finite collection is summarized here
#' using the corrected right-tail p-value.
#'
#' @param boot_stats Numeric vector of bootstrap statistics.
#' @param stat Observed statistic.
#'
#' @return A number between 0 and 1.
#' @noRd
bootstrap_pvalue_right_tail <- function(boot_stats, stat) {
  boot_stats <- as.numeric(boot_stats)
  stat <- as.numeric(stat)[1L]
  nboot <- length(boot_stats)
  if (!nboot) {
    stop("boot_stats must be non-empty")
  }
  (1 + sum(boot_stats >= stat)) / (1 + nboot)
}

# Method 1: centered multinomial bootstrap for KSD U-statistics.

#' Generate multinomial bootstrap weights
#'
#' Draws `n` counts from `n` equally likely categories for each bootstrap
#' replicate, then divides the counts by `n`.
#'
#' @param n Number of sample rows.
#' @param nboot Number of bootstrap replicates.
#'
#' @return Numeric matrix with `n` rows and `nboot` columns.
#' @noRd
generate_multinomial_weights <- function(n, nboot) {
  size <- validate_bootstrap_size(n, nboot)
  stats::rmultinom(
    size$nboot,
    size = size$n,
    prob = rep(1 / size$n, size$n)
  ) / size$n
}

#' Generate centered multinomial bootstrap weights
#'
#' Starts with multinomial weights and subtracts `1 / n` from every entry so
#' each column has mean zero.
#'
#' @param n Number of sample rows.
#' @param nboot Number of bootstrap replicates.
#'
#' @return Numeric matrix with `n` rows and `nboot` columns.
#' @noRd
generate_centered_multinomial_weights <- function(n, nboot) {
  size <- validate_bootstrap_size(n, nboot)
  generate_multinomial_weights(size$n, size$nboot) - (1 / size$n)
}

# Method 2: wild bootstrap sign weights for KSD V-statistics.

#' Generate independent sign weights
#'
#' Draws Rademacher weights, meaning each entry is independently `-1` or `1`
#' with equal probability.
#'
#' @param n Number of sample rows.
#' @param nboot Number of bootstrap replicates.
#'
#' @return Numeric matrix with `n` rows and `nboot` columns.
#' @noRd
generate_rademacher_weights <- function(n, nboot) {
  size <- validate_bootstrap_size(n, nboot)
  matrix(
    sample(c(-1, 1), size$n * size$nboot, replace = TRUE),
    nrow = size$n,
    ncol = size$nboot
  )
}

#' Simulate one Markov sign sequence
#'
#' Creates a sequence of signs that starts at `1`. At each later position, the
#' sign flips with probability `p_change` and otherwise stays the same.
#'
#' @param n Length of the sign sequence.
#' @param p_change Probability of changing sign between neighboring positions.
#'
#' @return Numeric vector of length `n` containing `-1` and `1`.
#' @noRd
simulatepm <- function(n, p_change) {
  if (!is.numeric(n) || length(n) != 1 || !is.finite(n) || n < 2) {
    stop("n must be a numeric scalar >= 2")
  }
  if (!is.numeric(p_change) || length(p_change) != 1 ||
    !is.finite(p_change) || p_change <= 0 || p_change >= 1) {
    stop("p_change must be a scalar in (0, 1)")
  }

  n <- as.integer(n)
  out <- numeric(n)
  out[1] <- 1

  flips <- stats::runif(n - 1) < p_change
  for (i in 2:n) {
    out[i] <- if (flips[i - 1]) -out[i - 1] else out[i - 1]
  }
  out
}

#' Generate Markov sign weights
#'
#' Repeats `simulatepm()` to produce one Markov sign sequence per bootstrap
#' replicate. These weights are useful when the input sample has an ordering and
#' adjacent signs should be allowed to be correlated.
#'
#' @param n Number of sample rows.
#' @param nboot Number of bootstrap replicates.
#' @param p_change Probability of changing sign between neighboring rows.
#'
#' @return Numeric matrix with `n` rows and `nboot` columns.
#' @noRd
generate_markov_weights <- function(n, nboot, p_change) {
  size <- validate_bootstrap_size(n, nboot)
  if (size$n < 2) {
    stop("n must be a numeric scalar >= 2 for Markov bootstrap")
  }
  if (!is.numeric(p_change) || length(p_change) != 1 ||
    !is.finite(p_change) || p_change <= 0 || p_change >= 1) {
    stop("p_change must be a scalar in (0, 1)")
  }

  replicate(size$nboot, simulatepm(size$n, p_change))
}

#' Check a Markov sign-flip probability
#'
#' Checks that `change_prob` is a single finite number strictly between 0 and 1.
#'
#' @param change_prob Candidate sign-flip probability.
#'
#' @return A list with entry `change_prob`.
#' @noRd
resolve_markov_change_prob <- function(change_prob) {
  if (is.null(change_prob)) {
    stop("change_prob must be supplied explicitly for the Markov bootstrap and be numeric in (0, 1)")
  }

  if (is.character(change_prob)) {
    stop("change_prob must be numeric in (0, 1)")
  }

  if (!is.numeric(change_prob) || length(change_prob) != 1 ||
    !is.finite(change_prob) || change_prob <= 0 || change_prob >= 1) {
    stop("change_prob must be numeric in (0, 1)")
  }

  list(
    change_prob = as.numeric(change_prob)
  )
}

#' Resolve the sign-flip probability used by a bootstrap method
#'
#' Returns `NA` unless the selected bootstrap method is `"markov"`. For Markov
#' weights it checks and returns the requested sign-flip probability.
#'
#' @param boot_method Bootstrap method name.
#' @param change_prob Candidate sign-flip probability.
#'
#' @return A list with entry `change_prob`.
#' @noRd
resolve_change_probability_for_bootstrap <- function(boot_method, change_prob) {
  if (!identical(boot_method, "markov")) {
    return(list(
      change_prob = NA_real_
    ))
  }

  resolve_markov_change_prob(change_prob)
}

#' Generate bootstrap weights for KSD tests
#'
#' Returns an `n x nboot` matrix of weights. U-statistics use centered
#' multinomial weights; V-statistics use independent or Markov sign weights.
#'
#' @details
#' The available methods are:
#' `"multinomial_centered"`, which is used for the KSD U-statistic;
#' `"rademacher"`, which uses independent `-1` and `1` signs; and
#' `"markov"`, which uses signs that change with probability `change_prob`.
#' The older name `"weighted"` is accepted as an alias for
#' `"multinomial_centered"`.
#'
#' @param n Number of sample rows.
#' @param nboot Number of bootstrap replicates.
#' @param boot_method Bootstrap method name.
#' @param change_prob Markov sign-flip probability, used only when
#'   `boot_method = "markov"`.
#' @param method Deprecated alias for `boot_method`.
#'
#' @return Numeric matrix with `n` rows and `nboot` columns.
#' @noRd
generate_bootstrap_weights <- function(n, nboot, boot_method = NULL,
                                       change_prob = NULL, method = NULL) {
  if (!is.null(method)) {
    if (!is.null(boot_method) && !identical(boot_method, method)) {
      stop("Specify only one of boot_method or method")
    }
    boot_method <- method
  }
  if (is.null(boot_method) || !is.character(boot_method) || length(boot_method) != 1) {
    stop("boot_method must be a single character string")
  }

  if (identical(boot_method, "weighted")) {
    boot_method <- "multinomial_centered"
  }

  switch(boot_method,
    multinomial_centered = generate_centered_multinomial_weights(n, nboot),
    rademacher = generate_rademacher_weights(n, nboot),
    markov = generate_markov_weights(n, nboot, change_prob),
    stop("Unknown boot_method")
  )
}
