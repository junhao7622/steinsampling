# Bootstrap weight generators for KSD U/V tests.

#' @keywords internal
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

#' Right-tailed bootstrap p-value with Davison & Hinkley (1997) (1 + B_ge)/(B+1).
#'
#' @keywords internal
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

#' @keywords internal
generate_multinomial_weights <- function(n, nboot) {
  size <- validate_bootstrap_size(n, nboot)
  stats::rmultinom(
    size$nboot,
    size = size$n,
    prob = rep(1 / size$n, size$n)
  ) / size$n
}

#' @keywords internal
generate_centered_multinomial_weights <- function(n, nboot) {
  size <- validate_bootstrap_size(n, nboot)
  generate_multinomial_weights(size$n, size$nboot) - (1 / size$n)
}

# Method 2: wild bootstrap sign weights for KSD V-statistics.

#' @keywords internal
generate_rademacher_weights <- function(n, nboot) {
  size <- validate_bootstrap_size(n, nboot)
  matrix(
    sample(c(-1, 1), size$n * size$nboot, replace = TRUE),
    nrow = size$n,
    ncol = size$nboot
  )
}

#' @keywords internal
simulatepm <- function(n, p_change) {
  if (!is.numeric(n) || length(n) != 1 || !is.finite(n) || n < 2) {
    stop("n must be a numeric scalar >= 2")
  }
  if (!is.numeric(p_change) || length(p_change) != 1 ||
      !is.finite(p_change) || p_change < 0 || p_change > 1) {
    stop("p_change must be a scalar in [0, 1]")
  }

  n <- as.integer(n)
  out <- numeric(n)
  out[1] <- 1
  if (n == 1) {
    return(out)
  }

  flips <- stats::runif(n - 1) < p_change
  for (i in 2:n) {
    out[i] <- if (flips[i - 1]) -out[i - 1] else out[i - 1]
  }
  out
}

#' @keywords internal
generate_markov_weights <- function(n, nboot, p_change) {
  size <- validate_bootstrap_size(n, nboot)
  if (size$n < 2) {
    stop("n must be a numeric scalar >= 2 for Markov bootstrap")
  }
  if (!is.numeric(p_change) || length(p_change) != 1 ||
      !is.finite(p_change) || p_change < 0 || p_change > 1) {
    stop("p_change must be a scalar in [0, 1]")
  }

  replicate(size$nboot, simulatepm(size$n, p_change))
}

#' Resolve Markov sign-flip probability.
#' @keywords internal
resolve_markov_change_prob <- function(change_prob) {
  if (is.null(change_prob)) {
    stop("change_prob must be numeric in [0, 1]")
  }

  if (is.character(change_prob)) {
    stop("change_prob must be numeric in [0, 1]")
  }

  if (!is.numeric(change_prob) || length(change_prob) != 1 ||
      !is.finite(change_prob) || change_prob < 0 || change_prob > 1) {
    stop("change_prob must be numeric in [0, 1]")
  }

  list(
    change_prob = as.numeric(change_prob)
  )
}

#' @keywords internal
resolve_change_probability_for_bootstrap <- function(boot_method, change_prob) {
  if (!identical(boot_method, "markov")) {
    return(list(
      change_prob = NA_real_
    ))
  }

  resolve_markov_change_prob(change_prob)
}

#' Unified bootstrap weight entry point.
#'
#' Returns an n x nboot matrix. For U-statistics, "multinomial_centered"
#' returns multinomial frequencies centered by 1/n. For V-statistics,
#' "rademacher" and "markov" return wild-bootstrap signs.
#' @keywords internal
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

  switch(
    boot_method,
    multinomial_centered = generate_centered_multinomial_weights(n, nboot),
    rademacher = generate_rademacher_weights(n, nboot),
    markov = generate_markov_weights(n, nboot, change_prob),
    stop("Unknown boot_method")
  )
}
