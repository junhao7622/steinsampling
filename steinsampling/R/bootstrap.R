# Bootstrap weights and the right-tail p-value for the KSD tests. KSD-U uses
# centered multinomial weights and KSD-V uses Rademacher or Markov signs.

# Count the observed statistic as one more draw, so the p-value is never 0. FSSD
# simulates its null without this correction and does not use this helper.
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

# Each column is one bootstrap replicate. For Markov weights, cumprod turns the
# flip indicators into a sign sequence that starts at 1.
.generate_bootstrap_weights <- function(n, nboot, boot_method,
                                        change_prob = NULL) {
  n <- validate_integer(n, "n")
  nboot <- validate_integer(nboot, "nboot")
  if (identical(boot_method, "markov")) {
    change_prob <- .resolve_markov_change_prob(change_prob)
  }

  switch(boot_method,
    # Draw n cell indices for each column with sample.int and count them. The
    # counts are multinomial with equal cell probabilities. The offset sends
    # each block of n draws to its own column.
    multinomial_centered = {
      cells <- as.double(n) * nboot
      draws <- sample.int(n, cells, replace = TRUE) +
        rep(seq.int(0, by = n, length.out = nboot), each = n)
      matrix(tabulate(draws, nbins = cells), nrow = n) / n - (1 / n)
    },
    rademacher =
      matrix(sample(c(-1, 1), n * nboot, replace = TRUE), nrow = n, ncol = nboot),
    markov =
      replicate(nboot, cumprod(c(1, 1 - 2 * (stats::runif(n - 1) < change_prob)))),
    stop("unknown `boot_method`", call. = FALSE)
  )
}

.validate_bootstrap_weights <- function(W_mat, n) {
  W_mat <- as.matrix(W_mat)
  if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1 ||
    any(!is.finite(W_mat))) {
    stop("W_mat must be a finite numeric matrix with nrow(K0) rows and at least one column",
         call. = FALSE)
  }
  W_mat
}
