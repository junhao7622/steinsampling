# Shared helper functions used across the package.


#' Check sample input and return a numeric matrix
#'
#' Converts vectors, matrices, and data frames into the common sample format
#' used throughout the package: one observation per row and one variable per
#' column.
#'
#' @param x Samples supplied as a numeric vector, matrix, or data frame.
#' @param min_rows Smallest allowed number of sample rows.
#'
#' @return A numeric matrix with samples in rows and variables in columns.
#' @noRd
validate_samples <- function(x, min_rows = 2) {
  if (!is.numeric(min_rows) || length(min_rows) != 1 || !is.finite(min_rows) ||
    min_rows < 1 || abs(min_rows - round(min_rows)) > sqrt(.Machine$double.eps)) {
    stop("min_rows must be a positive integer")
  }
  min_rows <- as.integer(round(min_rows))

  if (is.null(dim(x))) {
    if (!is.numeric(x)) stop("X must be numeric")
    x <- matrix(x, ncol = 1)
  } else if (is.data.frame(x)) {
    if (!all(vapply(x, is.numeric, logical(1L)))) stop("X must be numeric")
    x <- as.matrix(x)
  } else {
    x <- as.matrix(x)
  }

  if (!is.numeric(x) || any(!is.finite(x))) {
    stop("X must contain only finite numeric values")
  }
  if (nrow(x) < min_rows) {
    stop(sprintf("X must contain at least %d samples", min_rows))
  }

  x
}

#' Check that a score function matches the samples
#'
#' Calls the score function on the sample matrix and checks that the returned
#' object has the same shape. A score is the gradient of the target log density,
#' so row `i` of the output should describe row `i` of `x_mat`.
#'
#' @param score_function Function that returns the score for each row of `x_mat`.
#' @param x_mat Numeric sample matrix with samples in rows.
#'
#' @return A numeric matrix with the same shape as `x_mat`.
#' @noRd
validate_scores <- function(score_function, x_mat) {
  if (!is.function(score_function)) {
    stop("score_function must be a function")
  }

  n <- nrow(x_mat)
  d <- ncol(x_mat)
  grads <- score_function(x_mat)

  if (is.null(dim(grads))) {
    if (d != 1 || length(grads) != n) {
      stop("score_function output must be n x d (or length n when d = 1)")
    }
    grads <- matrix(as.numeric(grads), ncol = 1)
  } else {
    grads <- as.matrix(grads)
  }

  if (!is.numeric(grads) || nrow(grads) != n || ncol(grads) != d ||
    any(!is.finite(grads))) {
    stop("score_function output must contain finite numeric values with shape n x d")
  }

  grads
}

#' Check a positive integer argument
#'
#' Accepts numeric input only when it is finite, positive, and equal to an
#' integer up to floating-point rounding error.
#'
#' @param x Value to check.
#' @param arg_name Name used in the error message.
#'
#' @return The checked value as an integer.
#' @noRd
validate_positive_integer <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) ||
    x <= 0 || abs(x - round(x)) > sqrt(.Machine$double.eps)) {
    stop(sprintf("%s must be a positive integer", arg_name))
  }
  as.integer(round(x))
}

#' Check a nonnegative integer argument
#'
#' Accepts numeric input only when it is finite, at least zero, and equal to an
#' integer up to floating-point rounding error.
#'
#' @param x Value to check.
#' @param arg_name Name used in the error message.
#'
#' @return The checked value as an integer.
#' @noRd
validate_nonnegative_integer <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) ||
    x < 0 || abs(x - round(x)) > sqrt(.Machine$double.eps)) {
    stop(sprintf("%s must be a nonnegative integer", arg_name))
  }
  as.integer(round(x))
}

#' Run code with a local random seed
#'
#' Evaluates an expression with a temporary seed, then restores the caller's
#' random-number state.
#'
#' @details
#' This is useful for examples and simulation routines that need repeatable
#' random choices without changing the random seed seen by the caller after the
#' function returns. If the suggested package `withr` is installed, it is used;
#' otherwise the same behavior is implemented directly.
#'
#' @param seed Seed value. Use `NULL` to leave the random state unchanged.
#' @param expr Code to evaluate.
#'
#' @return The value produced by `expr`.
#' @noRd
with_local_seed <- function(seed, expr) {
  if (is.null(seed)) {
    return(force(expr))
  }
  seed <- as.integer(seed)
  if (requireNamespace("withr", quietly = TRUE)) {
    return(withr::with_seed(seed, expr))
  }
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit(
    {
      if (had_seed) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    },
    add = TRUE
  )
  set.seed(seed)
  force(expr)
}

#' Largest non-diagonal entry in a kernel matrix
#'
#' Looks only below the diagonal of a square matrix and returns the largest
#' value found there. This is used as a quick check for kernels whose
#' off-diagonal entries have effectively vanished.
#'
#' @param k_mat Numeric kernel matrix.
#'
#' @return Largest non-diagonal value, or `Inf` when no such value is available.
#' @noRd
max_offdiag_from_kernel_matrix <- function(k_mat) {
  if (!is.matrix(k_mat) || nrow(k_mat) < 2) {
    return(Inf)
  }
  vals <- k_mat[lower.tri(k_mat, diag = FALSE)]
  if (length(vals) == 0) {
    return(Inf)
  }
  max(vals, na.rm = TRUE)
}

#' Choose whether to compute a large matrix in blocks
#'
#' Decides whether a KSD kernel matrix should be built all at once or in smaller
#' row and column blocks.
#'
#' @param n Number of sample rows.
#' @param block_size Requested block size. If `NULL`, a default is chosen.
#' @param block_threshold Automatic blocking is used when `n` is larger than
#'   this value.
#'
#' @return A list with `use_block_mode` and `block_size`.
#' @noRd
resolve_block_settings <- function(n, block_size, block_threshold) {
  if (!is.numeric(n) || length(n) != 1 || !is.finite(n) || n < 1) {
    stop("n must be a positive scalar")
  }
  if (!is.numeric(block_threshold) || length(block_threshold) != 1 ||
    !is.finite(block_threshold) || block_threshold < 1) {
    stop("block_threshold must be a positive scalar")
  }

  n <- as.integer(n)
  block_threshold <- as.integer(block_threshold)
  auto_block <- n > block_threshold
  if (is.null(block_size)) {
    block_size <- if (auto_block) min(1024L, n) else n
  }
  if (!is.numeric(block_size) || length(block_size) != 1 ||
    !is.finite(block_size) || block_size < 1) {
    stop("block_size must be NULL or a positive scalar")
  }
  block_size <- min(as.integer(block_size), n)

  list(
    use_block_mode = auto_block || block_size < n,
    block_size = block_size
  )
}

#' Median squared distance between sample pairs
#'
#' Computes the square of the median Euclidean distance between all sample-row
#' pairs. This is the exact median-heuristic squared scale used by the built-in
#' kernels when the user does not supply a scale.
#'
#' @details
#' If the sample rows are `z_1, ..., z_n` in `R^d`, the full calculation forms
#' the vector of all off-diagonal Euclidean distances
#' \deqn{
#'   D = \{ ||z_i - z_j||_2 : 1 \le i < j \le n \}.
#' }
#' If `d_ij = ||z_i - z_j||_2`, the return value is
#' `median({d_ij : i < j})^2`. The median is taken before squaring, matching the
#' bandwidth rule stated in the KSD-U and FSSD experiments. This order matters
#' when the number of pairwise distances is even. For matrix input, rows are
#' observations and columns are coordinates; a numeric vector is treated as an
#' `n x 1` sample.
#'
#' The value is a data-dependent bandwidth proxy. In this package it is used as
#' a squared scale for built-in kernels when the user supplies `scaling = NULL`.
#' For the Gaussian RBF kernel this means `h^2 = median(d_ij)^2` in
#' `exp(-||x - y||^2 / (2 h^2))`. For the IMQ kernel it means
#' `c^2 = median(d_ij)^2` in `(c^2 + ||x - y||^2)^beta`. The function does not
#' standardize columns, so variables should already be on a comparable scale
#' or the user should provide a kernel scale directly.
#'
#' All \eqn{n(n - 1) / 2} pairwise distances are used. The calculation is
#' therefore deterministic but requires quadratic time and memory in the row
#' count. For large samples, provide `scaling` explicitly to the calling KSD or
#' FSSD routine, or provide a fixed bandwidth when constructing a kernel.
#'
#' Repeated or nearly repeated rows can make the median equal to zero. A zero
#' squared scale would make the built-in kernels ill-defined, so the function
#' returns a small positive floor value and emits a warning in that case. This
#' warning is usually a signal that the sample contains many duplicates or that
#' the chain has not moved enough for a median-distance rule to be informative.
#'
#' @param Z Numeric vector, matrix, or data frame of samples.
#' @return
#' A positive numeric scalar. It is the square of the median finite
#' off-diagonal Euclidean distance. Use `sqrt(find_median_distance(Z))` only if
#' a distance-scale bandwidth is needed. If all finite pairwise distances
#' are zero, the returned value is `1e-5`; if no finite distance can be formed,
#' the returned fallback is `1`.
#' @examples
#' find_median_distance(c(-1, 0, 2))
#'
#' Z <- matrix(c(0, 0, 1, 0, 0, 2), ncol = 2, byrow = TRUE)
#' find_median_distance(Z)
#' @export
find_median_distance <- function(Z) {
  if (is.null(dim(Z))) {
    Z <- matrix(as.numeric(Z), ncol = 1)
  } else if (is.data.frame(Z)) {
    Z <- data.matrix(Z)
  } else {
    Z <- as.matrix(Z)
  }

  if (!is.numeric(Z) || nrow(Z) < 2) {
    stop("Z must be numeric with at least two rows")
  }

  dists <- as.numeric(stats::dist(Z, method = "euclidean"))
  dists <- dists[is.finite(dists)]
  if (length(dists) == 0) {
    return(1)
  }

  med <- stats::median(dists)^2
  if (!is.finite(med) || med < 0) {
    med <- 1
  }
  if (med == 0) {
    warning(
      "Median pairwise squared distance is zero (possible non-mixing/repeated states); using floor value 1e-5.",
      call. = FALSE
    )
    med <- 1e-5
  }

  med
}
