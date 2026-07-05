# Shared helper functions for GMM and KSD implementations.


#' Validate and coerce samples to an n x d numeric matrix
#'
#' @param x Numeric vector/matrix/data.frame.
#' @param min_rows Minimum required number of rows.
#'
#' @return Numeric matrix with at least `min_rows` rows.
#' @keywords internal
validate_samples <- function(x, min_rows = 2) {
  if (!is.numeric(min_rows) || length(min_rows) != 1 || !is.finite(min_rows) || min_rows < 1) {
    stop("min_rows must be a positive scalar")
  }
  min_rows <- as.integer(min_rows)

  if (is.null(dim(x))) {
    x <- matrix(as.numeric(x), ncol = 1)
  } else if (is.data.frame(x)) {
    x <- data.matrix(x)
  } else {
    x <- as.matrix(x)
  }

  if (!is.numeric(x)) {
    stop("X must be numeric")
  }
  if (nrow(x) < min_rows) {
    stop(sprintf("X must contain at least %d samples", min_rows))
  }

  x
}

#' Validate score function output shape against samples
#'
#' @param score_function Function mapping `x_mat` to score values.
#' @param x_mat Numeric sample matrix (`n x d`).
#'
#' @return Numeric score matrix with shape `n x d`.
#' @keywords internal
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

  if (!is.numeric(grads) || nrow(grads) != n || ncol(grads) != d) {
    stop("score_function output shape must match X (n x d)")
  }

  grads
}

#' @keywords internal
validate_positive_integer <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) ||
      x <= 0 || abs(x - round(x)) > sqrt(.Machine$double.eps)) {
    stop(sprintf("%s must be a positive integer", arg_name))
  }
  as.integer(round(x))
}

#' @keywords internal
validate_nonnegative_integer <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) ||
      x < 0 || abs(x - round(x)) > sqrt(.Machine$double.eps)) {
    stop(sprintf("%s must be a nonnegative integer", arg_name))
  }
  as.integer(round(x))
}

#' Run an expression under a fixed RNG seed
#'
#' Uses `withr::with_seed` when available; otherwise saves and restores
#' `.Random.seed` in the global environment.
#'
#' @param seed Seed value, or `NULL` to evaluate without changing the RNG state.
#' @param expr Expression to evaluate.
#'
#' @return The value of `expr`.
#' @export
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
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  force(expr)
}

#' Max off-diagonal value from a kernel matrix
#' @keywords internal
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

#' Resolve whether to use blocked O(n^2) accumulation
#' @keywords internal
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

# Deterministic subsampling without leaking RNG state to callers.
sample_rows_seeded <- function(sample_size, take_n, seed = NULL) {
  if (is.null(seed)) {
    return(sample.int(sample_size, take_n))
  }

  if (requireNamespace("withr", quietly = TRUE)) {
    return(withr::with_seed(as.integer(seed), sample.int(sample_size, take_n)))
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))
  sample.int(sample_size, take_n)
}

#' Median pairwise squared-distance heuristic
#'
#' Computes the median of pairwise squared Euclidean distances, optionally after
#' deterministic subsampling.
#'
#' @param Z Numeric vector, matrix, or data frame of samples.
#' @param max_samples Maximum number of rows used when `use_sampling = TRUE`.
#' @param use_sampling Logical; if `TRUE`, subsample rows when needed.
#' @param seed Optional seed for deterministic subsampling.
#'
#' @return A positive numeric scalar.
#' @export
find_median_distance <- function(Z,
                                 max_samples = 2000,
                                 use_sampling = TRUE,
                                 seed = NULL) {
  if (!is.numeric(max_samples) || length(max_samples) != 1 || !is.finite(max_samples) || max_samples < 2) {
    stop("max_samples must be a numeric scalar >= 2")
  }
  max_samples <- as.integer(max_samples)
  if (!is.logical(use_sampling) || length(use_sampling) != 1 || is.na(use_sampling)) {
    stop("use_sampling must be TRUE or FALSE")
  }
  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || !is.finite(seed))) {
    stop("seed must be NULL or a finite numeric scalar")
  }

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

  sample_size <- nrow(Z)
  if (use_sampling && sample_size > max_samples) {
    idx <- sample_rows_seeded(sample_size, max_samples, seed = seed)
    z_med <- Z[idx, , drop = FALSE]
  } else {
    z_med <- Z
  }

  sq_dists <- as.numeric(stats::dist(z_med, method = "euclidean"))^2
  sq_dists <- sq_dists[is.finite(sq_dists)]
  if (length(sq_dists) == 0) {
    return(1)
  }

  med <- stats::median(sq_dists)
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
