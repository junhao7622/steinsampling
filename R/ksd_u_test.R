#' Kernel Stein Discrepancy GOF Test (U-statistics)
#'
#' @param X Numeric vector or matrix of samples. A vector is treated as `n x 1`.
#' @param score_function Function of `X` returning score values.
#' @param boot_method Bootstrap method. `"multinomial_centered"` implements
#'   the centered multinomial bootstrap in Liu, Lee & Jordan (2016), Eq. 16.
#' @param scaling Positive kernel scaling parameter. NULL uses median heuristic.
#' @param nboot Number of bootstrap samples.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or an instantiated `SteinKernel`.
#' @param return_raw_boot Logical; if `TRUE`, return bootstrap samples.
#' @param median_max_samples,median_use_sampling,median_seed Controls for the
#'   median-distance heuristic when `scaling = NULL`.
#' @param block_size,block_threshold Controls for blocked matrix computation.
#' @param imq_beta IMQ kernel exponent.
#' @return An object of class `htest`.
#' @examples
#' X <- matrix(rnorm(10), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_u_test(X, score_function, nboot = 10)
#' @export
ksd_u_test <- function(X, score_function,
                       boot_method = "multinomial_centered",
                       scaling = NULL,
                       nboot = 1000,
                       kernel = c("gaussian_rbf", "imq"),
                       return_raw_boot = FALSE,
                       median_max_samples = 2000, median_use_sampling = TRUE,
                       median_seed = 123, block_size = NULL, block_threshold = 5000, imq_beta = -0.5) {

  data_name <- deparse(substitute(X))
  boot_method <- resolve_ksd_u_boot_method(boot_method)
  prep <- prepare_ksd_u_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    median_max_samples = median_max_samples,
    median_use_sampling = median_use_sampling,
    median_seed = median_seed,
    imq_beta = imq_beta
  )
  n <- nrow(prep$X)

  if (!is.numeric(nboot) || length(nboot) != 1 || !is.finite(nboot) || nboot <= 0) {
    stop("nboot must be a positive scalar")
  }
  nboot <- as.integer(nboot)

  W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method)
  block_settings <- resolve_block_settings(n, block_size, block_threshold)

  engine_res <- internal_u_compute_engine(
    X = prep$X,
    grads = prep$grads,
    kernel_obj = prep$kernel_obj,
    W_mat = W_mat,
    boot_method = boot_method,
    use_block = block_settings$use_block_mode,
    block_size = block_settings$block_size
  )

  res <- list(
    statistic = c(ksd_u = engine_res$stat),
    p.value = engine_res$pval,
    method = paste("Kernel Stein Discrepancy (U-statistics) -", prep$kernel_name),
    data.name = data_name,
    parameter = c(
      nboot = nboot,
      scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling,
      imq_beta = if (prep$kernel_name == "imq" && !is.null(prep$kernel_obj$beta)) prep$kernel_obj$beta else NA_real_
    )
  )

  if (isTRUE(return_raw_boot)) res$bootstrap_samples <- engine_res$raw_boots
  class(res) <- "htest"
  res
}

#' KSD U-statistic kernel matrix (Theorem 3.6)
#'
#' @param X Numeric vector or matrix of samples. A vector is treated as `n x 1`.
#' @param score_function Function of `X` returning score values.
#' @param scaling Positive kernel scaling parameter. NULL uses median heuristic.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or an instantiated `SteinKernel`.
#' @param median_max_samples,median_use_sampling,median_seed Controls for the
#'   median-distance heuristic when `scaling = NULL`.
#' @param imq_beta IMQ kernel exponent.
#' @return Numeric matrix with entries `u_q(x_i, x_j)`.
#' @examples
#' X <- matrix(rnorm(5), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_uq_matrix(X, score_function)
#' @export
ksd_uq_matrix <- function(X, score_function,
                          scaling = NULL,
                          kernel = c("gaussian_rbf", "imq"),
                          median_max_samples = 2000, median_use_sampling = TRUE,
                          median_seed = 123, imq_beta = -0.5) {
  prep <- prepare_ksd_u_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    median_max_samples = median_max_samples,
    median_use_sampling = median_use_sampling,
    median_seed = median_seed,
    imq_beta = imq_beta
  )

  ksd_uq_matrix_from_scores(prep$X, prep$grads, prep$kernel_obj)
}

#' KSD U-statistic (Eq. 14)
#'
#' @param U_mat Numeric square matrix with entries `u_q(x_i, x_j)`.
#' @return Numeric scalar `1 / (n * (n - 1)) * sum_{i != j} u_q(x_i, x_j)`.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_u_statistic(U)
#' @export
ksd_u_statistic <- function(U_mat) {
  U_mat <- validate_ksd_u_matrix(U_mat)
  n <- nrow(U_mat)
  diag(U_mat) <- 0
  sum(U_mat) / (n * (n - 1))
}

#' KSD U-statistic bootstrap samples (Eq. 16)
#'
#' @param U_mat Numeric square matrix with entries `u_q(x_i, x_j)`.
#' @param nboot Number of bootstrap samples.
#' @param W_mat Optional centered multinomial bootstrap weight matrix.
#' @param boot_method Bootstrap method. `"multinomial_centered"` implements
#'   the centered multinomial bootstrap in Liu, Lee & Jordan (2016), Eq. 16.
#' @return Numeric vector of bootstrap samples on the U-statistic scale.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_u_bootstrap(U, nboot = 5)
#' @export
ksd_u_bootstrap <- function(U_mat, nboot = 1000, W_mat = NULL,
                            boot_method = "multinomial_centered") {
  boot_method <- resolve_ksd_u_boot_method(boot_method)
  U_mat <- validate_ksd_u_matrix(U_mat)
  n <- nrow(U_mat)
  diag(U_mat) <- 0

  if (is.null(W_mat)) {
    if (!is.numeric(nboot) || length(nboot) != 1 || !is.finite(nboot) || nboot <= 0) {
      stop("nboot must be a positive scalar")
    }
    nboot <- as.integer(nboot)
    W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method)
  } else {
    W_mat <- as.matrix(W_mat)
    if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1) {
      stop("W_mat must be a numeric matrix with nrow(U_mat) rows and at least one column")
    }
  }

  # Liu, Lee & Jordan (2016), Eq. 16.
  as.numeric(colSums((U_mat %*% W_mat) * W_mat))
}

#' Prepare inputs for the KSD U-statistic routines
#'
#' Validates samples and score values, resolves the requested Stein kernel, and
#' applies the median-distance heuristic when no scaling is supplied.
#'
#' @param X Numeric vector or matrix of samples. A vector is treated as `n x 1`.
#' @param score_function Function of `X` returning score values.
#' @param scaling Positive kernel scaling parameter. `NULL` uses the median heuristic.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or an instantiated `SteinKernel`.
#' @param median_max_samples,median_use_sampling,median_seed Controls for the
#'   median-distance heuristic when `scaling = NULL`.
#' @param imq_beta IMQ kernel exponent.
#'
#' @return A list containing validated samples, scores, kernel object, kernel
#'   name, and scaling.
#' @export
prepare_ksd_u_inputs <- function(X, score_function, scaling = NULL,
                                 kernel = c("gaussian_rbf", "imq"),
                                 median_max_samples = 2000,
                                 median_use_sampling = TRUE,
                                 median_seed = 123, imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = 2)
  grads <- validate_scores(score_function, x_mat)

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_type_name(kernel_obj)
    kernel_scale <- kernel_scaling_value(kernel_obj)
    if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
      }
      kernel_obj <- instantiate_kernel("gaussian_rbf", scaling, imq_beta)
    } else {
      scaling <- kernel_scale
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
    }
    kernel_obj <- instantiate_kernel(kernel_name, scaling, imq_beta)
  }

  list(
    X = x_mat,
    grads = grads,
    kernel_obj = kernel_obj,
    kernel_name = kernel_name,
    scaling = scaling
  )
}

#' @keywords internal
resolve_ksd_u_boot_method <- function(boot_method) {
  if (!identical(boot_method, "multinomial_centered")) {
    stop("boot_method must be 'multinomial_centered'")
  }
  boot_method
}

#' @keywords internal
validate_ksd_u_matrix <- function(U_mat) {
  if (is.null(dim(U_mat))) {
    stop("U_mat must be a square numeric matrix")
  }
  U_mat <- as.matrix(U_mat)
  if (!is.numeric(U_mat) || nrow(U_mat) != ncol(U_mat) || nrow(U_mat) < 2) {
    stop("U_mat must be a square numeric matrix with at least two rows")
  }
  U_mat
}

#' @keywords internal
ksd_uq_matrix_from_scores <- function(X, grads, kernel_obj) {
  stein_kernel_matrix(kernel_obj, X, grads)
}


#' @keywords internal
internal_u_compute_engine <- function(X, grads, kernel_obj, W_mat, boot_method, use_block, block_size) {
  n <- nrow(X)

  if (use_block) {
    return(compute_u_blocked_pure(X, grads, kernel_obj, W_mat, block_size))
  }

  U_mat <- ksd_uq_matrix_from_scores(X, grads, kernel_obj)
  stat <- ksd_u_statistic(U_mat)
  boot_stats <- ksd_u_bootstrap(U_mat, W_mat = W_mat, boot_method = boot_method)

  list(
    stat = unname(stat),
    pval = bootstrap_pvalue_right_tail(boot_stats, stat),
    raw_boots = as.numeric(boot_stats)
  )
}

#' @keywords internal
compute_u_blocked_pure <- function(X, grads, kernel_obj, W_mat, block_size) {
  n <- nrow(X)
  nboot <- ncol(W_mat)
  i_starts <- seq(1, n, by = block_size)
  j_starts <- seq(1, n, by = block_size)

  stat_sum <- 0
  boot_stats <- numeric(nboot)

  for (i_start in i_starts) {
    i_end <- min(i_start + block_size - 1L, n)
    ii <- i_start:i_end
    X_i <- X[ii, , drop = FALSE]
    g_i <- grads[ii, , drop = FALSE]
    W_i <- W_mat[ii, , drop = FALSE]

    for (j_start in j_starts) {
      j_end <- min(j_start + block_size - 1L, n)
      jj <- j_start:j_end
      X_j <- X[jj, , drop = FALSE]
      g_j <- grads[jj, , drop = FALSE]
      W_j <- W_mat[jj, , drop = FALSE]

      m_block <- compute_u_block_pure(X_i, X_j, g_i, g_j, kernel_obj)

      if (i_start == j_start) {
        diag(m_block) <- 0
      }

      stat_sum <- stat_sum + sum(m_block)
      boot_stats <- boot_stats + colSums((m_block %*% W_j) * W_i)
    }
  }

  stat <- stat_sum / (n * (n - 1))
  list(
    stat = stat,
    pval = bootstrap_pvalue_right_tail(boot_stats, stat),
    raw_boots = as.numeric(boot_stats)
  )
}

#' @keywords internal
compute_u_block_pure <- function(X_i, X_j, g_i, g_j, kernel_obj) {
  stein_kernel_matrix(kernel_obj, X_i, g_i, X_j, g_j)
}
