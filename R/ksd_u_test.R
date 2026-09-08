# KSD-U test, statistic, and centered-multinomial bootstrap.

#' KSD-U goodness-of-fit test for independent observations
#'
#' Tests whether independent observations come from a target distribution
#' specified by its score function. The target density need not be normalized.
#'
#' @details
#' Let \eqn{s_p(x)=\nabla_x\log p(x)} and
#' \deqn{K_{ij}=k_{0,p}(X_i,X_j),}
#' where \eqn{k_{0,p}} is the Stein kernel defined in
#' [stein_kernel_matrix()]. The test uses
#' \deqn{U_n=\frac{1}{n(n-1)}\sum_{i\ne j}K_{ij},\qquad T_n=nU_n.}
#' The diagonal is excluded. Thus \eqn{U_n} is unbiased for the population
#' squared KSD, but its finite-sample value can be negative.
#'
#' Null calibration uses the centered multinomial bootstrap implemented by
#' [ksd_u_bootstrap()]. If \eqn{T_n^{*(1)},\ldots,T_n^{*(B)}} are the bootstrap
#' draws, the reported right-tail p-value is
#' \deqn{\widehat p=\frac{1+\sum_{b=1}^B
#' \mathbf{1}\{T_n^{*(b)}\ge T_n\}}{B+1}.}
#'
#' Warns when off-diagonal Stein-kernel magnitudes are negligible relative to
#' the diagonal, because the resulting bootstrap calibration is degenerate.
#'
#' @param X Numeric vector or matrix containing \eqn{n} observations. Rows are
#'   observations and columns are coordinates; a vector is treated as an
#'   \eqn{n\times 1} matrix.
#' @param score_function Function that accepts the checked \eqn{n\times d}
#'   sample matrix and returns the \eqn{n\times d} matrix with row
#'   \eqn{s_p(X_i)^\top}.
#' @param boot_method Bootstrap method. Currently only
#'   `"multinomial_centered"` is supported.
#' @param scaling Positive squared scale passed to the kernel unless a supplied
#'   `SteinKernel` already has one. For built-in RBF and IMQ kernels this is,
#'   respectively, \eqn{h^2} and \eqn{c^2}. `NULL` uses the squared median
#'   pairwise distance.
#' @param nboot Positive integer number of bootstrap draws \eqn{B}.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object.
#' @param return_raw_boot Logical. If `TRUE`, include the \eqn{B} bootstrap
#'   draws in the result.
#' @param block_size Positive integer number of rows per block. `NULL` uses
#'   `min(1024, n)` when `n > block_threshold` and `n` otherwise. Blocking
#'   changes memory use, not the test definition.
#' @param block_threshold Positive integer sample-size threshold above which
#'   blockwise computation is used.
#' @param imq_beta Finite exponent \eqn{\beta<0} of the built-in IMQ kernel.
#' @return
#' An `htest` object. `statistic` is \eqn{T_n=nU_n}; `p.value` is the
#' bootstrap p-value above. `parameter` records `nboot` and `scaling`, plus
#' `imq_beta` for an IMQ kernel; `kernel` contains the resolved `SteinKernel`
#' object. If `return_raw_boot = TRUE`, `bootstrap_samples` contains
#' \eqn{T_n^{*(1)},\ldots,T_n^{*(B)}}.
#' @references Liu, Lee, and Jordan (2016), A Kernelized Stein Discrepancy for
#' Goodness-of-Fit Tests.
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
                       block_size = NULL, block_threshold = 5000,
                       imq_beta = -0.5) {

  data_name <- deparse(substitute(X))
  boot_method <- .resolve_ksd_u_boot_method(boot_method)
  nboot <- validate_integer(nboot, "nboot")
  return_raw_boot <- validate_flag(return_raw_boot, "return_raw_boot")
  prep <- .prepare_ksd_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    imq_beta = imq_beta
  )
  n <- nrow(prep$X)

  block_settings <- .resolve_block_settings(n, block_size, block_threshold)
  W_mat <- .generate_bootstrap_weights(n, nboot, boot_method)
  nboot <- ncol(W_mat)

  engine_res <- .compute_ksd_u(
    X = prep$X,
    scores = prep$scores,
    kernel_obj = prep$kernel_obj,
    W_mat = W_mat,
    use_block = block_settings$use_block_mode,
    block_size = block_settings$block_size
  )
  .warn_if_degenerate_stein_matrix(engine_res$offdiag_max, engine_res$diag_max)

  res <- list(
    statistic = c(ksd_u = engine_res$statistic),
    p.value = engine_res$p_value,
    method = paste("Kernelized Stein Discrepancy (U-statistics) -", prep$kernel_name),
    data.name = data_name,
    parameter = c(
      nboot = nboot,
      scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling,
      if (prep$kernel_name == "imq" && !is.null(prep$kernel_obj$beta))
        c(imq_beta = prep$kernel_obj$beta)
    ),
    kernel = prep$kernel_obj
  )

  if (return_raw_boot) res$bootstrap_samples <- engine_res$bootstrap_samples
  class(res) <- "htest"
  res
}

#' Build the Stein-kernel matrix for the KSD tests
#'
#' @rdname ksd_matrix
#' @details
#' For observations \eqn{X_1,\ldots,X_n}, this function returns the full matrix
#' \deqn{K_{ij}=k_{0,p}(X_i,X_j).}
#' The diagonal is included. [ksd_u_statistic()] and [ksd_u_bootstrap()]
#' exclude it; [ksd_v_statistic()] and [ksd_v_bootstrap()] retain it.
#'
#' @inheritParams ksd_u_test
#' @param X Numeric vector or matrix containing \eqn{n} observations. Rows are
#'   observations; a vector is treated as an \eqn{n\times 1} matrix.
#' @param score_function Function that accepts the checked \eqn{n\times d}
#'   sample matrix and returns an \eqn{n\times d} score matrix.
#' @return
#' Numeric \eqn{n\times n} matrix \eqn{K}, including its diagonal.
#' @examples
#' X <- matrix(rnorm(5), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_uq_matrix(X, score_function)
#' @export
ksd_uq_matrix <- function(X, score_function,
                          scaling = NULL,
                          kernel = c("gaussian_rbf", "imq"),
                          imq_beta = -0.5) {
  prep <- .prepare_ksd_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    imq_beta = imq_beta
  )

  stein_kernel_matrix(prep$kernel_obj, prep$X, prep$scores)
}

#' Compute the KSD-U statistic
#'
#' @details
#' For \eqn{K_{ij}=k_{0,p}(X_i,X_j)}, this function returns
#' \deqn{T_n=nU_n=\frac{1}{n-1}\sum_{i\ne j}K_{ij}.}
#' The diagonal is excluded. The result can be negative. This function does
#' not compute bootstrap draws or a p-value.
#'
#' @param K0 Finite numeric \eqn{n\times n} Stein-kernel matrix with
#'   \eqn{n\ge 2}, such as the output of [ksd_uq_matrix()].
#' @return
#' One numeric value, \eqn{T_n=nU_n}.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_u_statistic(U)
#' @export
ksd_u_statistic <- function(K0) {
  K0 <- .validate_stein_matrix(K0)
  n <- nrow(K0)
  diag(K0) <- 0
  sum(K0) / (n - 1L)
}

#' Centered multinomial bootstrap for KSD-U
#'
#' @details
#' For bootstrap draw \eqn{b}, let
#' \deqn{w_i^{(b)}=\frac{N_i^{(b)}}{n}-\frac{1}{n},\qquad
#' (N_1^{(b)},\ldots,N_n^{(b)})\sim
#' \operatorname{Multinomial}\left(n;\frac1n,\ldots,\frac1n\right).}
#' The returned draw is
#' \deqn{T_n^{*(b)}=n\sum_{i\ne j}w_i^{(b)}w_j^{(b)}K_{ij}.}
#' The diagonal is excluded. The draws are on the same scale as
#' [ksd_u_statistic()]. This function does not compute a p-value.
#'
#' @param K0 Finite numeric \eqn{n\times n} Stein-kernel matrix with
#'   \eqn{n\ge 2}, such as the output of [ksd_uq_matrix()].
#' @param nboot Positive integer number of bootstrap draws. Ignored when
#'   `W_mat` is supplied.
#' @param W_mat Optional finite numeric \eqn{n\times B} multiplier matrix. If
#'   `NULL`, centered multinomial weights are generated. Supplied columns need
#'   not be centered.
#' @param boot_method Bootstrap method. Currently only
#'   `"multinomial_centered"` is supported.
#' @return
#' Numeric vector containing \eqn{T_n^{*(1)},\ldots,T_n^{*(B)}}.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_u_bootstrap(U, nboot = 5)
#' @export
ksd_u_bootstrap <- function(K0, nboot = 1000, W_mat = NULL,
                            boot_method = "multinomial_centered") {
  boot_method <- .resolve_ksd_u_boot_method(boot_method)
  K0 <- .validate_stein_matrix(K0)
  n <- nrow(K0)
  diag(K0) <- 0

  W_mat <- if (is.null(W_mat)) {
    .generate_bootstrap_weights(n, nboot, boot_method)
  } else {
    .validate_bootstrap_weights(W_mat, n)
  }

  # Liu, Lee & Jordan (2016), Eq. 16.
  n * as.numeric(colSums((K0 %*% W_mat) * W_mat))
}

.resolve_ksd_u_boot_method <- function(boot_method) {
  if (!identical(boot_method, "multinomial_centered")) {
    stop("boot_method must be 'multinomial_centered'", call. = FALSE)
  }
  boot_method
}

.compute_ksd_u <- function(X, scores, kernel_obj, W_mat, use_block, block_size) {
  if (use_block) {
    return(.compute_ksd_u_blocked(X, scores, kernel_obj, W_mat, block_size))
  }

  n <- nrow(X)
  K0 <- stein_kernel_matrix(kernel_obj, X, scores)
  diag_max <- max(abs(diag(K0)))
  diag(K0) <- 0
  offdiag_max <- max(abs(K0))
  u_statistic <- sum(K0) / (n * (n - 1L))
  bootstrap_u <- as.numeric(colSums((K0 %*% W_mat) * W_mat))
  statistic <- n * u_statistic
  bootstrap_samples <- n * bootstrap_u

  list(
    statistic = statistic,
    p_value = .bootstrap_pvalue_right_tail(bootstrap_samples, statistic),
    bootstrap_samples = bootstrap_samples,
    diag_max = diag_max,
    offdiag_max = offdiag_max
  )
}

# Accumulate U_n and bootstrap quadratic forms without storing the full K0.
.compute_ksd_u_blocked <- function(X, scores, kernel_obj, W_mat, block_size) {
  n <- nrow(X)
  nboot <- ncol(W_mat)
  i_starts <- seq(1, n, by = block_size)
  j_starts <- seq(1, n, by = block_size)

  stat_sum <- 0
  boot_stats <- numeric(nboot)
  diag_max <- 0
  offdiag_max <- 0

  for (i_start in i_starts) {
    i_end <- min(i_start + block_size - 1L, n)
    ii <- i_start:i_end
    X_i <- X[ii, , drop = FALSE]
    scores_i <- scores[ii, , drop = FALSE]
    W_i <- W_mat[ii, , drop = FALSE]

    for (j_start in j_starts) {
      j_end <- min(j_start + block_size - 1L, n)
      jj <- j_start:j_end
      X_j <- X[jj, , drop = FALSE]
      scores_j <- scores[jj, , drop = FALSE]
      W_j <- W_mat[jj, , drop = FALSE]

      K0_block <- stein_kernel_matrix(
        kernel_obj, X_i, scores_i, X_j, scores_j
      )

      if (i_start == j_start) {
        diag_max <- max(diag_max, max(abs(diag(K0_block))))
        diag(K0_block) <- 0
      }
      offdiag_max <- max(offdiag_max, max(abs(K0_block)))

      stat_sum <- stat_sum + sum(K0_block)
      boot_stats <- boot_stats + colSums((K0_block %*% W_j) * W_i)
    }
  }

  u_statistic <- stat_sum / (n * (n - 1L))
  statistic <- n * u_statistic
  bootstrap_samples <- n * boot_stats
  list(
    statistic = statistic,
    p_value = .bootstrap_pvalue_right_tail(bootstrap_samples, statistic),
    bootstrap_samples = as.numeric(bootstrap_samples),
    diag_max = diag_max,
    offdiag_max = offdiag_max
  )
}
