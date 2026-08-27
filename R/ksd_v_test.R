# KSD-V test, statistic, and wild bootstrap.

#' KSD-V goodness-of-fit test with wild-bootstrap calibration
#'
#' Tests observations against a target distribution specified by its score
#' function. Use Rademacher calibration for independent observations and
#' Markov calibration for ordered dependent observations.
#'
#' @details
#' Let \eqn{s_p(x)=\nabla_x\log p(x)} and
#' \deqn{K_{ij}=k_{0,p}(X_i,X_j),}
#' where \eqn{k_{0,p}} is defined in [stein_kernel_matrix()]. The test uses
#' \deqn{V_n=\frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^nK_{ij},\qquad
#' nV_n=\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^nK_{ij}.}
#' The diagonal is retained. For a positive-definite base kernel, \eqn{V_n} is
#' nonnegative but upward biased for the population squared KSD.
#'
#' `boot_method = "rademacher"` uses independent signs and is intended for
#' independent observations. `boot_method = "markov"` uses correlated signs
#' and requires rows of `X` to be in dependence order. The latter calibration
#' requires the dependence, moment, and kernel conditions of the wild-bootstrap
#' result; this function cannot verify them. Asymptotically, the sign-change
#' probabilities \eqn{a_n} must satisfy \eqn{a_n\to0} and
#' \eqn{na_n\to\infty}.
#'
#' If \eqn{nV_n^{*(1)},\ldots,nV_n^{*(B)}} are the bootstrap draws, the
#' reported right-tail p-value is
#' \deqn{\widehat p=\frac{1+\sum_{b=1}^B
#' \mathbf{1}\{nV_n^{*(b)}\ge nV_n\}}{B+1}.}
#'
#' @param X Numeric vector or matrix containing \eqn{n} observations. Rows are
#'   observations and columns are coordinates; a vector is treated as an
#'   \eqn{n\times 1} matrix. For Markov calibration, rows must follow the
#'   dependence order.
#' @param score_function Function that accepts the checked \eqn{n\times d}
#'   sample matrix and returns the \eqn{n\times d} matrix with row
#'   \eqn{s_p(X_i)^\top}.
#' @param boot_method Calibration method: `"rademacher"` for independent
#'   observations or `"markov"` for ordered dependent observations.
#' @param scaling Positive squared scale passed to the kernel. For the built-in
#'   RBF and IMQ kernels this is, respectively, \eqn{h^2} and \eqn{c^2}.
#'   `NULL` uses the squared median pairwise distance.
#' @param nboot Positive number of bootstrap draws \eqn{B}.
#' @param change_prob Sign-change probability \eqn{a_n} for Markov calibration.
#'   It must be supplied and lie strictly between 0 and 1. It is ignored for
#'   Rademacher calibration.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object.
#' @param return_raw_boot Logical. If `TRUE`, include the \eqn{B} bootstrap
#'   draws in the result.
#' @param block_size Positive number of rows per block. `NULL` uses
#'   `min(1024, n)` when `n > block_threshold` and `n` otherwise. Blocking
#'   changes memory use, not the test definition.
#' @param block_threshold Positive sample-size threshold above which blockwise
#'   computation is used.
#' @param imq_beta Finite exponent \eqn{\beta<0} of the built-in IMQ kernel.
#' @return
#' An `htest` object. `statistic` is \eqn{nV_n}; `p.value` is the bootstrap
#' p-value above. If `return_raw_boot = TRUE`, `bootstrap_samples` contains
#' \eqn{nV_n^{*(1)},\ldots,nV_n^{*(B)}}.
#' @examples
#' X <- matrix(rnorm(20), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_v_test(X, score_function, nboot = 10)
#' @export
ksd_v_test <- function(X, score_function,
                       boot_method = c("rademacher", "markov"),
                       scaling = NULL,
                       nboot = 1000,
                       change_prob = NULL,
                       kernel = c("gaussian_rbf", "imq"),
                       return_raw_boot = FALSE,
                       block_size = NULL, block_threshold = 5000,
                       imq_beta = -0.5) {
  data_name <- deparse(substitute(X))
  boot_method <- match.arg(boot_method)
  prep <- .prepare_ksd_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    imq_beta = imq_beta
  )
  n <- nrow(prep$X)

  # Wild-bootstrap columns are sign sequences; Markov signs retain sample-order dependence.
  W_mat <- .generate_bootstrap_weights(
    n, nboot, boot_method = boot_method, change_prob = change_prob
  )
  nboot <- as.integer(nboot)
  resolved_change_prob <- if (identical(boot_method, "markov")) {
    as.numeric(change_prob)
  } else {
    NA_real_
  }
  block_settings <- .resolve_block_settings(n, block_size, block_threshold)

  engine_res <- .compute_ksd_v(
    X = prep$X,
    scores = prep$scores,
    kernel_obj = prep$kernel_obj,
    W_mat = W_mat,
    use_block = block_settings$use_block_mode,
    block_size = block_settings$block_size
  )

  max_offdiag <- engine_res$max_offdiag
  if (any(is.finite(max_offdiag) & max_offdiag < 1e-12)) {
    warning("Vanishing off-diagonal kernel detected: bandwidth may be too small.", call. = FALSE)
  }

  res <- list(
    statistic = c(ksd_v = engine_res$statistic),
    p.value = engine_res$p_value,
    method = sprintf(
      "Kernelized Stein Discrepancy (V-statistics) - %s bootstrap, %s kernel",
      boot_method, prep$kernel_name
    ),
    data.name = data_name,
    parameter = c(
      nboot = nboot,
      scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling,
      change_prob = resolved_change_prob
    ),
    kernel = prep$kernel_obj
  )

  if (isTRUE(return_raw_boot)) res$bootstrap_samples <- engine_res$bootstrap_samples
  class(res) <- "htest"
  res
}

#' Build the Stein-kernel matrix for KSD-V
#'
#' @details
#' For observations \eqn{X_1,\ldots,X_n}, this function returns
#' \deqn{K_{ij}=k_{0,p}(X_i,X_j).}
#' The diagonal is included and is retained by [ksd_v_statistic()] and
#' [ksd_v_bootstrap()].
#'
#' @param X Numeric vector or matrix containing \eqn{n} observations. Rows are
#'   observations; a vector is treated as an \eqn{n\times 1} matrix.
#' @param score_function Function that accepts the checked \eqn{n\times d}
#'   sample matrix and returns an \eqn{n\times d} score matrix.
#' @param scaling Positive squared scale passed to the kernel. For the built-in
#'   RBF and IMQ kernels this is, respectively, \eqn{h^2} and \eqn{c^2}.
#'   `NULL` uses the squared median pairwise distance.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object.
#' @param imq_beta Finite exponent \eqn{\beta<0} of the built-in IMQ kernel.
#' @return
#' Numeric \eqn{n\times n} matrix \eqn{K}, including its diagonal.
#' @examples
#' X <- matrix(rnorm(5), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_vq_matrix(X, score_function)
#' @export
ksd_vq_matrix <- function(X, score_function,
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

#' Compute the KSD-V statistic
#'
#' @details
#' For \eqn{K_{ij}=k_{0,p}(X_i,X_j)}, this function returns
#' \deqn{nV_n=\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^nK_{ij}.}
#' All ordered pairs, including \eqn{i=j}, are included. This function does not
#' compute bootstrap draws or a p-value.
#'
#' @param K0 Numeric \eqn{n\times n} Stein-kernel matrix, such as the output of
#'   [ksd_vq_matrix()].
#' @return
#' One numeric value, \eqn{nV_n}.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_statistic(U)
#' @export
ksd_v_statistic <- function(K0) {
  K0 <- .validate_stein_matrix(K0)
  nrow(K0) * mean(K0)
}

#' Wild bootstrap for KSD-V
#'
#' @details
#' For sign vector \eqn{W^{(b)}}, the returned draw is
#' \deqn{nV_n^{*(b)}=\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^n
#' W_i^{(b)}W_j^{(b)}K_{ij}.}
#' Rademacher signs are independent and take values \eqn{-1} and \eqn{1} with
#' equal probability. Markov signs satisfy
#' \deqn{W_1^{(b)}=1,\qquad W_t^{(b)}=
#' \begin{cases}
#' -W_{t-1}^{(b)},&\text{with probability }a_n,\\
#' W_{t-1}^{(b)},&\text{with probability }1-a_n.
#' \end{cases}}
#' Here \eqn{a_n} is `change_prob`. All matrix entries, including the diagonal,
#' are used. The draws are on the same scale as [ksd_v_statistic()]. This
#' function does not compute a p-value.
#'
#' @param K0 Numeric \eqn{n\times n} Stein-kernel matrix, such as the output of
#'   [ksd_vq_matrix()].
#' @param nboot Positive number of bootstrap draws. Ignored when `W_mat` is
#'   supplied.
#' @param W_mat Optional numeric \eqn{n\times B} sign matrix. Row \eqn{i}
#'   corresponds to observation \eqn{i}; column \eqn{b} specifies bootstrap
#'   draw \eqn{b}. If `NULL`, signs are generated from `boot_method`.
#' @param boot_method Sign process used when `W_mat = NULL`: `"rademacher"` or
#'   `"markov"`.
#' @param change_prob Markov sign-change probability \eqn{a_n}. It must be
#'   supplied and lie strictly between 0 and 1 when `boot_method = "markov"`.
#' @return
#' Numeric vector containing \eqn{nV_n^{*(1)},\ldots,nV_n^{*(B)}}.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_bootstrap(U, nboot = 5)
#' @export
ksd_v_bootstrap <- function(K0, nboot = 1000, W_mat = NULL,
                            boot_method = c("rademacher", "markov"),
                            change_prob = NULL) {
  boot_method <- match.arg(boot_method)
  K0 <- .validate_stein_matrix(K0)
  n <- nrow(K0)

  if (is.null(W_mat)) {
    W_mat <- .generate_bootstrap_weights(
      n, nboot, boot_method = boot_method, change_prob = change_prob
    )
  } else {
    W_mat <- as.matrix(W_mat)
    if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1) {
      stop("W_mat must be a numeric matrix with nrow(K0) rows and at least one column", call. = FALSE)
    }
  }

  as.numeric(colSums((K0 %*% W_mat) * W_mat) / n)
}

.compute_ksd_v <- function(X, scores, kernel_obj, W_mat, use_block, block_size) {
  if (use_block) {
    return(.compute_ksd_v_blocked(X, scores, kernel_obj, W_mat, block_size))
  }

  # The vanishing-bandwidth diagnostic is specific to Gaussian RBF kernels.
  # Other valid Stein-kernel classes may implement only the complete
  # stein_kernel_matrix() method and need no base-kernel evaluation here.
  k_mat <- if (inherits(kernel_obj, "SteinKernel_gaussian_rbf")) {
    eval_kernel(kernel_obj, X)
  } else {
    NULL
  }
  n <- nrow(X)
  K0 <- stein_kernel_matrix(kernel_obj, X, scores)
  statistic <- sum(K0) / n
  bootstrap_samples <- as.numeric(
    colSums((K0 %*% W_mat) * W_mat) / n
  )

  list(
    statistic = statistic,
    p_value = .bootstrap_pvalue_right_tail(bootstrap_samples, statistic),
    bootstrap_samples = bootstrap_samples,
    max_offdiag = .max_offdiag_from_kernel_matrix(k_mat)
  )
}

# Accumulate n V_n and bootstrap quadratic forms without storing the full K0.
.compute_ksd_v_blocked <- function(X, scores, kernel_obj, W_mat, block_size) {
  n <- nrow(X)
  nboot <- ncol(W_mat)
  i_starts <- seq(1, n, by = block_size)
  j_starts <- seq(1, n, by = block_size)

  stat_sum <- 0
  boot_stats <- numeric(nboot)
  max_offdiag <- -Inf

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

      k_block <- if (inherits(kernel_obj, "SteinKernel_gaussian_rbf")) {
        eval_kernel(kernel_obj, X_i, X_j)
      } else {
        NULL
      }
      K0_block <- stein_kernel_matrix(
        kernel_obj, X_i, scores_i, X_j, scores_j
      )

      stat_sum <- stat_sum + sum(K0_block)
      # ksd_v_statistic() reports n V_n, so bootstrap blocks are accumulated on the same scale.
      boot_stats <- boot_stats + colSums((K0_block %*% W_j) * W_i) / n

      max_offdiag <- max(max_offdiag, .max_offdiag_from_kernel_matrix(
        k_block, diagonal_block = i_start == j_start
      ))
    }
  }

  stat <- stat_sum / n
  list(
    statistic = stat,
    p_value = .bootstrap_pvalue_right_tail(boot_stats, stat),
    bootstrap_samples = as.numeric(boot_stats),
    max_offdiag = max_offdiag
  )
}
