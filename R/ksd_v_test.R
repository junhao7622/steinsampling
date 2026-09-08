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
#' is valid only under the dependence, moment, and kernel assumptions of
#' Chwialkowski et al. (2016); this function does not check them. The sign-change
#' probabilities \eqn{a_n} must asymptotically satisfy \eqn{a_n\to0} and
#' \eqn{na_n\to\infty}.
#'
#' If \eqn{nV_n^{*(1)},\ldots,nV_n^{*(B)}} are the bootstrap draws, the
#' reported right-tail p-value is
#' \deqn{\widehat p=\frac{1+\sum_{b=1}^B
#' \mathbf{1}\{nV_n^{*(b)}\ge nV_n\}}{B+1}.}
#'
#' Warns when off-diagonal Stein-kernel magnitudes are negligible relative to
#' the diagonal, because the resulting bootstrap calibration is degenerate.
#'
#' @inheritParams ksd_u_test
#' @param X Numeric vector or matrix containing \eqn{n} observations. Rows are
#'   observations and columns are coordinates; a vector is treated as an
#'   \eqn{n\times 1} matrix. For Markov calibration, rows must follow the
#'   dependence order.
#' @param boot_method Calibration method: `"rademacher"` for independent
#'   observations or `"markov"` for ordered dependent observations.
#' @param change_prob Sign-change probability \eqn{a_n} for Markov calibration.
#'   It must be supplied and lie strictly between 0 and 1. It is ignored for
#'   Rademacher calibration.
#' @return
#' An `htest` object. `statistic` is \eqn{nV_n}; `p.value` is the bootstrap
#' p-value above. `parameter` records `nboot`, `scaling`, and `change_prob`,
#' plus `imq_beta` for an IMQ kernel; `kernel` contains the resolved
#' `SteinKernel` object. If `return_raw_boot = TRUE`, `bootstrap_samples`
#' contains \eqn{nV_n^{*(1)},\ldots,nV_n^{*(B)}}.
#' @references Chwialkowski, Strathmann, and Gretton (2016), A Kernel Test of
#' Goodness of Fit.
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
  nboot <- validate_integer(nboot, "nboot")
  return_raw_boot <- validate_flag(return_raw_boot, "return_raw_boot")
  resolved_change_prob <- if (identical(boot_method, "markov")) {
    .resolve_markov_change_prob(change_prob)
  } else {
    NA_real_
  }
  prep <- .prepare_ksd_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    imq_beta = imq_beta
  )
  n <- nrow(prep$X)

  block_settings <- .resolve_block_settings(n, block_size, block_threshold)
  # Wild-bootstrap columns are sign sequences; Markov signs retain sample-order dependence.
  W_mat <- .generate_bootstrap_weights(
    n, nboot, boot_method, change_prob = change_prob
  )
  nboot <- ncol(W_mat)

  engine_res <- .compute_ksd_v(
    X = prep$X,
    scores = prep$scores,
    kernel_obj = prep$kernel_obj,
    W_mat = W_mat,
    use_block = block_settings$use_block_mode,
    block_size = block_settings$block_size
  )
  .warn_if_degenerate_stein_matrix(engine_res$offdiag_max, engine_res$diag_max)

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
      change_prob = resolved_change_prob,
      if (prep$kernel_name == "imq" && !is.null(prep$kernel_obj$beta))
        c(imq_beta = prep$kernel_obj$beta)
    ),
    kernel = prep$kernel_obj
  )

  if (return_raw_boot) res$bootstrap_samples <- engine_res$bootstrap_samples
  class(res) <- "htest"
  res
}

#' @rdname ksd_matrix
#' @export
ksd_vq_matrix <- ksd_uq_matrix

#' Compute the KSD-V statistic
#'
#' @details
#' For \eqn{K_{ij}=k_{0,p}(X_i,X_j)}, this function returns
#' \deqn{nV_n=\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^nK_{ij}.}
#' All ordered pairs, including \eqn{i=j}, are included. This function does not
#' compute bootstrap draws or a p-value.
#'
#' @param K0 Finite numeric \eqn{n\times n} Stein-kernel matrix with
#'   \eqn{n\ge 2}, such as the output of [ksd_vq_matrix()].
#' @return
#' One numeric value, \eqn{nV_n}.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_statistic(U)
#' @export
ksd_v_statistic <- function(K0) {
  K0 <- .validate_stein_matrix(K0)
  sum(K0) / nrow(K0)
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
#' @param K0 Finite numeric \eqn{n\times n} Stein-kernel matrix with
#'   \eqn{n\ge 2}, such as the output of [ksd_vq_matrix()].
#' @param nboot Positive integer number of bootstrap draws. Ignored when
#'   `W_mat` is supplied.
#' @param W_mat Optional finite numeric \eqn{n\times B} multiplier matrix. If
#'   `NULL`, signs are generated from `boot_method`. Supplied entries need not
#'   be \eqn{\pm 1}.
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

  W_mat <- if (is.null(W_mat)) {
    .generate_bootstrap_weights(n, nboot, boot_method, change_prob = change_prob)
  } else {
    .validate_bootstrap_weights(W_mat, n)
  }

  as.numeric(colSums((K0 %*% W_mat) * W_mat) / n)
}

.compute_ksd_v <- function(X, scores, kernel_obj, W_mat, use_block, block_size) {
  if (use_block) {
    return(.compute_ksd_v_blocked(X, scores, kernel_obj, W_mat, block_size))
  }

  n <- nrow(X)
  K0 <- stein_kernel_matrix(kernel_obj, X, scores)
  statistic <- sum(K0) / n
  bootstrap_samples <- as.numeric(
    colSums((K0 %*% W_mat) * W_mat) / n
  )

  # Zero the diagonal only for the degeneracy diagnostic.
  k_diag <- diag(K0)
  diag(K0) <- 0

  list(
    statistic = statistic,
    p_value = .bootstrap_pvalue_right_tail(bootstrap_samples, statistic),
    bootstrap_samples = bootstrap_samples,
    diag_max = max(abs(k_diag)),
    offdiag_max = max(abs(K0))
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

      stat_sum <- stat_sum + sum(K0_block)
      # ksd_v_statistic() reports n V_n, so bootstrap blocks are accumulated on the same scale.
      boot_stats <- boot_stats + colSums((K0_block %*% W_j) * W_i) / n

      # Zero the diagonal only for the degeneracy diagnostic.
      if (i_start == j_start) {
        diag_max <- max(diag_max, max(abs(diag(K0_block))))
        diag(K0_block) <- 0
      }
      offdiag_max <- max(offdiag_max, max(abs(K0_block)))
    }
  }

  stat <- stat_sum / n
  list(
    statistic = stat,
    p_value = .bootstrap_pvalue_right_tail(boot_stats, stat),
    bootstrap_samples = as.numeric(boot_stats),
    diag_max = diag_max,
    offdiag_max = offdiag_max
  )
}
