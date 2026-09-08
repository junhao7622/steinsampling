# Alternative Stein kernels introduced for Stein Points.

# Public constructors

#' Create an inverse-log Stein kernel
#'
#' Creates the inverse-log base kernel and wraps it as a `SteinKernel` object.
#'
#' @details
#' The base kernel has the form
#' \deqn{k(x, y) = (\alpha + \log(1 + ||x - y||^2))^\beta.}
#' The parameter `alpha` must be positive and `beta` must be negative; the
#' default \eqn{\beta=-1} gives the plain inverse-log kernel. Compared with a
#' Gaussian RBF kernel, this kernel decays much more slowly as points move
#' apart and retains interactions between distant points.
#'
#' This kernel uses Euclidean distance and does not support preconditioning.
#'
#' @param alpha Positive offset parameter.
#' @param beta Negative exponent.
#'
#' @return A `SteinKernel` object for the inverse-log kernel.
#' @examples
#' stein_kernel_inverse_log(alpha = 2, beta = -0.5)
#' @export
stein_kernel_inverse_log <- function(alpha = 1, beta = -1) {
  .check_inverse_log_alpha_beta(alpha, beta)
  new_stein_kernel(
    "inverse_log",
    alpha = as.numeric(alpha), beta = as.numeric(beta),
    eval = .inverse_log_eval, grad_x = .inverse_log_grad_x,
    trace_mixed = .inverse_log_trace_mixed, k0_diag = .inverse_log_k0_diag
  )
}

#' Create a score-distance IMQ Stein kernel
#'
#' Constructs an IMQ kernel from distances between target-score vectors.
#'
#' @details
#' With \eqn{s_p(x)=\nabla_x\log p(x)}, the base kernel is
#' \deqn{k(x,y)=(\alpha+||s_p(x)-s_p(y)||^2)^\beta.}
#' Its Stein derivatives require `hess_log_p`, which returns an `n x d x d`
#' array containing one Hessian of the target log density per row of `X`.
#' Preconditioning is not supported.
#'
#' @param alpha Positive offset parameter.
#' @param beta Exponent in `(-1, 0)`.
#' @param hess_log_p Function returning an `n x d x d` Hessian array.
#'
#' @return A `SteinKernel` object for the score-distance IMQ kernel.
#' @examples
#' hess_log_p <- function(X) array(-1, dim = c(nrow(as.matrix(X)), 1, 1))
#' stein_kernel_imq_score(alpha = 1.5, beta = -0.25,
#'                       hess_log_p = hess_log_p)
#' @export
stein_kernel_imq_score <- function(alpha = 1, beta = -0.5, hess_log_p) {
  .check_imq_alpha_beta(alpha, beta)
  if (!is.function(hess_log_p)) {
    stop(
      "hess_log_p must be a function returning an n x d x d Hessian array",
      call. = FALSE
    )
  }
  new_stein_kernel(
    "imq_score",
    alpha = as.numeric(alpha), beta = as.numeric(beta),
    hess_log_p = hess_log_p,
    k0_matrix = .imq_score_k0_matrix, k0_diag = .imq_score_k0_diag
  )
}

.check_positive_alpha <- function(alpha) {
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
      alpha <= 0) {
    stop("alpha must be a finite positive scalar.", call. = FALSE)
  }
}

.check_imq_alpha_beta <- function(alpha, beta) {
  .check_positive_alpha(alpha)
  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) ||
      beta <= -1 || beta >= 0) {
    stop("beta must lie in (-1, 0).", call. = FALSE)
  }
}

.check_inverse_log_alpha_beta <- function(alpha, beta) {
  .check_positive_alpha(alpha)
  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) || beta >= 0) {
    stop("beta must be a finite negative scalar.", call. = FALSE)
  }
}


# Inverse-log operations

.inverse_log_eval <- function(k, X, Y, M) {
  (k$alpha + log1p(compute_cross_squared_distance(X, Y)))^k$beta
}

.inverse_log_grad_x <- function(k, X, Y, M) {
  d <- ncol(X); r <- compute_cross_squared_distance(X, Y)
  coef <- 2 * k$beta * (k$alpha + log1p(r))^(k$beta - 1) / (1 + r)
  arr <- array(0, c(nrow(X), nrow(Y), d))
  for (j in seq_len(d)) arr[, , j] <- coef * outer(X[, j], Y[, j], "-")
  arr
}

.inverse_log_trace_mixed <- function(k, X, Y, M) {
  d <- ncol(X); r <- compute_cross_squared_distance(X, Y)
  base <- k$alpha + log1p(r); b <- k$beta
  b * (b - 1) * base^(b - 2) * (-4 * r / (1 + r)^2) +
    b * base^(b - 1) * (-2 * d / (1 + r) + 4 * r / (1 + r)^2)
}

# At x = y, the distance terms vanish.
.inverse_log_k0_diag <- function(k, X, S_X, M) {
  a <- k$alpha
  b <- k$beta
  -2 * b * ncol(X) * a^(b - 1) + a^b * rowSums(S_X * S_X)
}


# Score-distance IMQ operations

# Measure distance between score vectors, not points.
.as_hessian_array <- function(H, n, d) {
  if (!is.numeric(H) || !identical(dim(H), c(n, d, d)) || any(!is.finite(H))) {
    stop("hess_log_p must return a finite numeric n x d x d array", call. = FALSE)
  }
  H
}

.imq_score_k0_matrix <- function(k, X, S_X, Y, S_Y, M) {
  a <- k$alpha; b <- k$beta
  H_X <- .as_hessian_array(k$hess_log_p(X), nrow(X), ncol(X))
  H_Y <- if (identical(X, Y)) H_X else
    .as_hessian_array(k$hess_log_p(Y), nrow(Y), ncol(Y))
  base <- a + compute_cross_squared_distance(S_X, S_Y)
  c0 <- base^b; c1 <- b * base^(b - 1); c2 <- b * (b - 1) * base^(b - 2)

  out <- matrix(0, nrow(X), nrow(Y))
  for (i in seq_len(nrow(X))) {
    Hi <- H_X[i, , ]; sx <- S_X[i, ]
    for (j in seq_len(nrow(Y))) {
      Hj <- H_Y[j, , ]; sy <- S_Y[j, ]
      delta <- sx - sy; HiHj <- Hi %*% Hj
      tr_u <- -2 * sum(diag(HiHj))
      ux_uy <- -4 * as.numeric(crossprod(delta, HiHj %*% delta))
      cs <- -2 * sum(sx * (Hj %*% delta)) + 2 * sum(sy * (Hi %*% delta))
      out[i, j] <- c2[i, j] * ux_uy + c1[i, j] * (tr_u + cs) +
                   c0[i, j] * sum(sx * sy)
    }
  }
  out
}

# At x = y, the trace uses the Hessian's squared Frobenius norm.
.imq_score_k0_diag <- function(k, X, S_X, M) {
  a <- k$alpha
  b <- k$beta
  H <- .as_hessian_array(k$hess_log_p(X), nrow(X), ncol(X))
  j_norm_sq <- vapply(
    seq_len(nrow(X)), function(i) sum(H[i, , ]^2), numeric(1)
  )
  -2 * b * a^(b - 1) * j_norm_sq + a^b * rowSums(S_X * S_X)
}
