# Stein kernels: the Langevin Stein kernel k0 and its four building-block terms.
#
# Usage:
#   k  <- stein_kernel(type = "gaussian_rbf", h = 1.3)
#   K  <- eval_kernel(k, X, Y)                          # k(x,y)
#   G  <- grad_x_kernel(k, X, Y)                        # ∇_x k   (n_x × n_y × d array)
#   Tr <- trace_mixed_kernel(k, X, Y)                   # ∇_x·∇_y k
#   Cr <- cross_kernel(k, X, grads, Y, grads_Y)         # cross / score-coupling term
#   K0 <- stein_kernel_matrix(k, X, grads, Y, grads_Y)  # assembled k0 (all four terms)
#
#   k0(x,y) = s_p(x)^T s_p(y) k + s_p(x)^T grad_y k
#             + s_p(y)^T grad_x k + trace(grad_x grad_y k)
#
# Conventions: gaussian_rbf k(x,y) = exp(-r(x,y) / (2 h²));
#              imq          k(x,y) = (c² + r(x,y))^β  (β < 0, c > 0; c = length scale).
#              r(x,y) = ||x-y||² unless precon = M is supplied, in which case
#              r(x,y) = (x-y)^T M (x-y).



# 1. ASSEMBLY -- stein_kernel_matrix(): the four terms combined into k0
#
# .default            : sums the four leaf generics (works for custom and any
#                       kernel exposing them).
# gaussian_rbf / imq  : fused fast path -- the squared distance and base kernel
#                       are computed ONCE and reused across the three terms.

#' Compute pairwise Stein kernel values
#'
#' Combines a base kernel, its derivatives, and the target scores to form the
#' Stein kernel `k0`. This is the common matrix object used by the package's KSD
#' tests, FSSD features, Stein thinning, Stein Points, and SP-MCMC.
#'
#' @details
#' Liu, Lee, and Jordan call this `u_q(x,x')` for target `q`; here the same
#' quantity is `k0(x,y)` for target `p`, since it also serves Stein thinning and
#' Stein Points. Given score `s_p(x)=\nabla_x\log p(x)` and base kernel `k`,
#' matrix entries are
#' \deqn{
#' k0(x, y) =
#' s_p(x)^T s_p(y) k(x, y)
#' + s_p(x)^T \nabla_y k(x, y)
#' + s_p(y)^T \nabla_x k(x, y)
#' + \nabla_x \cdot \nabla_y k(x, y).
#' }
#' These are, respectively, score-score coupling through `k`, two score-kernel
#' derivative couplings, and the mixed-derivative trace obtained by applying the
#' Langevin Stein operator in both arguments.
#'
#' Points and scores must share row order: `X[i, ]` pairs with `grads[i, ]`, and
#' supplied `Y[j, ]` with `grads_Y[j, ]`. Omitting `Y` reuses `X` and its scores
#' on both sides. This function never estimates scores; top-level functions such
#' as [ksd_u_test()] evaluate `score_function` first.
#'
#' KSD averages these entries; Stein Points and SP-MCMC use them to construct
#' support points; Stein thinning scores which fixed sample row to retain; and
#' FSSD applies the same Stein operator at finite locations through
#' [compute_tau()]. `Y` produces a rectangular `X`-by-`Y` matrix; omitting it
#' produces the square `X` matrix.
#'
#' This central layer prevents higher-level routines from reimplementing `k0`.
#' Gaussian RBF and position-distance IMQ use fused versions of the same
#' formula. [custom_stein_kernel()] objects use the default method, assembling
#' `k0` from three leaf functions when the base kernel depends only on `(X,Y)`.
#' A target-dependent kernel whose value or derivatives require scores instead
#' needs a full `stein_kernel_matrix.<class>()` method;
#' [stein_kernel_imq_score()] is the built-in example.
#'
#' In contrast, [compute_tau()] stores FSSD feature vectors
#' `s_p(x_i)k(x_i,v_j)+\nabla_x k(x_i,v_j)` before inner products, rather than
#' scalar pairwise `k0(x_i,y_j)`; both use the same base methods and score
#' convention.
#'
#' @param kernel A Stein kernel object.
#' @param X Numeric matrix with samples in rows.
#' @param grads Score matrix for `X`.
#' @param Y Optional second sample matrix. If `NULL`, `Y = X`.
#' @param grads_Y Optional score matrix for `Y`. Required when `Y` is supplied.
#' @param ... Additional arguments passed to kernel methods.
#'
#' @return
#' Numeric matrix. If `Y = NULL`, the matrix is `nrow(X) x nrow(X)` and entry
#' `(i, j)` is `k0(X[i, ], X[j, ])`. If `Y` is supplied, the matrix is
#' `nrow(X) x nrow(Y)` and entry `(i, j)` is `k0(X[i, ], Y[j, ])`. The matrix
#' supports each algorithm's use of the pairwise object: KSD
#' averaging, Stein Points/SP-MCMC accumulation, or Stein-thinning row choice.
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' grads <- -X
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' stein_kernel_matrix(kernel, X, grads)
#' @export
stein_kernel_matrix <- function(kernel, X, grads, Y = NULL, grads_Y = NULL, ...) {
  UseMethod("stein_kernel_matrix")
}

#' @export
stein_kernel_matrix.default <- function(kernel, X, grads, Y = NULL, grads_Y = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  X <- inp$X; grads <- inp$grads; Y <- inp$Y; grads_Y <- inp$grads_Y
  tcrossprod(grads, grads_Y) * eval_kernel(kernel, X, Y, ...) +
    cross_kernel(kernel, X, grads, Y, grads_Y, ...) +
    trace_mixed_kernel(kernel, X, Y, ...)
}

#' @export
stein_kernel_matrix.SteinKernel_gaussian_rbf <- function(kernel, X, grads, Y = NULL, grads_Y = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  X <- inp$X; grads <- inp$grads; Y <- inp$Y; grads_Y <- inp$grads_Y
  M <- kernel_precon(kernel, ncol(X))
  sq <- compute_cross_squared_distance(X, Y, M)
  h2 <- resolve_gaussian_rbf_h2(kernel, X, Y)
  k <- exp(-sq / (2 * h2))
  score_disp <- compute_cross_q(X, grads, Y, grads_Y, M)
  tr_M <- kernel_precon_trace(M, ncol(X))
  sq_m2 <- compute_cross_precon2_distance(X, Y, M)
  tcrossprod(grads, grads_Y) * k +
    (1 / h2) * k * score_disp +
    k * ((tr_M / h2) - (sq_m2 / (h2^2)))
}

#' @export
stein_kernel_matrix.SteinKernel_imq <- function(kernel, X, grads, Y = NULL, grads_Y = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  X <- inp$X; grads <- inp$grads; Y <- inp$Y; grads_Y <- inp$grads_Y
  M <- kernel_precon(kernel, ncol(X))
  sq <- compute_cross_squared_distance(X, Y, M)
  score_disp <- compute_cross_q(X, grads, Y, grads_Y, M)
  beta <- kernel$beta
  tr_M <- kernel_precon_trace(M, ncol(X))
  sq_m2 <- compute_cross_precon2_distance(X, Y, M)
  s <- kernel$c^2 + sq
  tcrossprod(grads, grads_Y) * s^beta +
    -2 * beta * s^(beta - 1) * score_disp +
    (-4 * beta * (beta - 1) * sq_m2 * s^(beta - 2) - 2 * beta * tr_M * s^(beta - 1))
}

# #############################################################################
# 2. THE FOUR STEIN TERMS (+ FSSD term grad_theta_v), grouped by term
# #############################################################################

# ---- Generics + defensive base methods --------------------------------------

#' Building blocks for Stein kernels
#'
#' These generic functions define the base kernel, its needed derivatives, and
#' the score-coupling terms. New kernel classes can implement these methods and
#' then work with the rest of the package.
#'
#' @details
#' These functions are the pieces used by the full Stein kernel calculation in
#' [stein_kernel_matrix()]. For rows `x_i = X[i, ]` and `y_j = Y[j, ]`,
#' `eval_kernel()` gives the base kernel matrix
#' \deqn{K_{ij} = k(x_i, y_j).}
#' `grad_x_kernel()` gives first derivatives with respect to the first
#' argument,
#' \deqn{G_{ij\ell} = \partial k(x_i, y_j) / \partial x_\ell.}
#' `trace_mixed_kernel()` gives the matched second-derivative sum
#' \deqn{T_{ij} = \sum_{\ell = 1}^d
#'   \partial^2 k(x_i, y_j) / \partial x_\ell \partial y_\ell.}
#' `cross_kernel()` gives the two score-derivative terms
#' \deqn{C_{ij} =
#' s_p(x_i)^T \nabla_y k(x_i, y_j)
#' + s_p(y_j)^T \nabla_x k(x_i, y_j).}
#' Then [stein_kernel_matrix()] returns
#' \deqn{(s_p(x_i)^T s_p(y_j)) K_{ij} + C_{ij} + T_{ij}.}
#'
#' `grad_theta_v_kernel()` is used by FSSD optimization. It gives the derivative
#' of the FSSD feature at one test location with respect to that location and,
#' for built-in kernels, with respect to the squared kernel scale.
#'
#' These generics are not parallel top-level algorithms; they are a method
#' contract. A user who moves sideways from a built-in kernel to a custom S3
#' kernel must supply methods with the same shapes and the same row ordering.
#' A user who creates a kernel with [custom_stein_kernel()] supplies the same
#' information as functions instead of S3 methods. Either path lets the
#' top-level algorithms call [stein_kernel_matrix()] and [compute_tau()] without
#' changing their own code.
#'
#' These generics are public because they are the contract for kernel
#' extension. A user who only runs [ksd_u_test()] or [stein_points()] normally
#' calls [stein_kernel()] or [custom_stein_kernel()] instead. A user implementing
#' a new S3 kernel class can implement these methods and then reuse the package
#' algorithms without changing those algorithms.
#'
#' The returned shapes are part of the contract: `eval_kernel()` and
#' `trace_mixed_kernel()` return matrices, `grad_x_kernel()` returns an
#' `nrow(X) x nrow(Y) x d` array, `cross_kernel()` returns the matrix of the two
#' score-derivative terms, and `grad_theta_v_kernel()` returns the derivatives
#' needed by FSSD optimization.
#'
#' @param obj A Stein kernel object.
#' @param X Numeric matrix with samples in rows.
#' @param Y Optional second sample matrix.
#' @param grads Score matrix for `X`.
#' @param grads_Y Optional score matrix for `Y`.
#' @param vj One FSSD test location.
#' @param grads_X Score matrix for `X`.
#' @param g_block Matrix used by the FSSD optimization gradient.
#' @param ... Additional arguments passed to methods.
#'
#' @return
#' The return value depends on the generic:
#' * `eval_kernel()` returns the base-kernel matrix with entry `k(X[i, ],
#'   Y[j, ])`.
#' * `grad_x_kernel()` returns an array whose `(i, j, l)` entry is the
#'   derivative of `k(X[i, ], Y[j, ])` with respect to coordinate `l` of
#'   `X[i, ]`.
#' * `trace_mixed_kernel()` returns the matrix of mixed-derivative traces
#'   `sum_l d^2 k(x, y) / dx_l dy_l`.
#' * `cross_kernel()` returns the matrix of the two score-derivative terms that
#'   enter the Stein kernel.
#' * `grad_theta_v_kernel()` returns the derivatives used when FSSD-opt changes
#'   a test location, and for built-in kernels, the squared kernel scale.
#'
#' These shapes are fixed because [stein_kernel_matrix()], [compute_tau()], and
#' FSSD optimization assemble their algebra from the same pieces. Returning
#' matrices and arrays rather than collapsed summaries lets higher-level
#' algorithms combine the pieces in different ways without asking each kernel
#' implementation to know about every algorithm.
#'
#' @name kernel_generics
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' grads <- -X
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' eval_kernel(kernel, X)
#' grad_x_kernel(kernel, X)
#' trace_mixed_kernel(kernel, X)
#' cross_kernel(kernel, X, grads)
#' grad_theta_v_kernel(kernel, X, vj = 0, grads_X = grads, g_block = grads)
NULL

#' @rdname kernel_generics
#' @export
eval_kernel <- function(obj, X, Y = NULL, ...) UseMethod("eval_kernel")
#' @rdname kernel_generics
#' @export
grad_x_kernel <- function(obj, X, Y = NULL, ...) UseMethod("grad_x_kernel")
#' @rdname kernel_generics
#' @export
trace_mixed_kernel <- function(obj, X, Y = NULL, ...) UseMethod("trace_mixed_kernel")
#' @rdname kernel_generics
#' @export
cross_kernel <- function(obj, X, grads, Y = NULL, grads_Y = NULL, ...) UseMethod("cross_kernel")
#' @rdname kernel_generics
#' @export
grad_theta_v_kernel <- function(obj, X, vj, grads_X, g_block, ...) UseMethod("grad_theta_v_kernel")

#' @export
eval_kernel.SteinKernel <- function(obj, X, Y = NULL, ...) stop("No eval_kernel() method for this SteinKernel")
#' @export
grad_x_kernel.SteinKernel <- function(obj, X, Y = NULL, ...) stop("No grad_x_kernel() method for this SteinKernel")
#' @export
trace_mixed_kernel.SteinKernel <- function(obj, X, Y = NULL, ...) stop("No trace_mixed_kernel() method for this SteinKernel")
#' @export
cross_kernel.SteinKernel <- function(obj, X, grads, Y = NULL, grads_Y = NULL, ...) stop("No cross_kernel() method for this SteinKernel")
#' @export
grad_theta_v_kernel.SteinKernel <- function(obj, X, vj, grads_X, g_block, ...) stop("No grad_theta_v_kernel() method for this SteinKernel")

# ---- Term 1: eval_kernel  k(x,y) -------------------------------------------

#' @export
eval_kernel.SteinKernel_gaussian_rbf <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X")
  Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
  M <- kernel_precon(obj, ncol(X))
  sq_dist <- compute_cross_squared_distance(X, Y, M)
  h2 <- resolve_gaussian_rbf_h2(obj, X, Y)
  exp(-sq_dist / (2 * h2))
}

#' @export
eval_kernel.SteinKernel_imq <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X")
  Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
  M <- kernel_precon(obj, ncol(X))
  sq_dist <- compute_cross_squared_distance(X, Y, M)
  (obj$c^2 + sq_dist)^obj$beta
}

#' @export
eval_kernel.SteinKernel_custom <- function(obj, X, Y = NULL, precon = NULL, ...) obj$eval_fn(X, Y, precon)

# ---- Term 2: grad_x_kernel  ∇_x k  (n_x × n_y × d array) -------------------

#' @export
grad_x_kernel.SteinKernel_gaussian_rbf <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X")
  Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
  n_x <- nrow(X); n_y <- nrow(Y); d <- ncol(X)
  h2 <- resolve_gaussian_rbf_h2(obj, X, Y)
  k_mat <- eval_kernel(obj, X, Y)
  grad_arr <- array(0, dim = c(n_x, n_y, d))
  M <- kernel_precon(obj, d)
  X_lin <- if (is.null(M)) X else X %*% M
  Y_lin <- if (is.null(M)) Y else Y %*% M
  coef <- -1 / h2
  for (j in seq_len(d)) {
    grad_arr[, , j] <- coef * outer(X_lin[, j], Y_lin[, j], "-") * k_mat
  }
  grad_arr
}

#' @export
grad_x_kernel.SteinKernel_imq <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X")
  Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
  n_x <- nrow(X); n_y <- nrow(Y); d <- ncol(X)
  M <- kernel_precon(obj, d)
  sq_dist <- compute_cross_squared_distance(X, Y, M)
  X_lin <- if (is.null(M)) X else X %*% M
  Y_lin <- if (is.null(M)) Y else Y %*% M
  factor_mat <- 2 * obj$beta * (obj$c^2 + sq_dist)^(obj$beta - 1)
  grad_arr <- array(0, dim = c(n_x, n_y, d))
  for (j in seq_len(d)) {
    grad_arr[, , j] <- factor_mat * outer(X_lin[, j], Y_lin[, j], "-")
  }
  grad_arr
}

#' @export
grad_x_kernel.SteinKernel_custom <- function(obj, X, Y = NULL, precon = NULL, ...) obj$grad_x_fn(X, Y, precon)

# ---- Term 3: trace_mixed_kernel  ∇_x·∇_y k ---------------------------------

#' @export
trace_mixed_kernel.SteinKernel_gaussian_rbf <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X")
  Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
  h2 <- resolve_gaussian_rbf_h2(obj, X, Y)
  M <- kernel_precon(obj, ncol(X))
  sq_m2 <- compute_cross_precon2_distance(X, Y, M)
  k_mat <- eval_kernel(obj, X, Y)
  tr_M <- kernel_precon_trace(M, ncol(X))
  k_mat * ((tr_M / h2) - (sq_m2 / (h2^2)))
}

#' @export
trace_mixed_kernel.SteinKernel_imq <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X")
  Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
  M <- kernel_precon(obj, ncol(X))
  sq_dist <- compute_cross_squared_distance(X, Y, M)
  sq_m2 <- compute_cross_precon2_distance(X, Y, M)
  tr_M <- kernel_precon_trace(M, ncol(X))
  s <- obj$c^2 + sq_dist
  -4 * obj$beta * (obj$beta - 1) * sq_m2 * s^(obj$beta - 2) -
    2 * obj$beta * tr_M * s^(obj$beta - 1)
}

#' @export
trace_mixed_kernel.SteinKernel_custom <- function(obj, X, Y = NULL, precon = NULL, ...) obj$trace_mixed_fn(X, Y, precon)

# ---- Term 4: cross_kernel  score-displacement coupling ----------------------

#' @export
cross_kernel.SteinKernel_gaussian_rbf <- function(obj, X, grads, Y = NULL, grads_Y = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  h2 <- resolve_gaussian_rbf_h2(obj, inp$X, inp$Y)
  k_mat <- eval_kernel(obj, inp$X, inp$Y)
  M <- kernel_precon(obj, ncol(inp$X))
  score_disp <- compute_cross_q(inp$X, inp$grads, inp$Y, inp$grads_Y, M)
  (1 / h2) * k_mat * score_disp
}

#' @export
cross_kernel.SteinKernel_imq <- function(obj, X, grads, Y = NULL, grads_Y = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  M <- kernel_precon(obj, ncol(inp$X))
  sq_dist <- compute_cross_squared_distance(inp$X, inp$Y, M)
  score_disp <- compute_cross_q(inp$X, inp$grads, inp$Y, inp$grads_Y, M)
  factor_mat <- (obj$c^2 + sq_dist)^(obj$beta - 1)
  -2 * obj$beta * factor_mat * score_disp
}

# Custom cross term is auto-derived from grad_x_fn (assumes symmetry).
#' @export
cross_kernel.SteinKernel_custom <- function(obj, X, grads, Y = NULL, grads_Y = NULL, precon = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  X <- inp$X; Y <- inp$Y; grads <- inp$grads; grads_Y <- inp$grads_Y
  n_x <- nrow(X); n_y <- nrow(Y); d <- ncol(X)

  grad_x <- grad_x_kernel(obj, X, Y, precon = precon)
  if (d == 1 && is.matrix(grad_x)) grad_x <- array(grad_x, dim = c(n_x, n_y, 1L))
  if (!is.array(grad_x) || !all(dim(grad_x) == c(n_x, n_y, d))) {
    stop("custom grad_x_fn must return an array with dim nrow(X) x nrow(Y) x ncol(X)")
  }
  grad_y_reverse <- grad_x_kernel(obj, Y, X, precon = precon)
  if (d == 1 && is.matrix(grad_y_reverse)) grad_y_reverse <- array(grad_y_reverse, dim = c(n_y, n_x, 1L))
  if (!is.array(grad_y_reverse) || !all(dim(grad_y_reverse) == c(n_y, n_x, d))) {
    stop("custom grad_x_fn must return an array with dim nrow(X) x nrow(Y) x ncol(X)")
  }
  out <- matrix(0, n_x, n_y)
  for (j in seq_len(d)) {
    grad_x_j <- grad_x[, , j]
    grad_y_j <- t(grad_y_reverse[, , j])
    out <- out +
      matrix(grads[, j], n_x, n_y) * grad_y_j +
      matrix(grads_Y[, j], n_x, n_y, byrow = TRUE) * grad_x_j
  }
  out
}

# ---- FSSD term: grad_theta_v_kernel  (gradient wrt feature location vj) ------

#' @export
grad_theta_v_kernel.SteinKernel_gaussian_rbf <- function(obj, X, vj, grads_X, g_block, ...) {
  X <- as_kernel_matrix(X, "X"); grads_X <- as.matrix(grads_X); g_block <- as.matrix(g_block)
  if (!is.numeric(vj) || length(vj) != ncol(X)) stop("vj must be a numeric vector of length ncol(X)")
  if (!is.numeric(grads_X) || nrow(grads_X) != nrow(X) || ncol(grads_X) != ncol(X)) {
    stop("grads_X must be a numeric matrix with the same shape as X")
  }
          if (!is.numeric(g_block) || !all(dim(g_block) == dim(X))) {
            stop("g_block must be a numeric matrix with the same dimensions as X")
          }
          if (is.null(obj$h2) || !is.numeric(obj$h2) || length(obj$h2) != 1 || !is.finite(obj$h2) || obj$h2 <= 0) {
            stop("SteinKernel_gaussian_rbf object must contain a positive scalar h2 for optimization")
          }
          h2 <- as.numeric(obj$h2)
          n <- nrow(X); d <- ncol(X)
          M <- kernel_precon(obj, d)
          vj_mat <- matrix(vj, nrow = n, ncol = d, byrow = TRUE)
          delta <- X - vj_mat
          M_delta <- if (is.null(M)) delta else delta %*% M
          sq_norm <- rowSums(M_delta * delta)
          k_xv <- exp(-sq_norm / (2 * h2))
          b <- grads_X - (1 / h2) * M_delta
          feature_weight <- rowSums(g_block * b)
          weighted_feature <- k_xv * feature_weight
          gM <- if (is.null(M)) g_block else g_block %*% M
          list(
            grad_vj = (1 / h2) * (colSums(M_delta * weighted_feature) + colSums(gM * k_xv)),
            grad_param = sum((sq_norm / (2 * h2^2)) * weighted_feature) +
              (1 / h2^2) * sum(k_xv * rowSums(g_block * M_delta))
          )
        }

#' @export
grad_theta_v_kernel.SteinKernel_imq <- function(obj, X, vj, grads_X, g_block, ...) {
  X <- as_kernel_matrix(X, "X"); grads_X <- as.matrix(grads_X); g_block <- as.matrix(g_block)
  if (!is.numeric(vj) || length(vj) != ncol(X)) stop("vj must be a numeric vector of length ncol(X)")
  if (!is.numeric(grads_X) || nrow(grads_X) != nrow(X) || ncol(grads_X) != ncol(X)) {
    stop("grads_X must be a numeric matrix with the same shape as X")
  }
  if (!is.numeric(g_block) || !all(dim(g_block) == dim(X))) {
    stop("g_block must be a numeric matrix with the same dimensions as X")
  }
  if (!is.numeric(obj$c) || length(obj$c) != 1 || !is.finite(obj$c) || obj$c <= 0) {
    stop("SteinKernel_imq object must contain a positive scalar c for optimization")
  }
  if (!is.numeric(obj$beta) || length(obj$beta) != 1 || !is.finite(obj$beta)) {
    stop("SteinKernel_imq object must contain a finite scalar beta")
          }
          beta <- as.numeric(obj$beta); c_param <- as.numeric(obj$c)
          n <- nrow(X); d <- ncol(X)
          M <- kernel_precon(obj, d)
          vj_mat <- matrix(vj, nrow = n, ncol = d, byrow = TRUE)
          delta <- X - vj_mat
          M_delta <- if (is.null(M)) delta else delta %*% M
          sq_norm <- rowSums(M_delta * delta)
          s <- c_param^2 + sq_norm
          t_coef <- 2 * beta * s^(beta - 1)
          u_coef <- 4 * beta * (beta - 1) * s^(beta - 2)
          a_vec <- rowSums(g_block * grads_X)
          b_vec <- rowSums(g_block * M_delta)
          gM <- if (is.null(M)) g_block else g_block %*% M
          grad_vj <- colSums(M_delta * (-t_coef * a_vec - u_coef * b_vec)) +
            colSums(gM * (-t_coef))
  # grad_param is d(obj)/d(offset) where offset = c^2; FSSD-opt optimizes this
  # offset directly (see fssd_kernel_with_scale / scale2_init), so no 2c factor.
  grad_param <- sum((t_coef / 2) * a_vec + (u_coef / 2) * b_vec)
  list(grad_vj = grad_vj, grad_param = grad_param)
}

# Local objective for the numeric custom-gradient path.
custom_local_objective <- function(obj, X, vj, grads_X, g_block) {
  d <- ncol(X)
  v_one <- matrix(vj, nrow = 1, ncol = d)
  k_xv <- eval_kernel(obj, X, v_one)
  grad_arr <- grad_x_kernel(obj, X, v_one)
  grad_k <- matrix(0, nrow = nrow(X), ncol = d)
  for (l in seq_len(d)) grad_k[, l] <- grad_arr[, 1, l]
  k_vec <- as.numeric(k_xv[, 1])
  xi <- grads_X * k_vec + grad_k
  sum(g_block * xi)
}

#' @export
grad_theta_v_kernel.SteinKernel_custom <- function(obj, X, vj, grads_X, g_block, ...) {
  X <- as_kernel_matrix(X, "X"); grads_X <- as.matrix(grads_X); g_block <- as.matrix(g_block)
  if (!is.numeric(vj) || length(vj) != ncol(X)) stop("vj must be a numeric vector of length ncol(X)")
  if (!is.numeric(grads_X) || nrow(grads_X) != nrow(X) || ncol(grads_X) != ncol(X)) {
    stop("grads_X must be a numeric matrix with the same shape as X")
  }
  if (!is.numeric(g_block) || !all(dim(g_block) == dim(X))) {
    stop("g_block must be a numeric matrix with the same dimensions as X")
  }
  if (!is.null(obj$grad_theta_v_fn)) {
    out <- obj$grad_theta_v_fn(X = X, vj = vj, grads_X = grads_X, g_block = g_block, obj = obj)
    if (!is.list(out) || is.null(out$grad_vj)) stop("grad_theta_v_fn must return a list with at least 'grad_vj'")
    grad_vj <- as.numeric(out$grad_vj)
    if (length(grad_vj) != ncol(X)) stop("grad_theta_v_fn$grad_vj must have length ncol(X)")
    grad_param <- if (is.null(out$grad_param)) 0 else as.numeric(out$grad_param)
    if (length(grad_param) != 1 || !is.finite(grad_param)) stop("grad_theta_v_fn$grad_param must be a finite scalar when provided")
    return(list(grad_vj = grad_vj, grad_param = grad_param))
  }
  mode <- if (!is.null(obj$custom_grad_mode)) obj$custom_grad_mode else "error"
  if (mode %in% c("error", "analytic")) stop("Custom kernel optimization requires grad_theta_v_fn in analytic mode.")
  if (!identical(mode, "numeric")) stop("Unknown custom_grad_mode for custom kernel optimization")
  eps <- if (!is.null(obj$numeric_grad_eps)) as.numeric(obj$numeric_grad_eps) else 1e-5
  if (!is.finite(eps) || eps <= 0) eps <- 1e-5
  grad_vj <- numeric(length(vj))
  for (k in seq_along(vj)) {
    v_plus <- vj; v_minus <- vj
    v_plus[k] <- v_plus[k] + eps
    v_minus[k] <- v_minus[k] - eps
    f_plus <- custom_local_objective(obj, X, v_plus, grads_X, g_block)
    f_minus <- custom_local_objective(obj, X, v_minus, grads_X, g_block)
    grad_vj[k] <- (f_plus - f_minus) / (2 * eps)
  }
  list(grad_vj = grad_vj, grad_param = 0)
}


# #############################################################################
# 3. CONSTRUCTORS & KERNEL UTILITIES
# #############################################################################

#' Create a built-in Stein kernel
#'
#' Creates one of the built-in base kernels and stores the parameters needed to
#' assemble the Stein kernel `k0`. The returned object can be passed to KSD,
#' FSSD, SVGD, Stein Points, SP-MCMC, and Stein thinning functions.
#'
#' @details
#' The Gaussian RBF base kernel is
#' \deqn{k(x, y) = \exp(-r(x, y) / (2 h^2)),}
#' where `h` is the bandwidth and `h^2` is the squared scale stored in the
#' object. The IMQ base kernel is
#' \deqn{k(x, y) = (c^2 + r(x, y))^\beta,}
#' where positive `c` is the length scale, `c^2` the squared offset, and `beta`
#' is negative. Both feed the same Stein-kernel machinery: [eval_kernel()],
#' [grad_x_kernel()], [trace_mixed_kernel()], [cross_kernel()], and
#' [stein_kernel_matrix()] dispatch to the corresponding formulas.
#'
#' RBF is common in KSD, FSSD, and SVGD; exponential squared-distance decay can
#' make off-diagonal entries nearly zero when `h` is too small. IMQ has
#' polynomial decay for `beta < 0`, is the Stein-thinning default, and is common
#' in Stein Points/SP-MCMC because their papers study its convergence. Switching
#' from RBF to IMQ keeps `X`, score, and algorithm but replaces `h^2` by `c^2`
#' and selects negative `beta`.
#'
#' By default,
#' \deqn{r(x, y) = ||x - y||^2.}
#' If `precon` is supplied, distances become
#' \deqn{r(x, y) = (x - y)^T precon (x - y).}
#' The base kernel, first derivatives, and mixed trace all use this `r`.
#' Top-level KSD/FSSD pass squared `scaling`: `h^2` for RBF and `c^2` for IMQ.
#' With `scaling = NULL`, [find_median_distance()] returns the squared exact
#' all-pair median; its quadratic time and memory cost favors fixed scales for
#' large samples.
#'
#' An S3 object stores parameters so the base kernel, first derivative, mixed
#' trace, and FSSD optimization derivative dispatch consistently through
#' [stein_kernel_matrix()], [compute_tau()], and the kernel generics.
#'
#' Specialized Stein Points constructors at this layer are
#' [stein_kernel_inverse_log()] for position-distance inverse-log and
#' [stein_kernel_imq_score()] for score-distance IMQ.
#'
#' @param type Kernel type, `"gaussian_rbf"` or `"imq"`.
#' @param sigma,h Gaussian RBF bandwidth. Provide at most one.
#' @param beta IMQ exponent.
#' @param c IMQ length scale.
#' @param precon Optional positive definite matrix used to rescale distances.
#'
#' @return
#' A list with class `"SteinKernel_gaussian_rbf"` or `"SteinKernel_imq"` plus
#' the common class `"SteinKernel"`. For Gaussian RBF kernels it stores `type`,
#' optional squared bandwidth `h2`, and optional `precon`; `h2 = NULL` invokes
#' the median squared-distance rule where supported. IMQ objects store `type`,
#' `beta`, `c`, and optional `precon`. Pass the result to
#' [stein_kernel_matrix()], [ksd_u_test()], [fssd_test()], [stein_points()],
#' [sp_mcmc()], and [stein_thinning()].
#' For adaptive RBF, KSD/FSSD freeze the input-data median scale, whereas SVGD
#' recomputes the paper's dynamic median each iteration. Stein Points, SP-MCMC,
#' coordinate descent, and Stein thinning require explicit fixed `h` because
#' their KSD objectives need one kernel for all candidate and diagonal values.
#' @examples
#' stein_kernel(type = "gaussian_rbf", h = 1)
#' stein_kernel(type = "imq", c = 1, beta = -0.5)
#' @export
stein_kernel <- function(type = c("gaussian_rbf", "imq"),
                         sigma = NULL, h = NULL, beta = -0.5, c = 1, precon = NULL) {
  type <- match.arg(type)
  if (!is.null(sigma) && !is.null(h)) stop("Provide at most one of sigma or h")
  bandwidth <- if (!is.null(h)) h else sigma

          validate_precon <- function(precon) {
            if (is.null(precon)) return(NULL)
            precon <- as.matrix(precon)
            if (!is.numeric(precon) || nrow(precon) != ncol(precon) || !all(is.finite(precon))) {
              stop("precon must be a finite square numeric matrix")
            }
            sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(precon)))
            if (max(abs(precon - t(precon))) > sym_tol) {
              stop("precon must be symmetric positive definite")
            }
            tryCatch(chol(precon), error = function(e) {
              stop("precon must be symmetric positive definite", call. = FALSE)
            })
            precon
          }

  if (type == "gaussian_rbf") {
    h2 <- NULL
    if (!is.null(bandwidth)) {
      if (!is.numeric(bandwidth) || length(bandwidth) != 1 || !is.finite(bandwidth) || bandwidth <= 0) {
        stop("Gaussian RBF bandwidth must be a positive scalar")
      }
      h2 <- as.numeric(bandwidth)^2
    }
    obj <- list(type = "gaussian_rbf", h2 = h2, precon = validate_precon(precon))
    class(obj) <- c("SteinKernel_gaussian_rbf", "SteinKernel")
    return(obj)
  }

  if (!is.numeric(beta) || length(beta) != 1 || !is.finite(beta) || beta >= 0) {
    stop("beta must be a finite negative scalar")
  }
  if (!is.numeric(c) || length(c) != 1 || !is.finite(c) || c <= 0) {
    stop("c must be a finite positive scalar")
  }
  obj <- list(type = "imq", beta = as.numeric(beta), c = as.numeric(c),
              precon = validate_precon(precon))
  class(obj) <- c("SteinKernel_imq", "SteinKernel")
  obj
}

#' Create a custom Stein kernel
#'
#' Wraps user-supplied functions for a base kernel and its derivatives in a
#' `SteinKernel` object. This lets custom kernels use the same package
#' interface as the built-in kernels.
#'
#' @details
#' A custom kernel must provide three functions. `eval_fn(X, Y, precon)` returns
#' the base kernel matrix
#' \deqn{K_{ij} = k(X[i, ], Y[j, ]).}
#' `grad_x_fn(X, Y, precon)` returns an array whose `(i, j, l)` entry is
#' \deqn{\partial k(X[i, ], Y[j, ]) / \partial X[i, l].}
#' `trace_mixed_fn(X, Y, precon)` returns the matrix
#' \deqn{T_{ij} = \sum_l
#' \partial^2 k(X[i, ], Y[j, ]) /
#' \partial X[i, l] \partial Y[j, l].}
#' These are precisely the base-kernel pieces from which
#' [stein_kernel_matrix()] builds the Liu et al. Stein kernel `k0`.
#'
#' The constructor validates computation, not KSD theorem assumptions. Shape
#' checks do not establish symmetry, positive definiteness, membership in the
#' target's Stein class, the boundary/integration-by-parts identity,
#' `C0`-universality, or required Lipschitz and moment bounds; these remain the
#' caller's responsibility.
#'
#' Liu et al.'s independent-sample U test requires a positive-definite base
#' kernel in the relevant Stein class, plus the smoothness, integrability, and
#' moment conditions supporting the Stein identity and degenerate bootstrap;
#' its null limit specifically assumes a finite second moment of induced `k0`.
#'
#' Chwialkowski et al.'s V test uses `C0`-universality so zero discrepancy
#' identifies the target; its wild bootstrap requires `k0` symmetric,
#' positive-definite, null-degenerate, Lipschitz, and satisfying
#' \eqn{E[k0(Z,Z)^2]<\infty}. Dependent observations additionally require the
#' stationary tau-mixing and tuning assumptions in [ksd_v_test()]. Correct
#' matrix and array shapes alone therefore do not define a valid KSD test.
#'
#' The cross term derives from `grad_x_fn(X,Y,precon)` and its swapped call,
#' appropriate for symmetric two-argument kernels whose second-argument
#' derivative follows by swapping and transposing. Nonsymmetric kernels need a
#' full S3 class with `cross_kernel()`.
#'
#' For an ordinary `(X,Y)` kernel, [stein_kernel_matrix()] directly assembles
#' the four Stein terms. Because the wrapper passes no target scores to the three
#' functions, score-dependent kernels such as [stein_kernel_imq_score()] need a
#' full S3 class with their own `stein_kernel_matrix()` method.
#'
#' FSSD optimization additionally needs `grad_theta_v_fn`, or
#' `custom_grad_mode = "numeric"` for slower numerical differentiation. The
#' analytic function accepts `X`, `vj`, `grads_X`, `g_block`, and `obj`, and
#' returns test-location derivative `grad_vj` plus optional scalar
#' kernel-parameter derivative `grad_param`.
#'
#' This functional alternative to a full S3 class stores explicit formulas and
#' lets existing methods call them with built-in matrix shapes. Relative to
#' [stein_kernel()], only how the base kernel and derivatives are supplied
#' changes: top-level algorithms still receive one `SteinKernel`, with samples
#' and scores interpreted row-wise.
#'
#' @param eval_fn Function that evaluates the base kernel.
#' @param grad_x_fn Function that evaluates the gradient with respect to `X`.
#' @param trace_mixed_fn Function that evaluates the mixed derivative trace.
#' @param grad_theta_v_fn Optional function used to optimize FSSD test
#'   locations.
#' @param custom_grad_mode How to handle missing FSSD optimization gradients.
#'
#' @return
#' A list with class `"SteinKernel_custom"` and `"SteinKernel"`. It stores the
#' `eval_fn`, `grad_x_fn`, `trace_mixed_fn`, optional `grad_theta_v_fn`, and
#' `custom_grad_mode`. Like [stein_kernel()] output, it can enter the same public
#' algorithms without exposing whether its implementation is built in or custom.
#' @examples
#' eval_fn <- function(X, Y = NULL, precon = NULL) {
#'   X <- as.matrix(X)
#'   Y <- if (is.null(Y)) X else as.matrix(Y)
#'   diff <- outer(X[, 1], Y[, 1], "-")
#'   exp(-0.5 * diff^2)
#' }
#' grad_fn <- function(X, Y = NULL, precon = NULL) {
#'   X <- as.matrix(X)
#'   Y <- if (is.null(Y)) X else as.matrix(Y)
#'   diff <- outer(X[, 1], Y[, 1], "-")
#'   array(-diff * exp(-0.5 * diff^2), dim = c(nrow(X), nrow(Y), 1))
#' }
#' trace_fn <- function(X, Y = NULL, precon = NULL) {
#'   X <- as.matrix(X)
#'   Y <- if (is.null(Y)) X else as.matrix(Y)
#'   diff <- outer(X[, 1], Y[, 1], "-")
#'   (1 - diff^2) * exp(-0.5 * diff^2)
#' }
#' custom_stein_kernel(eval_fn, grad_fn, trace_fn)
#' @export
custom_stein_kernel <- function(eval_fn, grad_x_fn, trace_mixed_fn,
                                grad_theta_v_fn = NULL,
                                custom_grad_mode = c("error", "analytic", "numeric")) {
  if (!is.function(eval_fn) || !is.function(grad_x_fn) || !is.function(trace_mixed_fn)) {
    stop("eval_fn, grad_x_fn, and trace_mixed_fn must be functions accepting (X, Y = NULL, precon = NULL)")
  }
  if (!is.null(grad_theta_v_fn) && !is.function(grad_theta_v_fn)) {
    stop("grad_theta_v_fn must be NULL or a function")
  }
  custom_grad_mode <- match.arg(custom_grad_mode)
  obj <- list(type = "custom", eval_fn = eval_fn, grad_x_fn = grad_x_fn,
              trace_mixed_fn = trace_mixed_fn, grad_theta_v_fn = grad_theta_v_fn,
              custom_grad_mode = custom_grad_mode)
  class(obj) <- c("SteinKernel_custom", "SteinKernel")
  obj
}

# Resolve a string / SteinKernel into an object with a scalar scale set.
instantiate_kernel <- function(kernel_choice, scale_val, beta = -0.5, precon = NULL) {
  if (inherits(kernel_choice, "SteinKernel")) return(kernel_choice)
  if (!is.character(kernel_choice) || length(kernel_choice) != 1) {
    stop("kernel must be 'gaussian_rbf', 'imq', or a SteinKernel object")
  }
  scale_val <- validate_kernel_squared_scale(scale_val)
  kernel_type <- match.arg(kernel_choice, c("gaussian_rbf", "imq"))
  if (kernel_type == "gaussian_rbf") {
    return(stein_kernel(type = "gaussian_rbf", h = sqrt(scale_val), precon = precon))
  }
  # scale_val is a squared scale (median of squared distances); the IMQ length
  # scale c satisfies c^2 = scale_val, so the offset in (c^2 + r(x,y))^beta
  # is on the same squared-distance scale as the RBF h^2 convention.
  stein_kernel(type = "imq", c = sqrt(scale_val), beta = beta, precon = precon)
}

# Validate a squared kernel scale before inserting it into an existing object.
validate_kernel_squared_scale <- function(scale_val) {
  if (!is.numeric(scale_val) || length(scale_val) != 1L ||
      !is.finite(scale_val) || scale_val <= 0) {
    stop("scaling must be a positive scalar")
  }
  as.numeric(scale_val)
}

# Sequential fixed-KSD algorithms require one RBF bandwidth throughout. SVGD
# is the intentional exception: its paper recomputes a dynamic bandwidth at
# every particle iteration before evaluating the update.
require_fixed_gaussian_rbf <- function(kernel, caller) {
  if (inherits(kernel, "SteinKernel_gaussian_rbf") && is.null(kernel$h2)) {
    stop(
      sprintf(
        "%s requires a Gaussian RBF kernel with fixed h; dynamic median bandwidth is supported only by SVGD.",
        caller
      ),
      call. = FALSE
    )
  }
  invisible(kernel)
}

# Squared scale carried by a built-in kernel (h2 for RBF, c^2 for IMQ).
kernel_scaling_value <- function(kernel_obj) {
  if (inherits(kernel_obj, "SteinKernel_gaussian_rbf")) {
    return(if (is.null(kernel_obj$h2)) NA_real_ else kernel_obj$h2)
  }
  if (inherits(kernel_obj, "SteinKernel_imq")) {
    # Report the squared scale (offset c^2) using the RBF h2 convention.
    return(if (is.null(kernel_obj$c)) NA_real_ else kernel_obj$c^2)
  }
  NA_real_
}

# Human-readable kernel type name.
kernel_type_name <- function(kernel_obj) {
  if (!inherits(kernel_obj, "SteinKernel")) return("unknown")
  if (inherits(kernel_obj, "SteinKernel_gaussian_rbf")) return("gaussian_rbf")
  if (inherits(kernel_obj, "SteinKernel_imq")) return("imq")
  if (inherits(kernel_obj, "SteinKernel_custom")) return("custom")
  "SteinKernel"
}


# #############################################################################
# 4. HELPERS (primitives shared by the terms above)
# #############################################################################

# Coerce input to an n x d numeric matrix (vector -> column, data.frame -> matrix).
as_kernel_matrix <- function(x, arg_name = "X") {
  if (is.null(dim(x))) {
    if (!is.numeric(x)) stop(sprintf("%s must be numeric with at least one row", arg_name))
    x <- matrix(x, ncol = 1)
  } else if (is.data.frame(x)) {
    if (!all(vapply(x, is.numeric, logical(1L)))) {
      stop(sprintf("%s must be numeric with at least one row", arg_name))
    }
    x <- as.matrix(x)
  } else {
    x <- as.matrix(x)
  }
  if (!is.numeric(x) || nrow(x) < 1 || any(!is.finite(x))) {
    stop(sprintf("%s must contain finite numeric values with at least one row", arg_name))
  }
  x
}

# Pairwise squared distances between rows of X and Y (Y = X if NULL). If M is
# supplied, distance is (x-y)^T M (x-y).
compute_cross_squared_distance <- function(X, Y = NULL, M = NULL) {
  X <- as_kernel_matrix(X, "X")
  if (!is.null(M)) {
    M <- as.matrix(M)
    if (nrow(M) != ncol(X) || ncol(M) != ncol(X)) {
      stop("M must be a d x d matrix matching ncol(X)")
    }
    X_M <- X %*% M
  } else {
    X_M <- X
  }
  if (is.null(Y)) {
    x_norm <- rowSums(X_M * X)
    x2 <- matrix(x_norm, nrow = length(x_norm), ncol = length(x_norm))
    return(x2 + t(x2) - 2 * tcrossprod(X_M, X))
  }
  Y <- as_kernel_matrix(Y, "Y")
  if (ncol(X) != ncol(Y)) {
    stop("X and Y must have the same number of columns")
  }
  Y_M <- if (is.null(M)) Y else Y %*% M
  x_norm <- rowSums(X_M * X)
  y_norm <- rowSums(Y_M * Y)
  outer(x_norm, y_norm, "+") - 2 * tcrossprod(X_M, Y)
}

# Squared Euclidean distance after applying M once: (x-y)^T M^2 (x-y).
compute_cross_precon2_distance <- function(X, Y = NULL, M = NULL) {
  if (is.null(M)) return(compute_cross_squared_distance(X, Y))
  compute_cross_squared_distance(as_kernel_matrix(X, "X") %*% M,
                                 if (is.null(Y)) NULL else as_kernel_matrix(Y, "Y") %*% M)
}

# A_ij = (s_p(x_i) - s_p(y_j))^T M (x_i - y_j): score/displacement coupling.
# With M = NULL this reduces to the usual Euclidean expression.
compute_cross_q <- function(X, grads, Y, grads_Y, M = NULL) {
  X_M <- if (is.null(M)) X else X %*% M
  Y_M <- if (is.null(M)) Y else Y %*% M
  vx <- rowSums(grads * X_M)
  vy <- rowSums(grads_Y * Y_M)
  sx_yt <- grads %*% t(Y_M)
  sy_xt <- grads_Y %*% t(X_M)
  sweep(matrix(vx, nrow(X), nrow(Y)), 2, vy, "+") - sx_yt - t(sy_xt)
}

# Resolve the gaussian_rbf bandwidth^2: stored h2 if present, else the exact
# all-pair median heuristic over X / rbind(X, Y).
resolve_gaussian_rbf_h2 <- function(obj, X, Y = NULL) {
  if (!is.null(obj$h2)) return(obj$h2)
  Z <- if (is.null(Y) || identical(X, Y)) X else rbind(X, Y)
  if (nrow(Z) < 2L) return(1)
  M <- kernel_precon(obj, ncol(Z))
  if (is.null(M)) {
    h2 <- find_median_distance(Z)
  } else {
    h2 <- find_median_distance(Z %*% t(chol(M)))
  }
  if (!is.finite(h2) || h2 <= 0) h2 <- 1
  h2
}

kernel_precon <- function(obj, d) {
  M <- obj$precon
  if (is.null(M)) return(NULL)
  M <- as.matrix(M)
  if (nrow(M) != d || ncol(M) != d) {
    stop("precon must be a d x d matrix matching ncol(X)")
  }
  M
}

kernel_precon_trace <- function(M, d) {
  if (is.null(M)) d else sum(diag(M))
}

# Validate / coerce the (X, grads, Y, grads_Y) tuple shared by cross & assembly.
validate_cross_inputs <- function(X, grads, Y = NULL, grads_Y = NULL) {
  X <- as_kernel_matrix(X, "X")
  grads <- as.matrix(grads)
  if (!is.numeric(grads) || nrow(grads) != nrow(X) || ncol(grads) != ncol(X)) {
    stop("grads must be a numeric matrix with the same shape as X")
  }
  if (is.null(Y)) {
    Y <- X
    grads_Y <- grads
  } else {
    Y <- as_kernel_matrix(Y, "Y")
    if (ncol(X) != ncol(Y)) stop("X and Y must have the same number of columns")
    if (is.null(grads_Y)) stop("grads_Y must be provided when Y is not NULL")
    grads_Y <- as.matrix(grads_Y)
    if (!is.numeric(grads_Y) || nrow(grads_Y) != nrow(Y) || ncol(grads_Y) != ncol(Y)) {
      stop("grads_Y must be a numeric matrix with the same shape as Y")
    }
  }
  list(X = X, grads = grads, Y = Y, grads_Y = grads_Y)
}
