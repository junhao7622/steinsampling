# Kernel constructors come first, then the public kernel operations, then the
# RBF and IMQ formulas, then distance and validation helpers.

#' Create a built-in Stein kernel
#'
#' Creates a Gaussian RBF or IMQ kernel with the chosen parameters.
#'
#' @details
#' With \eqn{r_M(x,y)=(x-y)^\top M(x-y)} and `M = precon` (identity when
#' `precon = NULL`), the base kernels are
#' \deqn{k_{\mathrm{RBF}}(x,y)=\exp\{-r_M(x,y)/(2h^2)\},\qquad
#'       k_{\mathrm{IMQ}}(x,y)=\{c^2+r_M(x,y)\}^{\beta}.}
#' [scale2_kernel()] reads or replaces the squared scale \eqn{h^2} or \eqn{c^2}.
#' For a different kernel formula, [custom_stein_kernel()] accepts user-supplied
#' calculations.
#'
#' Two further kernels, [stein_kernel_inverse_log()] and
#' [stein_kernel_imq_score()], are provided for [stein_points()] and [sp_mcmc()].
#' @param type `"gaussian_rbf"` or `"imq"`.
#' @param h Gaussian RBF bandwidth \eqn{h>0}. If `NULL`, the kernel has no fixed
#'   bandwidth.
#' @param beta Finite IMQ exponent \eqn{\beta<0}.
#' @param c Positive IMQ length scale \eqn{c}.
#' @param precon Optional symmetric positive-definite \eqn{M}.
#' @return `stein_kernel()` returns a `SteinKernel` object. `print()` displays
#' the kernel type, parameters, and the dimensions of the preconditioning
#' matrix, when present, and returns its argument invisibly.
#' @examples
#' kernel <- stein_kernel("gaussian_rbf", h = 2)
#' print(kernel)
#' stein_kernel("imq", c = 2, beta = -0.25)
#' @export
stein_kernel <- function(type = c("gaussian_rbf", "imq"),
                         h = NULL, beta = -0.5, c = 1,
                         precon = NULL) {
  type <- match.arg(type)
  precon <- validate_kernel_precon(precon)

  if (identical(type, "gaussian_rbf")) {
    # NA marks a scale that exists but is not set yet. NULL would mean the
    # kernel has no scale.
    scale2 <- NA_real_
    if (!is.null(h)) {
      validate_positive_scalar(h, "h")
      scale2 <- as.numeric(h)^2
    }
    return(new_stein_kernel(
      "gaussian_rbf", precon = precon, scale2 = scale2,
      eval = .rbf_eval, grad_x = .rbf_grad_x, trace_mixed = .rbf_trace_mixed,
      k0_diag = .rbf_k0_diag,
      fssd_grad = .rbf_fssd_grad
    ))
  }

  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) || beta >= 0) {
    stop("beta must be a finite negative scalar", call. = FALSE)
  }
  validate_positive_scalar(c, "c")
  new_stein_kernel(
    "imq", precon = precon, beta = as.numeric(beta), scale2 = as.numeric(c)^2,
    eval = .imq_eval, grad_x = .imq_grad_x, trace_mixed = .imq_trace_mixed,
    k0_diag = .imq_k0_diag,
    fssd_grad = .imq_fssd_grad
  )
}

#' Create a Stein kernel from callbacks
#'
#' Creates a kernel object from user-supplied functions that compute the
#' kernel value and its derivatives.
#'
#' @details
#' `eval`, `grad_x`, and `trace_mixed` supply the kernel value, its
#' first derivative, and its mixed-derivative trace. `M` may be `NULL`;
#' callbacks implement any preconditioning they support.
#' For \eqn{X\in\mathbb R^{n_X\times d}} and
#' \eqn{Y\in\mathbb R^{n_Y\times d}}, `eval` returns the
#' \eqn{n_X\times n_Y} matrix \eqn{k(X,Y)}, `grad_x` the
#' \eqn{n_X\times n_Y\times d} array \eqn{\nabla_x k(X,Y)}, and
#' `trace_mixed` the \eqn{n_X\times n_Y} matrix
#' \eqn{\mathrm{tr}\{\nabla_x\nabla_y^\top k(X,Y)\}}. Kernel symmetry supplies
#' \eqn{\nabla_y k} by reversing the arguments of `grad_x`.
#' [eval_kernel()], [grad_x_kernel()], and [trace_mixed_kernel()] evaluate these
#' callbacks on supplied points. [stein_kernel_matrix()] combines the kernel
#' calculations with target scores to form the Stein-kernel matrix.
#'
#' KSD uses all three callbacks; FSSD uses only `eval` and `grad_x`. For
#' FSSD-opt a custom kernel must provide `fssd_grad` or use
#' `custom_grad_mode = "numeric"`. A custom kernel can update its scale when
#' `scale2` is supplied; that path requires
#' `fssd_grad`, because numeric mode differentiates the test locations
#' only. [fssd_grad_kernel()] evaluates `fssd_grad` on supplied points,
#' and [fssd_opt_test()] uses the result to optimize the test locations.
#'
#' @param eval,grad_x,trace_mixed Callbacks `(k, X, Y, M)`, where `k` is the
#'   current kernel object and `M` is the preconditioning matrix.
#' @param fssd_grad Optional analytic FSSD-opt callback
#'   `(k, X, vj, grads_X, g_block, M)` returning `grad_vj` and, when the kernel
#'   exposes a scale, `grad_param`.
#' @param scale2 Optional positive squared scale, read inside callbacks with
#'   `scale2_kernel(k)`.
#' @param precon Optional symmetric positive-definite \eqn{M}.
#' @param custom_grad_mode `"analytic"` or location-only `"numeric"`.
#' @return A `SteinKernel` object containing the supplied callbacks.
#' Use `print(kernel)` to display the kernel type, parameters, and the
#' dimensions of the preconditioning matrix, when present.
#' @seealso [stein_kernel()] for built-in kernels.
#' @examples
#' rbf <- stein_kernel("gaussian_rbf", h = 1)
#' kernel <- custom_stein_kernel(
#'   eval = function(k, X, Y, M) eval_kernel(rbf, X, Y, precon = M),
#'   grad_x = function(k, X, Y, M) grad_x_kernel(rbf, X, Y, precon = M),
#'   trace_mixed = function(k, X, Y, M) trace_mixed_kernel(rbf, X, Y, precon = M)
#' )
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' all.equal(stein_kernel_matrix(kernel, X, -X),
#'           stein_kernel_matrix(rbf, X, -X))
#' @export
custom_stein_kernel <- function(eval, grad_x, trace_mixed,
                                fssd_grad = NULL,
                                scale2 = NULL,
                                precon = NULL,
                                custom_grad_mode = c("analytic", "numeric")) {
  custom_grad_mode <- match.arg(custom_grad_mode)
  if (!is.function(eval) || !is.function(grad_x) ||
      !is.function(trace_mixed)) {
    stop(paste0("eval, grad_x, and trace_mixed must be functions ",
                "accepting (k, X, Y, M)"), call. = FALSE)
  }
  if (!is.null(fssd_grad) && !is.function(fssd_grad)) {
    stop("fssd_grad must be NULL or a function", call. = FALSE)
  }
  if (!is.null(scale2)) scale2 <- validate_positive_scalar(scale2, "scale2")
  if (is.null(fssd_grad) && identical(custom_grad_mode, "numeric")) {
    if (!is.null(scale2)) {
      stop(paste0("numeric differentiation of the kernel scale is not implemented; ",
                  "supply fssd_grad with grad_param"), call. = FALSE)
    }
    fssd_grad <- .fssd_grad_numeric
  }

  new_stein_kernel(
    "custom", precon = validate_kernel_precon(precon), scale2 = scale2,
    eval = eval, grad_x = grad_x, trace_mixed = trace_mixed,
    fssd_grad = fssd_grad
  )
}

# NULL entries are dropped, so a kernel without a function has no entry for it.
# All kernels share one class and type names the formula.
new_stein_kernel <- function(type, precon = NULL, ...) {
  obj <- c(list(type = type, precon = precon), list(...))
  obj <- obj[!vapply(obj, is.null, logical(1L))]
  slots <- intersect(names(obj), .kernel_slots)
  bad <- slots[!vapply(obj[slots], is.function, logical(1L))]
  if (length(bad)) {
    stop(sprintf("kernel slots must be functions: %s",
                 paste(bad, collapse = ", ")), call. = FALSE)
  }
  class(obj) <- "SteinKernel"
  obj
}

.kernel_slots <- c("eval", "grad_x", "trace_mixed", "k0_matrix", "k0_diag",
                   "fssd_grad")

kernel_op <- function(kernel, slot, required = TRUE) {
  op <- kernel[[slot]]
  if (is.function(op)) return(op)
  if (!required) return(NULL)
  stop(sprintf("the %s kernel does not supply a `%s` operation",
               kernel$type, slot), call. = FALSE)
}

#' @export
print.SteinKernel <- function(x, ...) {
  shown <- intersect(c("scale2", "alpha", "beta"), names(x))
  pars <- vapply(shown, function(nm) {
    v <- x[[nm]]
    if (is.na(v)) sprintf("%s=unset", nm) else sprintf("%s=%g", nm, v)
  }, character(1L))
  cat(sprintf(
    "Stein kernel <%s>%s%s\n", x$type,
    if (length(pars)) paste0("  ", paste(pars, collapse = "  ")) else "",
    if (is.null(x$precon)) "" else sprintf("  precon=%dx%d",
                                           nrow(x$precon), ncol(x$precon))
  ))
  invisible(x)
}


# Each public operation below checks its arguments, resolves M, calls the
# kernel's own function and checks the result.

#' Evaluate a base kernel and its derivatives
#'
#' Returns the kernel value for each pair of points in `X` and `Y`, the
#' kernel's derivatives with respect to the coordinates of each point in `X`,
#' the mixed-derivative trace \eqn{H}, and the score-derivative coupling
#' \eqn{C}.
#'
#' @rdname kernel_operations
#' @details
#' [trace_mixed_kernel()] returns the matrix \eqn{H} with entries
#' \deqn{H_{ij}=\sum_{r=1}^{d}\partial_{x_r}\partial_{y_r}k(x_i,y_j)
#'   =\mathrm{tr}\{\nabla_x\nabla_y^\top k(x_i,y_j)\}.}
#'
#' [cross_kernel()] returns the matrix \eqn{C} with entries
#' \deqn{C_{ij}=s_p(x_i)^\top\nabla_y k(x_i,y_j)
#'   +s_p(y_j)^\top\nabla_x k(x_i,y_j),}
#' where \eqn{s_p(x)=\nabla\log p(x)}. It is derived from `grad_x` using
#' \eqn{\nabla_y k(x,y)=\nabla_x k(y,x)}, which
#' holds for every symmetric base kernel, so no separate callback is needed.
#' For its role in the complete kernel calculation, see [stein_kernel_matrix()].
#'
#' @param obj A `SteinKernel` object providing the requested operation.
#' @param X Numeric \eqn{n_X\times d} matrix.
#' @param Y Optional numeric \eqn{n_Y\times d} matrix; `NULL` uses `X`.
#' @param precon Optional preconditioner overriding `obj$precon`.
#' @return
#' [eval_kernel()] returns the numeric \eqn{n_X\times n_Y} matrix
#' \eqn{B_{ij}=k(x_i,y_j)}.
#' [grad_x_kernel()] returns the numeric \eqn{n_X\times n_Y\times d} array
#' \eqn{G_{ijr}=\partial k(x_i,y_j)/\partial x_r}.
#' [trace_mixed_kernel()] returns the numeric \eqn{n_X\times n_Y} matrix
#' \eqn{H}.
#' [cross_kernel()] returns the numeric \eqn{n_X\times n_Y} matrix \eqn{C}.
#' @seealso [stein_kernel()] for kernel construction.
#' @examples
#' k <- stein_kernel("gaussian_rbf", h = 1)
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' eval_kernel(k, X)
#' G <- grad_x_kernel(k, X)
#' dim(G)        # n_X x n_Y x d
#' G[, , 1]
#' trace_mixed_kernel(k, X)
#' S <- -X
#' all.equal(stein_kernel_matrix(k, X, S),
#'           tcrossprod(S) * eval_kernel(k, X) + cross_kernel(k, X, S) +
#'             trace_mixed_kernel(k, X))
#' @export
eval_kernel <- function(obj, X, Y = NULL, precon = NULL) {
  inp <- validate_kernel_xy(X, Y)
  M <- kernel_precon(obj, precon, ncol(inp$X))
  out <- kernel_op(obj, "eval")(obj, inp$X, inp$Y, M)
  validate_kernel_matrix(out, "eval", nrow(inp$X), nrow(inp$Y))
}

#' @rdname kernel_operations
#' @export
grad_x_kernel <- function(obj, X, Y = NULL, precon = NULL) {
  inp <- validate_kernel_xy(X, Y)
  M <- kernel_precon(obj, precon, ncol(inp$X))
  out <- kernel_op(obj, "grad_x")(obj, inp$X, inp$Y, M)
  validate_kernel_gradient(out, nrow(inp$X), nrow(inp$Y), ncol(inp$X))
}

#' @rdname kernel_operations
#' @export
trace_mixed_kernel <- function(obj, X, Y = NULL, precon = NULL) {
  inp <- validate_kernel_xy(X, Y)
  M <- kernel_precon(obj, precon, ncol(inp$X))
  out <- kernel_op(obj, "trace_mixed")(obj, inp$X, inp$Y, M)
  validate_kernel_matrix(out, "trace_mixed", nrow(inp$X), nrow(inp$Y))
}

#' @rdname kernel_operations
#' @param grads,grads_Y Score matrices matching `X` and `Y`.
#' @export
cross_kernel <- function(obj, X, grads, Y = NULL, grads_Y = NULL,
                         precon = NULL) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  X <- inp$X; grads <- inp$grads; Y <- inp$Y; grads_Y <- inp$grads_Y
  n_x <- nrow(X); n_y <- nrow(Y); d <- ncol(X)

  grad_x <- grad_x_kernel(obj, X, Y, precon = precon)
  # When Y is NULL, X and Y are the same matrix, so the swapped version is
  # the one just computed.
  grad_y_rev <- if (identical(X, Y)) {
    grad_x
  } else {
    grad_x_kernel(obj, Y, X, precon = precon)
  }

  out <- matrix(0, n_x, n_y)
  for (j in seq_len(d)) {
    out <- out +
      matrix(grads[, j], n_x, n_y) * t(matrix(grad_y_rev[, , j], n_y, n_x)) +
      matrix(grads_Y[, j], n_x, n_y, byrow = TRUE) *
        matrix(grad_x[, , j], n_x, n_y)
  }
  out
}

#' Assemble the pairwise Stein-kernel matrix
#'
#' Returns the Stein-kernel matrix \eqn{K}, with entries
#' \eqn{K_{ij}=k_{0,p}(x_i,y_j)}.
#'
#' @details
#' With \eqn{s_p(x)=\nabla_x\log p(x)},
#' \deqn{k_{0,p}(x,y)=s_p(x)^\top s_p(y)k(x,y)+s_p(x)^\top\nabla_y k(x,y)
#'   +s_p(y)^\top\nabla_x k(x,y)+\mathrm{tr}\{\nabla_x\nabla_y^\top k(x,y)\}.}
#' [eval_kernel()] supplies the base-kernel values, [cross_kernel()] the two
#' score-derivative terms, and [trace_mixed_kernel()] the mixed-derivative trace.
#' For kernel construction, see [stein_kernel()].
#' @param kernel A `SteinKernel` object.
#' @param X,Y Numeric point matrices; `Y = NULL` uses `X`.
#' @param grads,grads_Y Score matrices matching `X` and `Y`.
#' @param precon Optional preconditioner overriding `kernel$precon`.
#' @return Numeric \eqn{n_X\times n_Y} matrix \eqn{K}.
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' stein_kernel_matrix(stein_kernel("gaussian_rbf", h = 1), X, -X)
#' @export
stein_kernel_matrix <- function(kernel, X, grads, Y = NULL, grads_Y = NULL,
                                precon = NULL) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  M <- kernel_precon(kernel, precon, ncol(inp$X))

  whole <- kernel_op(kernel, "k0_matrix", required = FALSE)
  if (is.function(whole)) {
    out <- whole(kernel, inp$X, inp$grads, inp$Y, inp$grads_Y, M)
    return(validate_kernel_matrix(out, "k0_matrix", nrow(inp$X), nrow(inp$Y)))
  }

  # Revalidate the sum because finite terms can overflow when combined.
  out <- tcrossprod(inp$grads, inp$grads_Y) *
    eval_kernel(kernel, inp$X, inp$Y, precon = M) +
    cross_kernel(kernel, inp$X, inp$grads, inp$Y, inp$grads_Y, precon = M) +
    trace_mixed_kernel(kernel, inp$X, inp$Y, precon = M)
  validate_kernel_matrix(out, "k0_matrix", nrow(inp$X), nrow(inp$Y))
}

#' Diagonal of the pairwise Stein-kernel matrix
#'
#' @details
#' Returns \eqn{k_{0,p}(x_i,x_i)}. A kernel that provides `k0_diag` supplies
#' the closed form, which the sequential algorithms need because forming the
#' full \eqn{n\times n} matrix to read its diagonal is \eqn{O(n^2)} work for an
#' \eqn{O(n)} answer. Otherwise the diagonal is assembled one row at a
#' time, which is correct but slower.
#'
#' @param kernel A `SteinKernel` object.
#' @param X Numeric \eqn{n\times d} matrix.
#' @param S_X Numeric \eqn{n\times d} score matrix matching `X`.
#' @param precon Optional preconditioner overriding `kernel$precon`.
#' @return Numeric vector of length `nrow(X)`.
#' @noRd
k0_diag <- function(kernel, X, S_X, precon = NULL) {
  inp <- validate_cross_inputs(X, S_X)
  X <- inp$X; S_X <- inp$grads
  M <- kernel_precon(kernel, precon, ncol(X))

  fast <- kernel_op(kernel, "k0_diag", required = FALSE)
  out <- if (is.function(fast)) {
    as.numeric(fast(kernel, X, S_X, M))
  } else {
    vapply(seq_len(nrow(X)), function(i) {
      stein_kernel_matrix(kernel, X[i, , drop = FALSE], S_X[i, , drop = FALSE],
                          precon = M)[1L, 1L]
    }, numeric(1L))
  }
  if (length(out) != nrow(X) || any(!is.finite(out))) {
    stop("`k0_diag` must return one finite value per row of X", call. = FALSE)
  }
  out
}

#' Compute objective gradients for one FSSD test location
#'
#' Tells an optimizer how moving one test location or changing the kernel
#' scale changes its objective. The input `g_block` specifies how that
#' objective changes with each Stein feature; [fssd_opt_test()] computes
#' this input internally.
#'
#' @details
#' Let \eqn{C} be the objective computed from the Stein features
#' \deqn{\xi_p(x_i,v_j)=s_p(x_i)k(x_i,v_j)+\nabla_x k(x_i,v_j),}
#' where \eqn{s_p} is the target score. Supply
#' \eqn{\partial C/\partial\xi_{p,r}(x_i,v_j)} in `g_block[i, r]`.
#' The function applies the chain rule to obtain the derivatives of \eqn{C}
#' with respect to \eqn{v_j} and the squared kernel scale \eqn{\rho}.
#'
#' For an objective computed from the normalized features in [compute_tau()],
#' take its derivative columns for location \eqn{j} and divide by
#' \eqn{\sqrt{dJ}} before passing them as `g_block`; \eqn{J} is the number of
#' test locations.
#'
#' @param obj A `SteinKernel` object supporting FSSD gradients. A Gaussian RBF
#'   kernel must have a fixed positive bandwidth.
#' @param X Numeric \eqn{n\times d} matrix.
#' @param vj Test location, a numeric vector of length \eqn{d}.
#' @param grads_X Target-score matrix matching `X`.
#' @param g_block Numeric \eqn{n\times d} matrix of the derivatives defined
#'   below, one row per observation and one column per coordinate.
#' @param precon Optional preconditioner overriding `obj$precon`.
#' @return
#' A list with two entries:
#' * `grad_vj`: \eqn{\nabla_{v_j}C}, a numeric vector of length \eqn{d}.
#' * `grad_param`: this location's contribution to \eqn{\partial C/\partial\rho},
#'   one numeric value. Sum over locations when the scale is shared; zero
#'   for a kernel that exposes no squared scale.
#' @references
#' Jitkrittum, W., Xu, W., Szabó, Z., Fukumizu, K., and Gretton, A. (2017).
#' A Linear-Time Kernel Goodness-of-Fit Test.
#' Advances in Neural Information Processing Systems, 30, 262--271.
#' Curran Associates, Inc.
#' \url{https://papers.nips.cc/paper_files/paper/2017/hash/979d472a84804b9f647bc185a877a8b5-Abstract.html}.
#' @examples
#' # Target: N(0, 1); d = J = 1.
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' kernel <- stein_kernel("gaussian_rbf", h = 1)
#' vj <- 1.5
#' tau <- compute_tau(X, -X, matrix(vj, nrow = 1), kernel)
#' sum(tau)  # Objective C
#' g_block <- matrix(1, nrow(X), ncol(X))  # Derivative of sum: 1
#' fssd_grad_kernel(kernel, X, vj, grads_X = -X, g_block = g_block)
#' @export
fssd_grad_kernel <- function(obj, X, vj, grads_X, g_block, precon = NULL) {
  inp <- validate_cross_inputs(X, grads_X)
  X <- inp$X; grads_X <- inp$grads
  vj <- as.numeric(vj)
  g_block <- as.matrix(g_block)
  if (length(vj) != ncol(X) || any(!is.finite(vj)) ||
      !identical(dim(g_block), dim(X)) || any(!is.finite(g_block))) {
    stop("`vj` must have length ncol(X) and `g_block` must match X",
         call. = FALSE)
  }
  M <- kernel_precon(obj, precon, ncol(X))

  out <- kernel_op(obj, "fssd_grad")(obj, X, vj, grads_X, g_block, M)
  grad_vj <- as.numeric(out$grad_vj)
  if (length(grad_vj) != ncol(X) || any(!is.finite(grad_vj))) {
    stop("`fssd_grad` must return a finite grad_vj of length ncol(X)",
         call. = FALSE)
  }
  grad_param <- if (is.null(scale2_kernel(obj))) 0 else out$grad_param
  if (length(grad_param) != 1L || !is.finite(grad_param)) {
    stop("this kernel reports a scale, so `fssd_grad` must return a finite grad_param",
         call. = FALSE)
  }
  list(grad_vj = grad_vj, grad_param = as.numeric(grad_param))
}

#' Compute and set the squared kernel scale
#'
#' [find_median_distance()] computes the squared median Euclidean distance
#' between sample rows. [scale2_kernel()] returns a kernel's squared scale, or
#' an updated kernel when a new value is supplied.
#'
#' @rdname kernel_scales
#' @details
#' For rows \eqn{z_1,\ldots,z_n}, the scale returned by
#' [find_median_distance()] is
#' \deqn{\rho_{\mathrm{med}}
#' =\left\{\mathop{\mathrm{median}}_{i<j}
#' \lVert z_i-z_j\rVert_2\right\}^{2}.}
#' The median is taken before squaring, so \eqn{\rho_{\mathrm{med}}} is a
#' squared scale and enters as `scale2_kernel(obj, rho_med)`, not as an
#' unsquared bandwidth. All \eqn{n(n-1)/2} pairs are used, so
#' the calculation requires quadratic time and memory. Coordinates are not
#' standardized. If the median distance is zero, the function warns and
#' returns the floor value \eqn{10^{-5}}.
#' For its use in test setup, see [ksd_u_test()] and [ksd_v_test()].
#'
#' The squared scale read and set by [scale2_kernel()] is \eqn{h^2} for
#' Gaussian RBF and \eqn{c^2} for IMQ. It is `NULL` for a kernel that exposes
#' no scale, and `NA_real_` for one whose scale exists but is not set.
#' For kernel construction and parameter choices, see [stein_kernel()].
#'
#' @param Z Finite numeric vector, matrix, or data frame with at least two
#'   observations. Matrix rows are observations; a vector is treated as an
#'   \eqn{n\times 1} sample.
#' @param obj A `SteinKernel` object.
#' @param value Optional new squared scale.
#' @return
#' [find_median_distance()] returns the positive scalar
#' \eqn{\rho_{\mathrm{med}}}.
#' [scale2_kernel()] returns the squared scale, or the updated kernel when
#' `value` is supplied.
#' @examples
#' k <- stein_kernel("gaussian_rbf", h = 2)
#' scale2_kernel(k)  # 4 = h^2
#'
#' k <- scale2_kernel(k, 9)
#' scale2_kernel(k)  # 9
#'
#' rho <- find_median_distance(c(0, 1, 3))  # 4 = median(c(1, 2, 3))^2
#' k <- scale2_kernel(k, rho)
#' scale2_kernel(k)  # 4
#' @export
scale2_kernel <- function(obj, value = NULL) {
  if (!inherits(obj, "SteinKernel")) {
    stop("`kernel` must be a SteinKernel object", call. = FALSE)
  }
  if (is.null(obj$scale2)) {
    if (is.null(value)) return(NULL)
    stop("this kernel does not support scale optimization", call. = FALSE)
  }
  if (is.null(value)) return(as.numeric(obj$scale2))
  obj$scale2 <- validate_positive_scalar(value, "value")
  obj
}


# Callers pass checked X and Y and the resolved M, so these formulas do no
# checks.

.rbf_eval <- function(k, X, Y, M) {
  exp(-compute_cross_squared_distance(X, Y, M) /
        (2 * resolve_gaussian_rbf_scale2(k, X, Y, M)))
}

.imq_eval <- function(k, X, Y, M) {
  (k$scale2 + compute_cross_squared_distance(X, Y, M))^k$beta
}

.rbf_grad_x <- function(k, X, Y, M) {
  h2 <- resolve_gaussian_rbf_scale2(k, X, Y, M)
  k_mat <- exp(-compute_cross_squared_distance(X, Y, M) / (2 * h2))
  # as.numeric of k_mat has length n_X times n_Y, so R recycles it over the d
  # slices of the array.
  .displacement_array(X, Y, M, -1 / h2) * as.numeric(k_mat)
}

.imq_grad_x <- function(k, X, Y, M) {
  s <- k$scale2 + compute_cross_squared_distance(X, Y, M)
  .displacement_array(X, Y, M, 2 * k$beta * s^(k$beta - 1))
}

.rbf_trace_mixed <- function(k, X, Y, M) {
  h2 <- resolve_gaussian_rbf_scale2(k, X, Y, M)
  sq <- compute_cross_squared_distance(X, Y, M)
  sq_m2 <- if (is.null(M)) sq else compute_cross_precon2_distance(X, Y, M)
  exp(-sq / (2 * h2)) * (kernel_precon_trace(M, ncol(X)) / h2 - sq_m2 / h2^2)
}

.imq_trace_mixed <- function(k, X, Y, M) {
  sq <- compute_cross_squared_distance(X, Y, M)
  sq_m2 <- if (is.null(M)) sq else compute_cross_precon2_distance(X, Y, M)
  s <- k$scale2 + sq
  -4 * k$beta * (k$beta - 1) * sq_m2 * s^(k$beta - 2) -
    2 * k$beta * kernel_precon_trace(M, ncol(X)) * s^(k$beta - 1)
}


# The diagonal of k0, where x equals y and the distance is 0. For the RBF
# that leaves k = 1, and only two of the four terms of k0 remain.
.rbf_k0_diag <- function(k, X, S_X, M) {
  kernel_precon_trace(M, ncol(X)) / resolve_gaussian_rbf_scale2(k, X, X, M) +
    rowSums(S_X * S_X)
}

# For IMQ the distance on the diagonal is 0, so c^2 + r reduces to c^2.
.imq_k0_diag <- function(k, X, S_X, M) {
  a <- k$scale2
  b <- k$beta
  -2 * b * kernel_precon_trace(M, ncol(X)) * a^(b - 1) + a^b * rowSums(S_X * S_X)
}

.rbf_fssd_grad <- function(k, X, vj, grads_X, g_block, M) {
  # h has to be fixed. A bandwidth recomputed at each call would keep moving
  # the quantity FSSD is optimizing.
  h2 <- k$scale2
  if (is.na(h2)) {
    stop("FSSD-opt needs a Gaussian RBF kernel with fixed h", call. = FALSE)
  }
  delta <- X - matrix(vj, nrow(X), ncol(X), byrow = TRUE)
  M_delta <- if (is.null(M)) delta else delta %*% M
  sq_norm <- rowSums(M_delta * delta)
  k_xv <- exp(-sq_norm / (2 * h2))
  weighted <- k_xv * rowSums(g_block * (grads_X - (1 / h2) * M_delta))
  gM <- if (is.null(M)) g_block else g_block %*% M
  list(
    grad_vj = (1 / h2) * (colSums(M_delta * weighted) + colSums(gM * k_xv)),
    grad_param = sum((sq_norm / (2 * h2^2)) * weighted) +
      (1 / h2^2) * sum(k_xv * rowSums(g_block * M_delta))
  )
}

.imq_fssd_grad <- function(k, X, vj, grads_X, g_block, M) {
  beta <- k$beta
  delta <- X - matrix(vj, nrow(X), ncol(X), byrow = TRUE)
  M_delta <- if (is.null(M)) delta else delta %*% M
  s <- k$scale2 + rowSums(M_delta * delta)
  t_coef <- 2 * beta * s^(beta - 1)
  u_coef <- 4 * beta * (beta - 1) * s^(beta - 2)
  a_vec <- rowSums(g_block * grads_X)
  b_vec <- rowSums(g_block * M_delta)
  gM <- if (is.null(M)) g_block else g_block %*% M
  list(
    grad_vj = colSums(M_delta * (-t_coef * a_vec - u_coef * b_vec)) +
      colSums(gM * (-t_coef)),
    # The derivative is taken in c^2, not in c, so there is no factor of 2c.
    grad_param = sum((t_coef / 2) * a_vec + (u_coef / 2) * b_vec)
  )
}


# Central differences give the gradient in vj. A kernel with a scale must supply
# its own scale gradient, because this function does not estimate it.
.fssd_grad_numeric <- function(k, X, vj, grads_X, g_block, M,
                               eps = 1e-5) {
  objective <- function(v) {
    v_one <- matrix(v, nrow = 1L)
    k_xv <- eval_kernel(k, X, v_one, precon = M)
    grad_arr <- grad_x_kernel(k, X, v_one, precon = M)
    sum(g_block * (grads_X * as.numeric(k_xv[, 1L]) + grad_arr[, 1L, ]))
  }
  grad_vj <- vapply(seq_along(vj), function(r) {
    up <- dn <- vj
    up[r] <- up[r] + eps
    dn[r] <- dn[r] - eps
    (objective(up) - objective(dn)) / (2 * eps)
  }, numeric(1L))
  list(grad_vj = grad_vj, grad_param = 0)
}


instantiate_kernel <- function(kernel_choice, scale_val, beta = -0.5) {
  # scale_val is h^2 for RBF and c^2 for IMQ. Both are stored as scale2.
  base <- if (kernel_choice == "gaussian_rbf") {
    stein_kernel("gaussian_rbf")
  } else {
    stein_kernel("imq", beta = beta)
  }
  scale2_kernel(base, scale_val)
}

require_fixed_gaussian_rbf <- function(kernel, caller) {
  if (!inherits(kernel, "SteinKernel")) {
    stop(sprintf("%s requires a SteinKernel object", caller), call. = FALSE)
  }
  if (identical(kernel$type, "gaussian_rbf") && is.na(kernel$scale2)) {
    stop(sprintf("%s requires a Gaussian RBF kernel with fixed h", caller),
         call. = FALSE)
  }
  invisible(kernel)
}


.displacement_array <- function(X, Y, M, coef_mat) {
  X_lin <- if (is.null(M)) X else X %*% M
  Y_lin <- if (is.null(M)) Y else Y %*% M
  out <- array(0, dim = c(nrow(X), nrow(Y), ncol(X)))
  for (j in seq_len(ncol(X))) {
    out[, , j] <- coef_mat * outer(X_lin[, j], Y_lin[, j], "-")
  }
  out
}

# Pairwise squared distances under M, Euclidean when M is NULL. Differences are
# squared directly, because expanding the norm loses precision when points are
# far from the origin.
compute_cross_squared_distance <- function(X, Y = NULL, M = NULL) {
  if (!is.null(M)) {
    R <- chol(M)
    X <- tcrossprod(X, R)
    if (!is.null(Y)) Y <- tcrossprod(Y, R)
  }
  if (is.null(Y)) Y <- X
  out <- matrix(0, nrow(X), nrow(Y))
  for (j in seq_len(ncol(X))) out <- out + outer(X[, j], Y[, j], "-")^2
  out
}

# Squared distances under M^2 for every pair, needed by the trace term.
compute_cross_precon2_distance <- function(X, Y, M) {
  compute_cross_squared_distance(X %*% M, Y %*% M)
}


kernel_precon_trace <- function(M, d) if (is.null(M)) d else sum(diag(M))

resolve_gaussian_rbf_scale2 <- function(obj, X, Y = NULL, precon = NULL) {
  if (length(obj$scale2) == 1L && !is.na(obj$scale2)) return(obj$scale2)
  Z <- if (is.null(Y) || identical(X, Y)) X else rbind(X, Y)
  # One point has no pairwise distance, so the scale falls back to 1.
  if (nrow(Z) < 2L) return(1)
  M <- kernel_precon(obj, precon, ncol(Z))
  find_median_distance(if (is.null(M)) Z else Z %*% t(chol(M)))
}

# Use the precon passed to this call, else the kernel's own. Its size is checked
# here because the constructor does not know d.
kernel_precon <- function(obj, precon = NULL, d = NULL) {
  if (!inherits(obj, "SteinKernel")) {
    stop("`kernel` must be a SteinKernel object", call. = FALSE)
  }
  validate_kernel_precon(if (!is.null(precon)) precon else obj$precon, d)
}

validate_kernel_precon <- function(precon, d = NULL) {
  if (is.null(precon)) return(NULL)
  precon <- as.matrix(precon)
  if (!is.numeric(precon) || nrow(precon) != ncol(precon) ||
      !all(is.finite(precon))) {
    stop("precon must be a finite square numeric matrix", call. = FALSE)
  }
  if (!is.null(d) && !identical(dim(precon), c(as.integer(d), as.integer(d)))) {
    stop(sprintf("precon must be a %d x %d matrix for these points", d, d),
         call. = FALSE)
  }
  sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(precon)))
  if (max(abs(precon - t(precon))) > sym_tol) {
    stop("precon must be symmetric positive definite", call. = FALSE)
  }
  tryCatch(chol(precon),
           error = function(e) stop("precon must be symmetric positive definite",
                                    call. = FALSE))
  precon
}

validate_kernel_xy <- function(X, Y = NULL) {
  X <- .as_rows(X, "X")
  Y <- if (is.null(Y)) X else .as_rows(Y, "Y")
  if (ncol(X) != ncol(Y)) {
    stop(sprintf("X and Y must have the same number of columns (%d and %d)",
                 ncol(X), ncol(Y)), call. = FALSE)
  }
  list(X = X, Y = Y)
}

validate_kernel_matrix <- function(value, slot, n_x, n_y) {
  value <- as.matrix(value)
  if (!is.numeric(value) || !identical(dim(value), c(n_x, n_y))) {
    stop(sprintf("kernel `%s` must return a numeric %d x %d matrix",
                 slot, n_x, n_y), call. = FALSE)
  }
  if (any(!is.finite(value))) {
    stop(sprintf("kernel `%s` must return only finite values", slot),
         call. = FALSE)
  }
  value
}

validate_kernel_gradient <- function(value, n_x, n_y, d) {
  if (d == 1L && is.matrix(value) && identical(dim(value), c(n_x, n_y))) {
    value <- array(value, dim = c(n_x, n_y, 1L))
  }
  if (!is.numeric(value) || !identical(dim(value), c(n_x, n_y, d))) {
    stop(sprintf("kernel `grad_x` must return a numeric %d x %d x %d array",
                 n_x, n_y, d), call. = FALSE)
  }
  if (any(!is.finite(value))) {
    stop("kernel `grad_x` must return only finite values", call. = FALSE)
  }
  value
}

validate_cross_inputs <- function(X, grads, Y = NULL, grads_Y = NULL) {
  inp <- validate_kernel_xy(X, Y)
  X <- inp$X
  grads <- as.matrix(grads)
  if (!is.numeric(grads) || !identical(dim(grads), dim(X)) ||
      any(!is.finite(grads))) {
    stop("grads must be a finite numeric matrix with the same shape as X",
         call. = FALSE)
  }
  if (is.null(Y)) return(list(X = X, grads = grads, Y = X, grads_Y = grads))

  Y <- inp$Y
  if (is.null(grads_Y)) {
    stop("grads_Y must be provided when Y is not NULL", call. = FALSE)
  }
  grads_Y <- as.matrix(grads_Y)
  if (!is.numeric(grads_Y) || !identical(dim(grads_Y), dim(Y)) ||
      any(!is.finite(grads_Y))) {
    stop("grads_Y must be a finite numeric matrix with the same shape as Y",
         call. = FALSE)
  }
  list(X = X, grads = grads, Y = Y, grads_Y = grads_Y)
}
