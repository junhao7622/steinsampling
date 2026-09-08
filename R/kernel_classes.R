# Kernel constructors and calculations used by the Stein methods.

# Public constructors

#' Create a built-in Stein kernel
#'
#' @details
#' With \eqn{r_M(x,y)=(x-y)^\top M(x-y)} and `M = precon` (identity when
#' `precon = NULL`), the base kernels are
#' \deqn{k_{\mathrm{RBF}}(x,y)=\exp\{-r_M(x,y)/(2h^2)\},\qquad
#'       k_{\mathrm{IMQ}}(x,y)=\{c^2+r_M(x,y)\}^{\beta}.}
#' @param type `"gaussian_rbf"` or `"imq"`.
#' @param sigma,h Gaussian RBF bandwidth \eqn{h>0}. Supply at most one. If both
#'   are `NULL`, the kernel has no fixed bandwidth.
#' @param beta Finite IMQ exponent \eqn{\beta<0}.
#' @param c Positive IMQ length scale \eqn{c}.
#' @param precon Optional symmetric positive-definite \eqn{M}.
#' @return A `SteinKernel` object.
#' @examples
#' stein_kernel("gaussian_rbf", h = 2)
#' stein_kernel("imq", c = 2, beta = -0.25)
#' @export
stein_kernel <- function(type = c("gaussian_rbf", "imq"),
                         sigma = NULL, h = NULL, beta = -0.5, c = 1,
                         precon = NULL) {
  type <- match.arg(type)
  precon <- validate_kernel_precon(precon)

  if (identical(type, "gaussian_rbf")) {
    if (!is.null(sigma) && !is.null(h)) {
      stop("provide at most one of `sigma` or `h`", call. = FALSE)
    }
    bandwidth <- if (!is.null(h)) h else sigma
    # NA marks a scale that exists but is not set yet.
    scale2 <- NA_real_
    if (!is.null(bandwidth)) {
      if (!is.numeric(bandwidth) || length(bandwidth) != 1L ||
          !is.finite(bandwidth) || bandwidth <= 0) {
        stop("Gaussian RBF bandwidth must be a positive scalar", call. = FALSE)
      }
      scale2 <- as.numeric(bandwidth)^2
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
  if (!is.numeric(c) || length(c) != 1L || !is.finite(c) || c <= 0) {
    stop("c must be a finite positive scalar", call. = FALSE)
  }
  new_stein_kernel(
    "imq", precon = precon, beta = as.numeric(beta), scale2 = as.numeric(c)^2,
    eval = .imq_eval, grad_x = .imq_grad_x, trace_mixed = .imq_trace_mixed,
    k0_diag = .imq_k0_diag,
    fssd_grad = .imq_fssd_grad
  )
}

#' Create a Stein kernel from callbacks
#'
#' @details
#' `eval`, `grad_x`, and `trace_mixed` supply the kernel value, its
#' first derivative, and its mixed-derivative trace. Each receives `k`, `X`,
#' `Y`, and `M`; `M` may be `NULL` and each callback decides whether to
#' use it. For \eqn{X\in\mathbb R^{n_X\times d}} and
#' \eqn{Y\in\mathbb R^{n_Y\times d}}, `eval` returns the
#' \eqn{n_X\times n_Y} matrix \eqn{k(X,Y)}, `grad_x` the
#' \eqn{n_X\times n_Y\times d} array \eqn{\nabla_x k(X,Y)}, and
#' `trace_mixed` the \eqn{n_X\times n_Y} matrix
#' \eqn{\mathrm{tr}\{\nabla_x\nabla_y^\top k(X,Y)\}}. Kernel symmetry supplies
#' \eqn{\nabla_y k} by reversing the arguments of `grad_x`.
#'
#' KSD uses all three callbacks; FSSD uses only `eval` and `grad_x`. For
#' FSSD-opt a custom kernel must provide `fssd_grad` or use
#' `custom_grad_mode = "numeric"`. A custom kernel can update its scale when
#' `scale2` is supplied; that path requires
#' `fssd_grad`, because numeric mode differentiates the test locations
#' only.
#'
#' @param eval,grad_x,trace_mixed Callbacks `(k, X, Y, M)`.
#' @param fssd_grad Optional analytic FSSD-opt callback
#'   `(k, X, vj, grads_X, g_block, M)` returning `grad_vj` and, when the kernel
#'   exposes a scale, `grad_param`.
#' @param scale2 Optional positive squared scale.
#' @param precon Optional symmetric positive-definite \eqn{M}.
#' @param custom_grad_mode `"analytic"` or location-only `"numeric"`.
#' @return A `SteinKernel` object containing the supplied callbacks.
#' @examples
#' rbf <- stein_kernel("gaussian_rbf", h = 1)
#' custom_stein_kernel(
#'   eval        = function(k, X, Y, M) eval_kernel(rbf, X, Y),
#'   grad_x      = function(k, X, Y, M) grad_x_kernel(rbf, X, Y),
#'   trace_mixed = function(k, X, Y, M) trace_mixed_kernel(rbf, X, Y)
#' )
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
  if (!is.null(scale2)) scale2 <- validate_kernel_squared_scale(scale2)
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

# NULL entries are dropped. A missing name is how a kernel says it does not
# have that function. The class comes from `type`, so `inherits()` works.
new_stein_kernel <- function(type, precon = NULL, ...) {
  obj <- c(list(type = type, precon = precon), list(...))
  obj <- obj[!vapply(obj, is.null, logical(1L))]
  slots <- intersect(names(obj), .kernel_slots)
  bad <- slots[!vapply(obj[slots], is.function, logical(1L))]
  if (length(bad)) {
    stop(sprintf("kernel slots must be functions: %s",
                 paste(bad, collapse = ", ")), call. = FALSE)
  }
  class(obj) <- c(paste0("SteinKernel_", type), "SteinKernel")
  obj
}

.kernel_slots <- c("eval", "grad_x", "trace_mixed", "k0_matrix", "k0_diag",
                   "fssd_grad")

# Fetch one of the kernel's functions by name. Missing is fine when the
# function is optional.
kernel_op <- function(kernel, slot, required = TRUE) {
  op <- kernel[[slot]]
  if (is.function(op)) return(op)
  if (!required) return(NULL)
  stop(sprintf("the %s kernel does not supply a `%s` operation",
               kernel_type_name(kernel), slot), call. = FALSE)
}

#' Print a Stein kernel
#'
#' @param x A `SteinKernel` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' stein_kernel("imq", c = 1.5, beta = -0.75)
#' @export
print.SteinKernel <- function(x, ...) {
  shown <- intersect(c("scale2", "beta", "alpha"), names(x))
  pars <- vapply(shown, function(nm) {
    v <- x[[nm]]
    if (is.na(v)) sprintf("%s=unset", nm) else sprintf("%s=%g", nm, v)
  }, character(1L))
  cat(sprintf(
    "Stein kernel <%s>%s%s\n", kernel_type_name(x),
    if (length(pars)) paste0("  ", paste(pars, collapse = "  ")) else "",
    if (is.null(x$precon)) "" else sprintf("  precon=%dx%d",
                                           nrow(x$precon), ncol(x$precon))
  ))
  invisible(x)
}


# Public operations

# Each one does the same four things: check the arguments, work out M, call
# the kernel's own function, check the result.

#' Evaluate a base kernel
#'
#' @param obj A `SteinKernel` object.
#' @param X Numeric \eqn{n_X\times d} matrix.
#' @param Y Optional numeric \eqn{n_Y\times d} matrix; `NULL` uses `X`.
#' @param precon Optional preconditioner overriding `obj$precon`.
#' @param ... Ignored.
#' @return Numeric \eqn{n_X\times n_Y} matrix \eqn{B_{ij}=k(x_i,y_j)}.
#' @examples
#' eval_kernel(stein_kernel("gaussian_rbf", h = 1), matrix(c(-1, 0, 1), ncol = 1))
#' @export
eval_kernel <- function(obj, X, Y = NULL, precon = NULL, ...) {
  inp <- validate_kernel_xy(X, Y)
  M <- kernel_precon(obj, precon, ncol(inp$X))
  out <- kernel_op(obj, "eval")(obj, inp$X, inp$Y, M)
  validate_kernel_matrix(out, "eval", nrow(inp$X), nrow(inp$Y))
}

#' Differentiate a base kernel in its first argument
#'
#' @inheritParams eval_kernel
#' @return Numeric \eqn{n_X\times n_Y\times d} array
#'   \eqn{G_{ijr}=\partial k(x_i,y_j)/\partial x_r}.
#' @examples
#' k <- stein_kernel("gaussian_rbf", h = 1)
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' G <- grad_x_kernel(k, X)
#' dim(G)        # n_X x n_Y x d
#' G[, , 1]
#' @export
grad_x_kernel <- function(obj, X, Y = NULL, precon = NULL, ...) {
  inp <- validate_kernel_xy(X, Y)
  M <- kernel_precon(obj, precon, ncol(inp$X))
  out <- kernel_op(obj, "grad_x")(obj, inp$X, inp$Y, M)
  validate_kernel_gradient(out, nrow(inp$X), nrow(inp$Y), ncol(inp$X))
}

#' Mixed-derivative trace of a base kernel
#'
#' @inheritParams eval_kernel
#' @return Numeric \eqn{n_X\times n_Y} matrix
#'   \eqn{H_{ij}=\mathrm{tr}\{\nabla_x\nabla_y^\top k(x_i,y_j)\}}.
#' @examples
#' trace_mixed_kernel(stein_kernel("gaussian_rbf", h = 1),
#'                    matrix(c(-1, 0, 1), ncol = 1))
#' @export
trace_mixed_kernel <- function(obj, X, Y = NULL, precon = NULL, ...) {
  inp <- validate_kernel_xy(X, Y)
  M <- kernel_precon(obj, precon, ncol(inp$X))
  out <- kernel_op(obj, "trace_mixed")(obj, inp$X, inp$Y, M)
  validate_kernel_matrix(out, "trace_mixed", nrow(inp$X), nrow(inp$Y))
}

#' Score-derivative coupling of a base kernel
#'
#' @details
#' Derived from `grad_x` using \eqn{\nabla_y k(x,y)=\nabla_x k(y,x)}, which
#' holds for every symmetric base kernel, so no kernel supplies this term.
#'
#' @inheritParams eval_kernel
#' @param grads,grads_Y Score matrices matching `X` and `Y`.
#' @return Numeric \eqn{n_X\times n_Y} matrix.
#' @examples
#' k <- stein_kernel("gaussian_rbf", h = 1)
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' S <- -X
#' all.equal(stein_kernel_matrix(k, X, S),
#'           tcrossprod(S) * eval_kernel(k, X) + cross_kernel(k, X, S) +
#'             trace_mixed_kernel(k, X))
#' @export
cross_kernel <- function(obj, X, grads, Y = NULL, grads_Y = NULL,
                         precon = NULL, ...) {
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
#' @details
#' With \eqn{s_p(x)=\nabla_x\log p(x)},
#' \deqn{k_{0,p}(x,y)=s_p(x)^\top s_p(y)k(x,y)+s_p(x)^\top\nabla_y k(x,y)
#'   +s_p(y)^\top\nabla_x k(x,y)+\mathrm{tr}\{\nabla_x\nabla_y^\top k(x,y)\}.}
#' @param kernel A `SteinKernel` object.
#' @param X,Y Numeric point matrices; `Y = NULL` uses `X`.
#' @param grads,grads_Y Score matrices matching `X` and `Y`.
#' @param ... Optional `precon` overriding `kernel$precon`.
#' @return Numeric \eqn{n_X\times n_Y} matrix \eqn{K_{ij}=k_{0,p}(x_i,y_j)}.
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' stein_kernel_matrix(stein_kernel("gaussian_rbf", h = 1), X, -X)
#' @export
stein_kernel_matrix <- function(kernel, X, grads, Y = NULL, grads_Y = NULL,
                                ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  M <- kernel_precon(kernel, list(...)$precon, ncol(inp$X))

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
#' @param ... Ignored.
#' @return Numeric vector of length `nrow(X)`.
#' @noRd
k0_diag <- function(kernel, X, S_X, precon = NULL, ...) {
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

#' Gradient of the local FSSD-opt objective
#'
#' @details
#' With \eqn{\xi_p(x,v)=s_p(x)k(x,v)+\nabla_x k(x,v)} and
#' \eqn{\mathcal L_j(v_j)=\sum_i g_i^\top\xi_p(x_i,v_j)}, returns `grad_vj`
#' equal to \eqn{\nabla_{v_j}\mathcal L_j} and, when the kernel exposes a
#' squared scale \eqn{\rho}, `grad_param` equal to
#' \eqn{\partial\mathcal L_j/\partial\rho}.
#'
#' @param obj A `SteinKernel` object.
#' @param X Numeric \eqn{n\times d} matrix.
#' @param vj Test location, a numeric vector of length \eqn{d}.
#' @param grads_X Score matrix matching `X`.
#' @param g_block Numeric \eqn{n\times d} matrix with rows \eqn{g_i^\top}.
#' @param precon Optional preconditioner overriding `obj$precon`.
#' @param ... Ignored.
#' @return List with `grad_vj` and `grad_param`.
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' grad_theta_v_kernel(stein_kernel("gaussian_rbf", h = 1), X = X, vj = 1.5,
#'                     grads_X = -X, g_block = matrix(1, nrow(X), ncol(X)))
#' @export
grad_theta_v_kernel <- function(obj, X, vj, grads_X, g_block, precon = NULL,
                                ...) {
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
  grad_param <- if (is.null(kernel_scale2(obj))) 0 else out$grad_param
  if (length(grad_param) != 1L || !is.finite(grad_param)) {
    stop("this kernel reports a scale, so `fssd_grad` must return a finite grad_param",
         call. = FALSE)
  }
  list(grad_vj = grad_vj, grad_param = as.numeric(grad_param))
}

#' Read or replace a kernel's squared scale
#'
#' @details
#' The squared scale is \eqn{h^2} for Gaussian RBF and \eqn{c^2} for IMQ.
#' Returns `NULL` for a kernel that exposes no scale, and `NA_real_` for one
#' whose scale exists but is not set.
#'
#' @param obj A `SteinKernel` object.
#' @param value Optional new squared scale.
#' @return The squared scale, or the updated kernel when `value` is supplied.
#' @examples
#' k <- stein_kernel("gaussian_rbf", h = 2)
#' kernel_scale2(k)                             # h^2
#' kernel_scale2(kernel_scale2(k, 9))
#' kernel_scale2(stein_kernel("imq", c = 3))    # c^2
#' @export
kernel_scale2 <- function(obj, value = NULL) {
  if (!inherits(obj, "SteinKernel")) {
    stop("`kernel` must be a SteinKernel object", call. = FALSE)
  }
  if (is.null(obj$scale2)) {
    if (is.null(value)) return(NULL)
    stop("this kernel does not support scale optimization", call. = FALSE)
  }
  if (is.null(value)) return(as.numeric(obj$scale2))
  obj$scale2 <- validate_kernel_squared_scale(value)
  obj
}


# Built-in kernel formulas

# X and Y are already checked here, and M is already worked out.

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
  # The vector repeats the n_X by n_Y kernel weights across coordinate pages.
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

# The same for IMQ, where distance 0 leaves c^2 in place of (c^2 + r).
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


# Custom kernels

# The gradient in vj, by nudging each coordinate up and down. The gradient in
# the scale is not done this way: a kernel with a scale must supply its own.
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


# Internal helpers

instantiate_kernel <- function(kernel_choice, scale_val, beta = -0.5,
                               precon = NULL) {
  if (inherits(kernel_choice, "SteinKernel")) return(kernel_choice)
  if (!is.character(kernel_choice) || length(kernel_choice) != 1) {
    stop("kernel must be 'gaussian_rbf', 'imq', or a SteinKernel object",
         call. = FALSE)
  }
  kernel_type <- match.arg(kernel_choice, c("gaussian_rbf", "imq"))
  # `scale_val` is h^2 for RBF and c^2 for IMQ; both are stored as `scale2`.
  base <- if (kernel_type == "gaussian_rbf") {
    stein_kernel("gaussian_rbf", precon = precon)
  } else {
    stein_kernel("imq", beta = beta, precon = precon)
  }
  kernel_scale2(base, scale_val)
}

require_fixed_gaussian_rbf <- function(kernel, caller) {
  if (!inherits(kernel, "SteinKernel")) {
    stop(sprintf("%s requires a SteinKernel object.", caller), call. = FALSE)
  }
  if (inherits(kernel, "SteinKernel_gaussian_rbf") && is.na(kernel$scale2)) {
    stop(sprintf("%s requires a Gaussian RBF kernel with fixed h.", caller),
         call. = FALSE)
  }
  invisible(kernel)
}

kernel_scaling_value <- function(kernel_obj) {
  scale2 <- kernel_scale2(kernel_obj)
  if (is.null(scale2)) NA_real_ else as.numeric(scale2)
}

kernel_type_name <- function(kernel_obj) kernel_obj$type


# Shared distance code

.displacement_array <- function(X, Y, M, coef_mat) {
  X_lin <- if (is.null(M)) X else X %*% M
  Y_lin <- if (is.null(M)) Y else Y %*% M
  out <- array(0, dim = c(nrow(X), nrow(Y), ncol(X)))
  for (j in seq_len(ncol(X))) {
    out[, , j] <- coef_mat * outer(X_lin[, j], Y_lin[, j], "-")
  }
  out
}

# Pairwise (x-y)^T M (x-y), or squared Euclidean distance when M is NULL.
# Apply chol(M) when needed and sum squared coordinate differences directly,
# avoiding cancellation in ||x||^2 + ||y||^2 - 2x^T y at large offsets.
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

# (x-y)^T M^2 (x-y) for every pair.
compute_cross_precon2_distance <- function(X, Y = NULL, M = NULL) {
  if (is.null(M)) return(compute_cross_squared_distance(X, Y))
  compute_cross_squared_distance(X %*% M, if (is.null(Y)) NULL else Y %*% M)
}


kernel_precon_trace <- function(M, d) if (is.null(M)) d else sum(diag(M))

resolve_gaussian_rbf_scale2 <- function(obj, X, Y = NULL, precon = NULL) {
  if (length(obj$scale2) == 1L && !is.na(obj$scale2)) return(obj$scale2)
  Z <- if (is.null(Y) || identical(X, Y)) X else rbind(X, Y)
  if (nrow(Z) < 2L) return(1)
  M <- kernel_precon(obj, precon, ncol(Z))
  h2 <- if (is.null(M)) {
    find_median_distance(Z)
  } else {
    find_median_distance(Z %*% t(chol(M)))
  }
  if (!is.finite(h2) || h2 <= 0) 1 else h2
}

# M for this call: the one passed in, or the kernel's own. The size check is
# here because the constructor does not yet know d.
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

validate_kernel_squared_scale <- function(scale_val) {
  if (!is.numeric(scale_val) || length(scale_val) != 1L ||
      !is.finite(scale_val) || scale_val <= 0) {
    stop("a squared kernel scale must be a positive finite scalar",
         call. = FALSE)
  }
  as.numeric(scale_val)
}

validate_kernel_xy <- function(X, Y = NULL) {
  X <- .as_rows(X, "X")
  Y <- if (is.null(Y)) X else .as_rows(Y, "Y")
  if (ncol(X) != ncol(Y)) {
    stop("X and Y must have the same number of columns", call. = FALSE)
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
  X <- .as_rows(X, "X")
  grads <- as.matrix(grads)
  if (!is.numeric(grads) || !identical(dim(grads), dim(X)) ||
      any(!is.finite(grads))) {
    stop("grads must be a finite numeric matrix with the same shape as X",
         call. = FALSE)
  }
  if (is.null(Y)) return(list(X = X, grads = grads, Y = X, grads_Y = grads))

  Y <- .as_rows(Y, "Y")
  if (ncol(X) != ncol(Y)) {
    stop("X and Y must have the same number of columns", call. = FALSE)
  }
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
