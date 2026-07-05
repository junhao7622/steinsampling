#' Stein Variational Gradient Descent.
#'
#' Constructs an SVGD updater using Algorithm 1 from Liu & Wang (2016). The
#' default Gaussian RBF kernel uses the SVGD median-bandwidth rule at each
#' iteration when no explicit bandwidth is supplied.
#'
#' @param kernel `SteinKernel` object used by SVGD. Defaults to Gaussian RBF.
#'
#' @return A list-like SVGD object with `svgd_kernel()` and `update()` methods.
#' @examples
#' svgd()
#' @export
svgd <- function(kernel = stein_kernel(type = "gaussian_rbf")) {
  obj <- list(kernel = kernel)
  obj$svgd_kernel <- function(theta, kernel_obj = NULL) {
    if (is.null(kernel_obj)) kernel_obj <- obj$kernel
    compute_svgd_kernel(theta, kernel_obj)
  }
  obj$update <- function(x0, lnprob, n_iter = 1000, stepsize = 1e-3,
                         kernel = NULL, alpha = 0.9, adj_grad = NULL,
                         trace_iters = NULL) {
    update_svgd(obj, x0, lnprob,
               n_iter = n_iter, stepsize = stepsize,
               kernel = kernel, alpha = alpha, adj_grad = adj_grad,
               trace_iters = trace_iters)
  }
  obj
}

#' Run Stein Variational Gradient Descent updates.
#'
#' Applies SVGD to a set of particles. By default the update direction is
#' adjusted with the AdaGrad-with-momentum rule used in the SVGD experiments.
#' Passing `adj_grad` replaces only this adjusted-gradient calculation; the
#' SVGD direction, bandwidth, and history are still computed inside this
#' function.
#'
#' @param object SVGD object created by [svgd()].
#' @param x0 Numeric matrix of initial particles (`n` x `d`). A vector is
#'   treated as one-dimensional particles.
#' @param lnprob Function returning the score `nabla_x log p(x)` evaluated at
#'   the current particles. It must return an object with the same dimensions
#'   as `x0`.
#' @param n_iter Number of SVGD iterations.
#' @param stepsize Outer SVGD step size. Do not multiply by this inside a custom
#'   `adj_grad`; the update always uses `theta <- theta + stepsize * direction`.
#' @param kernel Optional `SteinKernel` overriding `object$kernel`.
#' @param alpha Momentum parameter for the running average of squared SVGD
#'   directions used by the default AdaGrad adjustment.
#' @param adj_grad Optional function defining a custom adjusted-gradient rule.
#'   If `NULL`, use the default AdaGrad rule
#'   `grad / (fudge_factor + sqrt(historical_grad))`. If supplied, the function
#'   is called by [custom_adjusted_gradient()] with named arguments `grad`,
#'   `historical_grad`, `iter`, `theta`, `stepsize`, `alpha`, and
#'   `fudge_factor`, and must return an update direction with the same
#'   dimensions as `theta`.
#' @param trace_iters Optional integer vector of iterations to save. If `NULL`,
#'   return only the final particles.
#'
#' @return If `trace_iters` is `NULL`, a numeric matrix of final particles.
#'   Otherwise a list with `theta` and named `trace` entries.
#' @examples
#' obj <- svgd()
#' x0 <- matrix(rnorm(5), ncol = 1)
#' score <- function(theta) -theta
#' update_svgd(obj, x0, score, n_iter = 2, stepsize = 0.01)
#' @export
update_svgd <- function(object, x0, lnprob, n_iter = 1000, stepsize = 1e-3,
                        kernel = NULL, alpha = 0.9, adj_grad = NULL,
                        trace_iters = NULL) {
  theta <- as.matrix(x0)
  kernel_obj <- if (is.null(kernel)) object$kernel else kernel
  fudge_factor <- 1e-6
  historical_grad <- 0
  trace_enabled <- !is.null(trace_iters)

  if (!is.null(adj_grad) && !is.function(adj_grad)) {
    stop("adj_grad must be NULL or a function.", call. = FALSE)
  }

  if (trace_enabled) {
    trace_iters <- sort(unique(as.integer(trace_iters)))
    trace <- vector("list", length(trace_iters))
    names(trace) <- as.character(trace_iters)
  }

  for (iter in seq_len(n_iter)) {
    lnpgrad <- lnprob(theta)

    kernel_out <- compute_svgd_kernel(theta, kernel_obj)
    kxy <- kernel_out$Kxy
    dxkxy <- kernel_out$dxkxy

    grad_theta <- (kxy %*% lnpgrad + dxkxy) / nrow(theta)

    if (iter == 1) {
      historical_grad <- historical_grad + grad_theta^2
    } else {
      historical_grad <- alpha * historical_grad + (1 - alpha) * (grad_theta^2)
    }

    if (is.null(adj_grad)) {
      direction <- grad_theta / (fudge_factor + sqrt(historical_grad))
    } else {
      direction <- custom_adjusted_gradient(
        adj_grad = adj_grad,
        grad = grad_theta,
        historical_grad = historical_grad,
        iter = iter,
        theta = theta,
        stepsize = stepsize,
        alpha = alpha,
        fudge_factor = fudge_factor
      )
    }

    theta <- theta + stepsize * direction

    if (trace_enabled) {
      trace_pos <- match(iter, trace_iters, nomatch = 0L)
      if (trace_pos > 0L) trace[[trace_pos]] <- theta
    }
  }

  if (trace_enabled) {
    return(list(theta = theta, trace = trace))
  }

  theta
}

#' Apply a user-defined SVGD adjusted-gradient function.
#'
#' This helper documents and enforces the named-argument contract for custom
#' adjusted-gradient rules passed to [update_svgd()]. It does not update the
#' particles directly; it returns the direction used in
#' `theta <- theta + stepsize * direction`.
#'
#' @param adj_grad User function that returns the update direction.
#' @param grad Current raw SVGD direction, also denoted `phi` in Algorithm 1.
#' @param historical_grad AdaGrad-style running average of `grad^2`, maintained
#'   by [update_svgd()].
#' @param iter Current iteration index.
#' @param theta Current particle matrix.
#' @param stepsize Outer SVGD step size.
#' @param alpha Momentum parameter used to update `historical_grad`.
#' @param fudge_factor Small numerical stabilizer used by the default AdaGrad
#'   rule.
#'
#' @return Numeric matrix giving the update direction.
#' @examples
#' adj <- function(grad, historical_grad, ...) grad / (1 + sqrt(historical_grad))
#' custom_adjusted_gradient(
#'   adj, grad = matrix(1), historical_grad = matrix(1),
#'   iter = 1, theta = matrix(0), stepsize = 0.1,
#'   alpha = 0.9, fudge_factor = 1e-6
#' )
#' @export
custom_adjusted_gradient <- function(adj_grad, grad, historical_grad, iter,
                                     theta, stepsize, alpha, fudge_factor) {
  adj_grad(
    grad = grad,
    historical_grad = historical_grad,
    iter = iter,
    theta = theta,
    stepsize = stepsize,
    alpha = alpha,
    fudge_factor = fudge_factor
  )
}

svgd_median_bandwidth <- function(theta) {
  n <- nrow(theta)
  if (n <= 1L) return(1)

  med2 <- find_median_distance(theta)
  sqrt(med2 / (2 * log(n + 1)))
}

compute_svgd_kernel <- function(theta, kernel_obj) {
  theta <- as.matrix(theta)
  if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && is.null(kernel_obj$h2)) {
    kernel_obj <- stein_kernel(
      type = "gaussian_rbf",
      h = svgd_median_bandwidth(theta)
    )
  }

  n <- nrow(theta)
  Kxy <- eval_kernel(kernel_obj, theta)
  grad_arr <- grad_x_kernel(kernel_obj, theta)

  dxkxy <- matrix(0, n, ncol(theta))
  for (j in seq_len(ncol(theta))) {
    dxkxy[, j] <- colSums(matrix(grad_arr[, , j], nrow = n, ncol = n))
  }

  list(Kxy = Kxy, dxkxy = dxkxy)
}
