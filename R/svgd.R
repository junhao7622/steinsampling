#' Create an SVGD updater
#'
#' Creates a small object for Stein Variational Gradient Descent, the particle
#' update method of Liu and Wang (2016). SVGD moves a fixed set of particles so
#' that their empirical distribution better approximates the target described by
#' a score function.
#'
#' @details
#' SVGD represents the current approximation by particles, not by a parametric
#' density. Each update asks for the target score
#' \eqn{s_p(x) = \nabla_x \log p(x)}, combines those scores through a pairwise
#' kernel, and adds a derivative term that spreads nearby particles apart. The
#' target density only needs to be known up to a normalizing constant because
#' the update uses the score rather than the density value.
#'
#' The returned object stores a Stein kernel and provides two convenience
#' functions. `svgd_kernel()` computes the kernel matrix and derivative term for
#' a set of particles. `update()` moves particles by calling [update_svgd()] with
#' the stored kernel. The particle state is always an `n x d` matrix: rows are
#' particles and columns are coordinates. The score function used in updates
#' must return the same shape.
#'
#' For the default Gaussian RBF kernel,
#' \deqn{k(x, y) = \exp(-||x - y||^2 / (2 h^2)).}
#' If `h` is not fixed, each update computes all \eqn{n(n-1)/2} pairwise
#' Euclidean distances between the current particles, lets `med` be their
#' median, and sets
#' \deqn{h = \frac{med}{\sqrt{2\log n}}.}
#' The paper writes its RBF as
#' \eqn{\exp(-||x-y||^2/h_{paper})} with
#' \eqn{h_{paper}=med^2/\log n}. Since this package writes the denominator as
#' \eqn{2h^2}, the displayed conversion is exact. No particle subsampling is
#' used for this median. The bandwidth is recomputed from the current particles
#' at every iteration, as in the paper's experiments, so this exact rule costs
#' quadratic time and storage in the particle count. For one particle the
#' pairwise rule is undefined and the implementation uses `h = 1`; this choice
#' is immaterial because the RBF repulsion is then zero. For multiple particles,
#' a zero median is not silently regularized: supply a fixed positive-bandwidth
#' kernel if repeated particles make the paper's rule degenerate.
#'
#' The object returned by `svgd()` is deliberately small: it is a list containing
#' the kernel and two closures. Use `obj$update(...)` for a convenient state-free
#' update call, or use [update_svgd()] directly when the kernel or trace options
#' should be supplied explicitly.
#'
#' The object does not store particles. SVGD particles are passed into each
#' update call and returned as a new matrix. This avoids hidden mutable state:
#' two runs with the same initial `x0`, score function, and settings can be
#' compared directly from their returned matrices.
#'
#' This is parallel to the other Stein methods only at the score/kernel layer.
#' KSD builds scalar Stein-kernel values `k0(x,y)` for testing, FSSD builds
#' finite-location features \eqn{\tau(x)}, Stein thinning uses `k0` to compress
#' an existing sample by choosing row indices, and Stein Points/SP-MCMC
#' construct support points. SVGD instead transports all particles at every
#' iteration. It uses [eval_kernel()] and [grad_x_kernel()] from the same kernel
#' object, but it does not assemble the full pairwise scalar `k0` through
#' [stein_kernel_matrix()]. Moving from a KSD or Stein Points workflow to SVGD
#' keeps the target score function and kernel choice, but changes the required
#' input from a test sample or candidate optimizer to an initial particle matrix
#' `x0` and an iteration count.
#'
#' @param kernel `SteinKernel` object used by SVGD. Defaults to Gaussian RBF.
#'
#' @return
#' A list with three entries:
#' * `kernel`: the stored `SteinKernel` object.
#' * `svgd_kernel(theta, kernel_obj = NULL)`: function returning the pairwise
#'   kernel matrix `Kxy` and summed derivative term `dxkxy` for particles
#'   `theta`.
#' * `update(...)`: function that calls [update_svgd()] using the stored kernel
#'   unless another kernel is supplied.
#'
#' Returning closures is a convenience layer over [update_svgd()]. It keeps the
#' chosen kernel with the updater, while still letting the user pass particle
#' matrices explicitly.
#' @examples
#' obj <- svgd()
#' x0 <- matrix(seq(-2, 2, length.out = 5), ncol = 1)
#' target_score <- function(theta) -theta
#' obj$update(x0, target_score, n_iter = 5, stepsize = 0.05)
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

#' Run SVGD particle updates
#'
#' Applies the SVGD update repeatedly to a particle matrix. The target enters
#' only through `lnprob`, which should return the score at each particle.
#'
#' @details
#' The name `lnprob` is kept for compatibility with common SVGD code, but the
#' function must return the score \eqn{\nabla_x \log p(x)}, not the scalar log
#' density. Rows of `theta` are particles and columns are coordinates, so
#' `lnprob(theta)` should return an `n x d` matrix with the same shape.
#'
#' At each iteration the particles `theta` are moved by
#' \deqn{\theta \leftarrow \theta + \mathrm{stepsize} \times \mathrm{direction}.}
#' Before the optional adaptive scaling, the raw SVGD direction for
#' particle `theta_i` is the empirical version of the paper's optimal Stein
#' direction:
#' \deqn{
#' g(\theta_i) =
#' \frac{1}{n} \sum_{j=1}^n
#' \left[
#' k(\theta_j, \theta_i) s_p(\theta_j)
#' + \nabla_{\theta_j} k(\theta_j, \theta_i)
#' \right].
#' }
#' Here the score at particle `j` is returned by `lnprob`. The first term moves
#' particles toward regions where the target density is high, smoothed through
#' the kernel. The derivative term acts as a repulsive force, keeping particles
#' from collapsing to the same mode. This raw direction is the quantity specified
#' by Algorithm 1. That algorithm leaves the iteration step size
#' \eqn{\epsilon_l} unspecified; the experiments state that AdaGrad is used,
#' and the Bayesian neural-network experiment refers to AdaGrad with momentum,
#' but the paper does not give the coordinatewise recurrence implemented here.
#' Thus the default scaling below is an experimental/implementation-level choice,
#' not part of the analytic definition of the SVGD direction. In implementation
#' terms, `kxy %*% lnpgrad` is the matrix form of
#' \eqn{\sum_j k(\theta_j,\theta_i)s_p(\theta_j)}, and `dxkxy` stores
#' \eqn{\sum_j \nabla_{\theta_j}k(\theta_j,\theta_i)} for each destination
#' particle `i`.
#'
#' Concretely, if `g_l` is the raw direction, the default keeps the exponential
#' moving average
#' \deqn{H_1=g_1^2,\qquad
#' H_l=\alpha H_{l-1}+(1-\alpha)g_l^2}
#' coordinatewise and uses
#' \eqn{g_l/(\delta+\sqrt{H_l})}, with
#' \eqn{\delta=10^{-6}}, before multiplying by the fixed outer `stepsize`.
#' This is closer to RMSProp, or to an AdaGrad-with-momentum implementation,
#' than to classical cumulative-sum AdaGrad. To apply Algorithm 1 without this
#' package scaling, pass an identity adjustment such as
#' `adj_grad = function(grad, ...) grad`. A custom `adj_grad` may otherwise
#' replace only the scaling step. If `trace_iters` is supplied, copies of the
#' particle matrix are saved after those iterations.
#'
#' @param object SVGD object created by [svgd()].
#' @param x0 Numeric matrix of initial particles. A vector is treated as
#'   one-dimensional particles.
#' @param lnprob Function returning the target score at the current particles,
#'   despite the historical name. It should return
#'   \eqn{\nabla_x \log p(x)}, not `log p(x)`, with the same dimensions as `x0`.
#' @param n_iter Number of SVGD iterations.
#' @param stepsize SVGD step size. Custom `adj_grad` functions should not
#'   multiply by this value; it is applied by `update_svgd()`.
#' @param kernel Optional `SteinKernel` overriding `object$kernel`.
#' @param alpha Exponential-decay/momentum coefficient used by the package's
#'   default RMSProp-style scaling. It is an implementation setting, not a
#'   parameter of the raw direction in Algorithm 1.
#' @param adj_grad Optional function for custom scaling of the SVGD direction.
#'   It receives the arguments documented for [custom_adjusted_gradient()] and
#'   must return a matrix with the same dimensions as `theta`.
#' @param trace_iters Optional integer vector of iterations to save. If `NULL`,
#'   return only the final particles.
#'
#' @return
#' If `trace_iters = NULL`, an `n x d` numeric matrix containing the final
#' particle locations after `n_iter` updates. If `trace_iters` is supplied, a
#' list with:
#' * `theta`: final `n x d` particle matrix.
#' * `trace`: list of saved particle matrices; the element names are the
#'   iteration numbers requested in `trace_iters`.
#'
#' The default return is only the final matrix because SVGD is usually used as
#' an iterative transport map from initial particles to final particles. The
#' optional trace is returned only when requested because storing every particle
#' matrix can be large. Keeping the final matrix in `theta` and the selected
#' intermediate matrices in `trace` makes the two use cases explicit: downstream
#' algorithms normally use the final particles, while diagnostics and plots use
#' the saved trajectory.
#' @examples
#' obj <- svgd()
#' x0 <- matrix(seq(-2, 2, length.out = 5), ncol = 1)
#' target_score <- function(theta) -theta
#' update_svgd(obj, x0, target_score, n_iter = 5, stepsize = 0.05)
#' @export
update_svgd <- function(object, x0, lnprob, n_iter = 1000, stepsize = 1e-3,
                        kernel = NULL, alpha = 0.9, adj_grad = NULL,
                        trace_iters = NULL) {
  theta <- validate_samples(x0, min_rows = 1)
  kernel_obj <- if (is.null(kernel)) object$kernel else kernel
  fudge_factor <- 1e-6
  historical_grad <- 0
  trace_enabled <- !is.null(trace_iters)

  if (!is.function(lnprob)) {
    stop("lnprob must be a function returning the target score.", call. = FALSE)
  }
  if (!is.null(adj_grad) && !is.function(adj_grad)) {
    stop("adj_grad must be NULL or a function.", call. = FALSE)
  }

  if (trace_enabled) {
    trace_iters <- sort(unique(as.integer(trace_iters)))
    trace <- vector("list", length(trace_iters))
    names(trace) <- as.character(trace_iters)
  }

  for (iter in seq_len(n_iter)) {
    # `lnprob` is the target score s_p(theta), not the log density itself.
    lnpgrad <- validate_scores(lnprob, theta)

    kernel_out <- compute_svgd_kernel(theta, kernel_obj)
    kxy <- kernel_out$Kxy
    dxkxy <- kernel_out$dxkxy

    # Empirical form of Liu and Wang's optimal Stein direction:
    # kernel-smoothed attraction toward high-density regions plus repulsion.
    grad_theta <- (kxy %*% lnpgrad + dxkxy) / nrow(theta)

    # Package-level RMSProp/AdaGrad-with-momentum scaling. Algorithm 1 specifies
    # grad_theta and epsilon_l, but not this exponential moving-average rule.
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

    # The step size is applied once, after optional custom direction scaling.
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

#' Apply a custom SVGD direction adjustment
#'
#' Calls a user-supplied adjustment function for the scaling part of an SVGD
#' update. It receives the raw SVGD direction and the running squared-gradient
#' history, then returns the direction that [update_svgd()] will multiply by the
#' step size.
#'
#' @details
#' SVGD itself determines the raw direction `grad`. This helper deliberately
#' touches only the rescaling step. Custom adjustment functions are useful when
#' the raw direction should be rescaled in a way other than the package's default
#' RMSProp-style rule. The returned matrix is multiplied by `stepsize` later
#' inside [update_svgd()].
#'
#' This wrapper exists so custom adjustment functions receive a stable set of
#' arguments. It keeps user code from depending on local variable names inside
#' [update_svgd()], while making clear that custom adjustments should return a
#' direction, not a fully stepped particle matrix.
#'
#' @param adj_grad User function that returns the adjusted update direction.
#' @param grad Current raw SVGD direction.
#' @param historical_grad Exponential moving average of `grad^2`, maintained by
#'   [update_svgd()].
#' @param iter Current iteration index.
#' @param theta Current particle matrix.
#' @param stepsize SVGD step size.
#' @param alpha Exponential-decay/momentum coefficient used to update
#'   `historical_grad`.
#' @param fudge_factor Small numerical stabilizer used by the default
#'   RMSProp-style rule.
#'
#' @return
#' Numeric matrix with the same shape as `grad` and `theta`. The matrix is the
#' adjusted direction; [update_svgd()] multiplies it by `stepsize` and adds it
#' to the particles.
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
  theta <- validate_samples(theta, min_rows = 1)
  n <- nrow(theta)
  if (n <= 1L) return(1)

  # Liu and Wang use h_paper = median(distance)^2 / log(n) in
  # exp(-r / h_paper). Our RBF is exp(-r / (2 h^2)), hence the conversion below.
  # Compute the exact all-particle median: no random subsampling or squared-
  # distance median approximation is used.
  med <- stats::median(as.numeric(stats::dist(theta, method = "euclidean")))
  if (!is.finite(med) || med <= 0) {
    stop(
      "SVGD median bandwidth is zero or non-finite; supply a Gaussian RBF kernel with fixed h > 0.",
      call. = FALSE
    )
  }
  med / sqrt(2 * log(n))
}

compute_svgd_kernel <- function(theta, kernel_obj) {
  theta <- validate_samples(theta, min_rows = 1)
  if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && is.null(kernel_obj$h2)) {
    # Preserve preconditioning and every other kernel setting; only the dynamic
    # bandwidth changes from one particle iteration to the next.
    kernel_obj$h2 <- svgd_median_bandwidth(theta)^2
  }

  n <- nrow(theta)
  Kxy <- eval_kernel(kernel_obj, theta)
  grad_arr <- grad_x_kernel(kernel_obj, theta)

  # grad_arr[j, i, ] is ∇_{theta_j} k(theta_j, theta_i). Summing over j gives
  # the repulsive derivative term for each destination particle i.
  dxkxy <- matrix(0, n, ncol(theta))
  for (j in seq_len(ncol(theta))) {
    dxkxy[, j] <- colSums(matrix(grad_arr[, , j], nrow = n, ncol = n))
  }

  list(Kxy = Kxy, dxkxy = dxkxy)
}
