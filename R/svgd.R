#' Create an SVGD updater
#'
#' Creates a small object for Stein Variational Gradient Descent, the particle
#' update method of Liu and Wang (2016). SVGD moves a fixed set of particles so
#' that their empirical distribution better approximates the target described by
#' a score function.
#'
#' @details
#' SVGD represents its approximation by particles rather than a parametric
#' density. Each update takes target scores
#' \eqn{s_p(x) = \nabla_x \log p(x)}, combines them through a pairwise
#' kernel, and adds a derivative term that separates nearby particles. Using
#' scores removes the need for the target's normalizing constant.
#'
#' The object stores a Stein kernel and two functions: `svgd_kernel()` computes
#' its particle kernel matrix and derivative term, while `update()` calls
#' [update_svgd()] to move particles. States and scores are matching `n x d`
#' matrices, with particles in rows and coordinates in columns.
#'
#' For the default Gaussian RBF kernel,
#' \deqn{k(x, y) = \exp(-||x - y||^2 / (2 h^2)).}
#' Without fixed `h`, each update takes the median `med` of all
#' \eqn{n(n-1)/2} current-particle Euclidean distances and sets
#' \deqn{h = \frac{med}{\sqrt{2\log n}}.}
#' The paper's parameterization is \eqn{\exp(-||x-y||^2/h_{paper})}, with
#' \eqn{h_{paper}=med^2/\log n}; the conversion is exact because this package
#' uses denominator \eqn{2h^2}. The median is not subsampled and, as in the
#' experiments, is recomputed each iteration, costing quadratic time and
#' storage. For one particle the undefined rule uses `h = 1`, immaterial because
#' RBF repulsion is zero. For multiple repeated particles, a zero median is not
#' regularized; supply a fixed positive bandwidth.
#'
#' The small list stores only the kernel and closures, not particles. Pass
#' particles into each call and receive a new matrix, avoiding hidden mutable
#' state and making identical `x0`, score, and settings directly comparable.
#' Use `obj$update(...)` for the state-free convenience call or [update_svgd()]
#' when supplying kernel or trace options explicitly.
#'
#' Other Stein methods share only its score/kernel layer: KSD tests scalar
#' `k0(x,y)`, FSSD builds finite-location \eqn{\tau(x)}, Stein thinning uses
#' `k0` to choose existing rows, and Stein Points/SP-MCMC construct support
#' points. SVGD transports every particle, using [eval_kernel()] and
#' [grad_x_kernel()] but not full scalar `k0` from [stein_kernel_matrix()]. It
#' keeps the target score and kernel choice while replacing a test sample or
#' candidate optimizer by initial particles `x0` and an iteration count.
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
#' These closures keep the chosen kernel while particles remain explicit.
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
#' Here `lnprob` returns the score at particle `j`. The first term is
#' kernel-smoothed attraction toward high density; the derivative term repels
#' particles to prevent mode collapse. This raw direction is Algorithm 1, which
#' leaves step size \eqn{\epsilon_l} unspecified. The experiments mention
#' AdaGrad, and the Bayesian neural-network experiment AdaGrad with momentum,
#' but the paper gives no coordinatewise recurrence matching this package.
#' Default scaling is therefore an experimental/implementation choice, not part
#' of the analytic SVGD direction. In code, `kxy %*% lnpgrad` represents
#' \eqn{\sum_j k(\theta_j,\theta_i)s_p(\theta_j)}, and `dxkxy` stores
#' \eqn{\sum_j\nabla_{\theta_j}k(\theta_j,\theta_i)} for destination `i`.
#'
#' For raw direction `g_l`, the default keeps coordinatewise
#' \deqn{H_1=g_1^2,\qquad
#' H_l=\alpha H_{l-1}+(1-\alpha)g_l^2}
#' and uses \eqn{g_l/(\delta+\sqrt{H_l})}, \eqn{\delta=10^{-6}}, before fixed
#' outer `stepsize`. This resembles RMSProp or AdaGrad with momentum more than
#' classical cumulative AdaGrad. For unscaled Algorithm 1, pass
#' `adj_grad = function(grad, ...) grad`; other `adj_grad` functions replace only
#' scaling. `trace_iters` saves particle copies after requested iterations.
#'
#' @param object SVGD object created by [svgd()].
#' @param x0 Numeric matrix of initial particles. A vector is treated as
#'   one-dimensional particles.
#' @param lnprob Function returning the target score at the current particles,
#'   despite the historical name. It should return
#'   \eqn{\nabla_x \log p(x)}, not `log p(x)`, with the same dimensions as `x0`.
#' @param n_iter Nonnegative integer number of SVGD iterations.
#' @param stepsize Positive finite SVGD step size. Custom `adj_grad` functions
#'   should not multiply by this value; it is applied by `update_svgd()`.
#' @param kernel Optional `SteinKernel` overriding `object$kernel`.
#' @param alpha Exponential-decay/momentum coefficient in `[0, 1)` used by the
#'   package's default RMSProp-style scaling. It is an implementation setting,
#'   not a parameter of the raw direction in Algorithm 1.
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
#' The default final matrix supports ordinary transport use; traces are opt-in
#' because particle matrices can be large. `theta` serves downstream algorithms,
#' while selected `trace` states serve diagnostics and plots.
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
  n_iter <- validate_nonnegative_integer(n_iter, "n_iter")
  if (!is.numeric(stepsize) || length(stepsize) != 1L ||
    !is.finite(stepsize) || stepsize <= 0) {
    stop("stepsize must be a positive finite scalar.", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L ||
    !is.finite(alpha) || alpha < 0 || alpha >= 1) {
    stop("alpha must be a finite scalar in [0, 1).", call. = FALSE)
  }
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
    if (!is.numeric(trace_iters) || any(!is.finite(trace_iters)) ||
      any(trace_iters < 1) ||
      any(abs(trace_iters - round(trace_iters)) > sqrt(.Machine$double.eps)) ||
      any(trace_iters > n_iter)) {
      stop("trace_iters must contain integer iteration numbers between 1 and n_iter.",
           call. = FALSE)
    }
    trace_iters <- sort(unique(as.integer(round(trace_iters))))
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
    direction <- as.matrix(direction)
    if (!is.numeric(direction) || !identical(dim(direction), dim(theta)) ||
      any(!is.finite(direction))) {
      stop("adj_grad must return a finite numeric matrix with the same dimensions as theta.",
           call. = FALSE)
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
