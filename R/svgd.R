# Stein variational gradient descent. `svgd()` moves a supplied particle
# matrix; the two private helpers compute its adaptive RBF bandwidth and the
# base-kernel attraction and repulsion terms.

#' Transport particles with Stein variational gradient descent
#'
#' Moves a fixed ensemble of particles toward a target distribution specified
#' by its score function. SVGD uses a base kernel and its first derivative; it
#' does not construct the pairwise Stein kernel \eqn{k_{0,p}}.
#'
#' @details
#' Let \eqn{x_1^{(t)},\ldots,x_m^{(t)}} be the particles at iteration \eqn{t}
#' and \eqn{s_p(x)=\nabla_x\log p(x)}. The raw direction for particle \eqn{i}
#' is
#' \deqn{g_i^{(t)}=\frac{1}{m}\sum_{j=1}^m
#' \left\{k_\rho(x_j^{(t)},x_i^{(t)})s_p(x_j^{(t)})+
#' \nabla_{x_j}k_\rho(x_j^{(t)},x_i^{(t)})\right\}.}
#' The first term moves particles toward regions of larger target density; the
#' second repels nearby particles. Each update costs \eqn{O(m^2d)}.
#'
#' The update is
#' \deqn{x_i^{(t+1)}=x_i^{(t)}+\epsilon\widetilde g_i^{(t)},}
#' where \eqn{\epsilon} is `step_size`. By default, operations being entrywise,
#' \deqn{H^{(1)}=g^{(1)}\odot g^{(1)},\qquad
#' H^{(t)}=\alpha H^{(t-1)}+(1-\alpha)g^{(t)}\odot g^{(t)},}
#' and
#' \deqn{\widetilde g^{(t)}=
#' \frac{g^{(t)}}{10^{-6}+\sqrt{H^{(t)}}}.}
#' Supplying `adj_grad` replaces this adjustment but not the outer
#' multiplication by `step_size`. Use
#' `adj_grad = function(grad, ...) grad` for the unadjusted direction.
#'
#' For a Gaussian RBF kernel without a fixed bandwidth, let \eqn{\rho_t} be the
#' median pairwise Euclidean distance between the current particles. The
#' bandwidth is recomputed at every iteration as
#' \deqn{h_t=\frac{\rho_t}{\sqrt{2\log m}}.}
#' For one particle, \eqn{h_t=1}. A zero or non-finite median requires a fixed
#' positive bandwidth. Fixed-bandwidth RBF and other kernels retain their
#' supplied parameters.
#'
#' @param x0 Finite numeric vector or matrix containing the initial particles.
#'   Rows are particles and columns are coordinates; a vector is treated as an
#'   \eqn{m\times1} matrix.
#' @param score_function Function that accepts the current \eqn{m\times d}
#'   particle matrix and returns an \eqn{m\times d} matrix with row
#'   \eqn{s_p(x_i)^\top}.
#' @param kernel A `SteinKernel` object describing the base kernel. SVGD calls
#'   [eval_kernel()] and [grad_x_kernel()], not [stein_kernel_matrix()].
#' @param n_iter Nonnegative integer number of particle updates.
#' @param step_size Positive finite update size \eqn{\epsilon}.
#' @param alpha Number in \eqn{[0,1)} controlling the running squared-gradient
#'   average in the default adjustment.
#' @param adj_grad Optional adjustment function. It receives `grad`,
#'   `historical_grad`, `iter`, `theta`, `step_size`, `alpha`, and
#'   `fudge_factor`, and must return a finite numeric matrix with the same
#'   dimensions as the particles.
#' @param trace_iters Optional integer vector of iterations to retain. Each
#'   requested matrix is stored after that iteration's update.
#'
#' @return
#' An object of class `"svgd"` containing:
#' * `X`: final \eqn{m\times d} particle matrix.
#' * `D`: target scores evaluated at `X`.
#' * `trace`: requested post-update matrices named by iteration, or `NULL`.
#' * `n_eval`, `cum_n_eval`: per-iteration and cumulative score-evaluation
#'   counts for the update loop. The final evaluation used for `D` is not
#'   included.
#' * `kernel`, `n_iter`, and `step_size`: supplied update settings.
#' * `method` and `call`: method label and matched call.
#' @examples
#' x0 <- matrix(seq(-2, 2, length.out = 5), ncol = 1)
#' target_score <- function(x) -x
#' svgd(x0, target_score, n_iter = 5, step_size = 0.05)
#' @export
svgd <- function(x0, score_function,
                 kernel = stein_kernel(type = "gaussian_rbf"),
                 n_iter = 1000, step_size = 1e-3,
                 alpha = 0.9, adj_grad = NULL, trace_iters = NULL) {
  particles <- .as_rows(x0, "x0")
  n_iter <- validate_integer(n_iter, "n_iter", min_value = 0L)
  if (!is.function(score_function)) {
    stop("score_function must return the target score", call. = FALSE)
  }
  if (!inherits(kernel, "SteinKernel")) {
    stop("kernel must be a SteinKernel object", call. = FALSE)
  }
  if (!is.numeric(step_size) || length(step_size) != 1L ||
      !is.finite(step_size) || step_size <= 0) {
    stop("step_size must be a positive finite scalar", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L ||
      !is.finite(alpha) || alpha < 0 || alpha >= 1) {
    stop("alpha must be a finite scalar in [0, 1)", call. = FALSE)
  }
  if (!is.null(adj_grad) && !is.function(adj_grad)) {
    stop("adj_grad must be NULL or a function", call. = FALSE)
  }

  trace_enabled <- !is.null(trace_iters)
  if (trace_enabled) {
    if (!is.numeric(trace_iters) || any(!is.finite(trace_iters)) ||
        any(trace_iters < 1) ||
        any(abs(trace_iters - round(trace_iters)) > sqrt(.Machine$double.eps)) ||
        any(trace_iters > n_iter)) {
      stop(
        "trace_iters must contain integer iteration numbers between 1 and n_iter",
        call. = FALSE
      )
    }
    trace_iters <- sort(unique(as.integer(round(trace_iters))))
    trace <- vector("list", length(trace_iters))
    names(trace) <- as.character(trace_iters)
  }

  fudge_factor <- 1e-6
  historical_grad <- 0

  for (iter in seq_len(n_iter)) {
    scores <- .as_score_matrix(score_function(particles), particles)
    kernel_terms <- .compute_svgd_kernel(particles, kernel)
    raw_direction <- (
      kernel_terms$kernel_matrix %*% scores + kernel_terms$repulsion
    ) / nrow(particles)

    if (iter == 1L) {
      historical_grad <- raw_direction^2
    } else {
      historical_grad <-
        alpha * historical_grad + (1 - alpha) * raw_direction^2
    }

    direction <- if (is.null(adj_grad)) {
      raw_direction / (fudge_factor + sqrt(historical_grad))
    } else {
      adj_grad(
        grad = raw_direction,
        historical_grad = historical_grad,
        iter = iter,
        theta = particles,
        step_size = step_size,
        alpha = alpha,
        fudge_factor = fudge_factor
      )
    }
    direction <- as.matrix(direction)
    if (!is.numeric(direction) || !identical(dim(direction), dim(particles)) ||
        any(!is.finite(direction))) {
      stop(
        "adj_grad must return a finite numeric matrix with the same dimensions as the particles",
        call. = FALSE
      )
    }

    particles <- particles + step_size * direction

    if (trace_enabled) {
      trace_pos <- match(iter, trace_iters, nomatch = 0L)
      if (trace_pos > 0L) trace[[trace_pos]] <- particles
    }
  }

  # One further score evaluation so that `D` matches the returned particles;
  # the loop's own scores belong to the particles before the last update.
  final_scores <- .as_score_matrix(score_function(particles), particles)
  n_eval <- rep(nrow(particles), n_iter)

  structure(
    list(
      X = particles, D = final_scores,
      n_eval = n_eval, cum_n_eval = cumsum(n_eval),
      method = "svgd", kernel = kernel,
      trace = if (trace_enabled) trace else NULL,
      n_iter = n_iter, step_size = step_size,
      call = match.call()
    ),
    class = "svgd"
  )
}


.svgd_median_bandwidth <- function(particles) {
  n <- nrow(particles)
  if (n <= 1L) return(1)

  med <- stats::median(
    as.numeric(stats::dist(particles, method = "euclidean"))
  )
  if (!is.finite(med) || med <= 0) {
    stop(
      "SVGD median bandwidth is zero or non-finite; supply a Gaussian RBF kernel with fixed h > 0",
      call. = FALSE
    )
  }
  med / sqrt(2 * log(n))
}


.compute_svgd_kernel <- function(particles, kernel) {
  kernel_current <- kernel
  if (inherits(kernel_current, "SteinKernel_gaussian_rbf") &&
      is.null(kernel_current$h2)) {
    kernel_current$h2 <- .svgd_median_bandwidth(particles)^2
  }

  n <- nrow(particles)
  kernel_matrix <- eval_kernel(kernel_current, particles)
  grad_array <- grad_x_kernel(kernel_current, particles)
  repulsion <- matrix(0, n, ncol(particles))
  for (coordinate in seq_len(ncol(particles))) {
    repulsion[, coordinate] <- colSums(
      matrix(grad_array[, , coordinate], nrow = n, ncol = n)
    )
  }

  list(kernel_matrix = kernel_matrix, repulsion = repulsion)
}
