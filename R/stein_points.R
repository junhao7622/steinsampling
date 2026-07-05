# Stein Points: sequential point selection by KSD minimisation.

#' Stein Points sequence.
#'
#' Selects points one at a time. The first point is `x_init`, or the maximiser
#' of `log_p`. Later points minimise either the greedy or herding Stein
#' objective against the points already selected.
#'
#' @param score_function Function returning scores, `n x d`, for candidate rows.
#' @param kernel Built-in Stein kernel, S3 object with `stein_kernel_matrix()`,
#'   function `kernel(X, S_X, Y, S_Y)`, or list with `k0_matrix`. Custom
#'   kernels should return `k0(X_i, Y_j)` and should be symmetric Stein kernels.
#' @param n_points Number of points to select.
#' @param d Dimension of each point.
#' @param optimizer Function taking `(objective, X_curr)` and returning
#'   `x_min`, `d_min`, `f_min`, and `n_eval`.
#' @param method `"greedy"` minimises `k0(x,x) + 2 sum_i k0(x_i,x)`;
#'   `"herding"` minimises `sum_i k0(x_i,x)`.
#' @param log_p Optional log density used to choose the first point.
#' @param x_init Optional first point. If supplied, `log_p` is not used.
#' @param c2 Positive truncation constant when `truncation != "none"`.
#' @param truncation Optional Theorem 1 candidate filter based on
#'   `k0(x,x) <= R_j^2`. The main Stein Herding algorithm uses `"none"`.
#' @param seed Optional local RNG seed.
#'
#' @return List with selected points `X`, scores `D`, running `ksd`, evaluation
#'   counts, and method settings.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' opt <- fmin_grid(lb = -1, ub = 1, n0 = 3, grow = FALSE)
#' stein_points(score, kernel, n_points = 2, d = 1, optimizer = opt,
#'              log_p = log_p)
#' @export
stein_points <- function(score_function, kernel, n_points, d, optimizer,
                         method = c("greedy", "herding"),
                         log_p = NULL, x_init = NULL, c2 = NULL,
                         truncation = c("none", "upper", "lower", "linear"),
                         seed = NULL) {
  method <- match.arg(method); truncation <- match.arg(truncation)
  n_points <- validate_positive_integer(n_points, "n_points")
  d <- validate_positive_integer(d, "d")
  herding <- identical(method, "herding")
  .stein_points_check_kernel(kernel)

  use_trunc <- truncation != "none"
  if (use_trunc && (!is.numeric(c2) || length(c2) != 1L || c2 <= 0))
    stop("`c2` must be positive when truncation != 'none'.", call. = FALSE)
  if (!is.null(log_p) && !is.function(log_p))
    stop("`log_p` must be a function.", call. = FALSE)
  if (!is.null(x_init)) {
    x_init <- as.numeric(x_init)
    if (length(x_init) != d || !all(is.finite(x_init)))
      stop("`x_init` must be a finite numeric vector of length d.", call. = FALSE)
  }
  trunc_obj <- function(obj, j) if (!use_trunc) obj
    else .truncate(obj, kernel, .r_squared(j, n_points, c2, truncation))

  run <- function() {
    X <- D <- matrix(NA_real_, n_points, d)
    n_eval <- integer(n_points); ksd <- numeric(n_points)

    s1 <- .seed_x1(kernel, score_function, optimizer, log_p, x_init, d)
    X[1, ] <- s1$x; D[1, ] <- s1$grad
    n_eval[1] <- s1$n_eval; ss <- s1$k0_self
    ksd[1] <- sqrt(max(ss, 0))

    if (n_points >= 2L) {
      for (n in 2:n_points) {
        Xs <- X[seq_len(n - 1), , drop = FALSE]
        Ds <- D[seq_len(n - 1), , drop = FALSE]
        obj_n <- trunc_obj(
          if (herding) .obj_herding(kernel, score_function, Xs, Ds)
          else         .obj_greedy (kernel, score_function, Xs, Ds), n)
        res <- optimizer(obj_n, Xs)
        X[n, ] <- res$x_min; D[n, ] <- res$d_min; n_eval[n] <- res$n_eval

        ss <- ss + if (herding) {
          2 * res$f_min + .k0_diag(kernel, matrix(res$x_min, 1, d),
                                           matrix(res$d_min, 1, d))
        } else res$f_min
        ksd[n] <- sqrt(max(ss, 0)) / n
      }
    }
    list(X = X, D = D, ksd = ksd, n_eval = n_eval,
         cum_n_eval = cumsum(n_eval), method = method,
         kernel = kernel, truncation = truncation, c2 = c2)
  }
  if (is.null(seed)) run() else with_local_seed(seed, run())
}


#' Block coordinate descent refinement of an existing point set.
#'
#' @param X0 Initial point matrix.
#' @param score_function Function returning scores for candidate rows.
#' @param kernel Stein kernel object or compatible custom kernel.
#' @param n_iter Number of coordinate-descent updates.
#' @param optimizer Optimizer function used for each coordinate update.
#' @param seed Optional RNG seed.
#'
#' @return List with refined points, scores, evaluation counts, and kernel.
#' @examples
#' score <- function(X) -as.matrix(X)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' opt <- fmin_grid(lb = -1, ub = 1, n0 = 3, grow = FALSE)
#' X0 <- matrix(c(-0.5, 0.5), ncol = 1)
#' stein_codescent(X0, score, kernel, n_iter = 1, optimizer = opt)
#' @export
stein_codescent <- function(X0, score_function, kernel, n_iter, optimizer,
                            seed = NULL) {
  n_iter <- validate_nonnegative_integer(n_iter, "n_iter")
  .stein_points_check_kernel(kernel)
  X0 <- as.matrix(X0); n <- nrow(X0); d <- ncol(X0)
  run <- function() {
    X <- X0; D <- validate_scores(score_function, X); n_eval <- integer(n_iter)
    if (n_iter >= 1L) {
      for (it in seq_len(n_iter)) {
        j <- ((it - 1L) %% n) + 1L
        obj <- .obj_greedy(kernel, score_function,
                           X[-j, , drop = FALSE], D[-j, , drop = FALSE])
        res <- optimizer(obj, X[-j, , drop = FALSE])
        X[j, ] <- res$x_min; D[j, ] <- res$d_min; n_eval[it] <- res$n_eval
      }
      n_eval[1] <- n_eval[1] + n
    }
    list(X = X, D = D, n_eval = n_eval, cum_n_eval = cumsum(n_eval),
         kernel = kernel)
  }
  if (is.null(seed)) run() else with_local_seed(seed, run())
}


# ---- Extra Stein Points kernels ----------------------------------------

.check_positive_alpha <- function(alpha) {
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) || alpha <= 0)
    stop("alpha must be a finite positive scalar.", call. = FALSE)
}

# k1/k3 IMQ exponents use the range from the experiments.
.check_imq_alpha_beta <- function(alpha, beta) {
  .check_positive_alpha(alpha)
  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) ||
      beta <= -1 || beta >= 0)
    stop("beta must lie in (-1, 0).", call. = FALSE)
}

# k2 allows beta < 0; the paper's experiments use beta = -1.
.check_inverse_log_alpha_beta <- function(alpha, beta) {
  .check_positive_alpha(alpha)
  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) || beta >= 0)
    stop("beta must be a finite negative scalar.", call. = FALSE)
}

#' k2: inverse-log Stein kernel (alpha + log(1 + ||x - y||^2))^beta.
#'
#' @param alpha Positive offset parameter.
#' @param beta Negative exponent.
#'
#' @return A Stein kernel object.
#' @examples
#' stein_kernel_inverse_log(alpha = 1, beta = -1)
#' @export
stein_kernel_inverse_log <- function(alpha = 1, beta = -1) {
  .check_inverse_log_alpha_beta(alpha, beta)
  structure(list(alpha = as.numeric(alpha), beta = as.numeric(beta)),
            class = c("SteinKernel_inverse_log", "SteinKernel"))
}

#' k3: IMQ-score Stein kernel (alpha + ||score_p(x) - score_p(y)||^2)^beta.
#'
#' `hess_log_p(X)` must return an n x d x d array of Hess(log p)(x_i).
#'
#' @param alpha Positive offset parameter.
#' @param beta Exponent in `(-1, 0)`.
#' @param hess_log_p Hessian function for the log density.
#'
#' @return A Stein kernel object.
#' @examples
#' hess_log_p <- function(X) array(-1, dim = c(nrow(as.matrix(X)), 1, 1))
#' stein_kernel_imq_score(alpha = 1, beta = -0.5, hess_log_p = hess_log_p)
#' @export
stein_kernel_imq_score <- function(alpha = 1, beta = -0.5, hess_log_p) {
  .check_imq_alpha_beta(alpha, beta)
  if (!is.function(hess_log_p))
    stop("hess_log_p must be a function returning an n x d x d Hessian array",
         call. = FALSE)
  structure(list(alpha = as.numeric(alpha), beta = as.numeric(beta),
                 hess_log_p = hess_log_p),
            class = c("SteinKernel_imq_score", "SteinKernel"))
}


# ---- Basic optimisers ---------------------------------------------------

.adaptive_defaults <- function(lb, ub, mu0, Sigma0) {
  if (is.null(mu0))    mu0    <- (lb + ub) / 2
  if (is.null(Sigma0)) Sigma0 <- ((ub - lb) / 4)^2 * diag(length(lb))
  list(mu0 = mu0, Sigma0 = Sigma0)
}

#' Monte Carlo optimiser.
#'
#' Samples `n_mc` candidates in `[lb, ub]` and returns the one with smallest
#' objective value. Until the current set has more than `delay` points, samples
#' from `N(mu0, Sigma0)`; afterwards, samples around the current selected points.
#'
#' @param lb,ub Lower and upper bounds for candidate points.
#' @param n_mc Number of Monte Carlo candidates.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance.
#' @param sigsq Local proposal variance after the delay period.
#' @param delay Number of selected points before using local proposals.
#'
#' @return Optimizer function.
#' @examples
#' fmin_mc(lb = -1, ub = 1, n_mc = 5)
#' @export
fmin_mc <- function(lb, ub, n_mc = 20, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20) {
  p <- .adaptive_defaults(lb, ub, mu0, Sigma0)
  function(f, X_curr) {
    X_mc <- sample_proposal_box(n_mc, lb, ub, p$mu0, p$Sigma0,
                                sigsq, X_curr, delay)
    res  <- f(X_mc)
    i    <- which.min(res$f_vec)
    list(x_min = X_mc[i, ], d_min = res$D_new[i, ],
         f_min = res$f_vec[i], n_eval = n_mc)
  }
}

#' Multi-start Nelder-Mead optimiser.
#'
#' Draws `n_res` starting points, runs Nelder-Mead inside `[lb, ub]`, and returns
#' the best local solution.
#'
#' @param lb,ub Lower and upper bounds for candidate points.
#' @param n_res Number of random restarts.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance.
#' @param sigsq Local proposal variance after the delay period.
#' @param delay Number of selected points before using local proposals.
#' @param control Control list passed to `stats::optim()`.
#'
#' @return Optimizer function.
#' @examples
#' fmin_nm(lb = -1, ub = 1, n_res = 2)
#' @export
fmin_nm <- function(lb, ub, n_res = 3, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20, control = list(reltol = 1e-3)) {
  p <- .adaptive_defaults(lb, ub, mu0, Sigma0)
  span <- ub - lb
  if (any(!is.finite(span)) || any(span <= 0))
    stop("`ub` must exceed `lb` componentwise", call. = FALSE)
  to_x     <- function(th) lb + span * sin(th)^2
  to_theta <- function(x)  asin(sqrt(pmin(pmax((x - lb) / span, 0), 1)))

  function(f, X_curr) {
    X0   <- sample_proposal_box(n_res, lb, ub, p$mu0, p$Sigma0,
                                sigsq, X_curr, delay)
    f_th <- function(th) f(matrix(to_x(th), nrow = 1))$f_vec[1]

    best_x <- X0[1, ]; best_val <- Inf; n_eval <- 0L
    for (i in seq_len(n_res)) {
      opt <- stats::optim(to_theta(X0[i, ]), f_th,
                          method = "Nelder-Mead", control = control)
      n_eval <- n_eval + as.integer(opt$counts["function"])
      if (opt$value < best_val) {
        best_val <- opt$value
        best_x   <- to_x(opt$par)
      }
    }
    final <- f(matrix(best_x, nrow = 1))
    list(x_min = as.numeric(best_x), d_min = final$D_new[1, ],
         f_min = best_val, n_eval = n_eval + 1L)
  }
}

#' Grid-search optimiser.
#'
#' Evaluates the objective on a Cartesian grid in `[lb, ub]`. With `grow = TRUE`,
#' the grid resolution increases slowly as more points are selected.
#'
#' @param lb,ub Lower and upper bounds for candidate points.
#' @param n0 Initial grid size per dimension.
#' @param grow Logical; whether to increase grid size as points are selected.
#'
#' @return Optimizer function.
#' @examples
#' fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)
#' @export
fmin_grid <- function(lb, ub, n0 = 100, grow = TRUE) {
  d <- length(lb); if (length(n0) == 1L) n0 <- rep(n0, d)
  function(f, X_curr) {
    n_g  <- if (grow) n0 + as.integer(round(sqrt(nrow(X_curr) + 1))) else n0
    grid <- as.matrix(do.call(expand.grid,
              lapply(seq_len(d), function(j)
                seq(lb[j], ub[j], length.out = n_g[j]))))
    dimnames(grid) <- NULL
    res <- f(grid)
    i   <- which.min(res$f_vec)
    list(x_min = grid[i, ], d_min = res$D_new[i, ],
         f_min = res$f_vec[i], n_eval = nrow(grid))
  }
}

#' Draw samples from the optimiser proposal inside `[lb, ub]`.
#'
#' @param n Number of samples to draw.
#' @param lb,ub Lower and upper bounds.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance.
#' @param sigsq Local proposal variance.
#' @param X_curr Currently selected points.
#' @param delay Number of selected points before using local proposals.
#' @param max_oversample Maximum number of proposal batches.
#'
#' @return Numeric matrix of proposed samples.
#' @examples
#' sample_proposal_box(
#'   n = 3, lb = -1, ub = 1, mu0 = 0, Sigma0 = matrix(0.1),
#'   sigsq = 0.1, X_curr = matrix(0, ncol = 1), delay = 2
#' )
#' @export
sample_proposal_box <- function(n, lb, ub, mu0, Sigma0, sigsq, X_curr, delay,
                                max_oversample = 200L) {
  d <- length(lb); out <- matrix(NA_real_, n, d); filled <- 0L
  for (iter in seq_len(max_oversample)) {
    if (filled == n) break
    batch <- max((n - filled) * 2L, 8L)
    Z <- if (nrow(X_curr) <= delay) {
      sweep(matrix(stats::rnorm(batch * d), batch) %*% chol(Sigma0),
            2, mu0, "+")
    } else {
      idx <- sample.int(nrow(X_curr), batch, replace = TRUE)
      X_curr[idx, , drop = FALSE] +
        matrix(stats::rnorm(batch * d, sd = sqrt(sigsq)), batch)
    }
    inside <- rowSums(Z >= matrix(lb, batch, d, byrow = TRUE) &
                      Z <= matrix(ub, batch, d, byrow = TRUE)) == d
    Z_in <- Z[inside, , drop = FALSE]
    take <- min(nrow(Z_in), n - filled)
    if (take > 0L) {
      out[(filled + 1L):(filled + take), ] <- Z_in[seq_len(take), , drop = FALSE]
      filled <- filled + take
    }
  }
  if (filled < n)
    stop(sprintf(
      "Box-truncated proposal failed to fill %d/%d draws after %d batches; check (mu0, Sigma0) vs [lb, ub].",
      n - filled, n, max_oversample), call. = FALSE)
  out
}


# ---- First point --------------------------------------------------------
# Use `x_init` if supplied; otherwise minimise -log_p with the same optimiser.

.seed_x1 <- function(kernel, score_function, optimizer, log_p, x_init, d) {
  if (!is.null(x_init)) {
    grad <- validate_scores(score_function, matrix(x_init, 1, d))[1, ]
    return(list(x = x_init, grad = grad, n_eval = 1L,
                k0_self = as.numeric(.k0_diag(kernel,
                  matrix(x_init, 1, d), matrix(grad, 1, d)))))
  }
  if (is.null(log_p))
    stop("Supply `log_p` or `x_init` for the first point.", call. = FALSE)
  res <- optimizer(.obj_neg_log_p(score_function, log_p), matrix(0, 0, d))
  list(x = res$x_min, grad = res$d_min, n_eval = res$n_eval,
       k0_self = as.numeric(.k0_diag(kernel,
         matrix(res$x_min, 1, d), matrix(res$d_min, 1, d))))
}


# ---- Stein kernel access ------------------------------------------------

.stein_points_check_kernel <- function(kernel) {
  if (inherits(kernel, "SteinKernel_imq")) {
    if (!is.numeric(kernel$c) || length(kernel$c) != 1L ||
        !is.finite(kernel$c) || kernel$c <= 0) {
      stop("IMQ Stein Points kernel requires finite c > 0.", call. = FALSE)
    }
    if (!is.numeric(kernel$beta) || length(kernel$beta) != 1L ||
        !is.finite(kernel$beta) || kernel$beta <= -1 || kernel$beta >= 0) {
      stop("IMQ Stein Points kernel requires beta in (-1, 0).", call. = FALSE)
    }
  }
  invisible(kernel)
}

.as_k0_matrix <- function(K, X, Y) {
  K <- as.matrix(K)
  if (!is.numeric(K) || !identical(dim(K), c(nrow(X), nrow(Y)))) {
    stop("kernel must return a numeric matrix with nrow(X) rows and nrow(Y) columns.",
         call. = FALSE)
  }
  K
}

.as_k0_diag <- function(v, X) {
  v <- as.numeric(v)
  if (length(v) != nrow(X) || anyNA(v)) {
    stop("k0_diag must return a numeric vector of length nrow(X) without NAs.",
         call. = FALSE)
  }
  v
}

# Accepted kernel forms:
#   1. function(X, S_X, Y, S_Y)
#   2. list(k0_matrix = ..., optional k0_diag = ...)
#   3. object handled by stein_kernel_matrix()
.k0_matrix <- function(kernel, X, Y, S_X, S_Y) {
  if (is.function(kernel))
    return(.as_k0_matrix(kernel(X, S_X, Y, S_Y), X, Y))
  if (is.list(kernel) && is.function(kernel$k0_matrix))
    return(.as_k0_matrix(kernel$k0_matrix(X, S_X, Y, S_Y), X, Y))
  if (inherits(kernel, "SteinKernel_imq_score"))
    return(.k0_matrix_imq_score(kernel, X, Y, S_X, S_Y))
  .as_k0_matrix(stein_kernel_matrix(kernel, X, S_X, Y, S_Y), X, Y)
}

# Fast diagonals for built-ins; custom kernels fall back to diag(k0_matrix).
.k0_diag <- function(kernel, X, S_X) {
  d <- ncol(X); norm_s_sq <- rowSums(S_X * S_X)
  if (is.list(kernel) && is.function(kernel$k0_diag))
    return(.as_k0_diag(kernel$k0_diag(X, S_X), X))
  if (inherits(kernel, c("SteinKernel_imq", "SteinKernel_inverse_log"))) {
    a <- if (inherits(kernel, "SteinKernel_imq")) kernel$c^2 else kernel$alpha
    b <- kernel$beta
    tr_M <- if (inherits(kernel, "SteinKernel_imq")) {
      kernel_precon_trace(kernel_precon(kernel, d), d)
    } else d
    return(-2 * b * tr_M * a^(b - 1) + a^b * norm_s_sq)
  }
  if (inherits(kernel, "SteinKernel_gaussian_rbf")) {
    h2 <- resolve_gaussian_rbf_h2(kernel, X)
    tr_M <- kernel_precon_trace(kernel_precon(kernel, d), d)
    return(tr_M / h2 + norm_s_sq)
  }
  if (inherits(kernel, "SteinKernel_imq_score")) {
    a <- kernel$alpha; b <- kernel$beta
    H <- kernel$hess_log_p(X)
    j_norm_sq <- vapply(seq_len(nrow(X)),
                        function(i) sum(H[i, , ]^2), numeric(1))
    return(-2 * b * a^(b - 1) * j_norm_sq + a^b * norm_s_sq)
  }
  .as_k0_diag(diag(.k0_matrix(kernel, X, X, S_X, S_X)), X)
}

# k3 uses distances between score vectors, so it has a separate matrix path.
.k0_matrix_imq_score <- function(kernel, X, Y, S_X, S_Y) {
  a <- kernel$alpha; b <- kernel$beta
  H_X <- kernel$hess_log_p(X)
  H_Y <- if (identical(X, Y)) H_X else kernel$hess_log_p(Y)
  base <- a + compute_cross_squared_distance(S_X, S_Y)
  c0 <- base^b; c1 <- b * base^(b - 1); c2 <- b * (b - 1) * base^(b - 2)

  out <- matrix(0, nrow(X), nrow(Y))
  for (i in seq_len(nrow(X))) {
    Hi <- H_X[i, , ]; sx <- S_X[i, ]
    for (j in seq_len(nrow(Y))) {
      Hj <- H_Y[j, , ]; sy <- S_Y[j, ]
      delta <- sx - sy; HiHj <- Hi %*% Hj
      tr_u  <- -2 * sum(diag(HiHj))
      ux_uy <- -4 * as.numeric(crossprod(delta, HiHj %*% delta))
      cs    <- -2 * sum(sx * (Hj %*% delta)) + 2 * sum(sy * (Hi %*% delta))
      out[i, j] <- c2[i, j] * ux_uy + c1[i, j] * (tr_u + cs) +
                   c0[i, j] * sum(sx * sy)
    }
  }
  out
}


# ---- Optimiser objectives ----------------------------------------------

# First point: maximise log_p(x), implemented as minimise -log_p(x).
.obj_neg_log_p <- function(score_function, log_p) {
  function(X_new, D_new = NULL) {
    if (is.null(D_new)) D_new <- validate_scores(score_function, X_new)
    lp <- as.numeric(log_p(X_new))
    if (length(lp) != nrow(X_new) || any(is.na(lp)))
      stop("`log_p` must return a numeric vector of length nrow(X_new) ",
           "without NAs", call. = FALSE)
    list(f_vec = -lp, D_new = D_new)
  }
}

# Greedy target for a candidate x:
#   k0(x, x) + 2 * sum_i k0(x_i, x)
.obj_greedy <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, D_new = NULL) {
    if (is.null(D_new)) D_new <- validate_scores(score_function, X_new)
    K0    <- .k0_matrix(kernel, X_sel, X_new, D_sel, D_new)
    list(f_vec = 2 * colSums(K0) + .k0_diag(kernel, X_new, D_new),
         D_new = D_new)
  }
}

# Herding target for a candidate x:
#   sum_i k0(x_i, x)
.obj_herding <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, D_new = NULL) {
    if (is.null(D_new)) D_new <- validate_scores(score_function, X_new)
    list(f_vec = colSums(.k0_matrix(kernel, X_sel, X_new, D_sel, D_new)),
         D_new = D_new)
  }
}


# ---- Optional truncation ------------------------------------------------

# Radius schedule for the candidate filter k0(x, x) <= R_j^2.
.r_squared <- function(j, n_total, c2, mode) {
  upper <- 2 * log(max(n_total, 2L)) / c2
  lower <- 2 * log(j)                / c2
  switch(mode,
    upper  = upper, lower = lower,
    linear = lower + (upper - lower) * ((j - 1) / max(n_total - 1, 1L)))
}

# Set infeasible candidates to a huge objective value.
.truncate <- function(obj, kernel, r2) {
  if (!is.finite(r2) || r2 <= 0) return(obj)
  function(X_new, D_new = NULL) {
    res     <- obj(X_new, D_new)
    diag_k0 <- .k0_diag(kernel, X_new, res$D_new)
    bad     <- !is.finite(diag_k0) | diag_k0 > r2
    if (all(bad))
      stop(sprintf(
        paste0("Truncation infeasible: all %d candidate(s) have ",
               "k0(x, x) > R^2 = %.4g. ",
               "Widen the proposal (Sigma0/sigsq), increase the optimiser's ",
               "candidate count, decrease `c2`, or use `truncation = \"upper\"`."),
        nrow(X_new), r2), call. = FALSE)
    res$f_vec[bad] <- .Machine$double.xmax
    res
  }
}


# ---- S3 methods for SteinKernel_inverse_log (k2) ------------------------

#' @export
eval_kernel.SteinKernel_inverse_log <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X"); Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  (obj$alpha + log1p(compute_cross_squared_distance(X, Y)))^obj$beta
}

#' @export
grad_x_kernel.SteinKernel_inverse_log <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X"); Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  d <- ncol(X); r <- compute_cross_squared_distance(X, Y)
  coef <- 2 * obj$beta * (obj$alpha + log1p(r))^(obj$beta - 1) / (1 + r)
  arr <- array(0, c(nrow(X), nrow(Y), d))
  for (j in seq_len(d)) arr[, , j] <- coef * outer(X[, j], Y[, j], "-")
  arr
}

#' @export
trace_mixed_kernel.SteinKernel_inverse_log <- function(obj, X, Y = NULL, ...) {
  X <- as_kernel_matrix(X, "X"); Y <- if (is.null(Y)) X else as_kernel_matrix(Y, "Y")
  d <- ncol(X); r <- compute_cross_squared_distance(X, Y)
  base <- obj$alpha + log1p(r); b <- obj$beta
  b * (b - 1) * base^(b - 2) * (-4 * r / (1 + r)^2) +
    b * base^(b - 1) * (-2 * d / (1 + r) + 4 * r / (1 + r)^2)
}

#' @export
cross_kernel.SteinKernel_inverse_log <- function(obj, X, grads, Y = NULL,
                                                 grads_Y = NULL, ...) {
  i <- validate_cross_inputs(X, grads, Y, grads_Y)
  r <- compute_cross_squared_distance(i$X, i$Y)
  -2 * obj$beta * (obj$alpha + log1p(r))^(obj$beta - 1) *
    compute_cross_q(i$X, i$grads, i$Y, i$grads_Y) / (1 + r)
}
