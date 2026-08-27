# Stein Points, coordinate descent, and their continuous-space optimizers.
# Alternative kernel constructors used by these algorithms are defined below.

# ---- Public: complete algorithms -----------------------------------------

#' Select points by Stein discrepancy minimization
#'
#' Builds a point set sequentially. At each step, `optimizer` searches the
#' continuous state space for the next point under a greedy or herding Stein
#' objective.
#'
#' @details
#' Let \eqn{k_{0,p}} be the Stein kernel formed from `kernel` and
#' \eqn{s_p(x)=\nabla_x\log p(x)}. Given selected points
#' \eqn{x_1,\ldots,x_{j-1}}, the greedy rule chooses
#' \deqn{x_j \in \mathop{\mathrm{argmin}}_{x\in\mathbb{R}^d} Q_j(x),}
#' where
#' \deqn{Q_j(x)=\frac{1}{2}k_{0,p}(x,x)
#'       +\sum_{i=1}^{j-1}k_{0,p}(x_i,x).}
#' With `method = "herding"`, the self-interaction is omitted and the
#' objective is
#' \deqn{Q_j^{\mathrm{herd}}(x)
#'       =\sum_{i=1}^{j-1}k_{0,p}(x_i,x).}
#'
#' `optimizer` performs the numerical search for each \eqn{x_j}. The supplied
#' constructors [fmin_grid()], [fmin_mc()], and [fmin_nm()] use grid, Monte
#' Carlo, and Nelder--Mead search, respectively. The first point is `x_init`;
#' if `x_init` is `NULL`, `optimizer` instead maximizes `log_p`.
#'
#' The returned discrepancy after step \eqn{j} is
#' \deqn{\mathrm{KSD}_j
#' =\left\{\frac{1}{j^2}\sum_{a=1}^j\sum_{b=1}^j
#' k_{0,p}(x_a,x_b)\right\}^{1/2}.}
#' When `truncation != "none"`, the truncation rule filters candidates from
#' steps 2 through `n_points`; it is not applied to the first point.
#'
#' @param score_function Function taking an `n x d` matrix and returning the
#'   corresponding `n x d` matrix of target scores.
#' @param kernel Kernel used to construct \eqn{k_{0,p}}. It may be a built-in
#'   kernel object, a compatible custom object, a function
#'   `kernel(X, S_X, Y, S_Y)`, or a list with `k0_matrix`. A Gaussian RBF
#'   kernel must use a fixed positive bandwidth.
#' @param n_points Positive number of points to select.
#' @param d Positive state dimension.
#' @param optimizer Function taking `(objective, X_curr, t)` and returning
#'   `x_min`, its score `d_min`, the minimized objective `f_min`, and the
#'   evaluation count `n_eval`.
#' @param method Point-selection rule: `"greedy"` or `"herding"`.
#' @param log_p Optional log density used to choose the first point.
#' @param x_init Optional finite numeric vector of length `d` giving the first
#'   point. If supplied, `log_p` is not used.
#' @param c2 Positive truncation constant when `truncation != "none"`.
#' @param truncation Candidate-filtering rule. Use `"none"` for no filtering.
#' @param seed Optional local RNG seed.
#'
#' @return
#' An object of class `"stein_points"` with:
#' * `X` and `D`: `n_points x d` matrices of selected points and their scores.
#' * `ksd`: running discrepancy defined above.
#' * `n_eval` and `cum_n_eval`: per-step and cumulative evaluation counts.
#' * `method`, `kernel`, `truncation`, and `c2`: settings used by the run.
#' * `call`: matched function call.
#'
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
  cl <- match.call()
  method <- match.arg(method)
  truncation <- match.arg(truncation)
  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }
  n_points <- validate_integer(n_points, "n_points")
  d <- validate_integer(d, "d")
  herding <- identical(method, "herding")
  .stein_points_check_kernel(kernel)

  use_trunc <- truncation != "none"
  if (use_trunc && (!is.numeric(c2) || length(c2) != 1L || c2 <= 0))
    stop("`c2` must be positive when truncation != 'none'.", call. = FALSE)
  if (!is.null(log_p) && !is.function(log_p))
    stop("`log_p` must be a function.", call. = FALSE)
  if (is.null(log_p) && is.null(x_init)) {
    stop("supply `log_p` or `x_init` for the first point", call. = FALSE)
  }
  if (!is.null(x_init)) {
    x_init <- as.numeric(x_init)
    if (length(x_init) != d || !all(is.finite(x_init)))
      stop("`x_init` must be a finite numeric vector of length d.", call. = FALSE)
  }
  trunc_obj <- function(obj, j) {
    if (!use_trunc) {
      return(obj)
    }
    .truncate(obj, kernel, .r_squared(j, n_points, c2, truncation))
  }

  run <- function() {
    X <- D <- matrix(NA_real_, n_points, d)
    n_eval <- integer(n_points)
    ksd <- numeric(n_points)

    s1 <- .seed_x1(kernel, score_function, optimizer, log_p, x_init, d)
    X[1, ] <- s1$x; D[1, ] <- s1$grad
    n_eval[1] <- s1$n_eval; ss <- s1$k0_self
    if (ss < -sqrt(.Machine$double.eps)) warning("KSD squared value is negative.", call. = FALSE)
    ksd[1] <- sqrt(max(ss, 0))

    if (n_points >= 2L) {
      for (j in 2:n_points) {
        selected_X <- X[seq_len(j - 1L), , drop = FALSE]
        selected_scores <- D[seq_len(j - 1L), , drop = FALSE]
        obj_base <- if (herding) {
          .obj_herding(kernel, score_function, selected_X, selected_scores)
        } else {
          .obj_greedy(kernel, score_function, selected_X, selected_scores)
        }
        res <- optimizer(trunc_obj(obj_base, j), selected_X, t = j)
        X[j, ] <- res$x_min
        D[j, ] <- res$d_min
        n_eval[j] <- res$n_eval

        # The greedy value is already the increment to sum_{a,b<=n} k0; the
        # herding value omits the self term and counts each pair once.
        ss <- ss + if (herding) {
          2 * res$f_min +
            .k0_diag(kernel, matrix(res$x_min, 1, d), matrix(res$d_min, 1, d))
        } else {
          res$f_min
        }
        if (ss < -sqrt(.Machine$double.eps)) warning("KSD squared value is negative.", call. = FALSE)
        ksd[j] <- sqrt(max(ss, 0)) / j
      }
    }
    structure(
      list(X = X, D = D, ksd = ksd, n_eval = n_eval,
           cum_n_eval = cumsum(n_eval), method = method,
           kernel = kernel, call = cl, truncation = truncation, c2 = c2),
      class = "stein_points"
    )
  }
  with_local_seed(seed, run())
}

#' Refine Stein Points by coordinate descent
#'
#' Runs the budget-constrained refinement described with Stein Points: keep the
#' number of points fixed, then repeatedly optimize one existing point under the
#' same KSD objective.
#'
#' @details
#' At iteration `it`, row `((it - 1) %% nrow(X0)) + 1` is optimized while the
#' other rows are held fixed. For the row being replaced, the objective is the
#' greedy Stein Points objective built from all remaining rows:
#' \deqn{k0(x,x) + 2\sum_{i\ne r} k0(x_i,x),}
#' where `r` is the row currently being replaced. This is the same doubled
#' objective used by the greedy branch of [stein_points()], but the
#' selected set size is fixed instead of growing by one point. The update is
#' useful after a fixed budget of points has already been produced, because it
#' can improve the locations without increasing the point count.
#'
#' The paper defines each coordinate update by a global `argmin`, for which the
#' current row is itself feasible and the KSD therefore cannot increase. The
#' numerical grid, Monte Carlo, and Nelder--Mead searches are approximate and
#' need not return a better point. After each search, this implementation
#' evaluates the old and proposed rows under the same coordinate objective and
#' keeps the proposal only when its value is no larger. This incumbent fallback
#' restores the non-increase property without adding a score evaluation because
#' both scores are already cached. When `X0` contains a single row the
#' interaction sum is empty and the objective reduces to \eqn{k0(x,x)},
#' matching the special branch in the authors' MATLAB code.
#'
#' The optimizer interface is the one used by [stein_points()], so a grid,
#' Monte Carlo, or Nelder-Mead search serves both construction and refinement.
#'
#' @param X0 Initial point matrix.
#' @param score_function Function returning scores for candidate rows.
#' @param kernel Stein kernel object or compatible custom kernel.
#' @param n_iter Number of coordinate-descent updates.
#' @param optimizer Optimizer function used for each coordinate update.
#' @param seed Optional RNG seed.
#'
#' @return
#' An object of class `"stein_codescent"` with:
#' * `X`: refined point matrix with the same dimensions as `X0`.
#' * `D`: target scores at the refined points.
#' * `objective`: numeric vector; `objective[it]` is the retained value of the
#'   coordinate objective at update `it`.
#' * `n_eval`: evaluation counts for each coordinate-descent update.
#' * `cum_n_eval`: cumulative evaluation counts.
#' * `kernel`: kernel used to define the Stein objective.
#'
#' The `objective` values are not KSDs. They differ from \eqn{n^2 KSD^2} by the
#' kernel sum over the rows held fixed, which changes at every update, so they
#' are comparable within an update but not across updates.
#'
#' There is no `ksd` trajectory, because each update replaces an existing row
#' rather than extending a prefix. For the final KSD, build
#' `K0 <- stein_kernel_matrix(kernel, out$X, out$D)` and evaluate
#' `sqrt(sum(K0)) / nrow(out$X)`.
#' @examples
#' score <- function(X) -as.matrix(X)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' opt <- fmin_grid(lb = -1, ub = 1, n0 = 3, grow = FALSE)
#' X0 <- matrix(c(-0.5, 0.5), ncol = 1)
#' stein_codescent(X0, score, kernel, n_iter = 1, optimizer = opt)
#' @export
stein_codescent <- function(X0, score_function, kernel, n_iter, optimizer,
                            seed = NULL) {
  cl <- match.call()
  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }
  n_iter <- validate_integer(n_iter, "n_iter", min_value = 0L)
  .stein_points_check_kernel(kernel)
  X0 <- .as_rows(X0, "X0")
  n <- nrow(X0)
  run <- function() {
    X <- X0
    D <- .as_score_matrix(score_function(X), X)
    n_eval <- integer(n_iter)
    # The coordinate objective is already evaluated at the incumbent and the
    # proposal for the non-increase check, so recording the retained value
    # costs nothing. It is the greedy objective, not a KSD.
    objective <- numeric(n_iter)
    if (n_iter >= 1L) {
      for (it in seq_len(n_iter)) {
        j <- ((it - 1L) %% n) + 1L
        X_other <- X[-j, , drop = FALSE]
        D_other <- D[-j, , drop = FALSE]
        obj <- .obj_greedy(kernel, score_function, X_other, D_other)
        res <- optimizer(obj, X_other, t = it)

        X_old <- X[j, , drop = FALSE]
        D_old <- D[j, , drop = FALSE]
        X_new <- matrix(res$x_min, nrow = 1L)
        candidate_scores <- matrix(res$d_min, nrow = 1L)
        old_value <- obj(X_old, D_old)$objective_values[1L]
        new_value <- obj(X_new, candidate_scores)$objective_values[1L]
        if (is.finite(new_value) &&
            (!is.finite(old_value) || new_value <= old_value)) {
          X[j, ] <- X_new
          D[j, ] <- candidate_scores
          objective[it] <- new_value
        } else {
          objective[it] <- old_value
        }
        n_eval[it] <- res$n_eval
      }
      n_eval[1] <- n_eval[1] + n
    }
    structure(
      list(X = X, D = D, objective = objective, n_eval = n_eval,
           cum_n_eval = cumsum(n_eval), method = "codescent",
           kernel = kernel, call = cl),
      class = "stein_codescent"
    )
  }
  with_local_seed(seed, run())
}


# ---- Public: print methods -----------------------------------------------

#' Print a Stein point set
#'
#' Prints the point matrix size, evaluation count, and available diagnostic.
#'
#' @param x An object returned by [stein_points()], [stein_codescent()], or
#'   [svgd()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @examples
#' score <- function(X) -as.matrix(X)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' opt <- fmin_grid(lb = -1, ub = 1, n0 = 3, grow = FALSE)
#' stein_points(score, kernel, n_points = 3, d = 1, optimizer = opt, x_init = 0)
#' @export
print.stein_points <- function(x, ...) {
  last <- function(v) if (length(v)) v[length(v)] else NA_real_
  diagnostic <- if (!is.null(x$ksd)) {
    sprintf("  KSD=%.4g", last(x$ksd))
  } else if (!is.null(x$objective)) {
    sprintf("  obj=%.4g", last(x$objective))
  } else {
    ""
  }
  cat(sprintf(
    "%s  n=%d  d=%d%s  n_eval=%g\n",
    x$method, nrow(x$X), ncol(x$X), diagnostic, last(x$cum_n_eval)
  ))
  invisible(x)
}

#' @rdname print.stein_points
#' @export
print.stein_codescent <- print.stein_points

#' @rdname print.stein_points
#' @export
print.svgd <- print.stein_points


# ---- Public: candidate-search optimizers ---------------------------------

#' Create the grid search used by Stein Points
#'
#' Creates the deterministic grid-search function used in the Stein Points
#' appendix. It evaluates every point on a Cartesian grid and returns the
#' smallest objective value.
#'
#' @details
#' With `grow = TRUE`, the grid size follows the paper's idea of increasing the
#' grid resolution as the point set grows: the per-dimension size is
#' `n0 + round(sqrt(t))`, where `t` is the optimizer iteration supplied by
#' [stein_points()] or [stein_codescent()]. The method is simple
#' and deterministic once the grid is fixed, but it is practical mainly in low
#' dimension because the total number of candidates is the product of the grid
#' sizes across dimensions.
#'
#' This is the deterministic optimizer in the Stein Points family. It is useful
#' for small-dimensional examples and for debugging because the same inputs lead
#' to the same candidate set and the same selected point.
#'
#' @param lb,ub Lower and upper bounds for candidate points.
#' @param n0 Initial grid size per dimension.
#' @param grow Logical; whether to increase grid size as points are selected.
#'
#' @return
#' An optimizer function with signature `function(objective, X_curr, t)`. It
#' evaluates the objective on a Cartesian grid and returns `x_min`, `d_min`,
#' `f_min`, and `n_eval`. `x_min` is the grid row with the smallest objective,
#' `d_min` is the target score at that row, `f_min` is the objective value used
#' by [stein_points()] to update its KSD accumulator, and `n_eval` is the number
#' of grid rows scored.
#' @examples
#' fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)
#' @export
fmin_grid <- function(lb, ub, n0 = 100, grow = TRUE) {
  d <- length(lb)
  if (length(n0) == 1L) n0 <- rep(n0, d)
  function(f, X_curr, t = nrow(X_curr) + 1L) {
    n_g <- if (grow) n0 + as.integer(round(sqrt(t))) else n0
    grid <- as.matrix(do.call(expand.grid,
              lapply(seq_len(d), function(j)
                seq(lb[j], ub[j], length.out = n_g[j]))))
    dimnames(grid) <- NULL
    res <- f(grid)
    i <- .objective_min_index(res)
    list(x_min = grid[i, ], d_min = res$scores[i, ],
         f_min = res$objective_values[i], n_eval = nrow(grid))
  }
}

#' Create the Monte Carlo search used by Stein Points
#'
#' Creates a candidate-search function for [stein_points()]. It implements the
#' Monte Carlo search described in the Stein Points appendix: draw a finite set
#' of candidate points in the box and return the candidate with the smallest
#' supplied objective value.
#'
#' @details
#' The returned optimizer is a function used by [stein_points()] and
#' [stein_codescent()]. Early iterations draw candidates from a broad Gaussian
#' distribution truncated to `[lb, ub]`. Once `t` exceeds `delay`,
#' the proposal becomes local: it chooses one of the current points and draws a
#' Gaussian perturbation with variance `sigsq`, again keeping only candidates in
#' the box. This mirrors the adaptive proposal used in the paper experiments.
#'
#' The optimizer returned by `fmin_mc()` does not know anything about Stein
#' kernels. It receives an objective from [stein_points()], evaluates that
#' objective on sampled candidate rows, and returns the best candidate plus its
#' score and evaluation count. This design keeps stochastic search separate
#' from the mathematical Stein objective.
#'
#' @param lb,ub Lower and upper bounds for candidate points.
#' @param n_mc Number of Monte Carlo candidates.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance.
#' @param sigsq Local proposal variance after the delay period.
#' @param delay Number of optimization iterations before using local proposals.
#'
#' @return
#' An optimizer function with signature `function(objective, X_curr, t)`. It
#' returns a list with `x_min` (best candidate row), `d_min` (score at that
#' row), `f_min` (objective value), and `n_eval` (number of candidates scored).
#' This is the interface expected by [stein_points()] and [stein_codescent()].
#' @examples
#' fmin_mc(lb = -1, ub = 1, n_mc = 5)
#' @export
fmin_mc <- function(lb, ub, n_mc = 20, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20) {
  p <- .adaptive_defaults(lb, ub, mu0, Sigma0)
  function(f, X_curr, t = nrow(X_curr) + 1L) {
    X_mc <- .sample_proposal_box(n_mc, lb, ub, p$mu0, p$Sigma0,
                                 sigsq, X_curr, delay, t = t)
    res <- f(X_mc)
    i <- .objective_min_index(res)
    list(x_min = X_mc[i, ], d_min = res$scores[i, ],
         f_min = res$objective_values[i], n_eval = n_mc)
  }
}

#' Create the multi-start Nelder-Mead search used by Stein Points
#'
#' Creates a local search function for [stein_points()]. It draws several
#' starting points in the box, runs Nelder-Mead from each one, and returns the
#' best local solution found.
#'
#' @details
#' The Stein Points paper uses numerical optimization because the exact global
#' search is usually unavailable. This helper performs a practical multi-start
#' local search. A sine-squared transformation maps unconstrained Nelder-Mead
#' parameters back into `[lb, ub]`, so every objective evaluation stays inside
#' the requested search box. Each restart begins from the same boxed proposal
#' rule used by [fmin_mc()].
#'
#' Nelder-Mead is useful when the objective is smooth enough for local
#' improvement to beat the finite candidate sets of [fmin_grid()] and
#' [fmin_mc()].
#'
#' @param lb,ub Lower and upper bounds for candidate points.
#' @param n_res Number of random restarts.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance.
#' @param sigsq Local proposal variance after the delay period.
#' @param delay Number of optimization iterations before using local proposals.
#' @param control Control list passed to `stats::optim()`.
#'
#' @return
#' An optimizer function with signature `function(objective, X_curr, t)`. The
#' returned function runs `stats::optim()` from `n_res` starting points and
#' returns `x_min` (the selected point), `d_min` (its score), `f_min` (the
#' Stein objective value that updates the running KSD diagnostic), and
#' `n_eval` (objective evaluations charged to this search).
#' @examples
#' fmin_nm(lb = -1, ub = 1, n_res = 2)
#' @export
fmin_nm <- function(lb, ub, n_res = 3, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20, control = list(reltol = 1e-3)) {
  p <- .adaptive_defaults(lb, ub, mu0, Sigma0)
  span <- ub - lb
  if (any(!is.finite(span)) || any(span <= 0))
    stop("`ub` must exceed `lb` componentwise", call. = FALSE)
  to_x <- function(th) lb + span * sin(th)^2
  to_theta <- function(x) asin(sqrt(pmin(pmax((x - lb) / span, 0), 1)))

  function(f, X_curr, t = nrow(X_curr) + 1L) {
    X0 <- .sample_proposal_box(n_res, lb, ub, p$mu0, p$Sigma0,
                               sigsq, X_curr, delay, t = t)
    f_th <- function(th) {
      f(matrix(to_x(th), nrow = 1))$objective_values[1L]
    }

    best_x <- X0[1, ]; best_val <- Inf; n_eval <- 0L
    for (i in seq_len(n_res)) {
      opt <- stats::optim(to_theta(X0[i, ]), f_th,
                          method = "Nelder-Mead", control = control)
      n_eval <- n_eval + as.integer(opt$counts["function"])
      if (opt$value < best_val) {
        best_val <- opt$value
        best_x <- to_x(opt$par)
      }
    }
    final <- f(matrix(best_x, nrow = 1))
    i_final <- .objective_min_index(final)
    list(x_min = as.numeric(best_x), d_min = final$scores[i_final, ],
         f_min = final$objective_values[i_final], n_eval = n_eval + 1L)
  }
}

.adaptive_defaults <- function(lb, ub, mu0, Sigma0) {
  if (is.null(mu0)) mu0 <- (lb + ub) / 2
  if (is.null(Sigma0)) Sigma0 <- ((ub - lb) / 4)^2 * diag(length(lb))
  list(mu0 = mu0, Sigma0 = Sigma0)
}

# Select the smallest feasible objective value. Truncated objectives attach
# feasibility metadata so batch optimizers can distinguish a genuinely feasible
# minimum from the finite penalty used by NM; ordinary objectives carry none, so
# every candidate counts as feasible.
.objective_min_index <- function(res) {
  n <- length(res$objective_values)
  feasible <- if (is.null(res$feasible)) rep(TRUE, n) else res$feasible
  if (!is.logical(feasible) || length(feasible) != n) {
    stop("objective returned invalid feasibility metadata", call. = FALSE)
  }
  feasible[is.na(feasible)] <- FALSE
  if (!any(feasible)) {
    bound <- if (!is.null(res$truncation_r2) && is.finite(res$truncation_r2)) {
      sprintf(" R^2 = %.4g", res$truncation_r2)
    } else {
      " the requested bound"
    }
    stop(
      paste0(
        "Truncation infeasible: the optimizer found no candidate satisfying ",
        "k0(x, x) <=", bound, ". Widen the proposal or search box, increase ",
        "the candidate count, or decrease `c2`."
      ),
      call. = FALSE
    )
  }
  values <- res$objective_values
  values[!feasible] <- Inf
  which.min(values)
}

# Called by `fmin_mc()` and `fmin_nm()`: draw broad Gaussian proposals
# before `delay`, then local proposals around selected rows, rejecting outside
# the box.
.sample_proposal_box <- function(n, lb, ub, mu0, Sigma0, sigsq, X_curr, delay,
                                 t = nrow(X_curr) + 1L,
                                 max_oversample = 200L) {
  d <- length(lb)
  out <- matrix(NA_real_, n, d)
  filled <- 0L
  for (iter in seq_len(max_oversample)) {
    if (filled == n) break
    batch <- max((n - filled) * 2L, 8L)
    Z <- if (t <= delay || nrow(X_curr) == 0L) {
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


# ---- Public: alternative Stein kernels -----------------------------------

#' Create an inverse-log Stein kernel
#'
#' Creates the inverse-log base kernel used in the Stein Points experiments and
#' wraps it as a `SteinKernel` object.
#'
#' @details
#' The base kernel has the form
#' \deqn{k(x, y) = (\alpha + \log(1 + ||x - y||^2))^\beta.}
#' The parameter `alpha` must be positive and `beta` must be negative; the
#' inverse-log kernel in the Stein Points paper is the special case
#' \eqn{\beta=-1}. Compared with a Gaussian RBF kernel, this kernel decays much
#' more slowly as points move apart. That slower decay is useful in the Stein
#' Points setting because the greedy objective needs to see interactions
#' between already selected points and candidates that may be far away.
#'
#' The squared distance is the ordinary Euclidean one between sample rows.
#' Unlike the IMQ constructor in [stein_kernel()], this helper exposes no
#' preconditioning matrix; it reproduces the position-distance inverse-log
#' kernel of the Stein Points experiments. The result is an ordinary
#' `SteinKernel`, so [stein_points()] and [sp_mcmc()] need the same inputs as
#' for any other kernel: points, scores, and the kernel object.
#'
#' @param alpha Positive offset parameter.
#' @param beta Negative exponent.
#'
#' @return A `SteinKernel` object for the inverse-log kernel.
#' @examples
#' stein_kernel_inverse_log(alpha = 1, beta = -1)
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
#' Creates the score-distance IMQ kernel used in the Stein Points experiments.
#' This kernel measures distance between score vectors rather than distance
#' between point locations.
#'
#' @details
#' The base kernel is an IMQ kernel applied to differences between score
#' vectors, rather than differences between sample positions:
#' \deqn{k(x, y) = (\alpha + ||s_p(x) - s_p(y)||^2)^\beta.}
#' Here \eqn{s_p(x)=\nabla_x\log p(x)} is the target score. Two points are
#' close under this kernel when the target score field behaves similarly at the
#' two points, even if the points themselves are not close in Euclidean
#' distance. This is a different design choice from the IMQ option in
#' [stein_kernel()], which applies the IMQ kernel to `||x - y||^2` or its
#' preconditioned version.
#'
#' Because the base kernel depends on \eqn{s_p(x)}, its derivatives with respect
#' to `x` depend on the Hessian of the target log density. The `hess_log_p`
#' argument supplies that information. It should return an array whose first
#' dimension indexes rows of `X` and whose remaining two dimensions contain the
#' `d x d` Hessian matrix for each row. The package uses those Hessians when
#' computing `k0(x, y)` and the diagonal self-interaction terms used by
#' [stein_points()] and [sp_mcmc()].
#'
#' [stein_points()] still takes `score_function` for the greedy objective;
#' `hess_log_p` is needed in addition because differentiating
#' \eqn{k(s_p(x), s_p(y))} with respect to point locations uses derivatives of
#' the score field. That requirement is also why this kernel cannot be built
#' with [custom_stein_kernel()], whose leaf functions receive point matrices
#' but not target-score matrices.
#'
#' @param alpha Positive offset parameter.
#' @param beta Exponent in `(-1, 0)`.
#' @param hess_log_p Function returning a Hessian array with shape `n x d x d`.
#'
#' @return
#' A list with class `"SteinKernel_imq_score"` and `"SteinKernel"`. It stores
#' the IMQ parameters `alpha` and `beta` and the Hessian evaluator
#' `hess_log_p`. Pass the object to [stein_points()], [sp_mcmc()], or
#' [stein_kernel_matrix()] rather than extracting the fields manually.
#' @examples
#' hess_log_p <- function(X) array(-1, dim = c(nrow(as.matrix(X)), 1, 1))
#' stein_kernel_imq_score(alpha = 1, beta = -0.5, hess_log_p = hess_log_p)
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

# k1/k3 IMQ exponents use the range from the experiments.
.check_imq_alpha_beta <- function(alpha, beta) {
  .check_positive_alpha(alpha)
  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) ||
      beta <= -1 || beta >= 0) {
    stop("beta must lie in (-1, 0).", call. = FALSE)
  }
}

# k2 allows beta < 0; the paper's experiments use beta = -1.
.check_inverse_log_alpha_beta <- function(alpha, beta) {
  .check_positive_alpha(alpha)
  if (!is.numeric(beta) || length(beta) != 1L || !is.finite(beta) || beta >= 0) {
    stop("beta must be a finite negative scalar.", call. = FALSE)
  }
}


# ---- Internal: Stein kernel access ---------------------------------------

.stein_points_check_kernel <- function(kernel) {
  require_fixed_gaussian_rbf(kernel, "Stein Points")
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
  # Check SteinKernel first because it is also a list.
  if (inherits(kernel, "SteinKernel"))
    return(.as_k0_matrix(stein_kernel_matrix(kernel, X, S_X, Y, S_Y), X, Y))
  if (is.function(kernel))
    return(.as_k0_matrix(kernel(X, S_X, Y, S_Y), X, Y))
  if (is.list(kernel) && is.function(kernel$k0_matrix))
    return(.as_k0_matrix(kernel$k0_matrix(X, S_X, Y, S_Y), X, Y))
  .as_k0_matrix(stein_kernel_matrix(kernel, X, S_X, Y, S_Y), X, Y)
}

# Diagonal adapter for the same kernel forms accepted by `.k0_matrix()`.
.k0_diag <- function(kernel, X, S_X, ...) {
  if (inherits(kernel, "SteinKernel"))
    return(.as_k0_diag(k0_diag(kernel, X, S_X, ...), X))
  if (is.list(kernel) && is.function(kernel$k0_diag))
    return(.as_k0_diag(kernel$k0_diag(X, S_X), X))
  # Function- or list-form kernel with no diagonal of its own: evaluate one row
  # at a time so memory stays O(n) instead of forming the full n x n matrix.
  .as_k0_diag(
    vapply(seq_len(nrow(X)), function(i) {
      .k0_matrix(
        kernel,
        X[i, , drop = FALSE], X[i, , drop = FALSE],
        S_X[i, , drop = FALSE], S_X[i, , drop = FALSE]
      )[1L, 1L]
    }, numeric(1)),
    X
  )
}


# ---- Internal: first point -----------------------------------------------

# Use `x_init` if supplied; otherwise minimise -log_p with the same optimiser.
# The search evaluates only log_p; score_function is called once at the winner.

.seed_x1 <- function(kernel, score_function, optimizer, log_p, x_init, d) {
  if (!is.null(x_init)) {
    x_mat <- matrix(x_init, 1L, d)
    grad <- .as_score_matrix(score_function(x_mat), x_mat)[1L, ]
    return(list(x = x_init, grad = grad, n_eval = 1L,
                k0_self = as.numeric(.k0_diag(kernel,
                  matrix(x_init, 1, d), matrix(grad, 1, d)))))
  }
  res <- optimizer(.obj_neg_log_p(log_p), matrix(0, 0, d),
                   t = 1L)
  x_mat <- matrix(res$x_min, 1L, d)
  grad <- .as_score_matrix(score_function(x_mat), x_mat)[1L, ]
  list(x = res$x_min, grad = grad, n_eval = res$n_eval + 1L,
       k0_self = as.numeric(.k0_diag(kernel,
         matrix(res$x_min, 1, d), matrix(grad, 1, d))))
}


# ---- Internal: optimizer objectives --------------------------------------

# First point: maximise log_p(x), implemented as minimise -log_p(x). Scores are
# deliberately deferred until the optimizer has selected one winning point.
.obj_neg_log_p <- function(log_p) {
  function(X_new, scores = NULL) {
    lp <- as.numeric(log_p(X_new))
    if (length(lp) != nrow(X_new) || any(is.na(lp)))
      stop("`log_p` must return a numeric vector of length nrow(X_new) ",
           "without NAs", call. = FALSE)
    list(
      objective_values = -lp,
      scores = matrix(NA_real_, nrow = nrow(X_new), ncol = ncol(X_new))
    )
  }
}

# Greedy target for a candidate x:
#   k0(x, x) + 2 * sum_i k0(x_i, x)
# `k0_diag` is returned so `.truncate()` can filter candidates without a second
# self-kernel pass, which for a score kernel costs an extra Hessian sweep over
# the whole candidate batch. Truncation is then free relative to no truncation.
# With no retained points the interaction sum is empty and the objective is
# k0(x, x); `.k0_matrix()` rejects a zero-row argument, so it is skipped. That
# is the coordinate-descent case of a single-point design.
.obj_greedy <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, scores = NULL) {
    if (is.null(scores)) {
      scores <- .as_score_matrix(score_function(X_new), X_new)
    }
    diag_k0 <- .k0_diag(kernel, X_new, scores)
    cross <- if (nrow(X_sel) == 0L) 0 else
      2 * colSums(.k0_matrix(kernel, X_sel, X_new, D_sel, scores))
    list(
      objective_values = cross + diag_k0,
      scores = scores,
      k0_diag = diag_k0
    )
  }
}

# Herding target for a candidate x:
#   sum_i k0(x_i, x)
.obj_herding <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, scores = NULL) {
    if (is.null(scores)) {
      scores <- .as_score_matrix(score_function(X_new), X_new)
    }
    list(
      objective_values = colSums(
        .k0_matrix(kernel, X_sel, X_new, D_sel, scores)
      ),
      scores = scores
    )
  }
}


# ---- Internal: optional truncation ---------------------------------------

# Radius schedule for the candidate filter k0(x, x) <= R_j^2.
.r_squared <- function(j, n_total, c2, mode) {
  upper <- 2 * log(max(n_total, 2L)) / c2
  lower <- 2 * log(j) / c2
  switch(mode,
    upper = upper, lower = lower,
    linear = lower + (upper - lower) * ((j - 1) / max(n_total - 1, 1L)))
}

# Set infeasible candidates to a finite penalty and attach feasibility metadata.
# Batch optimizers reject an all-infeasible candidate set after evaluation;
# Nelder-Mead can probe an infeasible singleton and continue searching.
.truncate <- function(obj, kernel, r2) {
  if (!is.finite(r2) || r2 <= 0) return(obj)
  function(X_new, scores = NULL) {
    res <- obj(X_new, scores)
    diag_k0 <- if (is.null(res$k0_diag)) .k0_diag(kernel, X_new, res$scores)
               else res$k0_diag
    bad <- !is.finite(diag_k0) | diag_k0 > r2
    res$objective_values[bad] <- .Machine$double.xmax
    res$feasible <- !bad
    res$truncation_r2 <- r2
    res
  }
}


# ---- Kernel operations: inverse-log (k2) ---------------------------------

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


# r(x, x) = 0 gives log1p(0) = 0, leaving alpha in every power. This kernel
# exposes no preconditioner, so the trace term is simply d.
.inverse_log_k0_diag <- function(k, X, S_X, M) {
  a <- k$alpha
  b <- k$beta
  -2 * b * ncol(X) * a^(b - 1) + a^b * rowSums(S_X * S_X)
}


# ---- Kernel operations: score-distance IMQ (k3) --------------------------

# k0 for k3, which measures distance between score vectors, not points.
.imq_score_k0_matrix <- function(k, X, S_X, Y, S_Y, M) {
  a <- k$alpha; b <- k$beta
  H_X <- k$hess_log_p(X)
  H_Y <- if (identical(X, Y)) H_X else k$hess_log_p(Y)
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

# s_p(x) - s_p(x) = 0, so the base kernel collapses to alpha^beta and the mixed
# trace to the squared Frobenius norm of the target Hessian at each row.
.imq_score_k0_diag <- function(k, X, S_X, M) {
  a <- k$alpha
  b <- k$beta
  H <- k$hess_log_p(X)
  j_norm_sq <- vapply(
    seq_len(nrow(X)), function(i) sum(H[i, , ]^2), numeric(1)
  )
  -2 * b * a^(b - 1) * j_norm_sq + a^b * rowSums(S_X * S_X)
}
