# Stein Points, coordinate refinement, and candidate-search optimizers.

# Public complete algorithms

#' Construct points by Stein discrepancy minimization
#'
#' Builds a point set sequentially. At each step, `optimizer` searches the
#' continuous state space for the next point under a greedy or herding Stein
#' objective.
#'
#' @details
#' Let \eqn{k_{0,p}} be the Stein kernel formed from `kernel` and
#' \eqn{s_p(x)=\nabla_x\log p(x)}. The first point is `x_init`; if `x_init` is
#' `NULL`, `optimizer` instead maximizes `log_p`.
#'
#' Given selected points
#' \eqn{x_1,\ldots,x_{j-1}}, the greedy optimizer minimizes
#' \deqn{G_j(x)=k_{0,p}(x,x)
#'       +2\sum_{i=1}^{j-1}k_{0,p}(x_i,x).}
#' This is the amount added to the kernel sum used by the running KSD. With
#' `method = "herding"`, the optimizer instead minimizes
#' \deqn{H_j(x)=\sum_{i=1}^{j-1}k_{0,p}(x_i,x).}
#'
#' The supplied [fmin_grid()], [fmin_mc()], and [fmin_nm()] constructors use
#' grid, Monte Carlo, and Nelder-Mead search, respectively.
#'
#' The returned discrepancy after step \eqn{j} is
#' \deqn{\mathrm{KSD}_j
#' =\left\{\frac{1}{j^2}\sum_{a=1}^j\sum_{b=1}^j
#' k_{0,p}(x_a,x_b)\right\}^{1/2}.}
#' Truncation keeps candidates satisfying
#' \deqn{k_{0,p}(x,x) \le R_j^2.}
#' Write \eqn{U^2=2\log\{\max(n_points,2)\}/c2} and
#' \eqn{L_j^2=2\log(j)/c2}. `"upper"` uses \eqn{U^2}, `"lower"` uses
#' \eqn{L_j^2}, and `"linear"` uses
#' \eqn{L_j^2+(U^2-L_j^2)(j-1)/\max(n_points-1,1)}. Larger `c2` gives a smaller
#' radius. Truncation starts at step 2; `"none"` disables it.
#'
#' @param score_function Function taking an `n x d` matrix and returning the
#'   corresponding `n x d` matrix of target scores.
#' @param kernel A `SteinKernel` object used to construct \eqn{k_{0,p}}. A
#'   Gaussian RBF kernel must use a fixed positive bandwidth.
#' @param n_points Positive number of points to select.
#' @param d Positive state dimension.
#' @param optimizer Function taking `(objective, X_curr, t)` and returning
#'   `x_min` and its score `d_min` as length-`d` vectors, `f_min` as the value
#'   of the supplied objective at `x_min`, and a nonnegative integer `n_eval`.
#'   The returned vectors and objective value must be finite. During
#'   initialization from `log_p`, only `x_min` and `n_eval` are used; the
#'   selected point is scored separately. With truncation,
#'   the optimizer must select a feasible candidate; the selected point is
#'   checked before it is appended. An infeasible result raises an error. The
#'   selected point's objective is recomputed before updating the running KSD.
#' @param method Point-selection rule: `"greedy"` or `"herding"`.
#' @param log_p Optional log density used to choose the first point.
#' @param x_init Optional finite numeric vector of length `d` giving the first
#'   point. If supplied, `log_p` is not used.
#' @param c2 Finite positive denominator in the truncation radius. Ignored when
#'   `truncation = "none"`.
#' @param truncation Candidate filter defined above: `"none"`, `"upper"`,
#'   `"lower"`, or `"linear"`.
#' @param seed Optional local RNG seed.
#'
#' @return
#' An object of class `"stein_points"` with:
#' * `X` and `D`: `n_points x d` matrices of selected points and their scores.
#' * `ksd`: running discrepancy defined above.
#' * `n_eval`: evaluation count reported by `optimizer` at each step; see
#'   [fmin_grid()], [fmin_mc()], and [fmin_nm()] for what each one counts. The
#'   first entry is 1 when `x_init` is supplied; otherwise it adds 1 for scoring
#'   the selected first point. `cum_n_eval = cumsum(n_eval)`.
#' * `method`, `kernel`, `truncation`, and `c2`: settings used by the run.
#' * `call`: matched function call.
#' @references Chen et al. (2018), Stein Points.
#'
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' opt <- fmin_grid(lb = -1, ub = 1, n0 = 3, grow = FALSE)
#' stein_points(score, kernel, n_points = 2, d = 1, optimizer = opt, 
#'             log_p = log_p)
#' @export
stein_points <- function(score_function, kernel, n_points, d, optimizer,
                         method = c("greedy", "herding"),
                         log_p = NULL, x_init = NULL, c2 = NULL,
                         truncation = c("none", "upper", "lower", "linear"),
                         seed = NULL) {
  cl <- match.call()
  method <- match.arg(method)
  truncation <- match.arg(truncation)
  if (!is.function(score_function))
    stop("`score_function` must be a function", call. = FALSE)
  n_points <- validate_integer(n_points, "n_points")
  d <- validate_integer(d, "d")
  herding <- identical(method, "herding")
  .stein_points_check_kernel(kernel)

  use_trunc <- truncation != "none"
  if (use_trunc && (!is.numeric(c2) || length(c2) != 1L || !is.finite(c2) || c2 <= 0))
    stop("`c2` must be finite and positive when truncation != 'none'.", call. = FALSE)
  if (is.null(x_init)) {
    if (is.null(log_p))
      stop("supply `log_p` or `x_init` for the first point", call. = FALSE)
    if (!is.function(log_p)) stop("`log_p` must be a function.", call. = FALSE)
  } else {
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
    if (ss < -sqrt(.Machine$double.eps)) warning("Accumulated squared KSD is negative; using 0.", call. = FALSE)
    ksd[1] <- sqrt(max(ss, 0))

    if (n_points >= 2L) {
      for (j in 2:n_points) {
        selected_X <- X[seq_len(j - 1L), , drop = FALSE]
        selected_scores <- D[seq_len(j - 1L), , drop = FALSE]
        # Build the objective from the points already selected.
        obj_base <- if (herding) {
          .obj_herding(kernel, score_function, selected_X, selected_scores)
        } else {
          .obj_greedy(kernel, score_function, selected_X, selected_scores)
        }
        res <- .validate_optimizer_result(
          optimizer(trunc_obj(obj_base, j), selected_X, t = j), d
        )
        self <- NULL
        if (use_trunc || herding) {
          self <- k0_diag(kernel, matrix(res$x_min, 1L, d), matrix(res$d_min, 1L, d))
          if (use_trunc &&
              (!is.finite(self) || self > .r_squared(j, n_points, c2, truncation))) {
            stop("optimizer returned a point outside the truncation radius", call. = FALSE)
          }
        }
        # The algorithm owns its KSD invariant; do not trust an optimizer's
        # cached objective value for the selected point.
        selected <- obj_base(
          matrix(res$x_min, 1L, d), matrix(res$d_min, 1L, d)
        )
        f_min <- selected$objective_values[1L]
        X[j, ] <- res$x_min
        D[j, ] <- res$d_min
        n_eval[j] <- res$n_eval

        # Herding omits the self term and counts each cross term once.
        ss <- ss + if (herding) {
          2 * f_min + self
        } else {
          f_min
        }
        if (ss < -sqrt(.Machine$double.eps)) warning("Accumulated squared KSD is negative; using 0.", call. = FALSE)
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
#' Refines a completed Stein point set one row at a time while holding the
#' others fixed. A replacement is accepted only if the point-set KSD does not
#' increase.
#'
#' @details
#' At iteration `it`, row \eqn{r} `= ((it - 1) %% nrow(X0)) + 1` is replaced by
#' minimizing
#' \deqn{k0(x,x) + 2\sum_{i\ne r} k0(x_i,x)}
#' over `x`. This is the greedy objective of [stein_points()], except the set
#' size stays fixed. With a single row the sum is empty and the objective is
#' \eqn{k0(x,x)}.
#'
#' Because the optimizers are approximate, each proposal is compared with the
#' current row under the same objective before it is accepted. The optimizer
#' interface is the same as in [stein_points()].
#'
#' @param X0 Initial point matrix.
#' @param score_function Function returning scores for candidate rows.
#' @param kernel A `SteinKernel` object.
#' @param n_iter Number of coordinate-descent updates.
#' @param optimizer Optimizer function used for each coordinate update.
#' @param seed Optional RNG seed.
#'
#' @return
#' An object of class `"stein_codescent"` with:
#' * `X`: refined point matrix with the same dimensions as `X0`.
#' * `D`: target scores at the refined points.
#' * `objective`: retained value of the coordinate objective at each update.
#'   These are not KSDs and are not comparable across updates, because the
#'   objective omits the kernel sum over the fixed rows, which changes as the
#'   points move.
#' * `n_eval`: optimizer-reported evaluations per update; the first entry also
#'   counts the `nrow(X0)` initial scores.
#' * `cum_n_eval = cumsum(n_eval)`.
#' * `kernel`: kernel used to define the Stein objective.
#'
#' No running `ksd` is returned. Compute the final KSD as
#' `sqrt(sum(stein_kernel_matrix(kernel, out$X, out$D))) / nrow(out$X)`.
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
  if (!is.function(score_function))
    stop("`score_function` must be a function", call. = FALSE)
  n_iter <- validate_integer(n_iter, "n_iter", min_value = 0L)
  .stein_points_check_kernel(kernel)
  X0 <- .as_rows(X0, "X0")
  n <- nrow(X0)
  run <- function() {
    X <- X0
    D <- .as_score_matrix(score_function(X), X)
    n_eval <- integer(n_iter)
    # Store coordinate objectives, not KSD values.
    objective <- numeric(n_iter)
    if (n_iter >= 1L) {
      for (it in seq_len(n_iter)) {
        j <- ((it - 1L) %% n) + 1L
        X_other <- X[-j, , drop = FALSE]
        D_other <- D[-j, , drop = FALSE]
        obj <- .obj_greedy(kernel, score_function, X_other, D_other)
        res <- .validate_optimizer_result(
          optimizer(obj, X_other, t = it), ncol(X)
        )

        X_old <- X[j, , drop = FALSE]
        D_old <- D[j, , drop = FALSE]
        X_new <- matrix(res$x_min, nrow = 1L)
        candidate_scores <- matrix(res$d_min, nrow = 1L)
        old_value <- obj(X_old, D_old)$objective_values[1L]
        new_value <- obj(X_new, candidate_scores)$objective_values[1L]
        # Keep only replacements that do not increase the objective.
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


# Public print methods

#' Print a Stein point set
#'
#' Prints the point matrix size and final KSD or coordinate-update count, plus
#' the corresponding evaluation total.
#'
#' @param x An object returned by [stein_points()] or [stein_codescent()].
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
  } else {
    sprintf("  updates=%d", length(x$objective))
  }
  cat(sprintf(
    "%s  n=%d  d=%d%s  n_eval=%g\n",
    .point_set_label(x$method), nrow(x$X), ncol(x$X), diagnostic,
    last(x$cum_n_eval)
  ))
  invisible(x)
}

#' @rdname print.stein_points
#' @export
print.stein_codescent <- print.stein_points

#' Summarize a Stein point set
#'
#' Summarizes the point-set size, kernel, method-specific settings, and
#' evaluation cost for [stein_points()] or [stein_codescent()].
#'
#' @details
#' `evaluation_label` names what `n_eval_total` counts for this class.
#' Coordinate-descent objectives are not displayed because they change with
#' the row being updated and are not point-set KSDs.
#'
#' @param object A `"stein_points"` or `"stein_codescent"` object.
#' @param x A `"summary.stein_points"` object.
#' @param ... Ignored.
#'
#' @return
#' A `"summary.stein_points"` object with components `method`, `kernel`, `n`,
#' `d`, `evaluation_label`, `n_eval_total`, and `call`. Stein Points adds
#' `ksd_last` and `truncation`, plus `c2` when truncation is enabled.
#' Coordinate descent adds `n_iter`.
#' The print method returns `x` invisibly.
#' @seealso [print.stein_points()]
#' @examples
#' score_function <- function(x) -as.matrix(x)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' fit <- stein_points(score_function, kernel, n_points = 5, d = 1,
#'                     optimizer = fmin_grid(lb = -3, ub = 3, n0 = 20),
#'                     log_p = function(x) -0.5 * rowSums(as.matrix(x)^2))
#' summary(fit)
#' @export
summary.stein_points <- function(object, ...) {
  out <- .point_set_summary(object, "objective evaluations")
  ksd <- object$ksd
  if (length(ksd)) {
    out$ksd_last <- ksd[length(ksd)]
  }
  out$truncation <- object$truncation
  if (!is.null(object$truncation) && !identical(object$truncation, "none")) {
    out$c2 <- object$c2
  }
  out
}

#' @rdname summary.stein_points
#' @export
summary.stein_codescent <- function(object, ...) {
  out <- .point_set_summary(object, "objective evaluations")
  out$n_iter <- length(object$objective)
  out
}

#' @rdname summary.stein_points
#' @export
print.summary.stein_points <- function(x, ...) {
  cat(sprintf("%s  n=%d  d=%d  kernel=%s\n",
              .point_set_label(x$method), x$n, x$d, x$kernel))
  if (!is.null(x$n_iter)) {
    if (is.null(x$step_size)) {
      cat(sprintf("  updates: %d\n", x$n_iter))
    } else {
      cat(sprintf("  iterations: %d   step size: %g\n", x$n_iter, x$step_size))
    }
  }
  .print_point_set_body(x)
}

# Public candidate-search optimizers

#' Create the grid search used by Stein Points
#'
#' Creates a deterministic grid-search function for [stein_points()]. It
#' evaluates every point on a Cartesian grid and returns the smallest objective
#' value.
#'
#' @details
#' With `grow = TRUE`, the grid resolution increases as the point set grows: the
#' per-dimension size is
#' `n0 + round(sqrt(t))`, where `t` is the optimizer iteration supplied by
#' [stein_points()] or [stein_codescent()]. It is practical mainly in low
#' dimension because the total number of candidates is the product of the grid
#' sizes across dimensions.
#'
#' @param lb,ub Finite lower and upper bound vectors of equal length, with
#'   `ub > lb` componentwise.
#' @param n0 Positive integer grid size, recycled across dimensions, or a
#'   length-`d` vector of positive integer sizes.
#' @param grow A single logical value; whether to increase grid size as points
#'   are selected.
#'
#' @return
#' An optimizer function with signature `function(objective, X_curr, t)` that
#' returns `x_min`, `d_min`, `f_min`, and `n_eval`. `x_min` is the grid row with
#' the smallest objective,
#' `d_min` is the target score at that row, `f_min` is its objective value, and
#' `n_eval` is the number of grid rows scored.
#' @examples
#' fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)
#' @export
fmin_grid <- function(lb, ub, n0 = 100, grow = TRUE) {
  .validate_optimizer_bounds(lb, ub)
  d <- length(lb)
  if (length(n0) == 1L) n0 <- rep(n0, d)
  if (!is.numeric(n0) || length(n0) != d) {
    stop("`n0` must be a positive integer or a length-d vector of positive integers",
         call. = FALSE)
  }
  n0 <- vapply(n0, validate_integer, integer(1L), arg_name = "n0 entry")
  grow <- validate_flag(grow, "grow")
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
#' Creates a candidate-search function for [stein_points()]. It draws a finite
#' set of candidate points in the box and returns the candidate with the
#' smallest supplied objective value.
#'
#' @details
#' The returned optimizer is a function used by [stein_points()] and
#' [stein_codescent()]. Early iterations draw candidates from a broad Gaussian
#' distribution truncated to `[lb, ub]`. Once `t` exceeds `delay`,
#' the proposal becomes local: it chooses one of the current points and draws a
#' Gaussian perturbation with variance `sigsq`, again keeping only candidates in
#' the box.
#'
#' @param lb,ub Finite lower and upper bound vectors of equal length, with
#'   `ub > lb` componentwise.
#' @param n_mc Positive integer number of Monte Carlo candidates.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance. `mu0` must
#'   be a finite length-`d` vector and `Sigma0` a finite symmetric
#'   positive-definite `d x d` matrix.
#' @param sigsq Finite positive local proposal variance after the delay period.
#' @param delay Nonnegative integer number of optimization iterations before
#'   using local proposals.
#'
#' @return
#' An optimizer function with signature `function(objective, X_curr, t)`. It
#' returns a list with `x_min` (best candidate row), `d_min` (score at that
#' row), `f_min` (objective value), and `n_eval` (number of candidates scored).
#' @examples
#' fmin_mc(lb = -1, ub = 1, n_mc = 5)
#' @export
fmin_mc <- function(lb, ub, n_mc = 20, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20) {
  .validate_optimizer_bounds(lb, ub)
  n_mc <- validate_integer(n_mc, "n_mc")
  controls <- .validate_adaptive_controls(lb, ub, mu0, Sigma0, sigsq, delay)
  p <- controls$proposal
  sigsq <- controls$sigsq
  delay <- controls$delay
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
#' The returned optimizer performs a multi-start local search. A sine-squared
#' transformation maps unconstrained Nelder-Mead parameters back into
#' `[lb, ub]`, so every objective evaluation stays inside
#' the requested search box. Each restart begins from the same boxed proposal
#' rule used by [fmin_mc()].
#'
#' @param lb,ub Finite lower and upper bound vectors of equal length, with
#'   `ub > lb` componentwise.
#' @param n_res Positive integer number of random restarts.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance. `mu0` must
#'   be a finite length-`d` vector and `Sigma0` a finite symmetric
#'   positive-definite `d x d` matrix.
#' @param sigsq Finite positive local proposal variance after the delay period.
#' @param delay Nonnegative integer number of optimization iterations before
#'   using local proposals.
#' @param control List passed to `stats::optim()`.
#'
#' @return
#' An optimizer function with signature `function(objective, X_curr, t)` that
#' returns `x_min` (the selected point), `d_min` (its score), `f_min` (its
#' objective value), and `n_eval` (objective evaluations charged to the search).
#' @examples
#' fmin_nm(lb = -1, ub = 1, n_res = 2)
#' @export
fmin_nm <- function(lb, ub, n_res = 3, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20, control = list(reltol = 1e-3)) {
  .validate_optimizer_bounds(lb, ub)
  n_res <- validate_integer(n_res, "n_res")
  controls <- .validate_adaptive_controls(lb, ub, mu0, Sigma0, sigsq, delay)
  p <- controls$proposal
  sigsq <- controls$sigsq
  delay <- controls$delay
  if (!is.list(control)) stop("`control` must be a list", call. = FALSE)
  span <- ub - lb
  # Map unconstrained search coordinates back into the box.
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

.validate_optimizer_bounds <- function(lb, ub) {
  if (!is.numeric(lb) || !is.numeric(ub) || length(lb) < 1L ||
      length(lb) != length(ub) || any(!is.finite(c(lb, ub))) ||
      any(ub <= lb)) {
    stop("`lb` and `ub` must be finite vectors of equal positive length ",
         "with `ub > lb` componentwise", call. = FALSE)
  }
  invisible(NULL)
}

.adaptive_defaults <- function(lb, ub, mu0, Sigma0) {
  if (is.null(mu0)) mu0 <- (lb + ub) / 2
  if (is.null(Sigma0)) Sigma0 <- ((ub - lb) / 4)^2 * diag(length(lb))
  d <- length(lb)
  if (!is.numeric(mu0) || length(mu0) != d || any(!is.finite(mu0))) {
    stop("`mu0` must be a finite numeric vector of length d", call. = FALSE)
  }
  Sigma0 <- as.matrix(Sigma0)
  if (!is.numeric(Sigma0) || !identical(dim(Sigma0), c(d, d)) ||
      any(!is.finite(Sigma0))) {
    stop("`Sigma0` must be a finite d x d matrix", call. = FALSE)
  }
  sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(Sigma0)))
  if (max(abs(Sigma0 - t(Sigma0))) > sym_tol) {
    stop("`Sigma0` must be symmetric positive definite", call. = FALSE)
  }
  tryCatch(chol(Sigma0), error = function(e) {
    stop("`Sigma0` must be symmetric positive definite", call. = FALSE)
  })
  list(mu0 = as.numeric(mu0), Sigma0 = Sigma0)
}

.validate_adaptive_controls <- function(lb, ub, mu0, Sigma0, sigsq, delay) {
  if (!is.numeric(sigsq) || length(sigsq) != 1L || !is.finite(sigsq) ||
      sigsq <= 0) {
    stop("`sigsq` must be a finite positive scalar", call. = FALSE)
  }
  list(
    proposal = .adaptive_defaults(lb, ub, mu0, Sigma0),
    sigsq = as.numeric(sigsq),
    delay = validate_integer(delay, "delay", min_value = 0L)
  )
}

# Select the smallest candidate allowed by truncation.
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

# Switch from broad to selected-point proposals after `delay`.
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

# Internal Stein kernel access

.stein_points_check_kernel <- function(kernel) {
  require_fixed_gaussian_rbf(kernel, "Stein Points")
  if (inherits(kernel, "SteinKernel_imq")) {
    if (!is.numeric(kernel$scale2) || length(kernel$scale2) != 1L ||
        !is.finite(kernel$scale2) || kernel$scale2 <= 0) {
      stop("IMQ Stein Points kernel requires finite c > 0.", call. = FALSE)
    }
    if (!is.numeric(kernel$beta) || length(kernel$beta) != 1L ||
        !is.finite(kernel$beta) || kernel$beta <= -1 || kernel$beta >= 0) {
      stop("IMQ Stein Points kernel requires beta in (-1, 0).", call. = FALSE)
    }
  }
  invisible(kernel)
}


# Internal first point

# Score only the winning first-point candidate.
.seed_x1 <- function(kernel, score_function, optimizer, log_p, x_init, d) {
  if (!is.null(x_init)) {
    x_mat <- matrix(x_init, 1L, d)
    grad <- .as_score_matrix(score_function(x_mat), x_mat)[1L, ]
    return(list(x = x_init, grad = grad, n_eval = 1L,
                k0_self = as.numeric(k0_diag(kernel,
                  matrix(x_init, 1, d), matrix(grad, 1, d)))))
  }
  res <- .validate_optimizer_result(
    optimizer(.obj_neg_log_p(log_p), matrix(0, 0, d), t = 1L),
    d,
    initialization = TRUE
  )
  x_mat <- matrix(res$x_min, 1L, d)
  grad <- .as_score_matrix(score_function(x_mat), x_mat)[1L, ]
  list(x = res$x_min, grad = grad, n_eval = res$n_eval + 1L,
       k0_self = as.numeric(k0_diag(kernel,
         matrix(res$x_min, 1, d), matrix(grad, 1, d))))
}

.validate_optimizer_result <- function(res, d, initialization = FALSE) {
  bad_x <- !is.list(res) || !is.numeric(res$x_min) ||
    length(res$x_min) != d || any(!is.finite(res$x_min))
  if (bad_x) {
    msg <- if (initialization)
      "optimizer must return finite length-d x_min during initialization"
    else
      "optimizer must return finite x_min, d_min, and scalar f_min"
    stop(msg, call. = FALSE)
  }
  if (!initialization && (
      !is.numeric(res$d_min) || length(res$d_min) != d ||
      any(!is.finite(res$d_min)) || !is.numeric(res$f_min) ||
      length(res$f_min) != 1L || !is.finite(res$f_min))) {
    stop("optimizer must return finite x_min, d_min, and scalar f_min",
         call. = FALSE)
  }
  res$n_eval <- validate_integer(res$n_eval, "optimizer n_eval", min_value = 0L)
  res
}


# Internal optimizer objectives

# Maximize log_p by minimizing its negative.
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

# Greedy objective: k0(x, x) + 2 * sum_i k0(x_i, x).
.obj_greedy <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, scores = NULL) {
    if (is.null(scores)) {
      scores <- .as_score_matrix(score_function(X_new), X_new)
    }
    diag_k0 <- k0_diag(kernel, X_new, scores)
    # Skip the interaction matrix when no points remain.
    cross <- if (nrow(X_sel) == 0L) 0 else
      2 * colSums(stein_kernel_matrix(kernel, X_sel, D_sel, X_new, scores))
    list(
      objective_values = cross + diag_k0,
      scores = scores,
      k0_diag = diag_k0
    )
  }
}

# Herding objective: sum_i k0(x_i, x).
.obj_herding <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, scores = NULL) {
    if (is.null(scores)) {
      scores <- .as_score_matrix(score_function(X_new), X_new)
    }
    list(
      objective_values = colSums(
        stein_kernel_matrix(kernel, X_sel, D_sel, X_new, scores)
      ),
      scores = scores
    )
  }
}


# Internal optional truncation

# Set the truncation radius for step j.
.r_squared <- function(j, n_total, c2, mode) {
  upper <- 2 * log(max(n_total, 2L)) / c2
  lower <- 2 * log(j) / c2
  switch(mode,
    upper = upper, lower = lower,
    linear = lower + (upper - lower) * ((j - 1) / max(n_total - 1, 1L)))
}

# Use a finite penalty so Nelder-Mead can continue searching.
.truncate <- function(obj, kernel, r2) {
  if (!is.finite(r2) || r2 <= 0)
    stop("truncation radius must be finite and positive", call. = FALSE)
  function(X_new, scores = NULL) {
    res <- obj(X_new, scores)
    diag_k0 <- if (is.null(res$k0_diag)) k0_diag(kernel, X_new, res$scores)
               else res$k0_diag
    bad <- !is.finite(diag_k0) | diag_k0 > r2
    res$objective_values[bad] <- .Machine$double.xmax
    res$feasible <- !bad
    res$truncation_r2 <- r2
    res
  }
}
