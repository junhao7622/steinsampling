# stein_points and stein_codescent come first, then the optimizers fmin_grid,
# fmin_mc and fmin_nm, then the greedy and herding objectives.

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
#' Supply a custom `optimizer` with the interface described under that
#' argument to replace this search. To refine a completed point set, use
#' [stein_codescent()].
#'
#' The returned discrepancy after step \eqn{j} is
#' \deqn{\mathrm{KSD}_j
#' =\left\{\frac{1}{j^2}\sum_{a=1}^j\sum_{b=1}^j
#' k_{0,p}(x_a,x_b)\right\}^{1/2}.}
#' Truncation keeps candidates satisfying
#' \deqn{k_{0,p}(x,x) \le R_j^2.}
#' Write \eqn{U^2=2\log\{\max(n_{\mathrm{points}},2)\}/\mathrm{c2}} and
#' \eqn{L_j^2=2\log(j)/\mathrm{c2}}. `"upper"` uses \eqn{U^2}, `"lower"` uses
#' \eqn{L_j^2}, and `"linear"` uses
#' \eqn{L_j^2+(U^2-L_j^2)(j-1)/\max(n_{\mathrm{points}}-1,1)}, where
#' \eqn{n_{\mathrm{points}}} and \eqn{\mathrm{c2}} are the arguments of the
#' same name. Larger `c2` gives a smaller radius. Truncation starts at step 2;
#' `"none"` disables it.
#'
#' @param score_function Function taking an \eqn{n\times d} matrix and
#'   returning the corresponding \eqn{n\times d} matrix of target scores.
#' @param kernel A `SteinKernel` object used to construct \eqn{k_{0,p}}. A
#'   Gaussian RBF kernel must use a fixed positive bandwidth.
#' @param n_points Positive number of points to select.
#' @param d Positive state dimension.
#' @param optimizer Function taking `(objective, X_curr, t)` and returning
#'   `x_min` and its score `d_min` as length-`d` vectors and a nonnegative
#'   integer `n_eval`.
#'   `objective(X_new, scores = NULL)` is vectorized over the rows of `X_new`
#'   and returns a list: `objective_values` holds one value per candidate row,
#'   the value to minimize, and `scores` holds the scores it used, which supply
#'   `d_min` without scoring the selected point again. `X_curr` holds the points
#'   kept fixed at this step, and `t` is the step counter.
#'   The returned vectors must be finite. During
#'   initialization from `log_p`, only `x_min` and `n_eval` are used; the
#'   selected point is scored separately. With truncation,
#'   the optimizer must select a feasible candidate; the selected point is
#'   checked before it is appended. An infeasible result raises an error. The
#'   selected point's objective is recomputed before updating the running KSD.
#' @param method Point-selection rule: `"greedy"` or `"herding"`.
#' @param log_p Log-density function taking an \eqn{n\times d} matrix and
#'   returning \eqn{n} values; required when `x_init` is `NULL`, otherwise
#'   ignored. The density need not be normalized.
#' @param x_init Optional finite numeric vector of length `d` giving the first
#'   point.
#' @param c2 Finite positive denominator in the truncation radius. Ignored when
#'   `truncation = "none"`.
#' @param truncation Candidate filter described under Details: `"none"`,
#'   `"upper"`, `"lower"`, or `"linear"`.
#'
#' @return
#' `stein_points()` returns an object of class `"stein_points"` with:
#' * `X` and `D`: \eqn{n_{\mathrm{points}}\times d} matrices of selected points
#'   and their scores.
#' * `ksd`: running discrepancy defined above.
#' * `n_eval`: evaluation count reported by `optimizer` at each step; see
#'   [fmin_grid()], [fmin_mc()], and [fmin_nm()] for what each one counts. The
#'   first entry is 1 when `x_init` is supplied; otherwise it adds 1 for scoring
#'   the selected first point. `cum_n_eval = cumsum(n_eval)`.
#' * `method`, `kernel`, `truncation`, and `c2`: settings used by the run.
#' * `call`: matched function call.
#'
#' `print(fit)` displays the point-set size, final KSD, and evaluation count,
#' and returns `fit` invisibly. `summary(fit)` returns a
#' `"summary.stein_points"` object with components `method`, `kernel`, `n`,
#' `d`, `evaluation_label`, `n_eval_total`, `call`, `ksd_last`, and
#' `truncation`, plus `c2` when truncation is enabled. `evaluation_label`
#' names what `n_eval_total` counts. Printing that object returns it
#' invisibly.
#' @seealso [fmin_grid()], [fmin_mc()], and [fmin_nm()] for the supplied
#'   optimizers; [stein_codescent()] to refine a finished point set.
#' @references
#' Chen, W. Y., Mackey, L., Gorham, J., Briol, F.-X., and Oates, C. J. (2018).
#' Stein Points. Proceedings of the 35th International Conference on Machine
#' Learning, Proceedings of Machine Learning Research, 80, 844--853.
#' \url{https://proceedings.mlr.press/v80/chen18f.html}.
#'
#' @examples
#' # Target: N(0, 1).
#' score <- function(X) -X
#' kernel <- stein_kernel("imq")
#' opt <- fmin_grid(lb = -4, ub = 4, n0 = 401, grow = FALSE)
#' fit <- stein_points(score, kernel, n_points = 50, d = 1,
#'                     optimizer = opt, x_init = 0)
#' print(fit)
#' c(mean = mean(fit$X), second_moment = mean(fit$X^2))  # Targets: 0 and 1
#' @export
stein_points <- function(score_function, kernel, n_points, d, optimizer,
                         method = c("greedy", "herding"),
                         log_p = NULL, x_init = NULL, c2 = NULL,
                         truncation = c("none", "upper", "lower", "linear")) {
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
  if (use_trunc) validate_positive_scalar(c2, "c2")
  if (is.null(x_init)) {
    if (is.null(log_p))
      stop("supply `log_p` or `x_init` for the first point", call. = FALSE)
    if (!is.function(log_p)) stop("`log_p` must be a function", call. = FALSE)
  } else {
    x_init <- as.numeric(x_init)
    if (length(x_init) != d || !all(is.finite(x_init)))
      stop("`x_init` must be a finite numeric vector of length d", call. = FALSE)
  }

  X <- D <- matrix(NA_real_, n_points, d)
  n_eval <- integer(n_points)
  ksd <- numeric(n_points)

  s1 <- .seed_x1(kernel, score_function, optimizer, log_p, x_init, d)
  X[1, ] <- s1$x; D[1, ] <- s1$grad
  n_eval[1] <- s1$n_eval; ss <- s1$k0_self
  ksd[1] <- .running_ksd(ss, 1L)

  if (n_points >= 2L) {
    for (j in 2:n_points) {
      selected_X <- X[seq_len(j - 1L), , drop = FALSE]
      selected_scores <- D[seq_len(j - 1L), , drop = FALSE]
      obj_base <- if (herding) {
        .obj_herding(kernel, score_function, selected_X, selected_scores)
      } else {
        .obj_greedy(kernel, score_function, selected_X, selected_scores)
      }
      r2 <- if (use_trunc) .r_squared(j, n_points, c2, truncation)
      objective <- if (use_trunc) .truncate(obj_base, kernel, r2) else obj_base
      res <- .validate_optimizer_result(
        optimizer(objective, selected_X, t = j), d
      )
      self <- NULL
      if (use_trunc || herding) {
        self <- k0_diag(kernel, matrix(res$x_min, 1L, d), matrix(res$d_min, 1L, d))
        if (use_trunc &&
            (!is.finite(self) || self > r2)) {
          stop("optimizer returned a point outside the truncation radius", call. = FALSE)
        }
      }
      # Recompute the objective at the chosen point instead of trusting the
      # optimizer's value, so the running KSD stays exact for any custom
      # optimizer.
      selected <- obj_base(
        matrix(res$x_min, 1L, d), matrix(res$d_min, 1L, d)
      )
      f_min <- selected$objective_values[1L]
      X[j, ] <- res$x_min
      D[j, ] <- res$d_min
      n_eval[j] <- res$n_eval

      # The herding objective has no self term and counts each pair once, so the
      # kernel sum grows by twice f_min plus the self term.
      ss <- ss + if (herding) {
        2 * f_min + self
      } else {
        f_min
      }
      ksd[j] <- .running_ksd(ss, j)
    }
  }
  structure(
    list(X = X, D = D, ksd = ksd, n_eval = n_eval,
         cum_n_eval = cumsum(n_eval), method = method,
         kernel = kernel, call = cl, truncation = truncation, c2 = c2),
    class = "stein_points"
  )
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
#' \deqn{k_{0,p}(x,x) + 2\sum_{i\ne r} k_{0,p}(x_i,x)}
#' over `x`, where \eqn{k_{0,p}} is the Stein kernel formed from `kernel` and
#' the target score. This is the greedy objective of [stein_points()], except
#' the set size stays fixed. With a single row the sum is empty and the
#' objective is \eqn{k_{0,p}(x,x)}.
#'
#' Because the optimizers are approximate, each proposal is compared with the
#' current row under the same objective before it is accepted. The optimizer
#' interface is the same as in [stein_points()].
#'
#' @param X0 Initial \eqn{n\times d} point matrix; rows are points.
#' @param score_function Function taking an \eqn{n\times d} matrix and
#'   returning the corresponding \eqn{n\times d} matrix of target scores.
#' @param kernel A `SteinKernel` object.
#' @param n_iter Number of coordinate-descent updates.
#' @param optimizer Optimizer function used for each coordinate update, such
#'   as [fmin_grid()], [fmin_mc()], or [fmin_nm()].
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
#'
#' `print(fit)` displays the point-set size, update count, and evaluation
#' count, and returns `fit` invisibly. `summary(fit)` returns a
#' `"summary.stein_codescent"` object with components `method`, `kernel`, `n`,
#' `d`, `evaluation_label`, `n_eval_total`, `call`, and `n_iter`;
#' `evaluation_label` names what `n_eval_total` counts. Coordinate-descent
#' objectives are not displayed because they change with the row being
#' updated and are not point-set KSDs. Printing that object returns it
#' invisibly.
#' @references
#' Chen, W. Y., Mackey, L., Gorham, J., Briol, F.-X., and Oates, C. J. (2018).
#' Stein Points. Proceedings of the 35th International Conference on Machine
#' Learning, Proceedings of Machine Learning Research, 80, 844--853.
#' \url{https://proceedings.mlr.press/v80/chen18f.html}.
#' @examples
#' # Target: N(0, 1).
#' score <- function(X) -X
#' kernel <- stein_kernel("imq")
#' X0 <- matrix(seq(-0.5, 0.5, length.out = 20), ncol = 1)
#' opt <- fmin_grid(-4, 4, n0 = 401, grow = FALSE)
#' fit <- stein_codescent(X0, score, kernel, n_iter = 40, optimizer = opt)
#' ksd <- function(X) {
#'   sqrt(max(0, sum(stein_kernel_matrix(kernel, X, score(X))))) / nrow(X)
#' }
#' c(before = ksd(X0), after = ksd(fit$X))
#' @export
stein_codescent <- function(X0, score_function, kernel, n_iter, optimizer) {
  cl <- match.call()
  if (!is.function(score_function))
    stop("`score_function` must be a function", call. = FALSE)
  n_iter <- validate_integer(n_iter, "n_iter", min_value = 0L)
  .stein_points_check_kernel(kernel)
  X0 <- .as_rows(X0, "X0")
  n <- nrow(X0)
  X <- X0
  D <- .as_score_matrix(score_function(X), X)
  n_eval <- integer(n_iter)
  # Store coordinate objectives, not KSD values.
  objective <- numeric(n_iter)
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
    # Accept a replacement that does not increase the objective. A finite value
    # always replaces a non-finite one.
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
  if (n_iter >= 1L) n_eval[1] <- n_eval[1] + n
  structure(
    list(X = X, D = D, objective = objective, n_eval = n_eval,
         cum_n_eval = cumsum(n_eval), method = "codescent",
         kernel = kernel, call = cl),
    class = "stein_codescent"
  )
}


#' @export
print.stein_points <- function(x, ...) {
  cat(sprintf(
    "Stein Points (%s)  n=%d  d=%d  KSD=%.4g  n_eval=%.0f\n",
    x$method, nrow(x$X), ncol(x$X),
    .last_value(x$ksd), .last_value(x$cum_n_eval)
  ))
  invisible(x)
}

#' @export
print.stein_codescent <- function(x, ...) {
  cat(sprintf(
    "Stein codescent  n=%d  d=%d  updates=%d  n_eval=%.0f\n",
    nrow(x$X), ncol(x$X), length(x$objective), .last_value(x$cum_n_eval)
  ))
  invisible(x)
}

#' @export
summary.stein_points <- function(object, ...) {
  out <- .point_set_summary(object, "objective evaluations",
                            "summary.stein_points")
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

#' @export
summary.stein_codescent <- function(object, ...) {
  out <- .point_set_summary(object, "objective evaluations",
                            "summary.stein_codescent")
  out$n_iter <- length(object$objective)
  out
}

#' @export
print.summary.stein_points <- function(x, ...) {
  cat(sprintf("Stein Points (%s)  n=%d  d=%d  kernel=%s\n",
              x$method, x$n, x$d, x$kernel))
  if (!is.null(x$ksd_last)) {
    cat(sprintf("  KSD: %.4g\n", x$ksd_last))
  }
  cat(sprintf("  %s: %.0f\n", x$evaluation_label, x$n_eval_total))
  if (!is.null(x$truncation) && !identical(x$truncation, "none")) {
    cat(sprintf("  truncation: %s  c2=%g\n", x$truncation, x$c2))
  }
  invisible(x)
}

#' @export
print.summary.stein_codescent <- function(x, ...) {
  cat(sprintf("Stein codescent  n=%d  d=%d  kernel=%s\n",
              x$n, x$d, x$kernel))
  cat(sprintf("  updates: %d\n", x$n_iter))
  cat(sprintf("  %s: %.0f\n", x$evaluation_label, x$n_eval_total))
  invisible(x)
}

#' Optimizers for Stein Points
#'
#' Creates a candidate-search function for [stein_points()] and
#' [stein_codescent()]. The returned optimizer selects the point with the
#' smallest objective among grid points ([fmin_grid()]), random candidates
#' in the box ([fmin_mc()]), or local solutions from several Nelder-Mead runs
#' started in the box ([fmin_nm()]).
#'
#' @rdname stein_point_optimizers
#' @details
#' With `grow = TRUE`, the [fmin_grid()] resolution increases as the point set
#' grows: the per-dimension size is
#' `n0 + round(sqrt(t))`, where `t` is the optimizer iteration supplied by
#' [stein_points()] or [stein_codescent()]. It is practical mainly in low
#' dimension because the total number of candidates is the product of the grid
#' sizes across dimensions.
#'
#' With [fmin_mc()], early iterations draw candidates from a broad
#' Gaussian distribution truncated to `[lb, ub]`. Once `t` exceeds `delay`,
#' the proposal becomes local: it chooses one of the current points and draws a
#' Gaussian perturbation with variance `sigsq`, again keeping only candidates in
#' the box.
#'
#' The optimizer returned by [fmin_nm()] performs a multi-start local search. A
#' sine-squared transformation maps unconstrained Nelder-Mead parameters back
#' into `[lb, ub]`, so every objective evaluation stays inside
#' the requested search box. Each restart begins from the same boxed proposal
#' rule used by [fmin_mc()].
#'
#' @param lb,ub Finite lower and upper bound vectors of equal length, with
#'   `ub > lb` componentwise.
#' @param n0 Positive integer grid size, recycled across dimensions, or a
#'   length-`d` vector of positive integer sizes.
#' @param grow A single logical value; whether to increase grid size as points
#'   are selected.
#'
#' @return
#' An optimizer function `function(f, X_curr, t = nrow(X_curr) + 1L)`, where
#' `f` is the objective described in [stein_points()]. It returns a list with
#' `x_min`, `d_min`, `f_min`, and `n_eval`. `d_min` is the target score
#' at the selected point and `f_min` is its objective value. The searches
#' differ in what they select and what `n_eval` counts:
#' * [fmin_grid()]: `x_min` is the grid row with the smallest objective, and
#'   `n_eval` is the number of grid rows scored.
#' * [fmin_mc()]: `x_min` is the best candidate row, and `n_eval` is the number
#'   of candidates scored.
#' * [fmin_nm()]: `x_min` is the selected point, and `n_eval` is the number of
#'   objective evaluations charged to the search.
#' @references
#' Chen, W. Y., Mackey, L., Gorham, J., Briol, F.-X., and Oates, C. J. (2018).
#' Stein Points. Proceedings of the 35th International Conference on Machine
#' Learning, Proceedings of Machine Learning Research, 80, 844--853.
#' \url{https://proceedings.mlr.press/v80/chen18f.html}.
#' @examples
#' # Target: N(0, 1) in each coordinate.
#' set.seed(1)
#' lb <- c(-4, -4)
#' ub <- c(4, 4)
#' opts <- list(grid = fmin_grid(lb, ub, n0 = 21, grow = FALSE),
#'              mc = fmin_mc(lb, ub, n_mc = 200),
#'              nm = fmin_nm(lb, ub, n_res = 3))
#' vapply(opts, function(opt) {
#'   fit <- stein_points(function(X) -X, stein_kernel("imq"),
#'                       n_points = 20, d = 2, optimizer = opt, x_init = c(0, 0))
#'   tail(fit$ksd, 1)
#' }, numeric(1))
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

#' @rdname stein_point_optimizers
#' @param n_mc Positive integer number of Monte Carlo candidates.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance. `NULL`
#'   uses the box midpoint and diagonal variances `((ub - lb) / 4)^2`,
#'   respectively. Supplied values must be a finite length-`d` vector and a
#'   finite symmetric positive-definite \eqn{d\times d} matrix.
#' @param sigsq Finite positive local proposal variance after the delay period.
#' @param delay Nonnegative integer number of optimization iterations before
#'   using local proposals.
#' @export
fmin_mc <- function(lb, ub, n_mc = 20, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20) {
  .validate_optimizer_bounds(lb, ub)
  n_mc <- validate_integer(n_mc, "n_mc")
  p <- .validate_adaptive_controls(lb, ub, mu0, Sigma0, sigsq, delay)
  function(f, X_curr, t = nrow(X_curr) + 1L) {
    X_mc <- .sample_proposal_box(n_mc, lb, ub, p$mu0, p$Sigma0,
                                 p$sigsq, X_curr, p$delay, t = t)
    res <- f(X_mc)
    i <- .objective_min_index(res)
    list(x_min = X_mc[i, ], d_min = res$scores[i, ],
         f_min = res$objective_values[i], n_eval = nrow(X_mc))
  }
}

#' @rdname stein_point_optimizers
#' @param n_res Positive integer number of random restarts.
#' @param control List passed to `stats::optim()`.
#' @export
fmin_nm <- function(lb, ub, n_res = 3, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20, control = list(reltol = 1e-3)) {
  .validate_optimizer_bounds(lb, ub)
  n_res <- validate_integer(n_res, "n_res")
  p <- .validate_adaptive_controls(lb, ub, mu0, Sigma0, sigsq, delay)
  if (!is.list(control)) stop("`control` must be a list", call. = FALSE)
  span <- ub - lb
  # Nelder-Mead searches over theta without bounds. lb plus span times sin^2
  # theta always lands inside the box.
  to_x <- function(th) lb + span * sin(th)^2
  to_theta <- function(x) asin(sqrt(pmin(pmax((x - lb) / span, 0), 1)))

  function(f, X_curr, t = nrow(X_curr) + 1L) {
    X0 <- .sample_proposal_box(n_res, lb, ub, p$mu0, p$Sigma0,
                               p$sigsq, X_curr, p$delay, t = t)
    f_th <- function(th) {
      f(matrix(to_x(th), nrow = 1))$objective_values[1L]
    }

    best_x <- X0[1, ]; best_val <- Inf; n_eval <- 0L
    for (i in seq_len(nrow(X0))) {
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

.validate_adaptive_controls <- function(lb, ub, mu0, Sigma0, sigsq, delay) {
  sigsq <- validate_positive_scalar(sigsq, "sigsq")
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
  list(mu0 = as.numeric(mu0), Sigma0 = Sigma0, sigsq = sigsq,
       delay = validate_integer(delay, "delay", min_value = 0L))
}

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
        "k0(x, x) <=", bound, "; widen the proposal or search box, increase ",
        "the candidate count, or decrease `c2`"
      ),
      call. = FALSE
    )
  }
  values <- res$objective_values
  values[!feasible] <- Inf
  which.min(values)
}

# Before step delay, draw around mu0. After it, draw around randomly chosen
# selected points.
.sample_proposal_box <- function(n, lb, ub, mu0, Sigma0, sigsq, X_curr, delay,
                                 t = nrow(X_curr) + 1L,
                                 max_oversample = 200L) {
  d <- length(lb)
  out <- matrix(NA_real_, n, d)
  filled <- 0L
  drawn <- 0
  local <- t > delay && nrow(X_curr) > 0L
  for (iter in seq_len(max_oversample)) {
    if (filled == n) break
    batch <- max((n - filled) * 2L, 8L)
    drawn <- drawn + batch
    Z <- if (!local) {
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
  if (filled < n) {
    msg <- sprintf(
      "Only %d of %.0f proposals fell inside [lb, ub] after %d batches; %d were needed (step %s, %s)",
      filled, drawn, max_oversample, n, format(t),
      if (local) sprintf("local proposal, sigsq = %g", sigsq) else "initial Gaussian proposal")
    if (filled == 0L) stop(msg, call. = FALSE)
    warning(msg, "; continuing with the ", filled, " found", call. = FALSE)
    out <- out[seq_len(filled), , drop = FALSE]
  }
  out
}

.stein_points_check_kernel <- function(kernel) {
  require_fixed_gaussian_rbf(kernel, "Stein Points")
  if (identical(kernel$type, "imq")) {
    if (!is.numeric(kernel$scale2) || length(kernel$scale2) != 1L ||
        !is.finite(kernel$scale2) || kernel$scale2 <= 0) {
      stop("IMQ Stein Points kernel requires finite c > 0", call. = FALSE)
    }
    if (!is.numeric(kernel$beta) || length(kernel$beta) != 1L ||
        !is.finite(kernel$beta) || kernel$beta >= 0) {
      stop("IMQ Stein Points kernel requires a finite negative beta", call. = FALSE)
    }
  }
  invisible(kernel)
}


# The first point is chosen by log_p alone, so only the chosen point needs a
# score evaluation.
.seed_x1 <- function(kernel, score_function, optimizer, log_p, x_init, d) {
  x <- x_init
  n_search <- 0L
  if (is.null(x)) {
    res <- .validate_optimizer_result(
      optimizer(.obj_neg_log_p(log_p), matrix(0, 0, d), t = 1L),
      d,
      initialization = TRUE
    )
    x <- res$x_min
    n_search <- res$n_eval
  }
  x_mat <- matrix(x, 1L, d)
  grad <- .as_score_matrix(score_function(x_mat), x_mat)[1L, ]
  list(x = x, grad = grad, n_eval = n_search + 1L,
       k0_self = k0_diag(kernel, x_mat, matrix(grad, 1L, d)))
}

.validate_optimizer_result <- function(res, d, initialization = FALSE) {
  bad_x <- !is.list(res) || !is.numeric(res$x_min) ||
    length(res$x_min) != d || any(!is.finite(res$x_min))
  if (initialization) {
    if (bad_x) {
      stop("optimizer must return finite length-d x_min during initialization",
           call. = FALSE)
    }
  } else if (bad_x || !is.numeric(res$d_min) || length(res$d_min) != d ||
             any(!is.finite(res$d_min))) {
    stop("optimizer must return finite x_min and d_min", call. = FALSE)
  }
  res$n_eval <- validate_integer(res$n_eval, "optimizer n_eval", min_value = 0L)
  res
}


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

# The greedy objective is k0 of x with itself plus twice the sum of k0 between x
# and each selected point.
.obj_greedy <- function(kernel, score_function, X_sel, D_sel) {
  function(X_new, scores = NULL) {
    if (is.null(scores)) {
      scores <- .as_score_matrix(score_function(X_new), X_new)
    }
    diag_k0 <- k0_diag(kernel, X_new, scores)
    cross <- if (nrow(X_sel) == 0L) 0 else
      2 * colSums(stein_kernel_matrix(kernel, X_sel, D_sel, X_new, scores))
    list(
      objective_values = cross + diag_k0,
      scores = scores,
      k0_diag = diag_k0
    )
  }
}

# The herding objective is the sum of k0 between x and each selected point.
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
