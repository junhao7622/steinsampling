# Stein Points: sequential point selection by KSD minimisation.

#' Build a Stein Points sequence by greedy KSD minimization
#'
#' Implements the Stein Points construction of Chen et al. The function builds a
#' point set whose empirical distribution is meant to approximate the target by
#' making the kernel Stein discrepancy small. The point-selection rule itself
#' introduces no randomness, but reproducibility depends on the numerical
#' search: grid search is deterministic, whereas Monte Carlo and randomly
#' initialized Nelder-Mead searches are stochastic unless their seed is fixed.
#'
#' @details
#' In the paper, the KSD of a point set is written through the Stein kernel
#' values `k0(x_i, x_l)`. With the package notation, for selected points
#' \eqn{x_1,\ldots,x_m},
#' \deqn{KSD_m^2 = \frac{1}{m^2}
#'        \sum_{a=1}^m \sum_{b=1}^m k0(x_a, x_b).}
#' The kernel `k0` is assembled by [stein_kernel_matrix()] from the target
#' score \eqn{s_p(x)=\nabla_x\log p(x)} and the selected base kernel. The first
#' point is either supplied by `x_init` or chosen by maximizing `log_p`, which
#' follows the paper's default choice of starting near a mode of the target.
#' After points \eqn{x_1,\ldots,x_{j-1}} have been selected, the greedy rule chooses
#' the next point by minimizing
#' \deqn{\frac{1}{2} k0(x, x) + \sum_{i=1}^{j-1} k0(x_i, x).}
#' The implementation scores candidates with the doubled but equivalent form
#' \deqn{k0(x, x) + 2 \sum_{i=1}^{j-1} k0(x_i, x),}
#' which has the same minimizer and is exactly the amount added to the running
#' double sum \eqn{\sum_{a,b\le j} k0(x_a,x_b)}. The herding rule drops the
#' self term in the selection objective and minimizes
#' \deqn{\sum_{i=1}^{j-1} k0(x_i, x).}
#' After a herding point is chosen, the package still adds both cross
#' interactions and the new self-interaction to the KSD diagnostic, so `ksd`
#' remains comparable between greedy and herding runs.
#'
#' The hard part of the paper algorithm is the global search over `x`. This
#' function decouples that search from the Stein objective. The `optimizer`
#' receives an objective function, the current point matrix `X_curr`, and the
#' iteration index `t`; it must
#' return a list containing `x_min` (the selected row), `d_min` (the score at
#' that row), `f_min` (the objective value), and `n_eval` (the number of target
#' or candidate evaluations it used). The helpers [fmin_grid()], [fmin_mc()],
#' and [fmin_nm()] implement the grid, Monte Carlo, and Nelder-Mead searches
#' used in the Stein Points experiments.
#'
#' The output records both the selected points and the running KSD diagnostic.
#' `X` is an `n_points x d` matrix, `D` is the matching score matrix with row
#' `D[j, ] = s_p(X[j, ])`, and
#' \deqn{ksd[j] =
#' \left\{\frac{1}{j^2}
#' \sum_{a=1}^j \sum_{b=1}^j k0(x_a,x_b)\right\}^{1/2}.}
#' If `truncation` is not `"none"`, candidates whose self-kernel value
#' `k0(x,x)` is outside the paper's admissible region are rejected before the
#' best one is chosen, matching the truncated variants used in the convergence
#' results.
#'
#' The design separates the Stein objective from the numerical search. This is
#' why `optimizer` is an argument and why [fmin_grid()], [fmin_mc()], and
#' [fmin_nm()] return optimizer functions rather than points directly. The
#' Stein objective knows the current selected points and the score function; the
#' optimizer only decides how to search for a good next candidate.
#'
#' The kernel input sits at the same layer as [stein_kernel_matrix()]. The three
#' Stein Points kernel families implemented here are:
#' \deqn{k_{\mathrm{IMQ}}(x,y) = (c^2 + r(x,y))^\beta,\quad
#'       r(x,y)=(x-y)^T M (x-y),}
#' from [stein_kernel()], usually with \eqn{\beta\in(-1,0)};
#' \deqn{k_{\log}(x,y) =
#'       (\alpha+\log(1+||x-y||^2))^\beta,}
#' from [stein_kernel_inverse_log()], with the paper's inverse-log choice
#' recovered at \eqn{\beta=-1}; and
#' \deqn{k_{\mathrm{score}}(x,y) =
#'       (\alpha+||s_p(x)-s_p(y)||^2)^\beta,}
#' from [stein_kernel_imq_score()]. Moving sideways among these choices keeps
#' the same greedy or herding point-selection loop, but changes what distance
#' the base kernel uses before the common Stein-kernel assembly: position
#' distance for IMQ, logarithmic position distance for inverse-log, and
#' score-vector distance for score-distance IMQ. The score-distance kernel also
#' requires a Hessian evaluator because derivatives of
#' \eqn{s_p(x)=\nabla_x\log p(x)} enter `k0`.
#'
#' @param score_function Function returning target scores for candidate rows.
#' @param kernel Built-in Stein kernel, compatible custom kernel, function
#'   `kernel(X, S_X, Y, S_Y)`, or list with `k0_matrix`.
#' @param n_points Number of points to select.
#' @param d Dimension of each point.
#' @param optimizer Function taking `(objective, X_curr, t)` and returning
#'   `x_min`, `d_min`, `f_min`, and `n_eval`.
#' @param method Point-selection rule: `"greedy"` or `"herding"`.
#' @param log_p Optional log density used to choose the first point.
#' @param x_init Optional first point. If supplied, `log_p` is not used.
#' @param c2 Positive truncation constant when `truncation != "none"`.
#' @param truncation Optional filter that removes candidates whose self-kernel
#'   value is too large. Use `"none"` for the basic Stein Points algorithm.
#' @param seed Optional local RNG seed.
#'
#' @return
#' A list with:
#' * `X`: `n_points x d` matrix of selected point locations.
#' * `D`: `n_points x d` matrix of target scores evaluated at `X`.
#' * `ksd`: numeric vector; `ksd[j]` is the square root of the empirical KSD
#'   double average for the first `j` selected points, using the same `k0`
#'   that drove selection.
#' * `n_eval`: number of objective or score evaluations charged at each
#'   selection step. This is kept separate from `ksd` because different
#'   optimizers can reach similar KSD values with very different budgets.
#' * `cum_n_eval`: cumulative sum of `n_eval`, used for budget-vs-accuracy
#'   comparisons across [fmin_grid()], [fmin_mc()], [fmin_nm()], and SP-MCMC.
#' * `method`: `"greedy"` or `"herding"`.
#' * `kernel`: kernel object or compatible kernel supplied by the user.
#' * `truncation`, `c2`: truncation rule and constant used for candidate
#'   filtering.
#'
#' The list is returned instead of only the point matrix because later analysis
#' often needs the scores `D`, the KSD path, and evaluation counts. For example,
#' [stein_codescent()] can refine `X`, `D` avoids recomputing target scores,
#' `ksd` shows the discrepancy path, and `n_eval`/`cum_n_eval` allow
#' comparisons between grid, Monte Carlo, Nelder-Mead, and SP-MCMC under the
#' same evaluation budget.
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
          else .obj_greedy(kernel, score_function, Xs, Ds), n)
        res <- optimizer(obj_n, Xs, t = n)
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


#' Refine Stein Points by coordinate descent
#'
#' Runs the budget-constrained refinement described with Stein Points: keep the
#' number of points fixed, then repeatedly replace one existing point by a point
#' that reduces the same KSD objective.
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
#' improves the locations without increasing the point count.
#'
#' This function is below [stein_points()] in the workflow. [stein_points()]
#' builds an initial sequence; `stein_codescent()` treats that sequence as a
#' fixed-size design and improves one coordinate row at a time. It uses the same
#' optimizer interface as [stein_points()], so a grid, Monte Carlo, or
#' Nelder-Mead search can be reused for both construction and refinement.
#'
#' @param X0 Initial point matrix.
#' @param score_function Function returning scores for candidate rows.
#' @param kernel Stein kernel object or compatible custom kernel.
#' @param n_iter Number of coordinate-descent updates.
#' @param optimizer Optimizer function used for each coordinate update.
#' @param seed Optional RNG seed.
#'
#' @return
#' A list with:
#' * `X`: refined point matrix with the same dimensions as `X0`.
#' * `D`: target scores at the refined points.
#' * `n_eval`: evaluation counts for each coordinate-descent update.
#' * `cum_n_eval`: cumulative evaluation counts.
#' * `kernel`: kernel used to define the Stein objective.
#'
#' The return shape keeps the same core pieces as [stein_points()] that are
#' needed downstream: refined locations, their target scores, the kernel, and
#' evaluation budgets. It does not store a full KSD trajectory because each
#' coordinate-descent step replaces an existing row rather than appending a new
#' prefix. To compute the final KSD, build
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
        res <- optimizer(obj, X[-j, , drop = FALSE], t = it)
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
#' This constructor is parallel to [stein_kernel()], but it is kept separate
#' because the inverse-log kernel is mainly part of the Stein Points paper's
#' specialized kernel family. Once constructed, the object still follows the
#' same `SteinKernel` interface: [stein_points()], [stein_codescent()],
#' [stein_kernel_matrix()], and the kernel generics can call the corresponding
#' S3 methods without knowing that the kernel came from this constructor.
#'
#' The squared distance in the formula is the ordinary Euclidean squared
#' distance between sample rows. Unlike the built-in IMQ constructor in
#' [stein_kernel()], this helper does not expose a preconditioning matrix; it is
#' intended to reproduce the position-distance inverse-log kernel used in the
#' Stein Points experiments.
#'
#' This is the second of the three Stein Points kernel choices implemented in
#' the package. The first is the position-distance IMQ kernel from
#' [stein_kernel()], with formula `(c^2 + r(x, y))^beta`. The third is
#' [stein_kernel_imq_score()], which replaces position distance by score-vector
#' distance and therefore needs Hessians. Moving from IMQ to inverse-log keeps
#' the same required inputs for [stein_points()] and [sp_mcmc()]: points,
#' scores, and the kernel object. Only the base-kernel formula changes.
#'
#' @param alpha Positive offset parameter.
#' @param beta Negative exponent.
#'
#' @return
#' A list with class `"SteinKernel_inverse_log"` and `"SteinKernel"`. It stores
#' `alpha` and `beta`, which are the only parameters needed by the S3 methods
#' for `eval_kernel()`, `grad_x_kernel()`, `trace_mixed_kernel()`, and
#' `cross_kernel()`. Returning the parameters in a kernel object, instead of
#' returning only a function, lets the rest of the package evaluate the base
#' kernel and all Stein-kernel derivative terms consistently.
#' @examples
#' stein_kernel_inverse_log(alpha = 1, beta = -1)
#' @export
stein_kernel_inverse_log <- function(alpha = 1, beta = -1) {
  .check_inverse_log_alpha_beta(alpha, beta)
  structure(list(alpha = as.numeric(alpha), beta = as.numeric(beta)),
            class = c("SteinKernel_inverse_log", "SteinKernel"))
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
#' Because the base kernel depends on \eqn{s_p(x)}, its derivatives with respect to
#' `x` depend on the Hessian of the target log density. The `hess_log_p`
#' argument supplies that information. It should return an array whose first
#' dimension indexes rows of `X` and whose remaining two dimensions contain the
#' `d x d` Hessian matrix for each row. The package uses those Hessians when
#' computing `k0(x, y)` and the diagonal self-interaction terms used by
#' [stein_points()] and [sp_mcmc()].
#'
#' This is the third Stein Points kernel family implemented here. It is parallel
#' to the position-distance IMQ kernel from [stein_kernel()] and the
#' inverse-log kernel from [stein_kernel_inverse_log()], but it requires more
#' target information. The top-level [stein_points()] call still takes
#' `score_function`, because the greedy objective needs scores at selected and
#' candidate points. This constructor additionally needs `hess_log_p`, because
#' differentiating \eqn{k(s_p(x), s_p(y))} with respect to point locations uses
#' derivatives of the score field.
#'
#' This constructor is separate from [stein_kernel()] because it requires a
#' target-specific Hessian function, not only a kernel bandwidth or length
#' scale. After construction it still behaves like a `SteinKernel` object, so
#' [stein_kernel_matrix()] dispatches to its complete score-aware method. This
#' is also why it cannot be represented by [custom_stein_kernel()], whose leaf
#' functions receive point matrices but not target-score matrices.
#'
#' @param alpha Positive offset parameter.
#' @param beta Exponent in `(-1, 0)`.
#' @param hess_log_p Function returning a Hessian array with shape `n x d x d`.
#'
#' @return
#' A list with class `"SteinKernel_imq_score"` and `"SteinKernel"`. It stores
#' the IMQ parameters `alpha` and `beta` and the Hessian evaluator
#' `hess_log_p`. These fields are kept together because evaluating this kernel
#' requires the score-distance formula and the Hessian-based derivative terms;
#' callers should pass the object to [stein_points()], [sp_mcmc()], or
#' [stein_kernel_matrix()] rather than extracting the fields manually.
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
  if (is.null(mu0)) mu0 <- (lb + ub) / 2
  if (is.null(Sigma0)) Sigma0 <- ((ub - lb) / 4)^2 * diag(length(lb))
  list(mu0 = mu0, Sigma0 = Sigma0)
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
#' The four fields are separated because the main Stein Points loop needs all
#' of them: `x_min` is appended to the design, `d_min` is cached so the score is
#' not recomputed, `f_min` updates the running KSD sum, and `n_eval` records the
#' search cost for budget comparisons.
#' @examples
#' fmin_mc(lb = -1, ub = 1, n_mc = 5)
#' @export
fmin_mc <- function(lb, ub, n_mc = 20, mu0 = NULL, Sigma0 = NULL,
                    sigsq = 1, delay = 20) {
  p <- .adaptive_defaults(lb, ub, mu0, Sigma0)
  function(f, X_curr, t = nrow(X_curr) + 1L) {
    X_mc <- sample_proposal_box(n_mc, lb, ub, p$mu0, p$Sigma0,
                                sigsq, X_curr, delay, t = t)
    res <- f(X_mc)
    i <- which.min(res$f_vec)
    list(x_min = X_mc[i, ], d_min = res$D_new[i, ],
         f_min = res$f_vec[i], n_eval = n_mc)
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
#' This optimizer is parallel to [fmin_grid()] and [fmin_mc()]: all three return
#' the same optimizer interface, but they search differently. Nelder-Mead is
#' useful when the objective is smooth enough for local improvement to beat a
#' finite candidate set.
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
#' returns `x_min`, `d_min`, `f_min`, and `n_eval`. The common return contract
#' lets [stein_points()] switch between optimizers without changing its main
#' loop. `x_min` is the selected point, `d_min` is its score, `f_min` is the
#' final Stein objective value used to update the running KSD diagnostic, and
#' `n_eval` is the number of objective evaluations charged to this search.
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
    X0 <- sample_proposal_box(n_res, lb, ub, p$mu0, p$Sigma0,
                              sigsq, X_curr, delay, t = t)
    f_th <- function(th) f(matrix(to_x(th), nrow = 1))$f_vec[1]

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
    list(x_min = as.numeric(best_x), d_min = final$D_new[1, ],
         f_min = best_val, n_eval = n_eval + 1L)
  }
}

#' Create the grid search used by Stein Points
#'
#' Creates the deterministic grid-search function used in the Stein Points
#' appendix. It evaluates every point on a Cartesian grid and returns the
#' smallest objective value.
#'
#' @details
#' With `grow = TRUE`, the grid size follows the paper's idea of increasing the
#' grid resolution as the point set grows: the per-dimension size is
#' `n0 + round(sqrt(j))`, where `j` is the next point index. The method is simple
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
#' of grid rows scored. Returning the same four fields as [fmin_mc()] and
#' [fmin_nm()] makes the deterministic grid search interchangeable with the
#' stochastic and local optimizers.
#' @examples
#' fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)
#' @export
fmin_grid <- function(lb, ub, n0 = 100, grow = TRUE) {
  d <- length(lb); if (length(n0) == 1L) n0 <- rep(n0, d)
  function(f, X_curr, t = nrow(X_curr) + 1L) {
    n_g <- if (grow) n0 + as.integer(round(sqrt(nrow(X_curr) + 1))) else n0
    grid <- as.matrix(do.call(expand.grid,
              lapply(seq_len(d), function(j)
                seq(lb[j], ub[j], length.out = n_g[j]))))
    dimnames(grid) <- NULL
    res <- f(grid)
    i <- which.min(res$f_vec)
    list(x_min = grid[i, ], d_min = res$D_new[i, ],
         f_min = res$f_vec[i], n_eval = nrow(grid))
  }
}

#' Draw boxed candidates for Stein Points searches
#'
#' Draws the bounded Gaussian proposals used by the Monte Carlo and
#' Nelder-Mead Stein Points searches. Rows outside the box are rejected, so all
#' returned candidates can be passed directly to a Stein Points objective.
#'
#' @details
#' While `t <= delay`, proposals are drawn
#' from a broad Gaussian with mean `mu0` and covariance `Sigma0`. After that,
#' the proposal becomes the adaptive mixture used in the paper's Monte Carlo
#' search: choose an existing point and add independent Gaussian noise with
#' variance `sigsq`. Rejection sampling keeps only rows inside the box.
#'
#' @param n Number of samples to draw.
#' @param lb,ub Lower and upper bounds.
#' @param mu0,Sigma0 Initial Gaussian proposal mean and covariance.
#' @param sigsq Local proposal variance used after the delay period.
#' @param X_curr Currently selected points.
#' @param delay Number of optimization iterations before using local proposals.
#' @param t Current optimization iteration. For sequential Stein Points this is
#'   the point index; for coordinate descent it is the update index.
#' @param max_oversample Maximum number of proposal batches.
#'
#' @return Numeric matrix of proposed candidate points.
#' @examples
#' sample_proposal_box(
#'   n = 3, lb = -1, ub = 1, mu0 = 0, Sigma0 = matrix(0.1),
#'   sigsq = 0.1, X_curr = matrix(0, ncol = 1), delay = 2, t = 1
#' )
#' @noRd
sample_proposal_box <- function(n, lb, ub, mu0, Sigma0, sigsq, X_curr, delay,
                                t = nrow(X_curr) + 1L,
                                max_oversample = 200L) {
  d <- length(lb); out <- matrix(NA_real_, n, d); filled <- 0L
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
  res <- optimizer(.obj_neg_log_p(score_function, log_p), matrix(0, 0, d),
                   t = 1L)
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
      tr_u <- -2 * sum(diag(HiHj))
      ux_uy <- -4 * as.numeric(crossprod(delta, HiHj %*% delta))
      cs <- -2 * sum(sx * (Hj %*% delta)) + 2 * sum(sy * (Hi %*% delta))
      out[i, j] <- c2[i, j] * ux_uy + c1[i, j] * (tr_u + cs) +
                   c0[i, j] * sum(sx * sy)
    }
  }
  out
}

#' @export
stein_kernel_matrix.SteinKernel_imq_score <- function(
    kernel, X, grads, Y = NULL, grads_Y = NULL, ...) {
  inp <- validate_cross_inputs(X, grads, Y, grads_Y)
  .k0_matrix_imq_score(kernel, inp$X, inp$Y, inp$grads, inp$grads_Y)
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
    K0 <- .k0_matrix(kernel, X_sel, X_new, D_sel, D_new)
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
  lower <- 2 * log(j) / c2
  switch(mode,
    upper = upper, lower = lower,
    linear = lower + (upper - lower) * ((j - 1) / max(n_total - 1, 1L)))
}

# Set infeasible candidates to a huge objective value.
.truncate <- function(obj, kernel, r2) {
  if (!is.finite(r2) || r2 <= 0) return(obj)
  function(X_new, D_new = NULL) {
    res <- obj(X_new, D_new)
    diag_k0 <- .k0_diag(kernel, X_new, res$D_new)
    bad <- !is.finite(diag_k0) | diag_k0 > r2
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
