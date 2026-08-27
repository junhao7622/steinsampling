# Stein Point MCMC, its transition kernels, and extension points.

# ---- Public: complete algorithm ------------------------------------------

#' Select Stein Points from short Markov chains
#'
#' Builds a point set sequentially. At each step, a short Markov chain produces
#' the candidates, and the candidate minimizing the greedy Stein objective is
#' selected.
#'
#' @details
#' Let \eqn{k_{0,p}} be the Stein kernel formed from `kernel` and
#' \eqn{s_p(x)=\nabla_x\log p(x)}. Given selected points
#' \eqn{x_1,\ldots,x_{j-1}}, define
#' \deqn{Q_j(x)=\frac{1}{2}k_{0,p}(x,x)
#'       +\sum_{i=1}^{j-1}k_{0,p}(x_i,x).}
#' At step \eqn{j}, the function generates a Markov-chain path
#' \eqn{Y_{j,1},\ldots,Y_{j,\nu_j}}, removes duplicate states, and selects
#' \deqn{x_j\in\mathop{\mathrm{argmin}}_{x\in\mathcal C_j}Q_j(x),}
#' where \eqn{\mathcal C_j} is the remaining candidate set. Thus
#' [stein_points()] and `sp_mcmc()` use the same greedy objective; they differ
#' in how candidates are obtained.
#'
#' The first path state is chosen from the previously selected points.
#' `criterion = "last"` uses the latest point, `"rand"` samples a selected
#' point, and `"infl"` uses the point whose removal gives the largest
#' remaining-set discrepancy. A path of length \eqn{\nu_j} makes
#' \eqn{\nu_j-1} transitions. Therefore `m_seq = 1` supplies only the starting
#' state and performs no search.
#'
#' Built-in paths use random-walk Metropolis (`mcmc = "rwm"`) or MALA
#' (`mcmc = "mala"`). `transition_fn`, `proposal_fn`, and a functional
#' `criterion` provide optional extension points. A custom transition must
#' return `X`, containing exactly the requested path rows; `D`, containing
#' matching scores or `NULL`; and `counts`, containing nonnegative `log_p`,
#' `score`, and `total` entries with `total = log_p + score`.
#'
#' The returned discrepancy after step \eqn{j} is
#' \deqn{\mathrm{KSD}_j
#' =\left\{\frac{1}{j^2}\sum_{a=1}^j\sum_{b=1}^j
#' k_{0,p}(x_a,x_b)\right\}^{1/2}.}
#'
#' @param score_function Function taking an `n x d` matrix and returning the
#'   corresponding `n x d` matrix of target scores.
#' @param log_p Function taking an `n x d` matrix and returning `n` log-density
#'   values.
#' @param kernel Kernel used to construct \eqn{k_{0,p}}. It may be a built-in
#'   kernel object, a compatible kernel function, or a list with `k0_matrix`.
#'   A Gaussian RBF kernel must use a fixed positive bandwidth.
#' @param n_points Positive total number of points, including `x_init`.
#' @param d Positive state dimension.
#' @param mcmc MCMC transition: `"rwm"` for Gaussian random-walk Metropolis or
#'   `"mala"` for MALA.
#' @param criterion Rule for choosing where the next MCMC chain starts:
#'   `"last"`, `"rand"`, `"infl"`, or a custom function.
#' @param m_seq Number of path states before duplicate removal. Supply one value
#'   for all steps, one value per selected point, or one value for each step
#'   after the first point.
#' @param h Step-size multiplier used by the built-in MCMC transitions. Both
#'   transitions give the proposal noise covariance `h * Sigma`.
#' @param Sigma Symmetric positive-definite proposal preconditioner shared by
#'   the drift and the noise of [mala()], and the proposal covariance scale of
#'   [rwm()]. If `NULL`, the identity matrix is used. This is the target-scale
#'   matrix itself, not its inverse; the `precon` matrix of an IMQ
#'   [stein_kernel()] is the inverse, so preconditioning both the transition
#'   and `k0` with the same covariance `S` means `Sigma = S` and
#'   `precon = solve(S)`.
#' @param x_init Finite numeric vector of length `d` giving the first point.
#' @param seed Optional RNG seed.
#' @param transition_fn Optional custom MCMC transition function. It receives
#'   the log density, score function, current state, step size, proposal matrix,
#'   and a requested row count \eqn{\nu_j}. It must return a list with exactly
#'   that many rows in `X`, the supplied current state in row 1, `D` set to a
#'   matching score matrix or explicitly to `NULL`, and `counts` containing
#'   `log_p`, `score`, and their sum `total`. The remaining \eqn{\nu_j-1} rows
#'   are states after successive Markov transitions.
#' @param proposal_fn Optional function that changes `h` or `Sigma` at each
#'   step. It should return a list containing replacement `h` and/or `Sigma`.
#' @param criterion_args Optional list of extra arguments passed to a custom
#'   `criterion` function.
#'
#' @return
#' An object of class `"sp_mcmc"` with:
#' * `X` and `D`: `n_points x d` matrices of selected points and their scores.
#' * `ksd`: running discrepancy defined above.
#' * `n_eval`, `cum_n_eval`, and `counts`: per-step, cumulative, and component
#'   evaluation counts.
#' * `selected_index`: row used to initialize each short path.
#' * `chain_d2_max`, `chain_d2_selected`, `chain_d2_last`, and `accept_rate`:
#'   short-path movement and acceptance summaries.
#' * `kernel`, `mcmc`, `criterion`, `m_seq`, `h`, and `Sigma`: settings used by
#'   the run.
#' * `call`: matched function call.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' sp_mcmc(score, log_p, kernel, n_points = 2, d = 1, m_seq = 2, h = 0.1,
#'         x_init = 0)
#' @export
sp_mcmc <- function(score_function, log_p, kernel, n_points, d,
                    mcmc = c("rwm", "mala"),
                    criterion = c("last", "rand", "infl"),
                    m_seq, h, Sigma = NULL,
                    x_init, seed = NULL,
                    transition_fn = NULL, proposal_fn = NULL,
                    criterion_args = list()) {
  cl <- match.call()

  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }
  if (!is.function(log_p)) {
    stop("`log_p` must be a function", call. = FALSE)
  }
  mcmc <- match.arg(mcmc)
  require_fixed_gaussian_rbf(kernel, "SP-MCMC")
  criterion_obj <- sp_mcmc_criterion(criterion, criterion_args)

  n_points <- validate_integer(n_points, "n_points")
  d <- validate_integer(d, "d")

  if (!is.numeric(x_init) || length(x_init) != d || any(!is.finite(x_init))) {
    stop("x_init must be a finite length-d numeric vector", call. = FALSE)
  }
  x_init <- as.numeric(x_init)

  if (!is.numeric(m_seq) || length(m_seq) < 1L) {
    stop(
      "m_seq must be a positive integer or a numeric vector of positive integers",
      call. = FALSE
    )
  }
  if (length(m_seq) == 1L) {
    m_seq <- validate_integer(m_seq, "m_seq")
  } else if (length(m_seq) == n_points - 1L) {
    m_seq <- vapply(m_seq, validate_integer, integer(1L),
                    arg_name = "m_seq entry")
    m_seq <- c(NA_integer_, m_seq)
  } else if (length(m_seq) == n_points) {
    m_seq <- vapply(m_seq, validate_integer, integer(1L),
                    arg_name = "m_seq entry")
  } else {
    stop("m_seq must be scalar, length n_points, or length n_points - 1", call. = FALSE)
  }

  run <- function() {
    .sp_mcmc_run(
      score_function = score_function,
      log_p = log_p,
      kernel = kernel,
      n_points = n_points,
      d = d,
      mcmc = mcmc,
      criterion_obj = criterion_obj,
      m_seq = m_seq,
      h = h,
      Sigma = Sigma,
      x_init = x_init,
      transition_fn = transition_fn,
      proposal_fn = proposal_fn,
      call = cl
    )
  }

  with_local_seed(seed, run())
}


# ---- Public: SP-MCMC composition contract --------------------------------

#' Store the current SP-MCMC state for a start rule
#'
#' Creates the state object passed to custom SP-MCMC start-point criteria. The
#' criterion uses this object to decide which selected point should initialize
#' the next short MCMC chain.
#'
#' @details
#' The state contains the current point set, scores, and Stein kernel matrix, as
#' well as diagnostic vectors accumulated so far. The matrix field satisfies
#' `K0[a, b] = k0(X[a, ], X[b, ])`, using the same target scores stored in `D`.
#' In the paper, the start rule is one of LAST, RAND, or INFL. This object makes
#' those rules explicit: LAST uses `state$X`, RAND samples from its rows, and
#' INFL uses `state$K0`. A custom rule should inspect the fields it needs and
#' return one row index from `state$X`.
#'
#' [sp_mcmc()] builds this object internally before every short chain; building
#' one directly lets a custom rule be tested outside the full algorithm.
#'
#' @param j Current greedy index.
#' @param X Current selected point matrix.
#' @param D Current score matrix.
#' @param K0 Current Stein kernel matrix.
#' @param ksd Optional running KSD values.
#' @param n_eval Optional evaluation counts.
#' @param counts Optional detailed evaluation-count matrix.
#' @param selected_index Optional selected start indices.
#' @param kernel Kernel used by SP-MCMC.
#' @param mcmc MCMC transition name.
#' @param criterion Start-rule label stored in the state.
#'
#' @return
#' A list of class `"sp_mcmc_state"` with the fields supplied as arguments:
#' current points `X`, scores `D`, Stein-kernel matrix `K0`, current step `j`,
#' running diagnostics, and method labels. It is a snapshot handed to a
#' start-point criterion, not a result object.
#' @examples
#' sp_mcmc_state(j = 2, X = matrix(0, ncol = 1), D = matrix(0, ncol = 1),
#'               K0 = matrix(1))
#' @export
sp_mcmc_state <- function(j, X, D, K0, ksd = NULL, n_eval = NULL,
                          counts = NULL, selected_index = NULL,
                          kernel = NULL, mcmc = NULL, criterion = NULL) {
  structure(
    list(
      j = j, X = X, D = D, K0 = K0, ksd = ksd, n_eval = n_eval,
      counts = counts, selected_index = selected_index,
      kernel = kernel, mcmc = mcmc, criterion = criterion
    ),
    class = "sp_mcmc_state"
  )
}

#' Create an SP-MCMC start-point rule
#'
#' `criterion` may be one of `"last"`, `"rand"`, `"infl"`, or a function that
#' accepts an `sp_mcmc_state` object and returns a one-based index into
#' `state$X`.
#'
#' @details
#' The SP-MCMC paper studies three rules for choosing where the next candidate
#' chain starts. `"last"` starts at the most recently selected point. `"rand"`
#' chooses a selected point uniformly at random. `"infl"` starts at a most
#' influential point, defined in the paper as one whose removal raises the KSD
#' of the remaining points the most. Removing point `a` lowers the double sum
#' \eqn{\sum_{b,c} k0(x_b,x_c)} by
#' \deqn{I_a = \sum_b K0_{ab} + \sum_b K0_{ba} - K0_{aa},}
#' and the remaining points share one divisor, so maximizing their KSD is the
#' same as taking the smallest `I_a`, which is what the rule computes.
#' `I_a` can be negative, and the most negative one marks the point
#' contributing most to the current KSD. A custom function can implement
#' another rule while still using the same SP-MCMC main loop.
#'
#' The result pairs a `select(state)` function with a `label` that [sp_mcmc()]
#' copies into its output, so a run records whether LAST, RAND, INFL, or a
#' custom rule produced it.
#'
#' @param criterion Start-rule name, custom function, or criterion object.
#' @param criterion_args Extra arguments passed to a custom criterion function.
#'
#' @return
#' A list with `label`, stored in the final [sp_mcmc()] output, and `select`, a
#' function taking an [sp_mcmc_state()] object and returning one row index into
#' `state$X`.
#' @examples
#' sp_mcmc_criterion("last")
#' @export
sp_mcmc_criterion <- function(criterion = c("last", "rand", "infl"),
                              criterion_args = list()) {
  if (is.list(criterion) && is.function(criterion$select)) {
    if (is.null(criterion$label)) criterion$label <- "custom"
    return(criterion)
  }
  if (is.function(criterion)) {
    fn <- criterion
    return(list(
      label = "custom",
      select = function(state) do.call(fn, c(list(state), criterion_args))
    ))
  }
  name <- match.arg(criterion, c("last", "rand", "infl"))
  fn <- switch(
    name,
    last = function(state) nrow(state$X),
    rand = function(state) sample.int(nrow(state$X), 1L),
    infl = function(state) .crit_infl(state$K0)
  )
  list(label = name, select = fn)
}

#' Choose the next SP-MCMC chain start
#'
#' Applies a criterion object to the current state and checks that the returned
#' value is a valid row index.
#'
#' @details
#' This helper is the validation step after [sp_mcmc_criterion()]. It keeps
#' custom start rules honest by requiring a single one-based row index into the
#' current point set. The selected row becomes the first state of the short MCMC
#' chain for the next SP-MCMC update.
#'
#' Pairing it with [sp_mcmc_state()] and [sp_mcmc_criterion()] checks that a
#' custom rule returns a legal index before the full algorithm is run. Inside
#' [sp_mcmc()] the returned index is also recorded in `selected_index`.
#'
#' @param criterion_obj Criterion object from `sp_mcmc_criterion()`.
#' @param state Current SP-MCMC state.
#'
#' @return
#' One integer: the one-based row index in `state$X` that should initialize the
#' next short candidate chain.
#' @examples
#' state <- sp_mcmc_state(j = 2, X = matrix(0, ncol = 1),
#'                        D = matrix(0, ncol = 1), K0 = matrix(1))
#' sp_mcmc_select_start(sp_mcmc_criterion("last"), state)
#' @export
sp_mcmc_select_start <- function(criterion_obj, state) {
  idx <- criterion_obj$select(state)
  if (is.list(idx) && !is.null(idx$index)) idx <- idx$index
  if (!is.numeric(idx) || length(idx) != 1L || !is.finite(idx) ||
      abs(idx - round(idx)) > sqrt(.Machine$double.eps)) {
    stop(
      "criterion must return one valid one-based integer index into the current point set",
      call. = FALSE
    )
  }
  idx <- as.integer(round(idx))
  if (idx < 1L || idx > nrow(state$X)) {
    stop(
      "criterion must return one valid one-based index into the current point set",
      call. = FALSE
    )
  }
  idx
}

#' Score SP-MCMC candidate points
#'
#' Scores candidate rows from a short MCMC path using the same greedy Stein
#' Points objective used inside [sp_mcmc()].
#'
#' @details
#' For each candidate `x`, the objective in the paper is
#' \deqn{\frac{1}{2} k0(x, x) + \sum_i k0(x_i, x)}
#' in the notation of the Stein Points paper. This helper returns the doubled
#' equivalent
#' \deqn{k0(x, x) + 2 \sum_i k0(x_i, x),}
#' where the sum runs over rows in `X_curr`. `objective_values` contains one
#' objective value per candidate row, in the same order as `cand_X`. If the MCMC
#' transition already returned candidate scores, pass them as `cand_D`;
#' otherwise this helper evaluates `score_function` on the candidate rows and
#' records how many score evaluations were needed.
#'
#' @param kernel Stein kernel object or compatible custom kernel.
#' @param score_function Function returning scores for candidate rows.
#' @param X_curr Current selected point matrix.
#' @param D_curr Current score matrix.
#' @param cand_X Candidate point matrix.
#' @param cand_D Optional candidate score matrix.
#'
#' @return
#' A list with:
#' * `objective_values`: doubled greedy objective value for each candidate row.
#'   The row with the smallest value is the point [sp_mcmc()] appends.
#' * `scores`: score matrix for the candidate rows, either reused from `cand_D`
#'   or computed from `score_function`; it is returned so the selected row can
#'   be appended without another score call.
#' * `score_evaluations`: number of candidate score evaluations performed inside
#'   this helper.
#' @examples
#' score <- function(X) -as.matrix(X)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' X_curr <- matrix(0, ncol = 1)
#' D_curr <- score(X_curr)
#' cand_X <- matrix(c(-0.5, 0.5), ncol = 1)
#' sp_mcmc_eval_candidates(kernel, score, X_curr, D_curr, cand_X)
#' @export
sp_mcmc_eval_candidates <- function(kernel, score_function, X_curr, D_curr,
                                    cand_X, cand_D = NULL) {
  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }
  # Reuse the Stein Points greedy objective; only the candidate source changes.
  obj <- .obj_greedy(kernel, score_function, X_curr, D_curr)
  score_evaluations <- 0L
  if (is.null(cand_D)) {
    out <- obj(cand_X)
    score_evaluations <- nrow(cand_X)
  } else {
    out <- obj(cand_X, cand_D)
  }
  out$score_evaluations <- score_evaluations
  out
}

.crit_infl <- function(K0_cache) {
  which.min(rowSums(K0_cache) + colSums(K0_cache) - diag(K0_cache))
}


# ---- Public: MCMC transition kernels -------------------------------------

#' Run a Metropolis-adjusted Langevin chain
#'
#' Runs the MALA transition used as one of the short candidate-chain kernels in
#' SP-MCMC. MALA uses the target score to drift proposals toward high-density
#' regions and then applies a Metropolis correction.
#'
#' @details
#' With the row-vector convention used by the code, the proposal from the
#' current state `x` is
#' \deqn{y = x + (h / 2) s_p(x) Sigma + \sqrt{h} z,
#'       \quad z \sim N(0, Sigma),}
#' so `Sigma` preconditions the drift and the noise alike and the proposal
#' noise has covariance `h * Sigma`. The acceptance step compares the target
#' log density and the two proposal densities, so accepted states leave the
#' distribution described by `log_p` invariant. The returned score matrix `D`
#' is included because SP-MCMC can reuse these scores when scoring the
#' candidate path.
#'
#' When `Sigma` is the identity matrix this reduces to the basic MALA proposal
#' \eqn{y = x + (h/2)\nabla\log p(x) + \sqrt{h}z}. Appendix A.5 of the SP-MCMC
#' paper prints \eqn{Sigma^{-1}} in the drift while keeping \eqn{N(0, Sigma)}
#' noise; the authors' released `mala.m` uses the same matrix in both places,
#' which is the standard preconditioned Langevin discretization, and this
#' implementation follows the released code. Taking \eqn{Sigma} to be an
#' approximation of the target covariance therefore scales the drift correctly.
#'
#' Two conventions are worth stating because they differ from the released
#' MATLAB code and from the Stein kernel. First, `mala.m` parameterizes the
#' chain by the square root of this step size, so its `h` corresponds to
#' `sqrt(h)` here. Second, `Sigma` is a proposal preconditioner and is
#' unrelated to the `precon` matrix of an IMQ Stein kernel: SP-MCMC
#' Equation 6 uses \eqn{Lambda^{-1}} inside `k0`, so a run that preconditions
#' both with the same target covariance passes `Sigma = S` here and
#' `precon = solve(S)` to [stein_kernel()].
#'
#' MALA is a transition kernel, not a full sampler interface. It is exported so
#' SP-MCMC users can inspect or replace the candidate-chain step. [sp_mcmc()]
#' calls it when `mcmc = "mala"` and then passes its chain output to
#' [sp_mcmc_eval_candidates()].
#'
#' @param log_p Function returning log density values.
#' @param score_function Function returning score values.
#' @param x0 Initial state vector.
#' @param h Positive step size. The proposal noise has covariance `h * Sigma`,
#'   so `h` is the square of the step size used by the authors' `mala.m`.
#' @param Sigma Symmetric positive-definite proposal preconditioner, used for
#'   both the drift and the noise. If `NULL`, the identity matrix is used.
#' @param m_iter Number of returned chain rows, including the initial state in
#'   row 1. The function therefore makes `m_iter - 1` Markov transitions.
#'
#' @return
#' A list with `X` (chain states, one row per iteration), `D` (scores at those
#' states), `log_p` (log density values), `accept` (0/1 acceptance indicators),
#' and evaluation-count fields. `D` is returned because SP-MCMC can reuse those
#' scores instead of calling the score function again for every candidate.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' mala(log_p, score, x0 = 0, h = 0.1, m_iter = 3)
#' @export
mala <- function(log_p, score_function, x0, h, Sigma = NULL, m_iter) {
  if (!is.function(score_function)) stop("score_function must be a function", call. = FALSE)
  setup <- .mcmc_setup(log_p, x0, h, Sigma, m_iter)
  x0 <- setup$x0
  d <- setup$d
  h <- setup$h
  Sigma <- setup$Sigma
  m_iter <- setup$m_iter
  U <- sqrt(h) * setup$U

  X <- matrix(0, m_iter, d)
  D <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L
  score_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  x0_mat <- matrix(x0, 1L, d)
  D[1L, ] <- .as_score_matrix(score_function(x0_mat), x0_mat)[1L, ]
  score_eval <- score_eval + 1L
  lp0 <- log_p(matrix(x0, 1L, d))
  if (!is.numeric(lp0) || length(lp0) != 1L || !is.finite(lp0)) {
    stop("log_p must return one finite value at x0", call. = FALSE)
  }
  lp[1L] <- as.numeric(lp0)
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      x_c <- X[i - 1L, ]
      d_c <- D[i - 1L, ]
      lp_c <- lp[i - 1L]

      m_x <- x_c + (h / 2) * as.numeric(d_c %*% Sigma)
      y <- m_x + as.numeric(stats::rnorm(d) %*% U)

      lp_y <- log_p(matrix(y, 1L, d))
      if (!is.numeric(lp_y) || length(lp_y) != 1L || is.na(lp_y) || lp_y == Inf) {
        stop("log_p must return one numeric value for each proposal", call. = FALSE)
      }
      lp_y <- as.numeric(lp_y)
      log_p_eval <- log_p_eval + 1L
      if (!is.finite(lp_y)) {
        X[i, ] <- x_c; D[i, ] <- d_c; lp[i] <- lp_c
        next
      }

      y_mat <- matrix(y, 1L, d)
      d_y <- as.numeric(.as_score_matrix(score_function(y_mat), y_mat)[1L, ])
      score_eval <- score_eval + 1L
      m_y <- y + (h / 2) * as.numeric(d_y %*% Sigma)

      q_old_given_y <- sum(forwardsolve(t(U), x_c - m_y)^2)
      q_y_given_old <- sum(forwardsolve(t(U), y - m_x)^2)
      log_a <- (lp_y - lp_c) + 0.5 * (q_y_given_old - q_old_given_y)

      if (!is.na(log_a) && (log_a >= 0 || log(stats::runif(1L)) < log_a)) {
        X[i, ] <- y
        D[i, ] <- d_y
        lp[i] <- lp_y
        ac[i] <- 1L
      } else {
        X[i, ] <- x_c
        D[i, ] <- d_c
        lp[i] <- lp_c
      }
    }
  }

  total_eval <- log_p_eval + score_eval
  list(
    X = X, D = D, log_p = lp, accept = ac, n_eval = total_eval,
    counts = list(log_p = log_p_eval, score = score_eval,
                  total = total_eval)
  )
}

#' Run a Gaussian random-walk Metropolis chain
#'
#' Runs the random-walk Metropolis kernel used as the other built-in SP-MCMC
#' candidate-chain transition.
#'
#' @details
#' From the current state `x`, the proposal is
#' \deqn{y = x + z, \quad z \sim N(0, h Sigma).}
#' The move is accepted with probability `min(1, exp(log_p(y) - log_p(x)))`.
#' Unlike MALA, this transition does not use the score to propose moves, so any
#' candidate scores needed by SP-MCMC are computed later by
#' [sp_mcmc_eval_candidates()].
#'
#' The first row of the returned chain is `x0`. Subsequent rows are accepted
#' proposals or repeats of the previous state after rejection. The acceptance
#' indicators therefore describe whether each transition moved, not whether a
#' row is unique. [sp_mcmc()] removes duplicate candidate states later before
#' scoring the path with the Stein Points objective.
#'
#' @param log_p Function returning log density values.
#' @param x0 Initial state vector.
#' @param h Positive step-size multiplier. The proposal covariance is
#'   `h * Sigma`.
#' @param Sigma Symmetric positive-definite proposal covariance scale. If
#'   `NULL`, the identity matrix is used.
#' @param m_iter Number of returned chain rows, including the initial state in
#'   row 1. The function therefore makes `m_iter - 1` Markov transitions.
#'
#' @return
#' A list with the same fields as [mala()]: `X` (chain states), `D` (explicitly
#' `NULL` because RWM does not evaluate scores), `log_p` (log density values),
#' `accept` (0/1 acceptance indicators), `n_eval`, and `counts`. The explicit
#' `D = NULL` tells [sp_mcmc()] to score the distinct candidate states after
#' the path is generated.
#' @examples
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' rwm(log_p, x0 = 0, h = 0.1, m_iter = 3)
#' @export
rwm <- function(log_p, x0, h, Sigma = NULL, m_iter) {
  setup <- .mcmc_setup(log_p, x0, h, Sigma, m_iter)
  x0 <- setup$x0
  d <- setup$d
  m_iter <- setup$m_iter
  U <- chol(setup$h * setup$Sigma)
  X <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  lp0 <- log_p(matrix(x0, 1L, d))
  if (!is.numeric(lp0) || length(lp0) != 1L || !is.finite(lp0)) {
    stop("log_p must return one finite value at x0", call. = FALSE)
  }
  lp[1L] <- as.numeric(lp0)
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      X[i, ] <- X[i - 1L, ]
      y <- X[i - 1L, ] + as.numeric(stats::rnorm(d) %*% U)
      lp_y <- log_p(matrix(y, 1L, d))
      if (!is.numeric(lp_y) || length(lp_y) != 1L || is.na(lp_y) || lp_y == Inf) {
        stop("log_p must return one numeric value for each proposal", call. = FALSE)
      }
      lp_y <- as.numeric(lp_y)
      log_p_eval <- log_p_eval + 1L
      log_a <- lp_y - lp[i - 1L]
      if (!is.na(log_a) && (log_a >= 0 || log(stats::runif(1L)) < log_a)) {
        X[i, ] <- y
        lp[i] <- lp_y
        ac[i] <- 1L
      } else {
        lp[i] <- lp[i - 1L]
      }
    }
  }

  list(
    X = X, D = NULL, log_p = lp, accept = ac, n_eval = log_p_eval,
    counts = list(log_p = log_p_eval, score = 0L, total = log_p_eval)
  )
}

# Validate the proposal inputs once for both public transition kernels.
.mcmc_setup <- function(log_p, x0, h, Sigma, m_iter) {
  if (!is.function(log_p)) stop("log_p must be a function", call. = FALSE)
  if (!is.numeric(x0) || length(x0) < 1L || any(!is.finite(x0))) {
    stop("x0 must be a finite numeric vector", call. = FALSE)
  }
  x0 <- as.numeric(x0)
  d <- length(x0)
  if (!is.numeric(h) || length(h) != 1L || !is.finite(h) || h <= 0) {
    stop("h must be a positive scalar", call. = FALSE)
  }
  if (is.null(Sigma)) Sigma <- diag(d) else Sigma <- as.matrix(Sigma)
  if (!is.numeric(Sigma) || any(!is.finite(Sigma)) ||
      nrow(Sigma) != d || ncol(Sigma) != d) {
    stop("Sigma must be a finite d x d matrix", call. = FALSE)
  }
  sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(Sigma)))
  if (max(abs(Sigma - t(Sigma))) > sym_tol) {
    stop("Sigma must be symmetric positive definite", call. = FALSE)
  }
  Sigma <- (Sigma + t(Sigma)) / 2
  U <- tryCatch(chol(Sigma), error = function(e) {
    stop("Sigma must be symmetric positive definite", call. = FALSE)
  })

  list(
    x0 = x0, d = d, h = as.numeric(h), Sigma = Sigma,
    m_iter = validate_integer(m_iter, "m_iter"),
    U = U
  )
}

# The SP-MCMC loop gives every transition the same six arguments. Keep the
# score placeholder inside this private adapter instead of exposing it in the
# public RWM interface.
.sp_mcmc_rwm_transition <- function(log_p, score_function, x0, h, Sigma,
                                    m_iter) {
  rwm(log_p = log_p, x0 = x0, h = h, Sigma = Sigma, m_iter = m_iter)
}


# ---- Internal: main loop and its helpers ---------------------------------

.sp_mcmc_run <- function(score_function, log_p, kernel, n_points, d,
                         mcmc, criterion_obj, m_seq, h, Sigma, x_init,
                         transition_fn = NULL, proposal_fn = NULL,
                         call = NULL) {

  chain_fn <- if (is.null(transition_fn)) {
    switch(mcmc, rwm = .sp_mcmc_rwm_transition, mala = mala)
  } else {
    transition_fn
  }
  get_m <- if (length(m_seq) == 1L) function(j) m_seq else function(j) m_seq[j]

  X <- D <- matrix(NA_real_, n_points, d)
  n_eval <- integer(n_points)
  ksd <- numeric(n_points)
  counts <- .sp_mcmc_count_matrix(n_points)
  selected_index <- rep(NA_integer_, n_points)
  chain_diagnostics <- matrix(
    NA_real_, nrow = n_points, ncol = 4L,
    dimnames = list(
      NULL,
      c("d2_max", "d2_selected", "d2_last", "accept_rate")
    )
  )
  n_stalled <- 0L

  X[1L, ] <- x_init
  x_init_mat <- matrix(x_init, 1L, d)
  D[1L, ] <- .as_score_matrix(
    score_function(x_init_mat), x_init_mat
  )[1L, ]
  K0_cache <- matrix(
    .k0_diag(kernel, matrix(x_init, 1L, d), matrix(D[1L, ], 1L, d)),
    nrow = 1L
  )
  ss <- as.numeric(K0_cache[1L, 1L])
  if (ss < -sqrt(.Machine$double.eps)) warning("KSD squared value is negative.", call. = FALSE)
  ksd[1L] <- sqrt(max(ss, 0))
  counts[1L, ] <- c(log_p = 0L, score = 1L, candidate_score = 0L,
                    transition_total = 0L, total = 1L)
  n_eval[1L] <- as.integer(counts[1L, "total"])

  if (n_points >= 2L) {
    for (j in 2:n_points) {
      X_curr <- X[seq_len(j - 1L), , drop = FALSE]
      D_curr <- D[seq_len(j - 1L), , drop = FALSE]
      state <- sp_mcmc_state(
        j = j, X = X_curr, D = D_curr, K0 = K0_cache,
        ksd = ksd[seq_len(j - 1L)],
        n_eval = n_eval[seq_len(j - 1L)],
        counts = counts[seq_len(j - 1L), , drop = FALSE],
        selected_index = selected_index[seq_len(j - 1L)],
        kernel = kernel, mcmc = mcmc, criterion = criterion_obj$label
      )
      i_star <- sp_mcmc_select_start(criterion_obj, state)
      selected_index[j] <- i_star
      start <- as.numeric(X_curr[i_star, ])

      m_j <- get_m(j)
      proposal <- .sp_mcmc_resolve_proposal(
        proposal_fn = proposal_fn,
        j = j,
        X_curr = X_curr,
        h = h,
        Sigma = Sigma,
        mcmc = mcmc
      )

      # nu_j counts all path states, not transitions.
      chain <- chain_fn(
        log_p, score_function, start, proposal$h, proposal$Sigma, m_j
      )
      chain <- .sp_mcmc_validate_chain(
        chain, d, expected_n = m_j, start = start
      )
      path <- .sp_mcmc_unique_path(chain)

      cand_X <- path$X
      cand_D_from_chain <- path$D
      # A path whose states are all identical leaves the starting point as the
      # only candidate, so the step re-appends an already selected point.
      if (nrow(cand_X) == 1L) n_stalled <- n_stalled + 1L
      cand_eval <- sp_mcmc_eval_candidates(
        kernel = kernel, score_function = score_function,
        X_curr = X_curr, D_curr = D_curr,
        cand_X = cand_X, cand_D = cand_D_from_chain
      )
      cand_D <- cand_eval$scores
      objective_values <- cand_eval$objective_values

      i_min <- which.min(objective_values)
      x_min <- as.numeric(cand_X[i_min, ])
      d_min <- as.numeric(cand_D[i_min, ])
      f_min <- as.numeric(objective_values[i_min])

      X[j, ] <- x_min
      D[j, ] <- d_min
      K0_cache <- .sp_mcmc_append_k0(
        kernel, X_curr, D_curr, K0_cache, x_min, d_min
      )
      ss <- ss + f_min
      if (ss < -sqrt(.Machine$double.eps)) warning("KSD squared value is negative.", call. = FALSE)
      ksd[j] <- sqrt(max(ss, 0)) / j

      chain_diagnostics[j, ] <- .sp_mcmc_chain_diagnostics(
        chain = chain,
        start = start,
        selected = x_min
      )

      auto_counts <- .sp_mcmc_eval_counts(
        chain = chain,
        candidate_score_eval = cand_eval$score_evaluations
      )
      counts[j, ] <- auto_counts
      n_eval[j] <- as.integer(auto_counts["total"])
    }
  }

  if (n_stalled > 0L) {
    warning(sprintf(
      paste0(
        "%d of %d SP-MCMC steps had a candidate path that never left its ",
        "starting point, so those steps duplicated an already selected point. ",
        "Increase `m_seq` (a value of 1 makes no transitions) or use a ",
        "proposal that moves more often."
      ),
      n_stalled, n_points - 1L
    ), call. = FALSE)
  }

  res <- list(
    X = X, D = D, ksd = ksd, n_eval = n_eval, cum_n_eval = cumsum(n_eval),
    counts = counts,
    method = "sp_mcmc", kernel = kernel, call = call,
    truncation = "none", c2 = NULL,
    mcmc = mcmc, criterion = criterion_obj$label, m_seq = m_seq,
    h = h, Sigma = Sigma, selected_index = selected_index,
    chain_d2_max = chain_diagnostics[, "d2_max"],
    chain_d2_selected = chain_diagnostics[, "d2_selected"],
    chain_d2_last = chain_diagnostics[, "d2_last"],
    accept_rate = chain_diagnostics[, "accept_rate"]
  )
  class(res) <- "sp_mcmc"
  res
}

.sp_mcmc_resolve_proposal <- function(proposal_fn, j, X_curr, h, Sigma, mcmc) {
  proposal <- if (is.null(proposal_fn)) {
    list(h = h, Sigma = Sigma)
  } else {
    proposal_fn(j = j, X_curr = X_curr, h = h, Sigma = Sigma, mcmc = mcmc)
  }
  if (is.null(proposal)) proposal <- list()
  if (!is.list(proposal)) proposal <- list(h = proposal)

  list(
    h = if (is.null(proposal$h)) h else proposal$h,
    Sigma = if (is.null(proposal$Sigma)) Sigma else proposal$Sigma
  )
}

.sp_mcmc_chain_diagnostics <- function(chain, start, selected) {
  # Figure 2 and the authors' jumpsize.m both measure from Y[j, 1].
  d2 <- rowSums((sweep(chain$X, 2L, start, "-"))^2)
  accept_rate <- if (!is.null(chain$accept) &&
    length(chain$accept) == nrow(chain$X) && nrow(chain$X) > 1L) {
    mean(chain$accept[-1L] == 1L)
  } else if (!is.null(chain$accept) &&
    length(chain$accept) == nrow(chain$X) - 1L &&
    length(chain$accept) > 0L) {
    mean(chain$accept == 1L)
  } else {
    NA_real_
  }

  c(
    d2_max = max(d2),
    d2_selected = sum((start - selected)^2),
    d2_last = d2[length(d2)],
    accept_rate = accept_rate
  )
}

.sp_mcmc_count_matrix <- function(n_points) {
  out <- matrix(0L, n_points, 5L)
  colnames(out) <- c("log_p", "score", "candidate_score",
                     "transition_total", "total")
  out
}

.sp_mcmc_validate_chain <- function(chain, d, expected_n = NULL, start = NULL) {
  if (!is.list(chain)) {
    stop("transition function must return a list", call. = FALSE)
  }
  Chn <- chain$X
  if (is.null(Chn)) stop("transition function must return list element X", call. = FALSE)
  Chn <- as.matrix(Chn)
  if (!is.numeric(Chn) || any(!is.finite(Chn))) {
    stop("transition function must return finite numeric states in X", call. = FALSE)
  }
  if (ncol(Chn) != d) {
    stop("transition function returned X with wrong dimension", call. = FALSE)
  }
  if (!is.null(expected_n) && nrow(Chn) != expected_n) {
    stop("transition function must return exactly ", expected_n,
         " rows, including the initial state", call. = FALSE)
  }
  if (!is.null(start) && !isTRUE(all.equal(
    as.numeric(Chn[1L, ]), as.numeric(start), tolerance = sqrt(.Machine$double.eps)
  ))) {
    stop("transition function must return the supplied initial state in row 1", call. = FALSE)
  }
  chain$X <- Chn

  if (!("D" %in% names(chain))) {
    stop("transition function must return D as a score matrix or NULL", call. = FALSE)
  }
  if (!is.null(chain$D)) {
    chain_D <- as.matrix(chain$D)
    if (!is.numeric(chain_D) || any(!is.finite(chain_D)) ||
      !identical(dim(chain_D), dim(Chn))) {
      stop(
        "transition function must return finite numeric D with the same dimensions as X",
        call. = FALSE
      )
    }
    chain$D <- chain_D
  }

  required_counts <- c("log_p", "score", "total")
  if (!is.list(chain$counts) ||
      any(!required_counts %in% names(chain$counts))) {
    stop(
      "transition function must return counts with log_p, score, and total",
      call. = FALSE
    )
  }
  count_values <- vapply(required_counts, function(name) {
    validate_integer(chain$counts[[name]], paste0("counts$", name),
                     min_value = 0L)
  }, integer(1L))
  names(count_values) <- required_counts
  if (count_values[["total"]] !=
      count_values[["log_p"]] + count_values[["score"]]) {
    stop("transition counts$total must equal counts$log_p + counts$score", call. = FALSE)
  }
  chain$counts <- as.list(count_values)
  if (!is.null(chain$n_eval)) {
    chain$n_eval <- validate_integer(chain$n_eval, "n_eval", min_value = 0L)
    if (chain$n_eval != count_values[["total"]]) {
      stop("transition n_eval must equal counts$total", call. = FALSE)
    }
  } else {
    chain$n_eval <- count_values[["total"]]
  }
  chain
}

.sp_mcmc_unique_path <- function(chain) {
  keep <- !duplicated(as.data.frame(chain$X))
  idx <- which(keep)
  list(
    X = chain$X[idx, , drop = FALSE],
    D = if (is.null(chain$D)) NULL else chain$D[idx, , drop = FALSE],
    idx = idx
  )
}

.sp_mcmc_append_k0 <- function(kernel, X_curr, D_curr, K0_curr, x_new, d_new) {
  x_new <- matrix(as.numeric(x_new), 1L, ncol(X_curr))
  d_new <- matrix(as.numeric(d_new), 1L, ncol(X_curr))
  new_row <- as.numeric(.k0_matrix(kernel, x_new, X_curr, d_new, D_curr))
  new_diag <- as.numeric(.k0_diag(kernel, x_new, d_new))
  rbind(cbind(K0_curr, new_row), c(new_row, new_diag))
}

.sp_mcmc_eval_counts <- function(chain, candidate_score_eval = 0L) {
  candidate_score_eval <- validate_integer(
    candidate_score_eval, "candidate score evaluations", min_value = 0L
  )
  transition_total <- chain$counts$total
  c(
    log_p = chain$counts$log_p,
    score = chain$counts$score,
    candidate_score = candidate_score_eval,
    transition_total = transition_total,
    total = transition_total + candidate_score_eval
  )
}


# ---- S3 methods ----------------------------------------------------------

#' @export
print.sp_mcmc <- function(x, ...) {
  cat(sprintf("SP-MCMC  [%s + %s]  n=%d  d=%d  KSD=%.4g  n_eval=%d\n",
              x$mcmc, x$criterion, nrow(x$X), ncol(x$X),
              x$ksd[length(x$ksd)], x$cum_n_eval[length(x$cum_n_eval)]))
  invisible(x)
}
