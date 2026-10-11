# sp_mcmc and its candidate objective come first, then the MALA and RWM
# transitions, then the start rules and the main loop.

#' Select Stein points from short Markov chains
#'
#' Builds a point set sequentially. At each step, a short Markov chain supplies
#' candidates, and the state that adds the least to the Stein-kernel sum is kept.
#'
#' @details
#' At step \eqn{j}, the function generates a path, removes repeated rows, and
#' appends the row minimizing the greedy objective
#' \deqn{G_j(x)=k_{0,p}(x,x)
#'       +2\sum_{i=1}^{j-1}k_{0,p}(x_i,x),}
#' where \eqn{k_{0,p}} is the Stein kernel formed from `kernel` and the target
#' score. The returned `ksd` is the running discrepancy defined in
#' [stein_points()]. [sp_mcmc_eval_candidates()] evaluates the greedy objective
#' for the candidate rows, reusing their scores when available.
#'
#' `criterion` chooses the selected point where each path starts. A path with
#' `m_seq` states makes `m_seq - 1` transitions. Built-in paths use [rwm()] or
#' [mala()] with proposal covariance `h * Sigma`; MALA also uses `Sigma` in its
#' drift. To use a covariance `S` for both the path and kernel distance, set
#' `Sigma = S` and the kernel's `precon = solve(S)`.
#'
#' @param score_function Function taking an \eqn{n\times d} matrix and
#'   returning the corresponding \eqn{n\times d} matrix of target scores.
#'   [sp_mcmc_eval_candidates()] uses it to score candidate rows.
#' @param log_p Function taking an \eqn{n\times d} matrix and returning `n`
#'   log-density values.
#' @param kernel A `SteinKernel` object. A Gaussian RBF kernel must have a
#'   fixed positive bandwidth.
#' @param n_points Total number of points, including `x_init`.
#' @param d State dimension.
#' @param mcmc MCMC transition: `"rwm"` for Gaussian random-walk Metropolis or
#'   `"mala"` for MALA.
#' @param criterion Path-start rule. `"last"` uses the newest point, `"rand"`
#'   samples uniformly, and `"infl"` uses the point whose removal maximizes the
#'   remaining-set KSD.
#'   A custom rule is a list containing `select` and an optional `label`.
#' @param m_seq Number of path states before duplicate removal. A scalar is
#'   reused; a vector of length `n_points - 1` configures each added point.
#' @param h Positive multiplier in the proposal covariance `h * Sigma`.
#' @param Sigma Symmetric positive-definite matrix used by the built-in
#'   transitions. If `NULL`, the identity matrix is used.
#' @param x_init Initial finite numeric vector of length `d`.
#' @param transition_fn Optional function replacing [rwm()] or [mala()].
#' @param proposal_fn Optional per-step update of `h` or `Sigma`; see Extensions.
#'
#' @section Extensions:
#' Supply a custom start rule as
#' `list(select = function(state) ..., label = "custom")`. `select` returns
#' one integer index into `state$X`; `label` defaults to `"custom"`.
#' `state` contains `j`, `X`, `D`, `K0`, recorded diagnostics, and run settings.
#'
#' A custom `transition_fn(log_p, score_function, x0, h, Sigma, m_iter)` returns
#' `X` and `counts`; `X` starts at `x0`, optional `D` holds the scores of `X`,
#' and optional `accept` is a length-`m_iter` vector of 0/1 indicators with
#' initial entry 0. `counts` contains nonnegative integer `log_p` and `score`
#' counts.
#'
#' `proposal_fn(j, X_curr, h, Sigma, mcmc)` returns a list containing `h`
#' and/or `Sigma`; an empty list keeps both inputs. Changes apply only at step `j`.
#'
#' [sp_mcmc_eval_candidates()] returns one greedy-objective value per row of
#' `cand_X`. Supply `cand_D` to reuse candidate scores; otherwise
#' `score_function` evaluates them.
#'
#' @details
#' `summary(fit)` reports `accept_rate` weighted by the number of transitions in
#' each path;
#' paths without acceptance diagnostics are omitted. `n_repeated` counts
#' selected points that repeat an earlier one; a candidate path contains its own
#' starting state, so repeats are expected. The reported `h` is the initial
#' step size, before any per-step proposal changes. `counts` contains the
#' `log_p`, `score`, and `candidate_score` column totals; `n_eval_total`
#' reports their total once.
#'
#' @return
#' An `"sp_mcmc"` object with:
#' * `X` and `D`: \eqn{n_{\mathrm{points}}\times d} matrices of selected points
#'   and their scores.
#' * `ksd`: running discrepancy after each point.
#' * `n_eval`: evaluations at each step; `cum_n_eval = cumsum(n_eval)`.
#' * `counts`: the `log_p`, `score`, and `candidate_score` counts at each step.
#' * `selected_index`: row of `X` that each path started from.
#' * `accept_rate`: share of accepted proposals in each path.
#' * `chain_d2_max`, `chain_d2_selected`, `chain_d2_last`: squared distances
#'   from each path's starting state to its farthest state, the state kept, and
#'   its last state.
#' * `n_repeated`: number of selected points that repeat an earlier one.
#' * `mcmc`, `transition`, `criterion`, `m_seq`, `h`, `Sigma`, and `kernel`:
#'   settings used by the run.
#' * `method` and `call`: method label and matched call.
#'
#' The first point has no path, so its diagnostic entries are `NA`. Recorded
#' `h` and `Sigma` are the original inputs, not step-specific replacements.
#'
#' [sp_mcmc_eval_candidates()] returns a list with:
#' * `objective_values`: one comparison value per candidate row.
#' * `scores`: candidate scores, reused from `cand_D` when supplied.
#' * `k0_diag`: Stein-kernel diagonal for the candidate rows.
#' * `score_evaluations`: rows scored here; zero when `cand_D` is supplied.
#'
#' `print(fit)` displays the transition method, selection rule, point-set
#' size, final KSD, and evaluation count, and returns `fit` invisibly.
#' `summary(fit)` returns a `"summary.sp_mcmc"` object with
#' `transition`, `criterion`, `m_seq`, `h`, `accept_rate`, `n_repeated`, and
#' `counts`. Printing that object returns it invisibly.
#' @seealso [stein_points()], [rwm()], [mala()]
#' @references
#' Chen, W. Y., Barp, A., Briol, F.-X., Gorham, J., Girolami, M.,
#' Mackey, L., and Oates, C. J. (2019).
#' Stein Point Markov Chain Monte Carlo.
#' Proceedings of the 36th International Conference on Machine Learning,
#' Proceedings of Machine Learning Research, 97, 1011--1021.
#' \url{https://proceedings.mlr.press/v97/chen19b.html}.
#' @examples
#' # Target: N(0, 1).
#' set.seed(1)
#' score <- function(X) -X
#' log_p <- function(X) -0.5 * rowSums(X^2)
#' kernel <- stein_kernel("imq")
#' fit <- sp_mcmc(score, log_p, kernel, n_points = 50, d = 1,
#'                m_seq = 50, h = 1, x_init = 0)
#' summary(fit)
#' c(mean = mean(fit$X), second_moment = mean(fit$X^2))  # Targets: 0 and 1
#'
#' X_curr <- matrix(0, ncol = 1)
#' D_curr <- score(X_curr)
#' cand_X <- matrix(c(-0.5, 0.5), ncol = 1)
#' sp_mcmc_eval_candidates(kernel, score, X_curr, D_curr, cand_X)
#' @export
sp_mcmc <- function(score_function, log_p, kernel, n_points, d,
                    mcmc = c("rwm", "mala"),
                    criterion = c("last", "rand", "infl"),
                    m_seq, h, Sigma = NULL,
                    x_init,
                    transition_fn = NULL, proposal_fn = NULL) {
  cl <- match.call()

  if (!is.function(score_function))
    stop("`score_function` must be a function", call. = FALSE)
  if (!is.function(log_p)) stop("`log_p` must be a function", call. = FALSE)
  validate_positive_scalar(h, "h")
  mcmc <- match.arg(mcmc)
  # transition is only the printed name. mcmc keeps its value because
  # proposal_fn receives it.
  transition <- if (is.null(transition_fn)) mcmc else {
    nm <- deparse(substitute(transition_fn))
    if (length(nm) == 1L && grepl("^[[:alnum:]._]+$", nm)) nm else "custom"
  }
  require_fixed_gaussian_rbf(kernel, "SP-MCMC")
  criterion_obj <- .sp_mcmc_criterion(criterion)

  n_points <- validate_integer(n_points, "n_points")
  d <- validate_integer(d, "d")

  if (!is.numeric(x_init) || length(x_init) != d || any(!is.finite(x_init)))
    stop("x_init must be a finite length-d numeric vector", call. = FALSE)
  x_init <- as.numeric(x_init)
  m_seq <- .validate_m_seq(m_seq, n_points)

  if (is.null(transition_fn)) {
    .prepare_mcmc_covariance(Sigma, d)
  }

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
    transition = transition,
    proposal_fn = proposal_fn,
    call = cl
  )
}


#' @rdname sp_mcmc
#' @param X_curr Current selected point matrix.
#' @param D_curr Current score matrix.
#' @param cand_X Candidate point matrix.
#' @param cand_D Optional candidate score matrix.
#' @export
sp_mcmc_eval_candidates <- function(kernel, score_function, X_curr, D_curr,
                                    cand_X, cand_D = NULL) {
  if (!is.function(score_function))
    stop("`score_function` must be a function", call. = FALSE)
  require_fixed_gaussian_rbf(kernel, "SP-MCMC candidate evaluation")
  obj <- .obj_greedy(kernel, score_function, X_curr, D_curr)
  out <- obj(cand_X, cand_D)
  out$score_evaluations <- if (is.null(cand_D)) nrow(cand_X) else 0L
  out
}


#' Markov transitions for SP-MCMC
#'
#' Runs the MALA or the Gaussian random-walk Metropolis transition used by
#' [sp_mcmc()].
#'
#' @rdname mcmc_transitions
#' @details
#' From the current state `x`, the [mala()] proposal is
#' \deqn{y = x + (h/2)\Sigma s_p(x) + z,
#'       \quad z \sim N(0, h\Sigma),}
#' so the proposal covariance is `h * Sigma`. The function evaluates `log_p`
#' at `y`; `-Inf` rejects the proposal. Otherwise it evaluates the score at `y`,
#' forms the reverse proposal, and applies the Metropolis correction. Row 1 of
#' the output is `x0`; `m_iter - 1` proposals follow.
#'
#' From the current state `x`, the [rwm()] proposal is
#' \deqn{y = x + z, \quad z \sim N(0, h\Sigma).}
#' The function evaluates `log_p(y)` and accepts with probability
#' `min(1, exp(log_p(y) - log_p(x)))`; `-Inf` is rejected. Row 1 of the output
#' is `x0`; `m_iter - 1` proposals follow. RWM does not evaluate the score.
#' [sp_mcmc_eval_candidates()] computes candidate scores later.
#'
#' @param log_p Function taking an \eqn{n\times d} matrix and returning `n`
#'   log-density values.
#' @param score_function Function taking an \eqn{n\times d} matrix and
#'   returning the corresponding \eqn{n\times d} matrix of target scores.
#' @param x0 Finite initial state vector.
#' @param h Positive step-size multiplier; the proposal covariance is
#'   `h * Sigma`.
#' @param Sigma Symmetric positive-definite proposal covariance scale. For
#'   [mala()] it preconditions both the drift and the noise. If `NULL`, the
#'   identity matrix is used.
#' @param m_iter Number of returned chain rows, including `x0` in row 1.
#'
#' @return
#' [mala()] returns a list with:
#' * `X`: chain states.
#' * `D`: scores at those states, reused by SP-MCMC.
#' * `log_p`: log-density values at those states.
#' * `accept`: 0/1 indicators, with 0 in row 1.
#' * `counts`: numbers of rows evaluated by `log_p` and `score_function`;
#'   `n_eval = counts$total`.
#'
#' [rwm()] returns a list with:
#' * `X`: chain states.
#' * `log_p`: log-density values at those states.
#' * `accept`: 0/1 indicators, with 0 in row 1.
#' * `D = NULL`, because RWM computes no scores.
#' * `counts`: row counts with `counts$score = 0`; `n_eval = counts$total =
#'   counts$log_p`.
#' @references
#' Chen, W. Y., Barp, A., Briol, F.-X., Gorham, J., Girolami, M.,
#' Mackey, L., and Oates, C. J. (2019).
#' Stein Point Markov Chain Monte Carlo.
#' Proceedings of the 36th International Conference on Machine Learning,
#' Proceedings of Machine Learning Research, 97, 1011--1021.
#' \url{https://proceedings.mlr.press/v97/chen19b.html}.
#' @examples
#' # Target: N(0, 1).
#' set.seed(1)
#' score <- function(X) -X
#' log_p <- function(X) -0.5 * rowSums(X^2)
#' x0 <- rnorm(1)
#' chains <- list(mala = mala(log_p, score, x0, h = 1, m_iter = 1000),
#'                rwm = rwm(log_p, x0, h = 1, m_iter = 1000))
#' # Acceptance rate over proposals (rows 2:1000).
#' vapply(chains, function(z) mean(z$accept[-1]), numeric(1))
#' @export
mala <- function(log_p, score_function, x0, h, Sigma = NULL, m_iter) {
  if (!is.function(score_function)) stop("`score_function` must be a function", call. = FALSE)
  setup <- .mcmc_setup(log_p, x0, h, Sigma, m_iter)
  x0 <- setup$x0
  d <- setup$d
  h <- setup$h
  Sigma <- setup$Sigma
  m_iter <- setup$m_iter
  U <- setup$proposal_chol

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
  lp[1L] <- .validate_initial_log_p(log_p(matrix(x0, 1L, d)))
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
        stop("log_p must return one finite or -Inf value for each proposal", call. = FALSE)
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

      # The Langevin proposal is not symmetric, so the acceptance ratio includes
      # both proposal densities.
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

#' @rdname mcmc_transitions
#' @export
rwm <- function(log_p, x0, h, Sigma = NULL, m_iter) {
  setup <- .mcmc_setup(log_p, x0, h, Sigma, m_iter)
  x0 <- setup$x0
  d <- setup$d
  m_iter <- setup$m_iter
  U <- setup$proposal_chol
  X <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  lp[1L] <- .validate_initial_log_p(log_p(matrix(x0, 1L, d)))
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      X[i, ] <- X[i - 1L, ]
      y <- X[i - 1L, ] + as.numeric(stats::rnorm(d) %*% U)
      lp_y <- log_p(matrix(y, 1L, d))
      if (!is.numeric(lp_y) || length(lp_y) != 1L || is.na(lp_y) || lp_y == Inf) {
        stop("log_p must return one finite or -Inf value for each proposal", call. = FALSE)
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

.mcmc_setup <- function(log_p, x0, h, Sigma, m_iter) {
  if (!is.function(log_p)) stop("`log_p` must be a function", call. = FALSE)
  if (!is.numeric(x0) || length(x0) < 1L || any(!is.finite(x0)))
    stop("x0 must be a finite numeric vector", call. = FALSE)
  x0 <- as.numeric(x0)
  d <- length(x0)
  validate_positive_scalar(h, "h")
  covariance <- .prepare_mcmc_covariance(Sigma, d)

  list(
    x0 = x0, d = d, h = as.numeric(h), Sigma = covariance$Sigma,
    m_iter = validate_integer(m_iter, "m_iter"),
    proposal_chol = sqrt(as.numeric(h)) * covariance$chol
  )
}

.prepare_mcmc_covariance <- function(Sigma, d) {
  if (is.null(Sigma)) Sigma <- diag(d) else Sigma <- as.matrix(Sigma)
  if (!is.numeric(Sigma) || any(!is.finite(Sigma)) ||
      nrow(Sigma) != d || ncol(Sigma) != d)
    stop("Sigma must be a finite d x d matrix", call. = FALSE)
  sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(Sigma)))
  if (max(abs(Sigma - t(Sigma))) > sym_tol)
    stop("Sigma must be symmetric positive definite", call. = FALSE)
  # Remove roundoff asymmetry before Cholesky factorization.
  Sigma <- (Sigma + t(Sigma)) / 2
  U <- tryCatch(chol(Sigma), error = function(e) {
    stop("Sigma must be symmetric positive definite", call. = FALSE)
  })
  list(Sigma = Sigma, chol = U)
}

.validate_m_seq <- function(x, n_points) {
  if (!is.numeric(x) || length(x) < 1L)
    stop("m_seq must be a positive integer or a numeric vector of positive integers",
         call. = FALSE)
  if (length(x) == 1L) validate_integer(x, "m_seq") else {
    if (length(x) != n_points - 1L)
      stop("m_seq must be scalar or length n_points - 1", call. = FALSE)
    vapply(x, validate_integer, integer(1L), arg_name = "m_seq entry")
  }
}

.validate_initial_log_p <- function(x) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x))
    stop("log_p must return one finite value at x0", call. = FALSE)
  as.numeric(x)
}

# rwm takes no score_function, so this wrapper gives it the signature that
# sp_mcmc calls.
.sp_mcmc_rwm_transition <- function(log_p, score_function, x0, h, Sigma,
                                    m_iter) {
  rwm(log_p = log_p, x0 = x0, h = h, Sigma = Sigma, m_iter = m_iter)
}


.sp_mcmc_criterion <- function(criterion = c("last", "rand", "infl")) {
  if (is.list(criterion) && is.function(criterion$select)) {
    fn <- criterion$select
    label <- if (is.null(criterion$label)) "custom" else criterion$label
  } else {
    if (!is.null(criterion) && !is.character(criterion)) {
      stop("`criterion` must be \"last\", \"rand\", \"infl\", or list(select = function(state) ...)",
           call. = FALSE)
    }
    name <- match.arg(criterion, c("last", "rand", "infl"))
    fn <- switch(
      name,
      last = function(state) nrow(state$X),
      rand = function(state) sample.int(nrow(state$X), 1L),
      infl = function(state) .crit_infl(state$K0)
    )
    return(list(label = name, select = fn, needs_k0 = name == "infl"))
  }
  list(label = label, select = fn, needs_k0 = TRUE)
}

.sp_mcmc_select_start <- function(criterion_obj, state) {
  idx <- criterion_obj$select(state)
  is_integer <- is.numeric(idx) && length(idx) == 1L && is.finite(idx) &&
    abs(idx - round(idx)) <= sqrt(.Machine$double.eps)
  if (!is_integer || idx < 1L || idx > nrow(state$X)) {
    stop("criterion must return one valid one-based index into the current point set",
         call. = FALSE)
  }
  as.integer(round(idx))
}

.crit_infl <- function(K0_cache) {
  # Removing point i lowers the kernel sum by its row sum plus column sum minus
  # its diagonal, so the smallest such contribution leaves the largest KSD.
  which.min(rowSums(K0_cache) + colSums(K0_cache) - diag(K0_cache))
}


.sp_mcmc_run <- function(score_function, log_p, kernel, n_points, d,
                         mcmc, criterion_obj, m_seq, h, Sigma, x_init,
                         transition_fn = NULL, transition = mcmc,
                         proposal_fn = NULL, call = NULL) {

  chain_fn <- if (is.null(transition_fn)) {
    switch(mcmc, rwm = .sp_mcmc_rwm_transition, mala = mala)
  } else {
    transition_fn
  }
  X <- D <- matrix(NA_real_, n_points, d)
  n_eval <- integer(n_points)
  ksd <- numeric(n_points)
  counts <- matrix(
    0L, nrow = n_points, ncol = 5L,
    dimnames = list(NULL, c("log_p", "score", "candidate_score",
                            "transition_total", "total"))
  )
  selected_index <- rep(NA_integer_, n_points)
  chain_diagnostics <- matrix(
    NA_real_, nrow = n_points, ncol = 4L,
    dimnames = list(
      NULL,
      c("d2_max", "d2_selected", "d2_last", "accept_rate")
    )
  )

  X[1L, ] <- x_init
  x_init_mat <- matrix(x_init, 1L, d)
  D[1L, ] <- .as_score_matrix(
    score_function(x_init_mat), x_init_mat
  )[1L, ]
  ss <- as.numeric(k0_diag(kernel, matrix(x_init, 1L, d), matrix(D[1L, ], 1L, d)))
  K0_cache <- if (criterion_obj$needs_k0) matrix(ss, nrow = 1L) else NULL
  ksd[1L] <- .running_ksd(ss, 1L)
  counts[1L, ] <- c(log_p = 0L, score = 1L, candidate_score = 0L,
                    transition_total = 0L, total = 1L)
  n_eval[1L] <- as.integer(counts[1L, "total"])

  if (n_points >= 2L) {
    for (j in 2:n_points) {
      X_curr <- X[seq_len(j - 1L), , drop = FALSE]
      D_curr <- D[seq_len(j - 1L), , drop = FALSE]
      state <- list(
        j = j, X = X_curr, D = D_curr, K0 = K0_cache,
        ksd = ksd[seq_len(j - 1L)],
        n_eval = n_eval[seq_len(j - 1L)],
        counts = counts[seq_len(j - 1L), , drop = FALSE],
        selected_index = selected_index[seq_len(j - 1L)],
        kernel = kernel, mcmc = mcmc, criterion = criterion_obj$label
      )
      i_star <- .sp_mcmc_select_start(criterion_obj, state)
      selected_index[j] <- i_star
      start <- as.numeric(X_curr[i_star, ])

      m_j <- if (length(m_seq) == 1L) m_seq else m_seq[j - 1L]
      proposal <- .sp_mcmc_resolve_proposal(
        proposal_fn = proposal_fn,
        j = j,
        X_curr = X_curr,
        h = h,
        Sigma = Sigma,
        mcmc = mcmc
      )

      # m_j counts the starting state, so the chain makes m_j minus 1
      # transitions.
      chain <- chain_fn(
        log_p, score_function, start, proposal$h, proposal$Sigma, m_j
      )
      chain <- .sp_mcmc_validate_chain(
        chain, d, expected_n = m_j, start = start
      )
      path <- .sp_mcmc_unique_path(chain)

      cand_X <- path$X
      cand_D_from_chain <- path$D
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
      if (criterion_obj$needs_k0) K0_cache <- .sp_mcmc_append_k0(
        kernel, X_curr, D_curr, K0_cache, x_min, d_min
      )
      # f_min is k0 of the new point with itself plus twice its k0 with each
      # earlier point, which is exactly the increase in the kernel sum.
      ss <- ss + f_min
      ksd[j] <- .running_ksd(ss, j)

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

  n_repeated <- nrow(X) - nrow(unique(X))

  res <- list(
    X = X, D = D, ksd = ksd, n_eval = n_eval, cum_n_eval = cumsum(n_eval),
    counts = counts,
    method = "sp_mcmc", kernel = kernel, call = call,
    mcmc = mcmc, transition = transition,
    criterion = criterion_obj$label, m_seq = m_seq,
    h = h, Sigma = Sigma, selected_index = selected_index,
    n_repeated = n_repeated,
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
  if (!is.list(proposal)) stop("proposal_fn must return a list", call. = FALSE)

  list(
    h = if (is.null(proposal$h)) h else proposal$h,
    Sigma = if (is.null(proposal$Sigma)) Sigma else proposal$Sigma
  )
}

.sp_mcmc_chain_diagnostics <- function(chain, start, selected) {
  d2 <- rowSums((sweep(chain$X, 2L, start, "-"))^2)
  # accept has one entry per row and row 1 is the start, so the rate skips the
  # first entry.
  accept_rate <- if (!is.null(chain$accept) && nrow(chain$X) > 1L) {
    mean(chain$accept[-1L] == 1L)
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

.sp_mcmc_validate_chain <- function(chain, d, expected_n, start) {
  if (!is.list(chain)) stop("transition function must return a list", call. = FALSE)
  if (is.null(chain$X)) stop("transition function must return list element X", call. = FALSE)
  Chn <- .as_rows(chain$X, "transition function X")
  if (ncol(Chn) != d)
    stop("transition function returned X with wrong dimension", call. = FALSE)
  if (nrow(Chn) != expected_n)
    stop("transition function must return exactly ", expected_n,
         " rows, including the initial state", call. = FALSE)
  if (!identical(as.numeric(Chn[1L, ]), as.numeric(start)))
    stop("transition function must return the supplied initial state in row 1", call. = FALSE)
  chain$X <- Chn
  if (!is.null(chain$accept) &&
      (length(chain$accept) != nrow(Chn) || any(!chain$accept %in% c(0, 1)) ||
       chain$accept[1L] != 0))
    stop("transition accept must align with X, contain 0/1, and start with 0", call. = FALSE)

  if (!is.null(chain$D)) {
    chain_D <- as.matrix(chain$D)
    if (!is.numeric(chain_D) || any(!is.finite(chain_D)) ||
        !identical(dim(chain_D), dim(Chn)))
      stop("transition function must return finite numeric D with the same dimensions as X",
           call. = FALSE)
    chain$D <- chain_D
  }

  required <- c("log_p", "score")
  if (!is.list(chain$counts) || any(!required %in% names(chain$counts)))
    stop("transition function must return counts with log_p and score",
         call. = FALSE)
  counts <- vapply(required, function(nm) {
    validate_integer(chain$counts[[nm]], paste0("counts$", nm), min_value = 0L)
  }, integer(1L))
  chain$counts <- c(as.list(counts), total = sum(counts))
  chain
}

.sp_mcmc_unique_path <- function(chain) {
  keep <- !duplicated(as.data.frame(chain$X))
  idx <- which(keep)
  list(
    X = chain$X[idx, , drop = FALSE],
    D = if (is.null(chain$D)) NULL else chain$D[idx, , drop = FALSE]
  )
}

.sp_mcmc_append_k0 <- function(kernel, X_curr, D_curr, K0_curr, x_new, d_new) {
  x_new <- matrix(as.numeric(x_new), 1L, ncol(X_curr))
  d_new <- matrix(as.numeric(d_new), 1L, ncol(X_curr))
  new_row <- as.numeric(
    stein_kernel_matrix(kernel, x_new, d_new, X_curr, D_curr)
  )
  new_diag <- as.numeric(k0_diag(kernel, x_new, d_new))
  # Symmetry makes the new row the new column.
  unname(rbind(cbind(K0_curr, new_row), c(new_row, new_diag)))
}

.sp_mcmc_eval_counts <- function(chain, candidate_score_eval) {
  transition_total <- chain$counts$total
  c(
    log_p = chain$counts$log_p,
    score = chain$counts$score,
    candidate_score = candidate_score_eval,
    transition_total = transition_total,
    total = transition_total + candidate_score_eval
  )
}


#' @export
print.sp_mcmc <- function(x, ...) {
  cat(sprintf("SP-MCMC  [%s + %s]  n=%d  d=%d  KSD=%.4g  n_eval=%.0f\n",
              x$transition, x$criterion, nrow(x$X), ncol(x$X),
              .last_value(x$ksd), .last_value(x$cum_n_eval)))
  invisible(x)
}


#' @export
summary.sp_mcmc <- function(object, ...) {
  out <- .point_set_summary(object, "total evaluations", "summary.sp_mcmc")
  if (length(object$ksd)) {
    out$ksd_last <- object$ksd[length(object$ksd)]
  }
  out$transition <- object$transition
  out$criterion <- object$criterion
  out$m_seq <- object$m_seq
  out$h <- object$h
  rates <- object$accept_rate[-1L]
  transitions <- rep_len(object$m_seq, length(rates)) - 1L
  observed <- !is.na(rates) & transitions > 0L
  out$accept_rate <- if (any(observed)) {
    stats::weighted.mean(rates[observed], transitions[observed])
  } else NA_real_
  out$counts <- colSums(object$counts[, c("log_p", "score", "candidate_score"),
                                    drop = FALSE])
  out$n_repeated <- object$n_repeated
  out
}

#' @export
print.summary.sp_mcmc <- function(x, ...) {
  cat(sprintf("SP-MCMC  n=%d  d=%d  kernel=%s\n", x$n, x$d, x$kernel))
  cat(sprintf(
    "  chain: %s, criterion %s, m_seq %s, initial step size %g\n",
    x$transition, x$criterion,
    paste(unique(range(x$m_seq)), collapse = "-"), x$h
  ))
  cat(sprintf("  acceptance rate: %.2f\n", x$accept_rate))
  cat(sprintf("  repeated points: %d of %d\n", x$n_repeated, x$n))
  cat("  evaluations:",
      paste(sprintf("%s=%d", names(x$counts), x$counts), collapse = "  "),
      "\n")
  if (!is.null(x$ksd_last)) {
    cat(sprintf("  KSD: %.4g\n", x$ksd_last))
  }
  cat(sprintf("  %s: %.0f\n", x$evaluation_label, x$n_eval_total))
  invisible(x)
}
