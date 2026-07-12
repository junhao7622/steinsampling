# Stein Point MCMC (Chen, Barp, Briol, Gorham, Girolami, Mackey, Oates, ICML 2019).

#' Generate Stein Points from short MCMC candidate paths
#'
#' Implements Stein Point Markov Chain Monte Carlo (SP-MCMC). At each Stein
#' Points step, the algorithm runs a short MCMC chain to produce a manageable
#' candidate set, then appends the candidate that most reduces the greedy KSD
#' objective.
#'
#' @details
#' SP-MCMC replaces Stein Points' global state-space search by a short Markov
#' chain. At step `j` it starts from an already selected point, makes
#' `m_seq[j]` Markov transitions, removes duplicate states among the resulting
#' `m_seq[j]` candidates, and scores each remaining candidate with the greedy
#' objective. The initial state is not a candidate. With target score
#' \eqn{s_p(x)=\nabla_x\log p(x)} and the `k0` from
#' [stein_kernel_matrix()], the prefix diagnostic is
#' \deqn{KSD_m^2 = \frac{1}{m^2}
#'        \sum_{a=1}^m \sum_{b=1}^m k0(x_a,x_b).}
#' The paper minimizes
#' \deqn{\frac{1}{2} k0(x, x) + \sum_{i=1}^{j-1} k0(x_i, x).}
#' The equivalent implementation form is
#' \deqn{k0(x, x) + 2 \sum_{i=1}^{j-1} k0(x_i, x),}
#' whose smallest candidate is appended.
#'
#' `x_init` fixes the first point, leaving `n_points - 1` short chains. Scalar
#' `m_seq` is recycled; a length-`n_points - 1` vector supplies each subsequent
#' length. Built-in transitions return states, log densities, acceptance
#' indicators, and reusable MALA scores. A cached
#' `K0[a,b]=k0(x_a,x_b)` lets start rules, candidate scores, and KSD diagnostics
#' share selected-point interactions.
#'
#' Start rules are `"last"` (latest point), `"rand"` (random selected point),
#' and `"infl"` (smallest current Stein-matrix influence). Built-in transitions
#' are the paper's Gaussian random-walk Metropolis and MALA. Custom `criterion`,
#' `transition_fn`, and `proposal_fn` expose these pieces without replacing the
#' main loop. Diagnostics split MCMC and candidate-scoring evaluations in
#' `counts`, accumulate them in `cum_n_eval`, and record acceptance rates and
#' squared-distance movement for each short path.
#'
#' [sp_mcmc_state()] builds start-rule state; [sp_mcmc_criterion()] turns
#' LAST/RAND/INFL or a custom function into a reusable selector;
#' [sp_mcmc_select_start()] validates its row index; and
#' [sp_mcmc_eval_candidates()] applies the Stein Points objective. Thus moving
#' from [stein_points()] to `sp_mcmc()` retains the greedy objective but replaces
#' its optimizer with [grw()], [rwm()], [mala()], or another MCMC transition.
#'
#' @param score_function Function returning target scores for sample rows.
#' @param log_p Function returning log density values for sample rows.
#' @param kernel A [stein_kernel()] object, compatible kernel function, or list
#'   with a `k0_matrix` function. A Gaussian RBF kernel must have a fixed
#'   positive bandwidth `h`; SP-MCMC evaluates one fixed Stein kernel across
#'   all short candidate paths.
#' @param n_points Total number of points to generate.
#' @param d State dimension.
#' @param mcmc MCMC transition: `"grw"` for Gaussian random-walk Metropolis,
#'   `"mala"` for MALA, or alias `"rwm"`.
#' @param criterion Rule for choosing where the next MCMC chain starts:
#'   `"last"`, `"rand"`, `"infl"`, or a custom function.
#' @param m_seq Number of Markov transitions, and hence candidates before
#'   duplicate removal, at each selection step. Supply one value, one value per
#'   point, or one value per update after the first point.
#' @param h Step-size multiplier used by the built-in MCMC transitions.
#' @param Sigma Symmetric positive-definite proposal covariance or
#'   preconditioning matrix. If `NULL`, the identity matrix is used.
#' @param x_init Fixed first point.
#' @param seed Optional RNG seed.
#' @param transition_fn Optional custom MCMC transition function. It receives
#'   the log density, score function, current state, step size, proposal matrix,
#'   and a requested row count of `m_seq[j] + 1`. It must return exactly that
#'   many rows in `X`, with the supplied current state in row 1. The remaining
#'   `m_seq[j]` rows are states after successive Markov transitions.
#' @param proposal_fn Optional function that changes `h` or `Sigma` at each
#'   step. It should return a list containing replacement `h` and/or `Sigma`.
#' @param n_eval_fn Optional function that overrides the per-step evaluation
#'   count.
#' @param criterion_args optional list of extra arguments passed to a custom
#'   `criterion` function.
#'
#' @return
#' A list of class `"sp_mcmc"` and `"stein_points"` with:
#' * `X`: `n_points x d` matrix of selected point locations.
#' * `D`: target scores at the selected points.
#' * `ksd`: running KSD diagnostic after each selected point, with `ksd[j]`
#'   equal to the square root of the empirical double average of `k0` over the
#'   first `j` selected rows.
#' * `n_eval`, `cum_n_eval`: per-step and cumulative evaluation counts.
#' * `counts`: matrix splitting each step's count into `log_p`, `score`,
#'   `candidate_score`, `transition_total`, and `total`.
#' * `method`, `kernel`, `mcmc`, `criterion`, `m_seq`, `h`, `Sigma`: algorithm
#'   settings used for the run.
#' * `selected_index`: index of the previously selected point used to start
#'   each short chain. This records the output of the LAST/RAND/INFL or custom
#'   start rule before the MCMC transition is run.
#' * `chain_d2_max`, `chain_d2_selected`, `chain_d2_last`: squared Euclidean
#'   distances from the chain start to the farthest candidate, selected
#'   candidate, and last candidate. `chain_d2_first_last` is the squared
#'   distance between the first and last candidates, as used in Figure 2 of the
#'   SP-MCMC paper.
#' * `accept_rate`: Metropolis acceptance rate for each short chain, useful for
#'   interpreting whether poor point choices came from the Stein objective or
#'   from a candidate chain that barely moved.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' sp_mcmc(score, log_p, kernel, n_points = 2, d = 1, m_seq = 2, x_init = 0)
#' @export
sp_mcmc <- function(score_function, log_p, kernel, n_points, d,
                    mcmc = c("grw", "mala", "rwm"),
                    criterion = c("last", "rand", "infl"),
                    m_seq, h = NULL, Sigma = NULL,
                    x_init, seed = NULL,
                    transition_fn = NULL, proposal_fn = NULL,
                    n_eval_fn = NULL,
                    criterion_args = list()) {

  mcmc <- match.arg(mcmc)
  if (identical(mcmc, "rwm")) mcmc <- "grw"
  require_fixed_gaussian_rbf(kernel, "SP-MCMC")
  criterion_obj <- sp_mcmc_criterion(criterion, criterion_args)

  n_points <- validate_positive_integer(n_points, "n_points")
  d <- validate_positive_integer(d, "d")

  if (!is.numeric(x_init) || length(x_init) != d || any(!is.finite(x_init))) {
    stop("x_init must be a finite length-d numeric vector")
  }
  x_init <- as.numeric(x_init)

  if (!is.numeric(m_seq) || length(m_seq) < 1L) {
    stop("m_seq must be a positive integer or a numeric vector of positive integers")
  }
  if (length(m_seq) == 1L) {
    m_seq <- validate_positive_integer(m_seq, "m_seq")
  } else if (length(m_seq) == n_points - 1L) {
    m_seq <- vapply(m_seq, validate_positive_integer, integer(1L),
                    arg_name = "m_seq entry")
    m_seq <- c(NA_integer_, m_seq)
  } else if (length(m_seq) == n_points) {
    m_seq <- vapply(m_seq, validate_positive_integer, integer(1L),
                    arg_name = "m_seq entry")
  } else {
    stop("m_seq must be scalar, length n_points, or length n_points - 1")
  }

  run <- function() {
    sp_mcmc_run(
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
      n_eval_fn = n_eval_fn
    )
  }

  if (is.null(seed)) run() else with_local_seed(seed, run())
}


# ---- SP-MCMC main loop and composable pieces -------------------------------

sp_mcmc_run <- function(score_function, log_p, kernel, n_points, d,
                        mcmc, criterion_obj, m_seq, h, Sigma, x_init,
                        transition_fn = NULL, proposal_fn = NULL,
                        n_eval_fn = NULL) {

  chain_fn <- if (is.null(transition_fn)) {
    switch(mcmc, grw = grw, mala = mala)
  } else {
    transition_fn
  }
  get_m <- if (length(m_seq) == 1L) function(j) m_seq else function(j) m_seq[j]
  get_proposal <- if (is.null(proposal_fn)) {
    function(j, X_curr, h, Sigma, mcmc) list(h = h, Sigma = Sigma)
  } else {
    proposal_fn
  }

  X <- D <- matrix(NA_real_, n_points, d)
  n_eval <- integer(n_points)
  ksd <- numeric(n_points)
  counts <- .sp_mcmc_count_matrix(n_points)
  selected_index <- rep(NA_integer_, n_points)
  chain_d2_max <- rep(NA_real_, n_points)
  chain_d2_selected <- rep(NA_real_, n_points)
  chain_d2_last <- rep(NA_real_, n_points)
  chain_d2_first_last <- rep(NA_real_, n_points)
  accept_rate <- rep(NA_real_, n_points)

  X[1L, ] <- x_init
  D[1L, ] <- validate_scores(score_function, matrix(x_init, 1L, d))[1L, ]
  K0_cache <- matrix(
    .k0_diag(kernel, matrix(x_init, 1L, d), matrix(D[1L, ], 1L, d)),
    nrow = 1L
  )
  ss <- as.numeric(K0_cache[1L, 1L])
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
      if (!is.finite(m_j) || m_j < 1L) {
        stop("m_seq entries must be integers >= 1")
      }

      proposal <- get_proposal(j = j, X_curr = X_curr, h = h,
                               Sigma = Sigma, mcmc = mcmc)
      if (is.null(proposal)) proposal <- list()
      if (!is.list(proposal)) proposal <- list(h = proposal)
      h_j <- if (is.null(proposal$h)) h else proposal$h
      Sigma_j <- if (is.null(proposal$Sigma)) Sigma else proposal$Sigma

      chain <- chain_fn(
        log_p, score_function, start, h_j, Sigma_j, m_j + 1L
      )
      chain <- sp_mcmc_validate_chain(
        chain, d, expected_n = m_j + 1L, start = start
      )
      candidate_chain <- sp_mcmc_drop_initial_state(chain)
      path <- sp_mcmc_unique_path(candidate_chain)

      cand_X <- path$X
      cand_D_from_chain <- path$D
      cand_eval <- sp_mcmc_eval_candidates(
        kernel = kernel, score_function = score_function,
        X_curr = X_curr, D_curr = D_curr,
        cand_X = cand_X, cand_D = cand_D_from_chain
      )
      cand_D <- cand_eval$D_new
      f_vec <- cand_eval$f_vec

      i_min <- which.min(f_vec)
      x_min <- as.numeric(cand_X[i_min, ])
      d_min <- as.numeric(cand_D[i_min, ])
      f_min <- as.numeric(f_vec[i_min])

      X[j, ] <- x_min
      D[j, ] <- d_min
      K0_cache <- sp_mcmc_append_k0(kernel, X_curr, D_curr, K0_cache,
                                    x_min, d_min)
      ss <- ss + f_min
      ksd[j] <- sqrt(max(ss, 0)) / j

      d2 <- rowSums((sweep(candidate_chain$X, 2L, start, "-"))^2)
      chain_d2_max[j] <- max(d2)
      chain_d2_selected[j] <- sum((start - x_min)^2)
      chain_d2_last[j] <- d2[length(d2)]
      chain_d2_first_last[j] <- sum(
        (candidate_chain$X[nrow(candidate_chain$X), ] -
          candidate_chain$X[1L, ])^2
      )
      accept_rate[j] <- if (!is.null(chain$accept) &&
        length(chain$accept) == nrow(chain$X)) {
        mean(chain$accept[-1L] == 1L)
      } else if (!is.null(chain$accept) &&
        length(chain$accept) == nrow(chain$X) - 1L) {
        mean(chain$accept == 1L)
      } else {
        NA_real_
      }

      auto_counts <- sp_mcmc_eval_counts(
        chain = chain,
        candidate_score_eval = cand_eval$score_eval,
        fallback_m = m_j
      )
      if (!is.null(n_eval_fn)) {
        override <- n_eval_fn(chain = chain, cand_X = cand_X, cand_D = cand_D,
                              m_j = m_j, mcmc = mcmc, j = j,
                              X_curr = X_curr)
        override <- validate_nonnegative_integer(override, "n_eval_fn result")
        auto_counts["total"] <- override
      }
      counts[j, ] <- auto_counts
      n_eval[j] <- as.integer(auto_counts["total"])
    }
  }

  res <- list(
    X = X, D = D, ksd = ksd, n_eval = n_eval, cum_n_eval = cumsum(n_eval),
    counts = counts,
    method = "sp_mcmc", kernel = kernel, truncation = "none", c2 = NULL,
    mcmc = mcmc, criterion = criterion_obj$label, m_seq = m_seq,
    h = h, Sigma = Sigma, selected_index = selected_index,
    chain_d2_max = chain_d2_max,
    chain_d2_selected = chain_d2_selected,
    chain_d2_last = chain_d2_last,
    chain_d2_first_last = chain_d2_first_last,
    chain_jumps = chain_d2_last,
    accept_rate = accept_rate
  )
  class(res) <- unique(c("sp_mcmc", "stein_points", class(res)))
  res
}


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
#' This constructor is exported for users who write custom SP-MCMC start rules.
#' The main [sp_mcmc()] loop creates this object internally before every short
#' chain. By exposing the same object, the package lets custom rules be tested
#' outside the full algorithm.
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
#' @param criterion Criterion label.
#'
#' @return
#' A list of class `"sp_mcmc_state"` with the fields supplied as arguments:
#' current points `X`, scores `D`, Stein-kernel matrix `K0`, current step `j`,
#' running diagnostics, and method labels. It is not a result object; it is a
#' snapshot passed to a start-point criterion so the criterion can decide where
#' the next short MCMC candidate chain should start without recomputing scores
#' or pairwise kernel values.
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
#' chooses a selected point uniformly at random. `"infl"` computes each point's
#' current influence from the Stein kernel matrix,
#' \deqn{I_a = \sum_b K0_{ab} + \sum_b K0_{ba} - K0_{aa},}
#' and starts at the point with the smallest `I_a`. A custom function can
#' implement another rule while still using the same SP-MCMC main loop.
#'
#' The returned object is a small strategy object: it stores a human-readable
#' `label` and a `select(state)` function. [sp_mcmc()] uses that object at every
#' iteration. This design keeps the choice of starting point independent of the
#' MCMC transition and the candidate-scoring rule.
#'
#' The label is not just cosmetic. It is copied into the final [sp_mcmc()]
#' output so that simulation results can record whether LAST, RAND, INFL, or a
#' custom rule was used. The `select` function is the executable part of the
#' rule. Splitting those two pieces lets a custom rule behave like the built-in
#' rules in both computation and reporting.
#'
#' @param criterion Criterion name, function, or criterion object.
#' @param criterion_args Extra arguments passed to a custom criterion function.
#'
#' @return
#' A list with `label` and `select`. `label` is stored in the final [sp_mcmc()]
#' output. `select` is a function that takes an [sp_mcmc_state()] object and
#' returns one row index into `state$X`. Returning this small object, rather
#' than returning an index immediately, is necessary because the state changes
#' at every SP-MCMC step.
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
#' It is exported mainly for custom criterion development. A user can build a
#' state with [sp_mcmc_state()], build a criterion with [sp_mcmc_criterion()],
#' and check that the criterion returns a legal index before running the full
#' SP-MCMC algorithm.
#'
#' Inside [sp_mcmc()], the returned index decides which already selected point
#' becomes the initial state of the next short candidate chain. That index is
#' also recorded in `selected_index` in the final output, because the start rule
#' is part of the algorithm's behavior and affects the candidate set that is
#' searched.
#'
#' @param criterion_obj Criterion object from `sp_mcmc_criterion()`.
#' @param state Current SP-MCMC state.
#'
#' @return
#' One integer: the one-based row index in `state$X` that should initialize the
#' next short candidate chain. The function returns only the checked index
#' because the state object already contains the point matrix, scores, and
#' diagnostic history.
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
    stop("criterion must return one valid one-based integer index into the current point set")
  }
  idx <- as.integer(round(idx))
  if (idx < 1L || idx > nrow(state$X)) {
    stop("criterion must return one valid one-based index into the current point set")
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
#' where the sum runs over rows in `X_curr`. The returned `f_vec` contains one
#' objective value per candidate row, in the same order as `cand_X`. If the MCMC
#' transition already returned candidate scores, pass them as `cand_D`;
#' otherwise this helper evaluates `score_function` on the candidate rows and
#' records how many score evaluations were needed.
#'
#' This function is the bridge between the MCMC transition and the Stein Points
#' objective. MCMC proposes a finite candidate set; this helper evaluates those
#' candidates using the current Stein Points objective and returns the pieces
#' needed for [sp_mcmc()] to append the best one.
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
#' * `f_vec`: doubled greedy objective value for each candidate row. The row
#'   with the smallest value is the point [sp_mcmc()] appends.
#' * `D_new`: score matrix for the candidate rows, either reused from `cand_D`
#'   or computed from `score_function`; it is returned so the selected row can
#'   be appended without another score call.
#' * `score_eval`: number of candidate score evaluations performed inside this
#'   helper.
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
  obj <- .obj_greedy(kernel, score_function, X_curr, D_curr)
  score_eval <- 0L
  if (is.null(cand_D)) {
    out <- obj(cand_X)
    score_eval <- nrow(cand_X)
  } else {
    out <- obj(cand_X, cand_D)
  }
  out$score_eval <- score_eval
  out
}


.sp_mcmc_count_matrix <- function(n_points) {
  out <- matrix(0L, n_points, 5L)
  colnames(out) <- c("log_p", "score", "candidate_score",
                     "transition_total", "total")
  out
}


sp_mcmc_validate_chain <- function(chain, d, expected_n = NULL, start = NULL) {
  if (!is.list(chain)) chain <- list(X = chain)
  Chn <- chain$X
  if (is.null(Chn)) stop("transition function must return list element X")
  Chn <- as.matrix(Chn)
  if (!is.numeric(Chn) || any(!is.finite(Chn))) {
    stop("transition function must return finite numeric states in X")
  }
  if (ncol(Chn) != d) stop("transition function returned X with wrong dimension")
  if (!is.null(expected_n) && nrow(Chn) != expected_n) {
    stop("transition function must return exactly ", expected_n,
         " rows, including the initial state")
  }
  if (!is.null(start) && !isTRUE(all.equal(
    as.numeric(Chn[1L, ]), as.numeric(start), tolerance = sqrt(.Machine$double.eps)
  ))) {
    stop("transition function must return the supplied initial state in row 1")
  }
  chain$X <- Chn

  if (!is.null(chain$D)) {
    chain_D <- as.matrix(chain$D)
    if (!is.numeric(chain_D) || any(!is.finite(chain_D)) ||
      !identical(dim(chain_D), dim(Chn))) {
      stop("transition function must return finite numeric D with the same dimensions as X")
    }
    chain$D <- chain_D
  }
  chain
}


sp_mcmc_drop_initial_state <- function(chain) {
  keep <- seq.int(2L, nrow(chain$X))
  chain$X <- chain$X[keep, , drop = FALSE]
  if (!is.null(chain$D)) chain$D <- chain$D[keep, , drop = FALSE]
  if (!is.null(chain$log_p) && length(chain$log_p) == length(keep) + 1L) {
    chain$log_p <- chain$log_p[keep]
  }
  if (!is.null(chain$accept) && length(chain$accept) == length(keep) + 1L) {
    chain$accept <- chain$accept[keep]
  }
  chain
}


sp_mcmc_unique_path <- function(chain) {
  keep <- !duplicated(as.data.frame(chain$X))
  idx <- which(keep)
  list(
    X = chain$X[idx, , drop = FALSE],
    D = if (is.null(chain$D)) NULL else chain$D[idx, , drop = FALSE],
    idx = idx
  )
}

sp_mcmc_append_k0 <- function(kernel, X_curr, D_curr, K0_curr, x_new, d_new) {
  x_new <- matrix(as.numeric(x_new), 1L, ncol(X_curr))
  d_new <- matrix(as.numeric(d_new), 1L, ncol(X_curr))
  new_row <- as.numeric(.k0_matrix(kernel, x_new, X_curr, d_new, D_curr))
  new_diag <- as.numeric(.k0_diag(kernel, x_new, d_new))
  rbind(cbind(K0_curr, new_row), c(new_row, new_diag))
}


sp_mcmc_eval_counts <- function(chain, candidate_score_eval = 0L,
                                fallback_m = NULL) {
  candidate_score_eval <- as.integer(candidate_score_eval)[1L]
  if (is.na(candidate_score_eval)) candidate_score_eval <- 0L
  out <- c(log_p = NA_integer_, score = NA_integer_,
           candidate_score = candidate_score_eval,
           transition_total = NA_integer_, total = NA_integer_)

  if (!is.null(chain$counts)) {
    raw <- unlist(chain$counts)
    raw <- raw[is.finite(raw)]
    raw_names <- names(raw)
    if ("log_p" %in% raw_names) out["log_p"] <- as.integer(raw[["log_p"]])
    if ("score" %in% raw_names) out["score"] <- as.integer(raw[["score"]])
    transition_total <- if ("total" %in% raw_names) {
      as.integer(raw[["total"]])
    } else {
      sum(as.integer(raw), na.rm = TRUE)
    }
  } else if (!is.null(chain$n_eval)) {
    transition_total <- as.integer(chain$n_eval)
  } else if (!is.null(fallback_m)) {
    transition_total <- as.integer(fallback_m)
  } else {
    transition_total <- 0L
  }

  out["transition_total"] <- transition_total[1L]
  out["total"] <- out["transition_total"] + candidate_score_eval
  out
}


# ---- MCMC kernels -----------------------------------------------------------

#' Run a Metropolis-adjusted Langevin chain
#'
#' Runs the MALA transition used as one of the short candidate-chain kernels in
#' SP-MCMC. MALA uses the target score to drift proposals toward high-density
#' regions and then applies a Metropolis correction.
#'
#' @details
#' From the current state `x`, MALA proposes a point near
#' \deqn{x + (h / 2) s_p(x) Sigma^{-1}.}
#' More explicitly, with the row-vector convention used by the code,
#' \deqn{y = x + (h / 2) s_p(x) Sigma^{-1} + \sqrt{h} z,
#'       \quad z \sim N(0, Sigma).}
#' Thus the proposal noise has covariance `h * Sigma`. The acceptance step
#' compares the target log density and the two proposal densities, so accepted
#' states leave the distribution described by `log_p` invariant. The returned
#' score matrix `D` is included because SP-MCMC can reuse these scores when
#' scoring the candidate path.
#'
#' When `Sigma` is the identity matrix this reduces to the basic MALA proposal
#' written in the SP-MCMC appendix,
#' \eqn{y = x + (h/2)\nabla\log p(x) + \sqrt{h}z}. The general `Sigma` argument
#' is the matrix used by this implementation's proposal calculation; it is
#' separate from the `M` or `precon` matrix that may be stored inside an IMQ
#' Stein kernel for computing `k0`.
#'
#' MALA is a transition kernel, not a full sampler interface. It is exported so
#' SP-MCMC users can inspect or replace the candidate-chain step. [sp_mcmc()]
#' calls it when `mcmc = "mala"` and then passes its chain output to
#' [sp_mcmc_eval_candidates()].
#'
#' @param log_p Function returning log density values.
#' @param score_function Function returning score values.
#' @param x0 Initial state vector.
#' @param h Positive step size.
#' @param Sigma Symmetric positive-definite proposal covariance or
#'   preconditioning matrix.
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
  if (!is.function(log_p)) stop("log_p must be a function")
  if (!is.function(score_function)) stop("score_function must be a function")
  if (!is.numeric(x0) || length(x0) < 1L || any(!is.finite(x0))) {
    stop("x0 must be a finite numeric vector")
  }
  x0 <- as.numeric(x0)
  d <- length(x0)
  if (!is.numeric(h) || length(h) != 1L || !is.finite(h) || h <= 0) {
    stop("h must be a positive scalar for MALA")
  }
  if (is.null(Sigma)) Sigma <- diag(d) else Sigma <- as.matrix(Sigma)
  if (!is.numeric(Sigma) || any(!is.finite(Sigma)) ||
    nrow(Sigma) != d || ncol(Sigma) != d) stop("Sigma must be a finite d x d matrix")
  sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(Sigma)))
  if (max(abs(Sigma - t(Sigma))) > sym_tol) {
    stop("Sigma must be symmetric positive definite")
  }
  Sigma <- (Sigma + t(Sigma)) / 2

  m_iter <- validate_positive_integer(m_iter, "m_iter")

  U_sigma <- tryCatch(chol(Sigma), error = function(e) {
    stop("Sigma must be symmetric positive definite", call. = FALSE)
  })
  Sigma_inv <- chol2inv(U_sigma)
  U <- sqrt(h) * U_sigma

  X <- matrix(0, m_iter, d)
  D <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L
  score_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  D[1L, ] <- validate_scores(score_function, matrix(x0, 1L, d))[1L, ]
  score_eval <- score_eval + 1L
  lp0 <- log_p(matrix(x0, 1L, d))
  if (!is.numeric(lp0) || length(lp0) != 1L || !is.finite(lp0)) {
    stop("log_p must return one finite value at x0")
  }
  lp[1L] <- as.numeric(lp0)
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      x_c <- X[i - 1L, ]
      d_c <- D[i - 1L, ]
      lp_c <- lp[i - 1L]

      m_x <- x_c + (h / 2) * as.numeric(d_c %*% Sigma_inv)
      y <- m_x + as.numeric(stats::rnorm(d) %*% U)

      lp_y <- log_p(matrix(y, 1L, d))
      if (!is.numeric(lp_y) || length(lp_y) != 1L || is.na(lp_y) || lp_y == Inf) {
        stop("log_p must return one numeric value for each proposal")
      }
      lp_y <- as.numeric(lp_y)
      log_p_eval <- log_p_eval + 1L
      if (!is.finite(lp_y)) {
        X[i, ] <- x_c; D[i, ] <- d_c; lp[i] <- lp_c
        next
      }

      d_y <- as.numeric(validate_scores(score_function, matrix(y, 1L, d))[1L, ])
      score_eval <- score_eval + 1L
      m_y <- y + (h / 2) * as.numeric(d_y %*% Sigma_inv)

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
#' \deqn{y = x + z, \quad z \sim N(0, S).}
#' The move is accepted with probability `min(1, exp(log_p(y) - log_p(x)))`.
#' Unlike MALA, this transition does not use the score to propose moves, so any
#' candidate scores needed by SP-MCMC are computed later by
#' [sp_mcmc_eval_candidates()].
#'
#' This is the lower-level random-walk Metropolis kernel. [grw()] wraps it to
#' match the transition signature used by [sp_mcmc()], where every transition
#' receives both `log_p` and `score_function`.
#'
#' The first row of the returned chain is `x0`. Subsequent rows are accepted
#' proposals or repeats of the previous state after rejection. The acceptance
#' indicators therefore describe whether each transition moved, not whether a
#' row is unique. [sp_mcmc()] removes duplicate candidate states later before
#' scoring the path with the Stein Points objective.
#'
#' @param log_p Function returning log density values.
#' @param x0 Initial state vector.
#' @param S Symmetric positive-definite Gaussian proposal covariance matrix.
#' @param m_iter Number of returned chain rows, including the initial state in
#'   row 1. The function therefore makes `m_iter - 1` Markov transitions.
#'
#' @return
#' A list with `X` (chain states), `log_p` (log density values), `accept` (0/1
#' acceptance indicators), and evaluation-count fields. It does not return
#' scores because the random-walk proposal does not use them. The evaluation
#' counts are returned so [sp_mcmc()] can report how much of the total budget
#' was spent proposing the candidate path before Stein scoring begins.
#' @examples
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' grwmetrop(log_p, x0 = 0, S = matrix(0.1), m_iter = 3)
#' @export
grwmetrop <- function(log_p, x0, S, m_iter) {
  if (!is.function(log_p)) stop("log_p must be a function")
  if (!is.numeric(x0) || length(x0) < 1L || any(!is.finite(x0))) {
    stop("x0 must be a finite numeric vector")
  }
  x0 <- as.numeric(x0)
  d <- length(x0)
  S <- as.matrix(S)
  if (!is.numeric(S) || any(!is.finite(S)) ||
    nrow(S) != d || ncol(S) != d) stop("S must be a finite d x d matrix")
  sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(S)))
  if (max(abs(S - t(S))) > sym_tol) stop("S must be symmetric positive definite")
  S <- (S + t(S)) / 2

  m_iter <- validate_positive_integer(m_iter, "m_iter")

  U <- tryCatch(chol(S), error = function(e) {
    stop("S must be symmetric positive definite", call. = FALSE)
  })
  X <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  lp0 <- log_p(matrix(x0, 1L, d))
  if (!is.numeric(lp0) || length(lp0) != 1L || !is.finite(lp0)) {
    stop("log_p must return one finite value at x0")
  }
  lp[1L] <- as.numeric(lp0)
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      X[i, ] <- X[i - 1L, ]
      y <- X[i - 1L, ] + as.numeric(stats::rnorm(d) %*% U)
      lp_y <- log_p(matrix(y, 1L, d))
      if (!is.numeric(lp_y) || length(lp_y) != 1L || is.na(lp_y) || lp_y == Inf) {
        stop("log_p must return one numeric value for each proposal")
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
    X = X, log_p = lp, accept = ac, n_eval = log_p_eval,
    counts = list(log_p = log_p_eval, score = 0L, total = log_p_eval)
  )
}


#' Run the Gaussian random-walk transition used by SP-MCMC
#'
#' This is a convenience wrapper around [grwmetrop()] using proposal covariance
#' `h * Sigma`, with `h = 1` when omitted.
#'
#' @details
#' The SP-MCMC main loop passes the log density, score function, current state,
#' step size, proposal matrix, and chain length to every transition. The
#' argument `score_function` is therefore present only for interface
#' compatibility with [mala()] and custom transition functions. The random-walk
#' proposal itself is \eqn{N(x, h Sigma)} and does not use the score.
#'
#' Use this wrapper, rather than [grwmetrop()], when supplying a custom
#' transition-like function to [sp_mcmc()] or when comparing GRW and MALA under
#' the same call signature.
#'
#' @param log_p Function returning log density values.
#' @param score_function Unused score function argument kept for transition
#'   interface compatibility.
#' @param x0 Initial state vector.
#' @param h Optional positive step-size multiplier.
#' @param Sigma Symmetric positive-definite proposal covariance or
#'   preconditioning matrix.
#' @param m_iter Number of returned chain rows, including the initial state in
#'   row 1. The function therefore makes `m_iter - 1` Markov transitions.
#'
#' @return
#' Output from [grwmetrop()], with chain states, log densities, acceptance
#' indicators, and evaluation counts. The score matrix is absent by design;
#' SP-MCMC computes candidate scores afterward when needed.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' grw(log_p, score, x0 = 0, h = 0.1, m_iter = 3)
#' @export
grw <- function(log_p, score_function, x0, h = NULL, Sigma = NULL, m_iter) {
  d <- length(x0)
  scale <- if (is.null(h)) 1 else h
  if (!is.numeric(scale) || length(scale) != 1L || !is.finite(scale) || scale <= 0) {
    stop("h must be a positive scalar for GRW")
  }
  base_Sigma <- if (is.null(Sigma)) diag(d) else as.matrix(Sigma)
  if (nrow(base_Sigma) != d || ncol(base_Sigma) != d) stop("Sigma must be d x d")
  S <- scale * base_Sigma
  out <- grwmetrop(log_p, x0, S, m_iter)
  out$D <- NULL
  out
}


#' Alias for the Gaussian random-walk transition
#'
#' Calls [grw()] with the same arguments. It is provided because the SP-MCMC
#' paper refers to the Gaussian random-walk Metropolis transition as RWM.
#'
#' @details
#' This function is intentionally thin. The package's implementation name for
#' the transition is [grw()], short for Gaussian random walk, while the paper
#' commonly uses RWM, short for random-walk Metropolis. Keeping both exported
#' names avoids forcing users to translate between the paper notation and the
#' package notation.
#'
#' No additional computation happens here: `rwm()` forwards the arguments to
#' [grw()], which in turn calls [grwmetrop()] with proposal covariance
#' `h * Sigma`. The return object therefore has exactly the same fields as
#' [grw()].
#'
#' @param log_p Function returning log density values.
#' @param score_function Unused score function argument kept for transition
#'   interface compatibility.
#' @param x0 Initial state vector.
#' @param h Optional positive step-size multiplier.
#' @param Sigma Symmetric positive-definite proposal covariance or
#'   preconditioning matrix.
#' @param m_iter Number of returned chain rows, including the initial state in
#'   row 1. The function therefore makes `m_iter - 1` Markov transitions.
#'
#' @return
#' Output from [grw()]: chain states, log-density values, acceptance indicators,
#' and evaluation counts. The alias exists for naming compatibility with the
#' SP-MCMC paper; it does not change the transition or add fields.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' rwm(log_p, score, x0 = 0, h = 0.1, m_iter = 3)
#' @export
rwm <- function(log_p, score_function, x0, h = NULL, Sigma = NULL, m_iter) {
  grw(log_p, score_function, x0, h, Sigma, m_iter)
}


# ---- Helpers: start-point criteria -----------------------------------------

.crit_infl <- function(K0_cache) {
  which.min(rowSums(K0_cache) + colSums(K0_cache) - diag(K0_cache))
}


#' @export
print.sp_mcmc <- function(x, ...) {
  cat(sprintf("SP-MCMC  [%s + %s]  n=%d  d=%d  KSD=%.4g  n_eval=%d\n",
              x$mcmc, x$criterion, nrow(x$X), ncol(x$X),
              x$ksd[length(x$ksd)], x$cum_n_eval[length(x$cum_n_eval)]))
  invisible(x)
}
