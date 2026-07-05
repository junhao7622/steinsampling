# Stein Point MCMC (Chen, Barp, Briol, Gorham, Girolami, Mackey, Oates, ICML 2019).

#' Stein Point MCMC using the greedy/MCMC selection loop from Section 3.1.
#'
#' `x_init` is the fixed initial point `x_1`.  For `j = 2, ..., n`, a
#' `P`-invariant Markov chain is initialized at the selected current point and
#' the next point is chosen from the realized path.
#'
#' @param score_function function(X) returning grad log p, X is n x d.
#' @param log_p          function(X) returning log p, length n.
#' @param kernel         a [stein_kernel()] object, a function
#'   `kernel(X, S_X, Y, S_Y)`, or a list with `k0_matrix`.
#' @param n_points       integer, total points to generate.
#' @param d              integer, state dimension.
#' @param mcmc           "grw" (Gaussian random-walk Metropolis), "mala",
#'   or alias "rwm".
#' @param criterion      start-point criterion: "last", "rand", "infl", or a
#'   function taking an `sp_mcmc_state` object and returning one selected index.
#' @param m_seq          chain length per greedy index; scalar, length n_points,
#'   or length n_points - 1 for the updates j = 2, ..., n_points.
#' @param h              MALA/GRW step size. For both built-in kernels, the
#'   proposal covariance is `h * Sigma`; if `h` is NULL for GRW it defaults to 1.
#' @param Sigma          covariance/preconditioner matrix Sigma in Appendix A.5.
#'   If NULL, the identity matrix is used by the built-in kernels.
#' @param x_init         Fixed first point x_1.
#' @param seed           optional RNG seed.
#' @param transition_fn  optional Markov transition function with signature
#'   `(log_p, score_function, x0, h, Sigma, m_iter)`.  Defaults to the built-in
#'   kernel selected by `mcmc`.
#' @param proposal_fn    optional function returning per-step proposal settings
#'   with signature `(j, X_curr, h, Sigma, mcmc)`.  Return a list containing
#'   replacement `h` and/or `Sigma`.
#' @param n_eval_fn      optional function returning the per-step evaluation
#'   count with signature `(chain, cand_X, cand_D, m_j, mcmc, j, X_curr)`.
#' @param criterion_args optional list of extra arguments passed to a custom
#'   `criterion` function.
#'
#' @return list with `X`, `D`, `ksd`, per-step evaluation counts, chain jump
#'   diagnostics, and S3 class c("sp_mcmc", "stein_points").
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' sp_mcmc(score, log_p, kernel, n_points = 2, d = 1, m_seq = 2, x_init = 0)
#' @export
sp_mcmc <- function(score_function, log_p, kernel, n_points, d,
                    mcmc      = c("grw", "mala", "rwm"),
                    criterion = c("last", "rand", "infl"),
                    m_seq, h = NULL, Sigma = NULL,
                    x_init, seed = NULL,
                    transition_fn = NULL, proposal_fn = NULL,
                    n_eval_fn = NULL,
                    criterion_args = list()) {

  mcmc <- match.arg(mcmc)
  if (identical(mcmc, "rwm")) mcmc <- "grw"
  criterion_obj <- sp_mcmc_criterion(criterion, criterion_args)

  n_points <- as.integer(n_points)
  if (!is.finite(n_points) || n_points < 1L) stop("n_points must be positive")
  d <- as.integer(d)
  if (!is.finite(d) || d < 1L) stop("d must be positive")

  x_init <- as.numeric(x_init)
  if (length(x_init) != d || any(!is.finite(x_init))) {
    stop("x_init must be a finite length-d numeric vector")
  }

  m_seq <- as.integer(m_seq)
  if (length(m_seq) != 1L) {
    if (length(m_seq) == n_points - 1L) {
      m_seq <- c(NA_integer_, m_seq)
    } else if (length(m_seq) < n_points) {
      stop("m_seq must be scalar, length n_points, or length n_points - 1")
    }
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
      if (!is.finite(m_j) || m_j < 2L) {
        stop("m_seq entries must be integers >= 2 because row 1 is the chain start")
      }

      proposal <- get_proposal(j = j, X_curr = X_curr, h = h,
                               Sigma = Sigma, mcmc = mcmc)
      if (is.null(proposal)) proposal <- list()
      if (!is.list(proposal)) proposal <- list(h = proposal)
      h_j <- if (is.null(proposal$h)) h else proposal$h
      Sigma_j <- if (is.null(proposal$Sigma)) Sigma else proposal$Sigma

      chain <- chain_fn(log_p, score_function, start, h_j, Sigma_j, m_j)
      chain <- sp_mcmc_validate_chain(chain, d)
      path <- sp_mcmc_unique_path(chain)

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

      d2 <- rowSums((sweep(chain$X, 2L, chain$X[1L, ], "-"))^2)
      chain_d2_max[j] <- max(d2)
      chain_d2_selected[j] <- sum((chain$X[1L, ] - x_min)^2)
      chain_d2_last[j] <- d2[length(d2)]
      accept_rate[j] <- if (!is.null(chain$accept) && length(chain$accept) >= 2L) {
        mean(chain$accept[-1L] == 1L)
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
        override <- as.integer(override)
        if (length(override) != 1L || !is.finite(override) || override < 0L) {
          stop("n_eval_fn must return one nonnegative finite integer")
        }
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
    chain_jumps = chain_d2_last,
    accept_rate = accept_rate
  )
  class(res) <- unique(c("sp_mcmc", "stein_points", class(res)))
  res
}


#' SP-MCMC state object passed to custom start criteria.
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
#' @return A list containing the current points `X`, scores `D`, Stein kernel
#'   matrix `K0`, running diagnostics, and method configuration.
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


#' Build an SP-MCMC start-point criterion.
#'
#' `criterion` may be one of `"last"`, `"rand"`, `"infl"`, or a function that
#' accepts an `sp_mcmc_state` object and returns a one-based index into
#' `state$X`.
#'
#' @param criterion Criterion name, function, or criterion object.
#' @param criterion_args Extra arguments passed to a custom criterion function.
#'
#' @return Criterion object with `label` and `select` fields.
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


#' Select and validate an SP-MCMC start index.
#'
#' @param criterion_obj Criterion object from `sp_mcmc_criterion()`.
#' @param state Current SP-MCMC state.
#'
#' @return One-based selected index.
#' @examples
#' state <- sp_mcmc_state(j = 2, X = matrix(0, ncol = 1),
#'                        D = matrix(0, ncol = 1), K0 = matrix(1))
#' sp_mcmc_select_start(sp_mcmc_criterion("last"), state)
#' @export
sp_mcmc_select_start <- function(criterion_obj, state) {
  idx <- criterion_obj$select(state)
  if (is.list(idx) && !is.null(idx$index)) idx <- idx$index
  idx <- as.integer(idx)
  if (length(idx) != 1L || !is.finite(idx) ||
      idx < 1L || idx > nrow(state$X)) {
    stop("criterion must return one valid one-based index into the current point set")
  }
  idx
}


#' Evaluate the greedy SP-MCMC candidate objective.
#'
#' Reuses the Greedy Stein Points increment
#' `k0(x, x) + 2 * sum_i k0(x_i, x)` while allowing MCMC transitions to supply
#' cached candidate scores.
#'
#' @param kernel Stein kernel object or compatible custom kernel.
#' @param score_function Function returning scores for candidate rows.
#' @param X_curr Current selected point matrix.
#' @param D_curr Current score matrix.
#' @param cand_X Candidate point matrix.
#' @param cand_D Optional candidate score matrix.
#'
#' @return Candidate objective values and scores.
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


# Lower-level optimizer hook. Most callers should use
# sp_mcmc_run() or sp_mcmc().
fmin_sp_mcmc <- function(mcmc, criterion, m_seq,
                         log_p, score_function,
                         h, Sigma,
                         kernel, d, tracker,
                         transition_fn = NULL, proposal_fn = NULL,
                         n_eval_fn = NULL) {
  if (identical(mcmc, "rwm")) mcmc <- "grw"
  criterion_obj <- sp_mcmc_criterion(criterion)
  function(f, X_curr) {
    j <- nrow(X_curr) + 1L
    if (j == 1L) stop("SP-MCMC requires a fixed first point x_init")
    D_curr <- validate_scores(score_function, X_curr)
    K0 <- stein_kernel_matrix(kernel, X_curr, D_curr)
    state <- sp_mcmc_state(j = j, X = X_curr, D = D_curr, K0 = K0,
                           kernel = kernel, mcmc = mcmc,
                           criterion = criterion_obj$label)
    i_star <- sp_mcmc_select_start(criterion_obj, state)
    start <- as.numeric(X_curr[i_star, ])
    chain_fn <- if (is.null(transition_fn)) switch(mcmc, grw = grw, mala = mala) else transition_fn
    get_m <- if (length(m_seq) == 1L) function(j) m_seq else function(j) m_seq[j]
    m_j <- get_m(j)
    proposal <- if (is.null(proposal_fn)) {
      list(h = h, Sigma = Sigma)
    } else {
      proposal_fn(j = j, X_curr = X_curr, h = h, Sigma = Sigma, mcmc = mcmc)
    }
    if (is.null(proposal)) proposal <- list()
    if (!is.list(proposal)) proposal <- list(h = proposal)
    h_j <- if (is.null(proposal$h)) h else proposal$h
    Sigma_j <- if (is.null(proposal$Sigma)) Sigma else proposal$Sigma
    chain <- sp_mcmc_validate_chain(
      chain_fn(log_p, score_function, start, h_j, Sigma_j, m_j), d
    )
    path <- sp_mcmc_unique_path(chain)
    obj <- if (is.null(path$D)) f(path$X) else f(path$X, path$D)
    i_min <- which.min(obj$f_vec)
    cand_D <- obj$D_new
    n_eval <- if (is.null(n_eval_fn)) {
      as.integer(sp_mcmc_eval_counts(chain, if (is.null(path$D)) nrow(path$X) else 0L,
                                     m_j)["total"])
    } else {
      as.integer(n_eval_fn(chain = chain, cand_X = path$X, cand_D = cand_D,
                           m_j = m_j, mcmc = mcmc, j = j, X_curr = X_curr))
    }
    list(x_min = as.numeric(path$X[i_min, ]),
         d_min = as.numeric(cand_D[i_min, ]),
         f_min = as.numeric(obj$f_vec[i_min]),
         n_eval = n_eval)
  }
}


.sp_mcmc_count_matrix <- function(n_points) {
  out <- matrix(0L, n_points, 5L)
  colnames(out) <- c("log_p", "score", "candidate_score",
                     "transition_total", "total")
  out
}


sp_mcmc_validate_chain <- function(chain, d) {
  if (!is.list(chain)) chain <- list(X = chain)
  Chn <- chain$X
  if (is.null(Chn)) stop("transition function must return list element X")
  Chn <- as.matrix(Chn)
  if (ncol(Chn) != d) stop("transition function returned X with wrong dimension")
  chain$X <- Chn

  if (!is.null(chain$D)) {
    chain_D <- as.matrix(chain$D)
    if (!identical(dim(chain_D), dim(Chn))) {
      stop("transition function returned D with dimension different from X")
    }
    chain$D <- chain_D
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

#' Metropolis-adjusted Langevin algorithm.
#'
#' Appendix A.5 parameterization: proposal covariance is
#' `h * Sigma` and the drift is `(h / 2) * Sigma^{-1} grad log p(x)`.
#'
#' @param log_p Function returning log density values.
#' @param score_function Function returning score values.
#' @param x0 Initial state vector.
#' @param h Positive step size.
#' @param Sigma Proposal covariance/preconditioner matrix.
#' @param m_iter Number of MCMC iterations.
#'
#' @return List containing chain states, scores, log densities, acceptance
#'   indicators, and evaluation counts.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' mala(log_p, score, x0 = 0, h = 0.1, m_iter = 3)
#' @export
mala <- function(log_p, score_function, x0, h, Sigma = NULL, m_iter) {
  d <- length(x0)
  if (!is.numeric(h) || length(h) != 1L || !is.finite(h) || h <= 0) {
    stop("h must be a positive scalar for MALA")
  }
  if (is.null(Sigma)) Sigma <- diag(d) else Sigma <- as.matrix(Sigma)
  if (nrow(Sigma) != d || ncol(Sigma) != d) stop("Sigma must be d x d")

  m_iter <- as.integer(m_iter)
  if (!is.finite(m_iter) || m_iter < 1L) stop("m_iter must be positive")

  U_sigma <- chol(Sigma)
  Sigma_inv <- chol2inv(U_sigma)
  U <- sqrt(h) * U_sigma

  X <- matrix(0, m_iter, d)
  D <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L
  score_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  D[1L, ] <- as.numeric(score_function(matrix(x0, 1L, d)))
  score_eval <- score_eval + 1L
  lp[1L] <- as.numeric(log_p(matrix(x0, 1L, d)))
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      x_c <- X[i - 1L, ]
      d_c <- D[i - 1L, ]
      lp_c <- lp[i - 1L]

      m_x <- x_c + (h / 2) * as.numeric(d_c %*% Sigma_inv)
      y <- m_x + as.numeric(stats::rnorm(d) %*% U)

      lp_y <- as.numeric(log_p(matrix(y, 1L, d)))
      log_p_eval <- log_p_eval + 1L
      if (!is.finite(lp_y)) {
        X[i, ] <- x_c; D[i, ] <- d_c; lp[i] <- lp_c
        next
      }

      d_y <- as.numeric(score_function(matrix(y, 1L, d)))
      score_eval <- score_eval + 1L
      m_y <- y + (h / 2) * as.numeric(d_y %*% Sigma_inv)

      q_old_given_y <- sum(forwardsolve(t(U), x_c - m_y)^2)
      q_y_given_old <- sum(forwardsolve(t(U), y - m_x)^2)
      log_a <- (lp_y - lp_c) + 0.5 * (q_y_given_old - q_old_given_y)

      if (log_a >= 0 || log(stats::runif(1L)) < log_a) {
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


#' Gaussian random-walk Metropolis.
#'
#' Uses `S` as the Gaussian proposal covariance matrix.
#'
#' @param log_p Function returning log density values.
#' @param x0 Initial state vector.
#' @param S Gaussian proposal covariance matrix.
#' @param m_iter Number of MCMC iterations.
#'
#' @return List containing chain states, log densities, acceptance indicators,
#'   and evaluation counts.
#' @examples
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' grwmetrop(log_p, x0 = 0, S = matrix(0.1), m_iter = 3)
#' @export
grwmetrop <- function(log_p, x0, S, m_iter) {
  d <- length(x0)
  S <- as.matrix(S)
  if (nrow(S) != d || ncol(S) != d) stop("S must be d x d")

  m_iter <- as.integer(m_iter)
  if (!is.finite(m_iter) || m_iter < 1L) stop("m_iter must be positive")

  U <- chol(S)
  X <- matrix(0, m_iter, d)
  lp <- numeric(m_iter)
  ac <- integer(m_iter)
  log_p_eval <- 0L

  X[1L, ] <- as.numeric(x0)
  lp[1L] <- as.numeric(log_p(matrix(x0, 1L, d)))
  log_p_eval <- log_p_eval + 1L

  if (m_iter >= 2L) {
    for (i in 2:m_iter) {
      X[i, ] <- X[i - 1L, ]
      y <- X[i - 1L, ] + as.numeric(stats::rnorm(d) %*% U)
      lp_y <- as.numeric(log_p(matrix(y, 1L, d)))
      log_p_eval <- log_p_eval + 1L
      log_a <- lp_y - lp[i - 1L]
      if (is.finite(log_a) && (log_a >= 0 || log(stats::runif(1L)) < log_a)) {
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


#' Wrapper used by [sp_mcmc()] for Gaussian random-walk Metropolis.
#'
#' Appendix A.5 parameterization: proposal covariance is
#' `h * Sigma`, with `h = 1` when omitted.
#'
#' @param log_p Function returning log density values.
#' @param score_function Unused score function argument kept for transition
#'   interface compatibility.
#' @param x0 Initial state vector.
#' @param h Optional positive step-size multiplier.
#' @param Sigma Proposal covariance/preconditioner matrix.
#' @param m_iter Number of MCMC iterations.
#'
#' @return List containing chain states, log densities, acceptance indicators,
#'   and evaluation counts.
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


#' Alias for [grw()].
#'
#' @param log_p Function returning log density values.
#' @param score_function Unused score function argument kept for transition
#'   interface compatibility.
#' @param x0 Initial state vector.
#' @param h Optional positive step-size multiplier.
#' @param Sigma Proposal covariance/preconditioner matrix.
#' @param m_iter Number of MCMC iterations.
#'
#' @return Output from `grw()`.
#' @examples
#' score <- function(X) -as.matrix(X)
#' log_p <- function(X) -0.5 * rowSums(as.matrix(X)^2)
#' rwm(log_p, score, x0 = 0, h = 0.1, m_iter = 3)
#' @export
rwm <- function(log_p, score_function, x0, h = NULL, Sigma = NULL, m_iter) {
  grw(log_p, score_function, x0, h, Sigma, m_iter)
}


# ---- Helpers: start-point criteria -----------------------------------------

.crit_last <- function(K0_cache) nrow(K0_cache)
.crit_rand <- function(K0_cache) sample.int(nrow(K0_cache), 1L)

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
