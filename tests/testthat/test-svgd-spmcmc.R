test_that("SVGD rebuilds its kernel each step without mutating it", {
  theta <- matrix(c(-2, -0.5, 1, 4), ncol = 1)
  med <- stats::median(as.numeric(stats::dist(theta)))
  expected <- med / sqrt(2 * log(nrow(theta)))

  expect_equal(steinsampling:::.svgd_median_bandwidth(theta), expected)
  expect_equal(steinsampling:::.svgd_median_bandwidth(3 * theta), 3 * expected)
  expect_error(
    steinsampling:::.svgd_median_bandwidth(matrix(0, nrow = 2, ncol = 1)),
    "fixed h > 0"
  )

  x0 <- matrix(c(-1, 0, 2), ncol = 1)
  precon <- matrix(2, nrow = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", precon = precon)
  at_median <- kernel
  at_median$scale2 <- steinsampling:::.svgd_median_bandwidth(x0)^2

  out <- steinsampling:::.compute_svgd_kernel(x0, kernel)
  traced <- svgd(x0, normal_score, n_iter = 2, step_size = 0.01,
                 trace_iters = 1:2)

  expect_equal(out$kernel_matrix, eval_kernel(at_median, x0))
  expect_equal(dim(out$repulsion), c(3L, 1L))
  # The lazy kernel is left unscaled, so the next step re-derives its bandwidth.
  expect_equal(kernel$precon, precon)
  expect_true(is.na(kernel$scale2))
  expect_identical(class(traced), "svgd")
  expect_equal(dim(traced$X), dim(x0))
  expect_length(traced$trace, 2L)

  # Zero iterations is a no-op, and a custom direction must keep the shape.
  fixed <- stein_kernel(type = "gaussian_rbf", h = 1)
  expect_equal(svgd(x0, normal_score, fixed, n_iter = 0)$X, x0)
  expect_error(
    svgd(x0, normal_score, fixed, n_iter = 1, adj_grad = function(...) 1),
    "same dimensions"
  )
  expect_error(svgd(x0, normal_score, fixed, n_iter = 1, alpha = 1), "alpha")
})

test_that("MALA and RWM share a chain contract but not their score cost", {
  set.seed(10)
  mala_res <- mala(normal_log_p, normal_score, x0 = 0, h = 0.1, m_iter = 3)
  rwm_res <- rwm(normal_log_p, x0 = 0, h = 0.1, m_iter = 3)

  expect_equal(dim(mala_res$X), c(3L, 1L))
  expect_equal(dim(rwm_res$X), c(3L, 1L))
  expect_identical(names(mala_res), names(rwm_res))
  # RWM never touches the score, so its slot is present but empty.
  expect_equal(dim(mala_res$D), c(3L, 1L))
  expect_null(rwm_res$D)
  expect_gt(mala_res$counts$score, 0L)
  expect_identical(rwm_res$counts$score, 0L)
  expect_error(rwm(normal_log_p, x0 = 0, m_iter = 2), "h")
})

test_that("sp_mcmc accepts custom start rules and scores candidate sets", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  farthest <- function(state, w) which.max(rowSums(state$X^2) * w)
  args <- list(normal_score, normal_log_p, kernel, n_points = 5, d = 1,
               m_seq = 3, h = 0.3, x_init = 0, seed = 11)
  run <- function(...) do.call(sp_mcmc, c(args, list(...)))

  # Rule parameters live in the closure; the list preserves its display name.
  w <- 2
  rule <- list(label = "farthest", select = function(state) farthest(state, w))
  expect_equal(run(criterion = rule)$criterion, "farthest")
  expect_equal(run(criterion = list(select = function(state) 1L))$selected_index,
               c(NA_integer_, rep(1L, 4L)))
  expect_error(run(criterion = list(select = function(state) 1.5)), "integer index")
  expect_error(run(criterion = list(select = function(state) 3)), "one-based index")

  cand_X <- matrix(c(-0.5, 0.5), ncol = 1)
  score_it <- function(kernel_obj = kernel, ...) sp_mcmc_eval_candidates(
    kernel_obj, normal_score, X_curr = matrix(0, ncol = 1),
    D_curr = matrix(0, ncol = 1), cand_X = cand_X, ...
  )
  candidates <- score_it()
  reused <- score_it(cand_D = normal_score(cand_X))

  expect_equal(length(candidates$objective_values), 2L)
  expect_equal(dim(candidates$scores), c(2L, 1L))
  expect_equal(candidates$score_evaluations, 2L)
  # Supplying `cand_D` gives the same objective at no score cost.
  expect_equal(reused$objective_values, candidates$objective_values)
  expect_equal(reused$score_evaluations, 0L)
  expect_error(score_it(stein_kernel(type = "gaussian_rbf")), "fixed h")
})

test_that("sp_mcmc returns a stateful chain and validates its controls", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  bad <- function(...) sp_mcmc(
    normal_score, normal_log_p, kernel, d = 1, h = 0.1, x_init = 0, ...
  )

  run <- sp_mcmc(normal_score, normal_log_p, kernel, n_points = 2, d = 1,
                 mcmc = "rwm", criterion = "last", m_seq = 2, h = 0.1,
                 x_init = 0, seed = 1)
  expect_identical(class(run), "sp_mcmc")
  expect_equal(dim(run$X), c(2L, 1L))
  expect_equal(dim(run$D), c(2L, 1L))
  expect_gt(sum(run$n_eval), 0)

  scheduled <- sp_mcmc(
    normal_score, normal_log_p, kernel, n_points = 3, d = 1,
    m_seq = c(1, 1), h = 0.1, x_init = 0
  )
  expect_identical(scheduled$m_seq, c(1L, 1L))
  expect_true(is.na(summary(scheduled)$accept_rate))

  bad_calls <- list(
    list(quote(bad(n_points = 2.5, m_seq = 2)), "positive integer"),
    list(quote(bad(n_points = 2, m_seq = 1.5)), "positive integer"),
    # Length n_points is no longer a second spelling of the same schedule.
    list(quote(bad(n_points = 3, m_seq = c(2, 2, 2))),
         "m_seq must be scalar or length n_points - 1"),
    list(quote(bad(n_points = 2, m_seq = 2, mcmc = "grw")), "should be one of"),
    list(quote(sp_mcmc(normal_score, normal_log_p, kernel, n_points = 1, d = 1,
                       m_seq = 1, h = "bad", x_init = 0)), "positive scalar"),
    list(quote(sp_mcmc(normal_score, normal_log_p,
                       stein_kernel(type = "gaussian_rbf"), n_points = 2, d = 1,
                       m_seq = 2, h = 0.1, x_init = 0)), "fixed h")
  )
  for (cs in bad_calls) expect_error(eval(cs[[1]]), cs[[2]])
})

test_that("SP-MCMC walks the paper's m_seq-state path and checks the contract", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  requested_rows <- NA_integer_
  counts0 <- list(log_p = 0L, score = 0L, total = 0L)
  moving <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    requested_rows <<- m_iter
    X <- matrix(c(x0, 1, 3), ncol = 1)
    list(X = X, D = score_function(X), accept = c(0L, 1L, 0L),
         counts = list(log_p = 3L, score = 3L, total = 6L))
  }
  # Duplicate states collapse, so only the distinct rows are scored.
  repeating <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    list(X = matrix(c(x0, x0, 2), ncol = 1), D = NULL, accept = c(0L, 0L, 1L),
         counts = list(log_p = 3L, score = 0L, total = 3L))
  }
  chain <- function(X, ...) function(log_p, score_function, x0, h, Sigma, m_iter) {
    c(list(X = X(x0), D = NULL), list(...))
  }
  spm <- function(fn, m_seq = 3) sp_mcmc(
    normal_score, normal_log_p, kernel, n_points = 2, d = 1,
    m_seq = m_seq, h = 0.1, x_init = 0, transition_fn = fn
  )

  run <- spm(moving)
  expect_equal(requested_rows, 3L)
  expect_equal(run$X[2, 1], 1)
  expect_equal(run$chain_d2_max[2], 9)
  expect_equal(run$chain_d2_last[2], 9)
  expect_equal(run$accept_rate[2], 0.5)
  repeated_counts <- spm(repeating)$counts
  expect_equal(
    unname(repeated_counts[2, c("candidate_score", "transition_total",
                                "total")]),
    c(2L, 3L, 5L)
  )

  # A one-state path makes no transition, so the only candidate is the start.
  # The run completes, but it must say that it selected nothing.
  one_move <- chain(function(x0) matrix(x0, ncol = 1), accept = 0L,
                    counts = counts0)
  never_moved <- spm(one_move, 1)
  expect_equal(never_moved$X[2, 1], 0)
  # Repeats are legal, so they are reported as a diagnostic, not a warning.
  expect_identical(never_moved$n_repeated, 1L)
  expect_true(is.na(never_moved$accept_rate[2]))

  expect_error(spm(one_move, 2), "exactly 2 rows")
  expect_error(
    spm(chain(function(x0) matrix(c(x0 + 1, x0 + 2), ncol = 1),
              counts = counts0), 2),
    "initial state in row 1"
  )
  expect_error(
    spm(chain(function(x0) matrix(c(x0, x0 + 1), ncol = 1), n_eval = 2L), 2),
    "return counts"
  )
})

test_that("sp_mcmc reports the transition it ran and its evaluation totals", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  fit <- sp_mcmc(normal_score, normal_log_p, kernel, n_points = 4, d = 1,
                 m_seq = 2, h = 0.5, x_init = 0, transition_fn = mala)
  s <- summary(fit)

  # The display name follows the transition actually used; `mcmc` still names
  # the built-in rule handed to `proposal_fn`.
  expect_identical(fit$transition, "mala")
  expect_identical(fit$mcmc, "rwm")
  expect_match(paste(capture.output(print(fit)), collapse = "\n"),
               "[mala + last]", fixed = TRUE)

  # `n_eval` sums the evaluation categories, so the label must say so.
  expect_identical(s$evaluation_label, "total evaluations")
  expect_match(paste(capture.output(print(s)), collapse = "\n"),
               "repeated points:", fixed = TRUE)
  expect_match(paste(capture.output(print(s)), collapse = "\n"),
               "total evaluations", fixed = TRUE)
  expect_identical(unname(s$n_eval_total), unname(sum(fit$counts[, "total"])))
  expect_named(s$counts, c("log_p", "score", "candidate_score"))
  expect_null(s$n_distinct_starts)

  # Different path lengths contribute different numbers of proposals.
  fit$m_seq <- c(2L, 10L, 3L)
  fit$accept_rate <- c(NA_real_, 1, 0, NA_real_)
  expect_equal(summary(fit)$accept_rate, 0.1)

  # A path with no transitions has no acceptance rate to report.
  stuck <- sp_mcmc(normal_score, normal_log_p, kernel, n_points = 4, d = 1,
                   m_seq = 1, h = 1e-12, x_init = 0)
  expect_equal(nrow(unique(stuck$X)), 1L)
  expect_true(is.na(summary(stuck)$accept_rate))
  expect_null(summary(stuck)$n_distinct_starts)
})
