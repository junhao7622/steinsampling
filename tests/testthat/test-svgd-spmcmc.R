test_that("SVGD kernel and updates return particle matrices", {
  x0 <- matrix(c(-0.5, 0.5), ncol = 1)
  kernel <- stein_kernel(type = "gaussian_rbf")
  kernel_eval <- steinsampling:::.compute_svgd_kernel(x0, kernel)
  updated <- svgd(x0, normal_score, n_iter = 2, step_size = 0.01)
  traced <- svgd(
    x0, normal_score, n_iter = 2, step_size = 0.01, trace_iters = 1:2
  )

  expect_equal(dim(kernel_eval$kernel_matrix), c(2L, 2L))
  expect_equal(dim(kernel_eval$repulsion), c(2L, 1L))
  expect_equal(dim(updated$X), c(2L, 1L))
  expect_equal(dim(traced$X), c(2L, 1L))
  expect_length(traced$trace, 2L)
  expect_identical(class(updated), "svgd")
})

test_that("SVGD validates iteration and scaling controls", {
  x0 <- matrix(c(-0.5, 0.5), ncol = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)

  expect_equal(svgd(x0, normal_score, kernel, n_iter = 0)$X, x0)
  expect_error(svgd(x0, normal_score, kernel, n_iter = 1.5), "nonnegative integer")
  expect_error(svgd(x0, normal_score, kernel, n_iter = 1, step_size = 0), "step_size")
  expect_error(svgd(x0, normal_score, kernel, n_iter = 1, alpha = 1), "alpha")
  expect_error(
    svgd(x0, normal_score, kernel, n_iter = 1, trace_iters = 2),
    "trace_iters"
  )
  expect_error(
    svgd(
      x0, normal_score, kernel, n_iter = 1,
      adj_grad = function(...) 1
    ),
    "same dimensions"
  )
})

test_that("SVGD uses the paper's exact dynamic median bandwidth", {
  theta <- matrix(c(-2, -0.5, 1, 4), ncol = 1)
  med <- stats::median(as.numeric(stats::dist(theta)))
  expected <- med / sqrt(2 * log(nrow(theta)))

  expect_equal(steinsampling:::.svgd_median_bandwidth(theta), expected)
  expect_equal(
    steinsampling:::.svgd_median_bandwidth(3 * theta),
    3 * expected
  )
  expect_error(
    steinsampling:::.svgd_median_bandwidth(matrix(0, nrow = 2, ncol = 1)),
    "fixed h > 0"
  )
})

test_that("SVGD dynamic RBF bandwidth preserves preconditioning", {
  theta <- matrix(c(-1, 0, 2), ncol = 1)
  precon <- matrix(2, nrow = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", precon = precon)
  h <- steinsampling:::.svgd_median_bandwidth(theta)
  expected_kernel <- kernel
  expected_kernel$h2 <- h^2

  out <- steinsampling:::.compute_svgd_kernel(theta, kernel)

  expect_equal(out$kernel_matrix, eval_kernel(expected_kernel, theta))
  expect_equal(kernel$precon, precon)
  expect_null(kernel$h2)
})

test_that("SVGD applies a custom direction adjustment", {
  x0 <- matrix(c(-0.5, 0.5), ncol = 1)
  adjusted <- svgd(
    x0,
    normal_score,
    kernel = stein_kernel(type = "gaussian_rbf", h = 1),
    n_iter = 1,
    step_size = 0.1,
    adj_grad = function(grad, ...) grad
  )

  expect_equal(dim(adjusted$X), dim(x0))
})

test_that("MCMC kernels return chain shapes and evaluation counts", {
  set.seed(10)
  mala_res <- mala(normal_log_p, normal_score, x0 = 0, h = 0.1, m_iter = 3)
  rwm_res <- rwm(normal_log_p, x0 = 0, h = 0.1, m_iter = 3)

  expect_equal(dim(mala_res$X), c(3L, 1L))
  expect_equal(dim(mala_res$D), c(3L, 1L))
  expect_equal(dim(rwm_res$X), c(3L, 1L))
  expect_gt(mala_res$n_eval, 0)
  expect_gt(rwm_res$n_eval, 0)
  expect_identical(names(mala_res), names(rwm_res))
  expect_true("D" %in% names(rwm_res))
  expect_null(rwm_res$D)
  expect_gt(mala_res$counts$score, 0L)
  expect_identical(rwm_res$counts$score, 0L)
  expect_error(mala(normal_log_p, normal_score, x0 = 0, m_iter = 2), "h")
  expect_error(rwm(normal_log_p, x0 = 0, m_iter = 2), "h")
})

test_that("SP-MCMC helpers select starts and evaluate candidates", {
  set.seed(11)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  state <- sp_mcmc_state(
    j = 2,
    X = matrix(c(0, 0.5), ncol = 1),
    D = normal_score(matrix(c(0, 0.5), ncol = 1)),
    K0 = diag(2)
  )
  criterion <- sp_mcmc_criterion("last")
  start_idx <- sp_mcmc_select_start(criterion, state)
  candidates <- sp_mcmc_eval_candidates(
    kernel,
    normal_score,
    X_curr = matrix(0, ncol = 1),
    D_curr = matrix(0, ncol = 1),
    cand_X = matrix(c(-0.5, 0.5), ncol = 1)
  )

  expect_s3_class(state, "sp_mcmc_state")
  expect_equal(criterion$label, "last")
  expect_equal(start_idx, 2L)
  expect_equal(length(candidates$objective_values), 2L)
  expect_equal(dim(candidates$scores), c(2L, 1L))
  expect_error(
    sp_mcmc_select_start(sp_mcmc_criterion(function(state) 1.5), state),
    "integer index"
  )
})

test_that("sp_mcmc wrapper returns a stateful chain", {
  set.seed(11)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  run <- sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, mcmc = "rwm", criterion = "last",
    m_seq = 2, h = 0.1, x_init = 0, seed = 1
  )

  expect_s3_class(run, "sp_mcmc")
  expect_identical(class(run), "sp_mcmc")
  expect_equal(dim(run$X), c(2L, 1L))
  expect_equal(dim(run$D), c(2L, 1L))
  expect_gt(sum(run$n_eval), 0)
  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2.5, d = 1, m_seq = 2, h = 0.1, x_init = 0
  ), "positive integer")
  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 1.5, h = 0.1, x_init = 0
  ), "positive integer")
  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = c(1, 1, 1), h = 0.1, x_init = 0
  ), "m_seq must be scalar")
  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, mcmc = "grw", m_seq = 2, h = 0.1,
    x_init = 0
  ), "should be one of")
})

test_that("SP-MCMC uses the paper's m_seq-state candidate path", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  requested_rows <- NA_integer_
  transition <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    requested_rows <<- m_iter
    X <- matrix(c(x0, 1, 3), ncol = 1)
    list(
      X = X, D = score_function(X), accept = c(0L, 1L, 0L),
      counts = list(log_p = 3L, score = 3L, total = 6L)
    )
  }

  run <- sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, mcmc = "mala", criterion = "last",
    m_seq = 3, h = 0.1, x_init = 0, transition_fn = transition
  )

  expect_equal(requested_rows, 3L)
  expect_equal(run$X[2, 1], 1)
  expect_equal(run$chain_d2_max[2], 9)
  expect_equal(run$chain_d2_last[2], 9)
  expect_equal(run$accept_rate[2], 0.5)
})

test_that("SP-MCMC includes its initial state and removes duplicate candidates", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  transition <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    list(
      X = matrix(c(x0, x0, 2), ncol = 1), D = NULL,
      accept = c(0L, 0L, 1L),
      counts = list(log_p = 3L, score = 0L, total = 3L)
    )
  }

  run <- sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 3, h = 0.1, x_init = 0,
    transition_fn = transition
  )

  expect_equal(
    unname(run$counts[2, c("candidate_score", "transition_total", "total")]),
    c(2L, 3L, 5L)
  )
})

test_that("SP-MCMC supports one-state paths and checks custom chain shape", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  requested_rows <- NA_integer_
  one_move <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    requested_rows <<- m_iter
    list(
      X = matrix(x0, ncol = 1), D = NULL, accept = 0L,
      counts = list(log_p = 0L, score = 0L, total = 0L)
    )
  }

  # A one-state path makes no transition, so the only candidate is the start.
  # The run still completes, but it must say that it selected nothing.
  expect_warning(
    run <- sp_mcmc(
      normal_score, normal_log_p, kernel,
      n_points = 2, d = 1, m_seq = 1, h = 0.1, x_init = 0,
      transition_fn = one_move
    ),
    "never left its starting point"
  )
  expect_equal(requested_rows, 1L)
  expect_equal(run$X[2, 1], 0)
  expect_true(is.na(run$accept_rate[2]))
  expect_warning(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, mcmc = "rwm", m_seq = 1, h = 0.1,
    x_init = 0
  ), "never left its starting point")

  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0,
    transition_fn = one_move
  ), "exactly 2 rows")

  wrong_start <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    list(
      X = matrix(c(x0 + 1, x0 + 2), ncol = 1), D = NULL,
      counts = list(log_p = 0L, score = 0L, total = 0L)
    )
  }
  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0,
    transition_fn = wrong_start
  ), "initial state in row 1")
})

test_that("SP-MCMC requires counts from custom transitions", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  legacy_transition <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    list(X = matrix(c(x0, x0 + 1), ncol = 1), D = NULL, n_eval = 2L)
  }

  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0,
    transition_fn = legacy_transition
  ), "return counts")
})

test_that("SP-MCMC rejects a lazy Gaussian RBF", {
  lazy <- stein_kernel(type = "gaussian_rbf")

  expect_error(
    sp_mcmc(
      normal_score, normal_log_p, lazy,
      n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0
    ),
    "fixed h"
  )
})
