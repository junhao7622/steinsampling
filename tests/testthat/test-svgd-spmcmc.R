test_that("SVGD kernel and updates return particle matrices", {
  obj <- svgd()
  x0 <- matrix(c(-0.5, 0.5), ncol = 1)
  kernel_eval <- obj$svgd_kernel(x0)
  updated <- update_svgd(obj, x0, normal_score, n_iter = 2, stepsize = 0.01)
  traced <- obj$update(x0, normal_score, n_iter = 2, stepsize = 0.01, trace_iters = 1:2)

  expect_true(is.list(obj))
  expect_equal(dim(kernel_eval$Kxy), c(2L, 2L))
  expect_equal(dim(kernel_eval$dxkxy), c(2L, 1L))
  expect_equal(dim(updated), c(2L, 1L))
  expect_equal(dim(traced$theta), c(2L, 1L))
  expect_length(traced$trace, 2L)
})

test_that("SVGD validates iteration and scaling controls", {
  obj <- svgd(stein_kernel(type = "gaussian_rbf", h = 1))
  x0 <- matrix(c(-0.5, 0.5), ncol = 1)

  expect_equal(update_svgd(obj, x0, normal_score, n_iter = 0), x0)
  expect_error(update_svgd(obj, x0, normal_score, n_iter = 1.5), "nonnegative integer")
  expect_error(update_svgd(obj, x0, normal_score, n_iter = 1, stepsize = 0), "stepsize")
  expect_error(update_svgd(obj, x0, normal_score, n_iter = 1, alpha = 1), "alpha")
  expect_error(
    update_svgd(obj, x0, normal_score, n_iter = 1, trace_iters = 2),
    "trace_iters"
  )
  expect_error(
    update_svgd(
      obj, x0, normal_score, n_iter = 1,
      adj_grad = function(...) 1
    ),
    "same dimensions"
  )
})

test_that("SVGD uses the paper's exact dynamic median bandwidth", {
  theta <- matrix(c(-2, -0.5, 1, 4), ncol = 1)
  med <- stats::median(as.numeric(stats::dist(theta)))
  expected <- med / sqrt(2 * log(nrow(theta)))

  expect_equal(steinsampling:::svgd_median_bandwidth(theta), expected)
  expect_equal(
    steinsampling:::svgd_median_bandwidth(3 * theta),
    3 * expected
  )
  expect_error(
    steinsampling:::svgd_median_bandwidth(matrix(0, nrow = 2, ncol = 1)),
    "fixed h > 0"
  )
})

test_that("SVGD dynamic RBF bandwidth preserves preconditioning", {
  theta <- matrix(c(-1, 0, 2), ncol = 1)
  precon <- matrix(2, nrow = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", precon = precon)
  h <- steinsampling:::svgd_median_bandwidth(theta)
  expected_kernel <- kernel
  expected_kernel$h2 <- h^2

  out <- steinsampling:::compute_svgd_kernel(theta, kernel)

  expect_equal(out$Kxy, eval_kernel(expected_kernel, theta))
  expect_equal(kernel$precon, precon)
  expect_null(kernel$h2)
})

test_that("custom_adjusted_gradient applies the user hook", {
  custom <- custom_adjusted_gradient(
    function(grad, historical_grad, ...) grad / (1 + sqrt(historical_grad)),
    grad = matrix(1),
    historical_grad = matrix(1),
    iter = 1,
    theta = matrix(0),
    stepsize = 0.1,
    alpha = 0.9,
    fudge_factor = 1e-6
  )

  expect_equal(custom, matrix(0.5))
})

test_that("MCMC kernels return chain shapes and evaluation counts", {
  set.seed(10)
  mala_res <- mala(normal_log_p, normal_score, x0 = 0, h = 0.1, m_iter = 3)
  grwmetrop_res <- grwmetrop(normal_log_p, x0 = 0, S = matrix(0.1), m_iter = 3)
  grw_res <- grw(normal_log_p, normal_score, x0 = 0, h = 0.1, m_iter = 3)
  rwm_res <- rwm(normal_log_p, normal_score, x0 = 0, h = 0.1, m_iter = 3)

  expect_equal(dim(mala_res$X), c(3L, 1L))
  expect_equal(dim(mala_res$D), c(3L, 1L))
  expect_equal(dim(grwmetrop_res$X), c(3L, 1L))
  expect_equal(dim(grw_res$X), c(3L, 1L))
  expect_equal(dim(rwm_res$X), c(3L, 1L))
  expect_gt(mala_res$n_eval, 0)
  expect_gt(grwmetrop_res$n_eval, 0)
  expect_gt(grw_res$n_eval, 0)
  expect_gt(rwm_res$n_eval, 0)
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
  expect_equal(length(candidates$f_vec), 2L)
  expect_equal(dim(candidates$D_new), c(2L, 1L))
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
    n_points = 2, d = 1, mcmc = "grw", criterion = "last",
    m_seq = 2, h = 0.1, x_init = 0, seed = 1
  )

  expect_s3_class(run, "sp_mcmc")
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
})

test_that("SP-MCMC makes m_seq transitions and excludes the initial state", {
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
    m_seq = 2, h = 0.1, x_init = 0, transition_fn = transition
  )

  expect_equal(requested_rows, 3L)
  expect_equal(run$X[2, 1], 1)
  expect_equal(run$chain_d2_max[2], 9)
  expect_equal(run$chain_d2_last[2], 9)
  expect_equal(run$chain_d2_first_last[2], 4)
  expect_equal(run$accept_rate[2], 0.5)
})

test_that("SP-MCMC keeps a rejected first transition as a candidate", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  candidates <- NULL
  transition <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    list(X = matrix(c(x0, x0, 2), ncol = 1), accept = c(0L, 0L, 1L))
  }
  count_candidates <- function(chain, cand_X, ...) {
    candidates <<- cand_X
    nrow(cand_X)
  }

  sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0,
    transition_fn = transition, n_eval_fn = count_candidates
  )

  expect_equal(as.numeric(candidates), c(0, 2))
})

test_that("SP-MCMC accepts one transition and checks custom chain shape", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  requested_rows <- NA_integer_
  one_move <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    requested_rows <<- m_iter
    list(X = matrix(c(x0, x0 + 1), ncol = 1), accept = c(0L, 1L))
  }

  expect_no_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 1, h = 0.1, x_init = 0,
    transition_fn = one_move
  ))
  expect_equal(requested_rows, 2L)

  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0,
    transition_fn = one_move
  ), "exactly 3 rows")

  wrong_start <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    list(X = matrix(c(x0 + 1, x0 + 2, x0 + 3), ncol = 1))
  }
  expect_error(sp_mcmc(
    normal_score, normal_log_p, kernel,
    n_points = 2, d = 1, m_seq = 2, h = 0.1, x_init = 0,
    transition_fn = wrong_start
  ), "initial state in row 1")
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
