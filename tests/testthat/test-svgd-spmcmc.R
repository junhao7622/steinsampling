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
})
