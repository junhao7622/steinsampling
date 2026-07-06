test_that("optimizer factories return result records", {
  set.seed(7)
  grid <- fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)
  mc <- fmin_mc(lb = -1, ub = 1, n_mc = 5, mu0 = 0, Sigma0 = matrix(0.1), delay = 2)
  nm <- fmin_nm(
    lb = c(-1, -1), ub = c(1, 1), n_res = 1,
    mu0 = c(0, 0), Sigma0 = diag(0.1, 2),
    delay = 2, control = list(maxit = 5, reltol = 1e-2)
  )

  grid_res <- grid(toy_objective_1d, matrix(0, ncol = 1))
  mc_res <- mc(toy_objective_1d, matrix(0, ncol = 1))
  nm_res <- nm(toy_objective_2d, matrix(c(0, 0), ncol = 2))

  expect_named(grid_res, c("x_min", "d_min", "f_min", "n_eval"))
  expect_named(mc_res, c("x_min", "d_min", "f_min", "n_eval"))
  expect_named(nm_res, c("x_min", "d_min", "f_min", "n_eval"))
  expect_finite_numeric(grid_res$x_min)
  expect_finite_numeric(mc_res$x_min)
  expect_finite_numeric(nm_res$x_min)
  expect_gt(grid_res$n_eval, 0)
  expect_gt(mc_res$n_eval, 0)
  expect_gt(nm_res$n_eval, 0)
})

test_that("sample_proposal_box draws within requested bounds", {
  set.seed(8)
  draws <- sample_proposal_box(
    n = 5,
    lb = -1,
    ub = 1,
    mu0 = 0,
    Sigma0 = matrix(0.1),
    sigsq = 0.1,
    X_curr = matrix(0, ncol = 1),
    delay = 2
  )

  expect_equal(dim(draws), c(5L, 1L))
  expect_true(all(draws >= -1 & draws <= 1))
})

test_that("with_local_seed isolates reproducible RNG draws", {
  baseline <- with_local_seed(42L, stats::rnorm(4))
  repeat_draw <- with_local_seed(42L, stats::rnorm(4))
  different_draw <- with_local_seed(43L, stats::rnorm(4))

  expect_equal(baseline, repeat_draw)
  expect_false(isTRUE(all.equal(baseline, different_draw)))
})

test_that("stein_thinning returns valid selected indices", {
  set.seed(9)
  X <- small_x(5)
  S <- normal_score(X)

  thin_idx <- stein_thinning(
    X,
    S = S, m = 2, kernel = "gaussian_rbf",
    h = 1, pre_subsample = 5, verbose_rbf_warning = FALSE
  )

  expect_length(thin_idx, 2L)
  expect_true(all(thin_idx >= 1 & thin_idx <= nrow(X)))
})

test_that("stein_points returns point and score matrices", {
  set.seed(9)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  opt <- fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)

  points <- stein_points(
    normal_score, kernel,
    n_points = 2, d = 1,
    optimizer = opt, x_init = 0
  )

  expect_equal(dim(points$X), c(2L, 1L))
  expect_equal(dim(points$D), c(2L, 1L))
})

test_that("stein_codescent preserves point and score matrices", {
  set.seed(9)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  opt <- fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)

  refined <- stein_codescent(
    matrix(c(-0.5, 0.5), ncol = 1),
    normal_score, kernel,
    n_iter = 1, optimizer = opt
  )

  expect_equal(dim(refined$X), c(2L, 1L))
  expect_equal(dim(refined$D), c(2L, 1L))
})

test_that("score-space Stein kernel selects finite Stein points", {
  hess_log_p <- function(X) array(-1, dim = c(nrow(as.matrix(X)), 1L, 1L))
  kernel <- stein_kernel_imq_score(alpha = 1, beta = -0.5, hess_log_p = hess_log_p)
  opt <- fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)

  res <- stein_points(
    normal_score, kernel,
    n_points = 2, d = 1,
    optimizer = opt, x_init = 0
  )

  expect_s3_class(kernel, "SteinKernel_imq_score")
  expect_equal(dim(res$X), c(2L, 1L))
  expect_finite_numeric(res$X)
})
