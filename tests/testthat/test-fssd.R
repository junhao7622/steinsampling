test_that("FSSD primitives return tau, statistic, and null summary", {
  set.seed(5)
  X <- small_x(6)
  S <- normal_score(X)
  V <- matrix(0, ncol = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)

  tau <- compute_tau(X, S, V, kernel)
  stat <- compute_fssd_unbiased_stat(tau)
  null <- compute_fssd_null_pvalue(tau, fssd_stat = stat, n_simulations = 5)

  expect_equal(dim(tau), c(6L, 1L))
  expect_finite_numeric(tau)
  expect_length(stat, 1L)
  expect_finite_numeric(stat)
  expect_true(is.list(null))
  expect_true(null$p_value >= 0 && null$p_value <= 1)
  expect_finite_numeric(null$test_score)
  expect_finite_numeric(null$eigenvalues)
})

test_that("FSSD rand returns an htest with rand metadata", {
  set.seed(6)
  X <- small_x(10)

  rand <- fssd_rand_test(X, normal_score, J = 1, nboot = 5, scaling = 1, seed = 1)

  expect_valid_htest(rand)
  expect_equal(rand$info$variant, "rand")
})

test_that("FSSD dispatcher preserves the rand result contract", {
  set.seed(6)
  X <- small_x(10)

  dispatcher <- fssd_test(
    X, normal_score,
    variant = "rand", J = 1, nboot = 5,
    scaling = 1, seed = 1
  )

  expect_valid_htest(dispatcher)
  expect_equal(dispatcher$info$variant, "rand")
})

test_that("FSSD opt returns an htest with opt metadata", {
  set.seed(6)
  X <- small_x(10)

  opt <- fssd_opt_test(
    X, normal_score,
    J = 1, nboot = 5, scaling = 1,
    train_ratio = 0.5, eval_on = "all", maxit = 1, seed = 1
  )

  expect_valid_htest(opt)
  expect_equal(opt$info$variant, "opt")
})

test_that("FSSD rand returns htest metadata and bounded p-value", {
  set.seed(101)
  X <- small_x(6)

  fssd_rand <- fssd_rand_test(
    X, normal_score,
    J = 1, nboot = 5,
    scaling = 1, seed = 101
  )

  expect_htest_contract(fssd_rand)
  expect_named(fssd_rand$info, c("variant", "V", "kernel"))
  expect_equal(fssd_rand$info$variant, "rand")
  expect_equal(fssd_rand$info$kernel, "gaussian_rbf")
  expect_equal(unname(fssd_rand$parameter["nboot"]), 5)
  expect_equal(unname(fssd_rand$parameter["J"]), 1)
})

test_that("FSSD null helper returns bounded p-value summaries", {
  set.seed(102)
  tau <- matrix(c(-1, 0, 1, 2), ncol = 1)
  null <- compute_fssd_null_pvalue(
    tau,
    fssd_stat = compute_fssd_unbiased_stat(tau),
    n_simulations = 5
  )

  expect_named(null, c("p_value", "test_score", "eigenvalues"))
  expect_true(null$p_value >= 0 && null$p_value <= 1)
  expect_finite_numeric(null$test_score)
  expect_finite_numeric(null$eigenvalues)
})

test_that("FSSD rejects invalid control parameters", {
  X <- small_x(4)

  expect_error(fssd_rand_test(X, normal_score, J = 0, nboot = 2, scaling = 1), "positive")
  expect_error(fssd_opt_test(X, normal_score, J = 1, nboot = 2, train_ratio = 1), "train_ratio")
})
