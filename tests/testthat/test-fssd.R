test_that("FSSD primitives return tau, statistic, and null summary", {
  set.seed(5)
  X <- small_x(6)
  S <- normal_score(X)
  V <- matrix(0, ncol = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)

  tau <- compute_tau(X, S, V, kernel)
  stat <- fssd_statistic(tau)
  null <- fssd_null_pvalue(
    tau, statistic = stat, n_simulations = 5
  )

  expect_equal(dim(tau), c(6L, 1L))
  expect_finite_numeric(tau)
  expect_length(stat, 1L)
  expect_finite_numeric(stat)
  expect_true(is.list(null))
  expect_true(null$p_value >= 0 && null$p_value <= 1)
  expect_equal(null$statistic, stat)
  expect_finite_numeric(null$statistic)
  expect_finite_numeric(null$null_samples)
  expect_finite_numeric(null$eigenvalues)
})

test_that("FSSD dispatcher preserves the rand result contract", {
  set.seed(6)
  X <- small_x(10)

  dispatcher <- fssd_test(
    X, normal_score,
    variant = "rand", J = 1, n_simulations = 5,
    scaling = 1, seed = 1
  )

  expect_htest_contract(dispatcher)
  expect_equal(dispatcher$info$variant, "rand")
  expect_s3_class(dispatcher$kernel, "SteinKernel_gaussian_rbf")
})

test_that("FSSD dispatcher preserves explicit scaling for opt", {
  X <- small_x(10)
  dispatcher <- fssd_test(
    X, normal_score,
    variant = "opt", J = 1, n_simulations = 5,
    scaling = 1, train_ratio = 0.5, maxit = 1, seed = 1
  )

  expect_htest_contract(dispatcher)
  expect_equal(dispatcher$info$variant, "opt")
  expect_null(dispatcher$info$scale_grid)
})

test_that("FSSD opt returns an htest with opt metadata", {
  set.seed(6)
  X <- small_x(10)

  opt <- fssd_opt_test(
    X, normal_score,
    J = 1, n_simulations = 5, scaling = 1,
    train_ratio = 0.5, maxit = 1, seed = 1
  )

  expect_htest_contract(opt)
  expect_equal(opt$info$variant, "opt")
  expect_named(
    opt$parameter,
    c("n_simulations", "J", "train_ratio", "gamma", "scaling")
  )
  expect_named(
    opt$info,
    c(
      "variant", "V", "kernel", "n_train", "n_test", "scale2_opt",
      "objective_opt", "function_evaluations", "convergence", "scale_grid",
      "scale_grid_objectives"
    )
  )
  expect_equal(opt$info$n_train + opt$info$n_test, nrow(X))
  expect_null(opt$info$scale_grid)
  expect_null(opt$info$scale_grid_objectives)
})

test_that("FSSD opt uses an exact median grid only when scaling is absent", {
  X <- small_x(20)
  opt <- fssd_opt_test(
    X, normal_score,
    J = 1, n_simulations = 5,
    train_ratio = 0.5, maxit = 1, seed = 1
  )

  expect_length(opt$info$scale_grid, 5L)
  expect_length(opt$info$scale_grid_objectives, 5L)
})

test_that("FSSD RBF scale updates preserve the preconditioner", {
  precon <- matrix(2, nrow = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1, precon = precon)

  updated <- kernel_scale2(kernel, 4)

  expect_equal(updated$h2, 4)
  expect_equal(updated$precon, precon)
  expect_identical(class(updated), class(kernel))
  expect_equal(kernel$h2, 1)

  unscaled <- stein_kernel(type = "gaussian_rbf", precon = precon)
  prepared <- steinsampling:::.prepare_fssd_kernel(
    X = small_x(4), kernel = unscaled, scaling = 3
  )
  expect_equal(prepared$kernel_obj$h2, 3)
  expect_equal(prepared$kernel_obj$precon, precon)
})

test_that("FSSD rand returns htest metadata and bounded p-value", {
  set.seed(101)
  X <- small_x(6)

  fssd_rand <- fssd_rand_test(
    X, normal_score,
    J = 1, n_simulations = 5,
    scaling = 1, seed = 101
  )

  expect_htest_contract(fssd_rand)
  expect_named(fssd_rand$info, c("variant", "V", "kernel"))
  expect_equal(fssd_rand$info$variant, "rand")
  expect_equal(fssd_rand$info$kernel, "gaussian_rbf")
  expect_equal(unname(fssd_rand$parameter["n_simulations"]), 5)
  expect_equal(unname(fssd_rand$parameter["J"]), 1)
})

test_that("FSSD htest statistic uses the paper's n-scaled test statistic", {
  X <- small_x(6)
  fssd_rand <- fssd_rand_test(
    X, normal_score,
    J = 1, n_simulations = 5,
    scaling = 1, seed = 101
  )
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  tau <- compute_tau(X, normal_score(X), fssd_rand$info$V, kernel)

  expect_equal(
    unname(fssd_rand$statistic),
    fssd_statistic(tau)
  )
  expect_false("fssd_stat" %in% names(fssd_rand))
})

test_that("FSSD null helper returns bounded p-value summaries", {
  set.seed(102)
  tau <- matrix(c(-1, 0, 1, 2), ncol = 1)
  null <- fssd_null_pvalue(
    tau,
    statistic = fssd_statistic(tau),
    n_simulations = 5
  )

  expect_named(null, c("p_value", "statistic", "null_samples", "eigenvalues"))
  expect_true(null$p_value >= 0 && null$p_value <= 1)
  expect_finite_numeric(null$statistic)
  expect_finite_numeric(null$null_samples)
  expect_finite_numeric(null$eigenvalues)

  zero_tau <- matrix(0, nrow = 4, ncol = 1)
  zero_null <- fssd_null_pvalue(
    zero_tau,
    fssd_statistic(zero_tau),
    n_simulations = 5
  )
  expect_equal(zero_null$p_value, 1)
})

test_that("FSSD opt supports the documented minimum split", {
  X <- small_x(4)
  expect_no_error(fssd_opt_test(
    X, normal_score, J = 1, n_simulations = 2, scaling = 1,
    train_ratio = 0.2, maxit = 1, seed = 1
  ))
})

test_that("FSSD rejects invalid control parameters", {
  X <- small_x(4)

  expect_error(fssd_rand_test(X, normal_score, J = 0, n_simulations = 2, scaling = 1), "positive")
  expect_error(fssd_opt_test(X, normal_score, J = 1, n_simulations = 2, train_ratio = 1), "train_ratio")
})

test_that("FSSD keeps a positive IMQ scale", {
  result <- fssd_rand_test(
    small_x(8), normal_score, J = 1, n_simulations = 2,
    kernel = "imq", scaling = 1e-12, seed = 1
  )

  expect_equal(kernel_scale2(result$kernel), 1e-12)
})
