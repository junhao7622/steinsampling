test_that("FSSD primitives return tau, statistic, and a bounded null summary", {
  set.seed(5)
  X <- small_x(6)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)

  tau <- compute_tau(X, normal_score(X), matrix(0, ncol = 1), kernel)
  stat <- fssd_statistic(tau)
  null <- fssd_null_pvalue(tau, statistic = stat, n_simulations = 5)

  expect_equal(dim(tau), c(6L, 1L))
  expect_named(null, c("p_value", "statistic", "null_samples", "eigenvalues"))
  expect_equal(null$statistic, stat)
  expect_true(null$p_value >= 0 && null$p_value <= 1)
  expect_finite_numeric(null$eigenvalues)

  # A degenerate tau leaves the statistic at the bottom of the null draws.
  zero <- matrix(0, nrow = 4, ncol = 1)
  expect_equal(
    fssd_null_pvalue(zero, fssd_statistic(zero), n_simulations = 5)$p_value, 1
  )

  # The htest reports exactly the primitive computed at its own locations.
  rand <- fssd_rand_test(X, normal_score, J = 1, n_simulations = 5,
                         scaling = 1, seed = 101)
  expect_equal(unname(rand$statistic),
               fssd_statistic(compute_tau(X, normal_score(X), rand$info$V,
                                          kernel)))
  expect_named(rand$info, c("variant", "V", "kernel"))
  expect_equal(unname(rand$parameter[c("n_simulations", "J")]), c(5, 1))
})

test_that("fssd_test dispatches to both variants and keeps their contracts", {
  X <- small_x(10)
  common <- list(X, normal_score, J = 1, n_simulations = 5, seed = 1)

  rand <- do.call(fssd_test, c(common, list(variant = "rand", scaling = 1)))
  opt <- do.call(fssd_test, c(common, list(
    variant = "opt", scaling = 1, train_ratio = 0.5, maxit = 1
  )))
  searched <- do.call(fssd_opt_test, c(common, list(
    train_ratio = 0.5, maxit = 1
  )))

  expect_htest_contract(rand)
  expect_htest_contract(opt)
  expect_equal(rand$info$variant, "rand")
  expect_equal(opt$info$variant, "opt")
  expect_equal(opt$info$n_train + opt$info$n_test, nrow(X))
  expect_s3_class(rand$kernel, "SteinKernel_gaussian_rbf")
  # An explicit `scaling` skips the median grid; leaving it out searches one.
  expect_null(opt$info$scale_grid)
  expect_length(searched$info$scale_grid, 5L)
  expect_length(searched$info$scale_grid_objectives, 5L)

  # Forwarding must not rename the caller's data.
  my_sample <- small_x(8)
  for (v in c("opt", "rand")) {
    fit <- fssd_test(my_sample, normal_score, variant = v, J = 1,
                     n_simulations = 5, scaling = 1, seed = 1)
    expect_identical(fit$data.name, "my_sample")
  }
  expect_identical(
    fssd_rand_test(my_sample, normal_score, J = 1, n_simulations = 5,
                   scaling = 1, seed = 1)$data.name,
    "my_sample"
  )
})

test_that("FSSD scale updates preserve the preconditioner and its metric", {
  precon <- matrix(2, nrow = 1)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1, precon = precon)
  updated <- kernel_scale2(kernel, 4)

  expect_equal(updated$scale2, 4)
  expect_equal(updated$precon, precon)
  expect_equal(kernel$scale2, 1)

  prepared <- steinsampling:::.prepare_fssd_kernel(
    X = small_x(4),
    kernel = stein_kernel(type = "gaussian_rbf", precon = precon), scaling = 3
  )
  expect_equal(prepared$kernel_obj$scale2, 3)
  expect_equal(prepared$kernel_obj$precon, precon)

  # An unset bandwidth is resolved in the preconditioned metric, and KSD and
  # FSSD must agree on the value.
  Z <- matrix(c(0, 1, 4, 10), ncol = 1)
  M <- matrix(4, nrow = 1)
  lazy <- stein_kernel(type = "gaussian_rbf", precon = M)
  fssd <- steinsampling:::.prepare_fssd_kernel(Z, kernel = lazy)
  ksd <- steinsampling:::.prepare_ksd_inputs(Z, normal_score, kernel = lazy)

  expect_equal(fssd$scaling, find_median_distance(Z %*% t(chol(M))))
  expect_equal(fssd$scaling, ksd$scaling)
  expect_equal(fssd$kernel_obj$precon, M)
  expect_true(is.na(lazy$scale2))
})

test_that("FSSD honours its documented control parameters", {
  X <- small_x(4)

  expect_no_error(fssd_opt_test(
    X, normal_score, J = 1, n_simulations = 2, scaling = 1,
    train_ratio = 0.2, maxit = 1, seed = 1
  ))
  expect_error(
    fssd_rand_test(X, normal_score, J = 0, n_simulations = 2, scaling = 1),
    "positive"
  )
  expect_error(
    fssd_opt_test(X, normal_score, J = 1, n_simulations = 2, train_ratio = 1),
    "train_ratio"
  )
  # A tiny explicit scale must survive as given.
  expect_equal(
    kernel_scale2(fssd_rand_test(
      small_x(8), normal_score, J = 1, n_simulations = 2,
      kernel = "imq", scaling = 1e-12, seed = 1
    )$kernel),
    1e-12
  )
})

test_that("compute_tau needs a fixed kernel, so batches share one feature map", {
  set.seed(11)
  X <- matrix(rnorm(20), ncol = 2)
  V <- matrix(c(-3, 0, 3, 0), nrow = 2, byrow = TRUE)

  expect_error(compute_tau(X, -X, V, stein_kernel("gaussian_rbf")), "fixed h")

  fixed <- stein_kernel("gaussian_rbf", h = 1)
  expect_identical(
    compute_tau(X, -X, V, fixed),
    rbind(compute_tau(X[1:5, ], -X[1:5, ], V, fixed),
          compute_tau(X[6:10, ], -X[6:10, ], V, fixed))
  )
})

test_that("an overflowing feature covariance stops instead of returning p = 0", {
  tau <- matrix(c(-1e155, 1e155, -1e155, 1e155), ncol = 1)
  expect_error(fssd_null_pvalue(tau, statistic = 1, n_simulations = 5),
               "not finite")
})
