test_that("GOF statistics and their calibration draws share one scale", {
  K0 <- matrix(c(1.0, 0.2, 0.3, 0.2, 1.5, 0.4, 0.3, 0.4, 2.0), nrow = 3L,
               byrow = TRUE)
  K0_offdiag <- K0
  diag(K0_offdiag) <- 0
  W <- matrix(c(1, -1, 0, -1, 1, 0), nrow = 3L)
  tau <- matrix(c(-1, 0, 1, 2), ncol = 1L)
  statistic <- fssd_statistic(tau)

  # Statistics come back on their null-comparison scale.
  expect_equal(ksd_u_statistic(K0), sum(K0_offdiag) / (nrow(K0) - 1L))
  expect_equal(ksd_v_statistic(K0), sum(K0) / nrow(K0))
  expect_equal(statistic,
               (sum(colSums(tau)^2) - sum(rowSums(tau^2))) / (nrow(tau) - 1L))

  # Calibration draws use exactly the matching scale.
  expect_equal(ksd_u_bootstrap(K0, W_mat = W),
               as.numeric(nrow(K0) * colSums((K0_offdiag %*% W) * W)))
  expect_equal(ksd_v_bootstrap(K0, W_mat = W),
               as.numeric(colSums((K0 %*% W) * W) / nrow(K0)))

  set.seed(202)
  null <- fssd_null_pvalue(tau, statistic, n_simulations = 5L)
  expect_equal(null$statistic, statistic)
  expect_length(null$null_samples, 5L)
  expect_finite_numeric(null$null_samples)
})

test_that("kernel_scale2 reports, replaces, and reaches FSSD-opt", {
  expect_equal(kernel_scale2(stein_kernel(type = "gaussian_rbf", h = 2)), 4)
  expect_equal(kernel_scale2(stein_kernel(type = "imq", c = 3)), 9)
  expect_identical(kernel_scale2(stein_kernel(type = "gaussian_rbf")), NA_real_)
  # A kernel with no scale hook reports none and refuses to take one.
  expect_null(kernel_scale2(stein_kernel_inverse_log(alpha = 1, beta = -1)))
  expect_error(
    kernel_scale2(stein_kernel_inverse_log(alpha = 1, beta = -1), 9),
    "does not support scale optimization"
  )

  replaceable <- list(stein_kernel(type = "gaussian_rbf"),
                      stein_kernel(type = "imq", c = 1), custom_rbf(2))
  for (k in replaceable) expect_equal(kernel_scale2(kernel_scale2(k, 9)), 9)

  # Every route into FSSD-opt optimizes the same scale.
  builtin <- fssd_opt_scale2(stein_kernel(type = "gaussian_rbf", h = 1))
  expect_true(is.finite(builtin))
  expect_equal(fssd_opt_scale2(steinkernel_toy(1)), builtin)
  expect_equal(fssd_opt_scale2(custom_rbf(1)), builtin)
  expect_true(is.finite(fssd_opt_scale2(steinkernel_toy(NA_real_))))
  # Opting out stays the default: no scale2, no scale optimization.
  expect_true(is.na(fssd_opt_scale2(custom_rbf())))
})
