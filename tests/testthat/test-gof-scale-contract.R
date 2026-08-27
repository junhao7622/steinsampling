test_that("all GOF statistic primitives return their null-comparison scale", {
  K0 <- matrix(
    c(1.0, 0.2, 0.3,
      0.2, 1.5, 0.4,
      0.3, 0.4, 2.0),
    nrow = 3L,
    byrow = TRUE
  )
  K0_offdiag <- K0
  diag(K0_offdiag) <- 0

  expect_equal(ksd_u_statistic(K0), sum(K0_offdiag) / (nrow(K0) - 1L))
  expect_equal(ksd_v_statistic(K0), sum(K0) / nrow(K0))

  tau <- matrix(c(-1, 0, 1, 2), ncol = 1L)
  numerator <- sum(colSums(tau)^2) - sum(rowSums(tau^2))
  expect_equal(fssd_statistic(tau), numerator / (nrow(tau) - 1L))
})

test_that("GOF calibration draws match their observed-statistic scale", {
  K0 <- matrix(
    c(1.0, 0.2, 0.3,
      0.2, 1.5, 0.4,
      0.3, 0.4, 2.0),
    nrow = 3L,
    byrow = TRUE
  )
  W <- matrix(c(1, -1, 0, -1, 1, 0), nrow = 3L)
  K0_offdiag <- K0
  diag(K0_offdiag) <- 0

  expected_u <- nrow(K0) * colSums((K0_offdiag %*% W) * W)
  expected_v <- colSums((K0 %*% W) * W) / nrow(K0)
  expect_equal(ksd_u_bootstrap(K0, W_mat = W), as.numeric(expected_u))
  expect_equal(ksd_v_bootstrap(K0, W_mat = W), as.numeric(expected_v))

  tau <- matrix(c(-1, 0, 1, 2), ncol = 1L)
  statistic <- fssd_statistic(tau)
  set.seed(202)
  null <- fssd_null_pvalue(tau, statistic, n_simulations = 5L)
  expect_equal(null$statistic, statistic)
  expect_length(null$null_samples, 5L)
  expect_finite_numeric(null$null_samples)
})

test_that("kernel_scale2 reports and replaces a kernel's squared scale", {
  expect_equal(kernel_scale2(stein_kernel(type = "gaussian_rbf", h = 2)), 4)
  expect_identical(kernel_scale2(stein_kernel(type = "gaussian_rbf")), NA_real_)
  expect_equal(kernel_scale2(stein_kernel(type = "imq", c = 3)), 9)
  expect_null(kernel_scale2(stein_kernel_inverse_log(alpha = 1, beta = -1)))
  expect_null(kernel_scale2(custom_rbf()))
  expect_equal(kernel_scale2(custom_rbf(2, custom_rbf_at_scale)), 2)

  replaceable <- list(stein_kernel(type = "gaussian_rbf"),
                      stein_kernel(type = "imq", c = 1),
                      custom_rbf(2, custom_rbf_at_scale))
  for (k in replaceable) expect_equal(kernel_scale2(kernel_scale2(k, 9)), 9)
  expect_error(
    kernel_scale2(stein_kernel_inverse_log(alpha = 1, beta = -1), 9),
    "does not support scale optimization"
  )
})

test_that("kernels reaching FSSD-opt through the hook match the built-in path", {
  builtin <- fssd_opt_scale2(stein_kernel(type = "gaussian_rbf", h = 1))
  expect_true(is.finite(builtin))
  expect_equal(fssd_opt_scale2(steinkernel_toy(1)), builtin)
  expect_equal(fssd_opt_scale2(custom_rbf(1, custom_rbf_at_scale)), builtin)
  expect_true(is.finite(fssd_opt_scale2(steinkernel_toy(NA_real_))))
  # Opting out stays the default: no set_scale_fn, no scale optimization.
  expect_true(is.na(fssd_opt_scale2(custom_rbf())))
})

test_that("opting in requires a working hook and a scale derivative", {
  expect_error(custom_rbf(scale_init = 1), "needs set_scale_fn")
  inert <- custom_rbf()
  expect_error(
    custom_stein_kernel(inert$eval_fn, inert$grad_x_fn, inert$trace_mixed_fn,
                        scale_init = 1, set_scale_fn = function(obj, scale2) obj),
    "left every callback unchanged"
  )

  no_param <- custom_rbf(1, custom_rbf_at_scale)
  no_param$grad_theta_v_fn <- function(X, vj, grads_X, g_block, obj) {
    list(grad_vj = rep(0, ncol(X)))
  }
  expect_error(fssd_opt_scale2(no_param), "must return grad_param")

  numeric_mode <- custom_rbf(1, custom_rbf_at_scale)
  numeric_mode$grad_theta_v_fn <- NULL
  numeric_mode$custom_grad_mode <- "numeric"
  expect_error(fssd_opt_scale2(numeric_mode), "not implemented")
})
