normal_score <- function(X) -as.matrix(X)

normal_log_p <- function(X) {
  X <- as.matrix(X)
  -0.5 * rowSums(X * X)
}

small_x <- function(n = 6L) {
  matrix(seq(-1, 1, length.out = n), ncol = 1)
}

# The 1D two-component mixture reused across the GMM tests.
toy_gmm_1d <- function(sigma = array(1, c(1, 1, 2))) {
  gmm(nComp = 2, mu = c(-1, 1), sigma = sigma, weights = c(0.4, 0.6), d = 1)
}

# Callbacks that reproduce a Gaussian RBF through the custom-kernel API.
rbf_like_fns <- function(reference = stein_kernel(type = "gaussian_rbf", h = 1)) {
  list(
    eval = function(k, X, Y, M) eval_kernel(reference, X, Y),
    grad_x = function(k, X, Y, M) grad_x_kernel(reference, X, Y),
    trace_mixed = function(k, X, Y, M)
      trace_mixed_kernel(reference, X, Y)
  )
}

expect_finite_numeric <- function(x) {
  expect_true(is.numeric(x))
  expect_true(all(is.finite(x)))
}

expect_htest_contract <- function(x) {
  expect_s3_class(x, "htest")
  expect_true(all(c("statistic", "p.value", "method", "data.name", "parameter") %in% names(x)))
  expect_finite_numeric(x$statistic)
  expect_length(x$p.value, 1L)
  expect_true(is.finite(x$p.value))
  expect_true(x$p.value >= 0 && x$p.value <= 1)
  expect_type(x$method, "character")
  expect_length(x$method, 1L)
}

toy_objective_1d <- function(X) {
  X <- as.matrix(X)
  list(
    objective_values = as.numeric((X[, 1] - 0.25)^2),
    scores = -X
  )
}

toy_objective_2d <- function(X) {
  X <- as.matrix(X)
  center <- matrix(c(0.25, -0.25), nrow(X), 2L, byrow = TRUE)
  list(
    objective_values = rowSums((X - center)^2),
    scores = -X
  )
}

# ---- Scale-hook fixtures -------------------------------------------------
# Both fixtures delegate to Gaussian RBF, providing an exact scale-hook reference.

rbf_at <- function(h2) kernel_scale2(stein_kernel(type = "gaussian_rbf"), h2)

steinkernel_toy <- function(h2 = 1) {
  steinsampling:::new_stein_kernel(
    "toy", scale2 = h2,
    eval        = function(k, X, Y, M) eval_kernel(rbf_at(k$scale2), X, Y),
    grad_x      = function(k, X, Y, M) grad_x_kernel(rbf_at(k$scale2), X, Y),
    trace_mixed = function(k, X, Y, M) trace_mixed_kernel(rbf_at(k$scale2), X, Y),
    fssd_grad   = function(k, X, vj, grads_X, g_block, M)
      grad_theta_v_kernel(rbf_at(k$scale2), X, vj, grads_X, g_block)
  )
}

custom_rbf <- function(scale2 = NULL) {
  fixed <- function(k) if (is.null(k$scale2)) 1 else k$scale2
  custom_stein_kernel(
    eval        = function(k, X, Y, M) eval_kernel(rbf_at(fixed(k)), X, Y),
    grad_x      = function(k, X, Y, M) grad_x_kernel(rbf_at(fixed(k)), X, Y),
    trace_mixed = function(k, X, Y, M) trace_mixed_kernel(rbf_at(fixed(k)), X, Y),
    fssd_grad   = function(k, X, vj, grads_X, g_block, M)
      grad_theta_v_kernel(rbf_at(fixed(k)), X, vj, grads_X, g_block),
    scale2 = scale2
  )
}

fssd_opt_scale2 <- function(kernel) {
  set.seed(1)
  X <- matrix(rnorm(60), ncol = 1L)
  fssd_opt_test(X, normal_score, J = 2L, kernel = kernel)$info$scale2_opt
}
