normal_score <- function(X) -as.matrix(X)

normal_log_p <- function(X) {
  X <- as.matrix(X)
  -0.5 * rowSums(X * X)
}

small_x <- function(n = 6L) {
  matrix(seq(-1, 1, length.out = n), ncol = 1)
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

rbf_at <- function(h2) stein_kernel(type = "gaussian_rbf", h = sqrt(h2))

steinkernel_toy <- function(h2 = 1) {
  steinsampling:::new_stein_kernel(
    "toy", h2 = h2,
    eval        = function(k, X, Y, M) eval_kernel(rbf_at(k$h2), X, Y),
    grad_x      = function(k, X, Y, M) grad_x_kernel(rbf_at(k$h2), X, Y),
    trace_mixed = function(k, X, Y, M) trace_mixed_kernel(rbf_at(k$h2), X, Y),
    fssd_grad   = function(k, X, vj, grads_X, g_block, M)
      grad_theta_v_kernel(rbf_at(k$h2), X, vj, grads_X, g_block),
    scale_get   = function(k) k$h2,
    scale_set   = function(k, v) { k$h2 <- v; k }
  )
}

custom_rbf_at_scale <- function(obj, h2) {
  b <- rbf_at(h2)
  obj$eval_fn <- function(X, Y = NULL, precon = NULL) eval_kernel(b, X, Y)
  obj$grad_x_fn <- function(X, Y = NULL, precon = NULL) grad_x_kernel(b, X, Y)
  obj$trace_mixed_fn <- function(X, Y = NULL, precon = NULL) trace_mixed_kernel(b, X, Y)
  obj
}

custom_rbf <- function(scale_init = NULL, set_scale_fn = NULL) {
  cb <- custom_rbf_at_scale(list(), if (is.null(scale_init)) 1 else scale_init)
  custom_stein_kernel(
    cb$eval_fn, cb$grad_x_fn, cb$trace_mixed_fn,
    grad_theta_v_fn = function(X, vj, grads_X, g_block, obj) {
      h2 <- if (is.null(obj$scale_init)) 1 else obj$scale_init
      grad_theta_v_kernel(rbf_at(h2), X, vj, grads_X, g_block)
    },
    scale_init = scale_init, set_scale_fn = set_scale_fn
  )
}

fssd_opt_scale2 <- function(kernel) {
  set.seed(1)
  X <- matrix(rnorm(60), ncol = 1L)
  fssd_opt_test(X, normal_score, J = 2L, kernel = kernel)$info$scale2_opt
}
