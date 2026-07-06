test_that("Gaussian RBF kernel generics return finite shapes", {
  X <- small_x(4)
  S <- normal_score(X)
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)

  K <- eval_kernel(kernel, X)
  G <- grad_x_kernel(kernel, X)
  T <- trace_mixed_kernel(kernel, X)
  C <- cross_kernel(kernel, X, S)
  K0 <- stein_kernel_matrix(kernel, X, S)
  theta_grad <- grad_theta_v_kernel(
    kernel,
    X = X,
    vj = 0,
    grads_X = S,
    g_block = matrix(1, nrow(X), ncol(X))
  )

  expect_s3_class(kernel, "SteinKernel_gaussian_rbf")
  expect_equal(dim(K), c(4L, 4L))
  expect_equal(dim(G), c(4L, 4L, 1L))
  expect_equal(dim(T), c(4L, 4L))
  expect_equal(dim(C), c(4L, 4L))
  expect_equal(dim(K0), c(4L, 4L))
  expect_finite_numeric(K)
  expect_finite_numeric(G)
  expect_finite_numeric(T)
  expect_finite_numeric(C)
  expect_finite_numeric(K0)
  expect_length(theta_grad$grad_vj, 1L)
  expect_finite_numeric(theta_grad$grad_vj)
  expect_finite_numeric(theta_grad$grad_param)
})

test_that("IMQ and inverse-log kernels produce finite Stein matrices", {
  X <- small_x(4)
  S <- normal_score(X)
  imq <- stein_kernel(type = "imq", c = 1, beta = -0.5)
  inverse_log <- stein_kernel_inverse_log(alpha = 1, beta = -1)

  expect_s3_class(imq, "SteinKernel_imq")
  expect_s3_class(inverse_log, "SteinKernel_inverse_log")
  expect_equal(dim(stein_kernel_matrix(imq, X, S)), c(4L, 4L))
  expect_equal(dim(stein_kernel_matrix(inverse_log, X, S)), c(4L, 4L))
  expect_finite_numeric(stein_kernel_matrix(imq, X, S))
  expect_finite_numeric(stein_kernel_matrix(inverse_log, X, S))
})

test_that("custom Stein kernels satisfy the generic shape contract", {
  eval_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    diff <- outer(X[, 1], Y[, 1], "-")
    exp(-0.5 * diff^2)
  }
  grad_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    diff <- outer(X[, 1], Y[, 1], "-")
    array(-diff * exp(-0.5 * diff^2), dim = c(nrow(X), nrow(Y), 1L))
  }
  trace_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    diff <- outer(X[, 1], Y[, 1], "-")
    (1 - diff^2) * exp(-0.5 * diff^2)
  }

  kernel <- custom_stein_kernel(eval_fn, grad_fn, trace_fn, custom_grad_mode = "numeric")
  X <- small_x(3)
  S <- normal_score(X)

  expect_s3_class(kernel, "SteinKernel_custom")
  expect_equal(dim(eval_kernel(kernel, X)), c(3L, 3L))
  expect_equal(dim(grad_x_kernel(kernel, X)), c(3L, 3L, 1L))
  expect_equal(dim(trace_mixed_kernel(kernel, X)), c(3L, 3L))
  expect_equal(dim(cross_kernel(kernel, X, S)), c(3L, 3L))
  expect_equal(dim(stein_kernel_matrix(kernel, X, S)), c(3L, 3L))
  expect_length(grad_theta_v_kernel(kernel, X, 0, S, matrix(1, 3, 1))$grad_vj, 1L)
})

test_that("Gaussian RBF Stein kernel matches 1D oracle values", {
  X <- matrix(c(-1, 0, 1), ncol = 1)
  S <- -X
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  diff <- outer(X[, 1], X[, 1], "-")
  base <- exp(-0.5 * diff^2)

  expect_equal(eval_kernel(kernel, X), base)
  expect_equal(grad_x_kernel(kernel, X)[, , 1L], -diff * base)
  expect_equal(trace_mixed_kernel(kernel, X), (1 - diff^2) * base)
  expect_equal(cross_kernel(kernel, X, S), -diff^2 * base)
  expect_equal(
    stein_kernel_matrix(kernel, X, S),
    tcrossprod(S) * base - diff^2 * base + (1 - diff^2) * base
  )
  expect_equal(stein_kernel_matrix(kernel, X, S), t(stein_kernel_matrix(kernel, X, S)))
  expect_equal(eval_kernel(kernel, X), t(eval_kernel(kernel, X)))
})

test_that("find_median_distance matches pairwise squared-distance median", {
  X <- matrix(c(-1, 0, 1), ncol = 1)

  expected_sq_dist <- c(1, 4, 1)
  expect_equal(find_median_distance(X, use_sampling = FALSE), stats::median(expected_sq_dist))
})

test_that("kernel contracts reject bad gradients and missing custom gradients", {
  X <- small_x(4)
  eval_fn <- function(X, Y = NULL, precon = NULL) matrix(1, nrow(as.matrix(X)), nrow(as.matrix(X)))
  grad_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    array(0, dim = c(nrow(X), nrow(Y), ncol(X)))
  }
  trace_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    matrix(0, nrow(X), nrow(Y))
  }
  custom_no_opt_grad <- custom_stein_kernel(eval_fn, grad_fn, trace_fn)

  expect_error(
    stein_kernel_matrix(stein_kernel(type = "gaussian_rbf", h = 1), X, matrix(0, 3, 1)),
    "same shape"
  )
  expect_error(custom_stein_kernel(eval_fn, NULL, trace_fn), "must be functions")
  expect_error(
    grad_theta_v_kernel(custom_no_opt_grad, X, 0, normal_score(X), matrix(1, nrow(X), 1)),
    "requires grad_theta_v_fn"
  )
})
