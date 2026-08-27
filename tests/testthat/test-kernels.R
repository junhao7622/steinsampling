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
  expect_identical(kernel$custom_grad_mode, "numeric")
  expect_equal(dim(eval_kernel(kernel, X)), c(3L, 3L))
  expect_equal(dim(grad_x_kernel(kernel, X)), c(3L, 3L, 1L))
  expect_equal(dim(trace_mixed_kernel(kernel, X)), c(3L, 3L))
  expect_equal(dim(cross_kernel(kernel, X, S)), c(3L, 3L))
  expect_equal(dim(stein_kernel_matrix(kernel, X, S)), c(3L, 3L))
  expect_equal(
    stein_kernel_matrix(kernel, X, S),
    stein_kernel_matrix(stein_kernel(type = "gaussian_rbf", h = 1), X, S)
  )
  expect_length(grad_theta_v_kernel(kernel, X, 0, S, matrix(1, 3, 1))$grad_vj, 1L)

  # A custom kernel's `grad_param` is ignored, not validated.
  with_param <- custom_stein_kernel(
    eval_fn, grad_fn, trace_fn,
    grad_theta_v_fn = function(X, vj, grads_X, g_block, obj) {
      list(grad_vj = rep(1, ncol(X)), grad_param = NaN)
    }
  )
  expect_identical(
    grad_theta_v_kernel(with_param, X, 0, S, matrix(1, 3, 1))$grad_param, 0
  )

  expect_error(
    custom_stein_kernel(eval_fn, grad_fn, trace_fn, custom_grad_mode = "error"),
    "should be one of"
  )
})

test_that("score-distance IMQ exposes the public Stein matrix method", {
  X <- matrix(c(-1, 0, 0.5, 1, 1.5, -0.5), ncol = 2, byrow = TRUE)
  Y <- matrix(c(-0.75, 0.25, 1.25, -1), ncol = 2, byrow = TRUE)
  hess_log_p <- function(Z) {
    out <- array(0, dim = c(nrow(as.matrix(Z)), 2L, 2L))
    for (i in seq_len(dim(out)[1L])) out[i, , ] <- -diag(2)
    out
  }
  score_kernel <- stein_kernel_imq_score(1, -0.5, hess_log_p)
  position_kernel <- stein_kernel(type = "imq", c = 1, beta = -0.5)

  expect_equal(
    stein_kernel_matrix(score_kernel, X, -X, Y, -Y),
    stein_kernel_matrix(position_kernel, X, -X, Y, -Y),
    tolerance = 1e-12
  )
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

test_that("fused preconditioned Stein kernels match their public components", {
  X <- matrix(c(-1, 0.5, 0.25, -0.5, 1, 1.5), ncol = 2, byrow = TRUE)
  Y <- matrix(c(-0.75, 0.25, 1.25, -1), ncol = 2, byrow = TRUE)
  SX <- -X
  SY <- -Y
  M <- matrix(c(2, 0.3, 0.3, 1), nrow = 2)

  for (kernel in list(
    stein_kernel(type = "gaussian_rbf", h = 1.2, precon = M),
    stein_kernel(type = "imq", c = 0.8, beta = -0.5, precon = M)
  )) {
    assembled <- tcrossprod(SX, SY) * eval_kernel(kernel, X, Y) +
      cross_kernel(kernel, X, SX, Y, SY) +
      trace_mixed_kernel(kernel, X, Y)
    expect_equal(
      stein_kernel_matrix(kernel, X, SX, Y, SY),
      assembled,
      tolerance = 1e-12
    )
  }
})

test_that("find_median_distance squares the exact all-pair distance median", {
  X <- matrix(c(0, 1, 4, 10), ncol = 1)
  distances <- as.numeric(stats::dist(X))

  expect_equal(find_median_distance(X), stats::median(distances)^2)
  expect_equal(find_median_distance(X), 25)
  expect_false(isTRUE(all.equal(
    find_median_distance(X),
    stats::median(distances^2)
  )))
})

test_that("lazy RBF uses each row once for its exact median scale", {
  X <- matrix(c(0, 1, 4, 10), ncol = 1)
  lazy <- stein_kernel(type = "gaussian_rbf")
  fixed <- stein_kernel(type = "gaussian_rbf", h = 5)

  expect_equal(eval_kernel(lazy, X), eval_kernel(fixed, X), tolerance = 1e-12)
  expect_equal(
    stein_kernel_matrix(lazy, X, -X),
    stein_kernel_matrix(fixed, X, -X),
    tolerance = 1e-12
  )
})

test_that("public median defaults expose no sampling controls", {
  functions <- list(
    find_median_distance,
    ksd_u_test, ksd_uq_matrix,
    ksd_v_test, ksd_vq_matrix,
    fssd_test, fssd_rand_test, fssd_opt_test
  )
  removed <- c("max_samples", "use_sampling", "median_max_samples",
               "median_use_sampling", "median_seed")

  for (fun in functions) {
    expect_false(any(removed %in% names(formals(fun))))
  }
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

test_that("closed-form k0_diag matches the general Stein-matrix diagonal", {
  set.seed(404)
  X <- matrix(rnorm(12), ncol = 2L)
  S <- -X
  M <- matrix(c(2, 0.4, 0.4, 1.5), 2L, 2L)
  hess <- function(Z) {
    Z <- as.matrix(Z)
    array(rep(c(-1, 0.3, 0.3, -2), each = nrow(Z)), dim = c(nrow(Z), 2L, 2L))
  }

  kernels <- list(
    rbf = stein_kernel(type = "gaussian_rbf", h = 0.8),
    rbf_precon = stein_kernel(type = "gaussian_rbf", h = 0.8, precon = M),
    imq = stein_kernel(type = "imq", c = 1.3, beta = -0.4),
    imq_precon = stein_kernel(type = "imq", c = 1.3, beta = -0.4, precon = M),
    inverse_log = stein_kernel_inverse_log(alpha = 1.2, beta = -1),
    imq_score = stein_kernel_imq_score(alpha = 1.1, beta = -0.5, hess_log_p = hess)
  )

  for (name in names(kernels)) {
    kernel <- kernels[[name]]
    expect_equal(
      as.numeric(steinsampling:::k0_diag(kernel, X, S)),
      diag(stein_kernel_matrix(kernel, X, S)),
      info = name
    )
  }
})

test_that("k0_diag falls back without forming the full pairwise matrix", {
  set.seed(405)
  X <- matrix(rnorm(8), ncol = 2L)
  S <- -X
  reference <- stein_kernel(type = "gaussian_rbf", h = 1)
  expected <- diag(stein_kernel_matrix(reference, X, S))

  # Without k0_diag, the shared fallback evaluates one row at a time.
  eval_fn <- function(X, Y = NULL, precon = NULL) {
    eval_kernel(reference, X, Y)
  }
  grad_fn <- function(X, Y = NULL, precon = NULL) grad_x_kernel(reference, X, Y)
  trace_fn <- function(X, Y = NULL, precon = NULL) {
    trace_mixed_kernel(reference, X, Y)
  }
  custom <- custom_stein_kernel(eval_fn, grad_fn, trace_fn)
  expect_equal(
    as.numeric(steinsampling:::k0_diag(custom, X, S)), expected
  )

  # Function- and list-form kernels go through the `.k0_diag()` adapter.
  fn_kernel <- function(X, S_X, Y, S_Y) {
    stein_kernel_matrix(reference, X, S_X, Y, S_Y)
  }
  expect_equal(
    as.numeric(steinsampling:::.k0_diag(fn_kernel, X, S)), expected
  )
  expect_equal(
    as.numeric(steinsampling:::.k0_diag(list(k0_matrix = fn_kernel), X, S)),
    expected
  )
})
