test_that("built-in Stein kernels match their oracle decomposition", {
  X1 <- matrix(c(-1, 0, 1), ncol = 1)
  S1 <- -X1
  rbf <- stein_kernel(type = "gaussian_rbf", h = 1)
  diff <- outer(X1[, 1], X1[, 1], "-")
  base <- exp(-0.5 * diff^2)

  expect_s3_class(rbf, "SteinKernel_gaussian_rbf")
  expect_equal(eval_kernel(rbf, X1), base)
  expect_equal(grad_x_kernel(rbf, X1)[, , 1L], -diff * base)
  expect_equal(trace_mixed_kernel(rbf, X1), (1 - diff^2) * base)
  expect_equal(cross_kernel(rbf, X1, S1), -diff^2 * base)
  expect_equal(stein_kernel_matrix(rbf, X1, S1),
               tcrossprod(S1) * base - diff^2 * base + (1 - diff^2) * base)

  theta_grad <- grad_theta_v_kernel(rbf, X = X1, vj = 0, grads_X = S1,
                                    g_block = matrix(1, nrow(X1), ncol(X1)))
  expect_length(theta_grad$grad_vj, 1L)
  expect_finite_numeric(theta_grad$grad_param)

  # The four-term identity must also hold under a preconditioner, in 2-d.
  X <- matrix(c(-1, 0.5, 0.25, -0.5, 1, 1.5), ncol = 2, byrow = TRUE)
  Y <- matrix(c(-0.75, 0.25, 1.25, -1), ncol = 2, byrow = TRUE)
  M <- matrix(c(2, 0.3, 0.3, 1), nrow = 2)
  for (kernel in list(
    stein_kernel(type = "gaussian_rbf", h = 1.2, precon = M),
    stein_kernel(type = "imq", c = 0.8, beta = -0.5, precon = M)
  )) {
    expect_equal(
      stein_kernel_matrix(kernel, X, -X, Y, -Y),
      tcrossprod(-X, -Y) * eval_kernel(kernel, X, Y) +
        cross_kernel(kernel, X, -X, Y, -Y) + trace_mixed_kernel(kernel, X, Y),
      tolerance = 1e-12
    )
  }

  # With s_p(x) = -x the score and position distances coincide, so the
  # score-distance IMQ has an exact reference.
  hess_log_p <- function(Z) {
    out <- array(0, dim = c(nrow(as.matrix(Z)), 2L, 2L))
    for (i in seq_len(dim(out)[1L])) out[i, , ] <- -diag(2)
    out
  }
  expect_equal(
    stein_kernel_matrix(stein_kernel_imq_score(1, -0.5, hess_log_p),
                        X, -X, Y, -Y),
    stein_kernel_matrix(stein_kernel(type = "imq", c = 1, beta = -0.5),
                        X, -X, Y, -Y),
    tolerance = 1e-12
  )

  # The lazy bandwidth squares the median distance, not the median squared
  # distance, and reproduces the fixed kernel at that scale.
  Z <- matrix(c(0, 1, 4, 10), ncol = 1)
  distances <- as.numeric(stats::dist(Z))
  expect_equal(find_median_distance(Z), stats::median(distances)^2)
  expect_equal(find_median_distance(Z), 25)
  expect_false(isTRUE(all.equal(find_median_distance(Z),
                                stats::median(distances^2))))
  expect_equal(
    stein_kernel_matrix(stein_kernel(type = "gaussian_rbf"), Z, -Z),
    stein_kernel_matrix(stein_kernel(type = "gaussian_rbf", h = 5), Z, -Z),
    tolerance = 1e-12
  )
})

test_that("custom Stein kernels satisfy the generic contract", {
  fns <- rbf_like_fns()
  kernel <- custom_stein_kernel(fns$eval, fns$grad_x, fns$trace_mixed,
                                custom_grad_mode = "numeric")
  X <- small_x(3)
  S <- normal_score(X)

  expect_s3_class(kernel, "SteinKernel_custom")
  # Matching the built-in RBF values implies the whole shape contract.
  expect_equal(stein_kernel_matrix(kernel, X, S),
               stein_kernel_matrix(stein_kernel(type = "gaussian_rbf", h = 1),
                                   X, S))
  expect_length(grad_theta_v_kernel(kernel, X, 0, S, matrix(1, 3, 1))$grad_vj, 1L)

  # A custom kernel's `grad_param` is ignored, not validated.
  with_param <- custom_stein_kernel(
    fns$eval, fns$grad_x, fns$trace_mixed,
    fssd_grad = function(k, X, vj, grads_X, g_block, M) {
      list(grad_vj = rep(1, ncol(X)), grad_param = NaN)
    }
  )
  expect_identical(
    grad_theta_v_kernel(with_param, X, 0, S, matrix(1, 3, 1))$grad_param, 0
  )

  expect_error(custom_stein_kernel(fns$eval, NULL, fns$trace_mixed),
               "must be functions")
  expect_error(
    grad_theta_v_kernel(
      custom_stein_kernel(fns$eval, fns$grad_x, fns$trace_mixed),
      X, 0, S, matrix(1, nrow(X), 1)
    ),
    "does not supply a `fssd_grad` operation"
  )
})

test_that("k0_diag matches the Stein-matrix diagonal for every kernel", {
  set.seed(404)
  X <- matrix(rnorm(12), ncol = 2L)
  S <- -X
  M <- matrix(c(2, 0.4, 0.4, 1.5), 2L, 2L)
  hess <- function(Z) {
    Z <- as.matrix(Z)
    array(rep(c(-1, 0.3, 0.3, -2), each = nrow(Z)), dim = c(nrow(Z), 2L, 2L))
  }
  reference <- stein_kernel(type = "gaussian_rbf", h = 1)
  fns <- rbf_like_fns(reference)

  kernels <- list(
    rbf = stein_kernel(type = "gaussian_rbf", h = 0.8),
    rbf_precon = stein_kernel(type = "gaussian_rbf", h = 0.8, precon = M),
    imq = stein_kernel(type = "imq", c = 1.3, beta = -0.4),
    imq_precon = stein_kernel(type = "imq", c = 1.3, beta = -0.4, precon = M),
    inverse_log = stein_kernel_inverse_log(alpha = 1.2, beta = -1),
    imq_score = stein_kernel_imq_score(alpha = 1.1, beta = -0.5,
                                       hess_log_p = hess),
    # No closed form here, so `k0_diag()` falls back to a row at a time.
    custom = custom_stein_kernel(fns$eval, fns$grad_x, fns$trace_mixed)
  )
  for (name in names(kernels)) {
    expect_equal(
      as.numeric(steinsampling:::k0_diag(kernels[[name]], X, S)),
      diag(stein_kernel_matrix(kernels[[name]], X, S)),
      info = name
    )
  }
})

test_that("squared distances survive a large common offset", {
  # Integer coordinates: the true squared distances are exact integers, and
  # every offset below 2^53 is exactly representable, so the computation must
  # reproduce them exactly however far the points are shifted.
  Xi <- matrix(c(0, 3, 7, 1, 4, 9), ncol = 2L)
  truth <- matrix(c(0, 18, 113, 18, 0, 41, 113, 41, 0), 3L, 3L)
  for (off in c(0, 1e4, 1e8, 1e12)) {
    expect_identical(
      steinsampling:::compute_cross_squared_distance(Xi + off), truth,
      info = format(off)
    )
    expect_identical(
      steinsampling:::compute_cross_squared_distance(Xi + off, Xi + off), truth,
      info = format(off)
    )
  }

  # Whatever the offset leaves in the data, the computation itself must add nothing: the
  # shifted points and the values they round to give the same distances.
  set.seed(31)
  X <- matrix(rnorm(20), ncol = 2L)
  Y <- matrix(rnorm(10), ncol = 2L)
  for (off in c(1e8, 1e12)) {
    expect_identical(
      steinsampling:::compute_cross_squared_distance(X + off, Y + off),
      steinsampling:::compute_cross_squared_distance((X + off) - off + off,
                                                     (Y + off) - off + off),
      info = format(off)
    )
  }

  # Kernel values are translation invariant, up to the offset's own rounding.
  # `stein_kernel_matrix` is checked too: it also uses the M^2 distance, which
  # `eval_kernel` never reaches.
  M <- matrix(c(2, 0.4, 0.4, 1.5), 2L, 2L)
  for (kernel in list(
    stein_kernel(type = "gaussian_rbf", h = 1),
    stein_kernel(type = "gaussian_rbf", h = 1, precon = M),
    stein_kernel(type = "imq", c = 1, beta = -0.5),
    stein_kernel(type = "imq", c = 1, beta = -0.5, precon = M),
    stein_kernel_inverse_log(alpha = 1, beta = -1)
  )) {
    for (off in c(1e4, 1e8)) {
      tol <- if (off > 1e6) 1e-4 else 1e-8
      expect_equal(eval_kernel(kernel, X + off, Y + off),
                   eval_kernel(kernel, X, Y), tolerance = tol,
                   info = paste(kernel$type, off))
      expect_equal(stein_kernel_matrix(kernel, X + off, -X, Y + off, -Y),
                   stein_kernel_matrix(kernel, X, -X, Y, -Y), tolerance = tol,
                   info = paste(kernel$type, off))
    }
  }
})

test_that("squared distances are never negative", {
  # Every combination below made the expansion ||x||^2 + ||y||^2 - 2x'y return
  # negative distances, and with them kernel values above their own maximum.
  set.seed(32)
  X <- matrix(rnorm(20), ncol = 2L)
  rot <- qr.Q(qr(matrix(c(2, 1, -1, 3), 2L, 2L)))
  ill <- rot %*% diag(c(1e6, 1)) %*% t(rot)          # condition number 1e6
  points <- list(X, X * 1e6, X + 1e8, X + 1e12, rbind(X, X),
                 matrix(0, 5L, 2L),
                 matrix(c(1e8, 1e8 + 1e-8, 1e8, 1e8), 2L, 2L))
  precons <- list(NULL, matrix(c(2, 0.4, 0.4, 1.5), 2L, 2L), ill)
  for (Z in points) for (M in precons) {
    D <- steinsampling:::compute_cross_squared_distance(Z, NULL, M)
    expect_true(all(is.finite(D)) && min(D) >= 0)
    expect_identical(D, t(D))
    expect_true(all(diag(D) == 0))
    # A decaying kernel cannot exceed its value at zero distance.
    for (scale in c(1, 0.01)) {
      expect_lte(max(eval_kernel(
        stein_kernel(type = "gaussian_rbf", h = scale, precon = M), Z)), 1)
      expect_lte(max(eval_kernel(
        stein_kernel(type = "imq", c = scale, beta = -0.5, precon = M), Z)),
        scale^(-1) * (1 + 1e-12))
    }
  }
})

test_that("a malformed hess_log_p is refused by both score-kernel paths", {
  bad <- stein_kernel_imq_score(1, -0.5, hess_log_p = function(X)
    array(1, dim = c(nrow(as.matrix(X)), 1L, 2L)))
  X <- matrix(c(0, 1), ncol = 1)

  expect_error(stein_kernel_matrix(bad, X, -X), "n x d x d")
  expect_error(steinsampling:::k0_diag(bad, X, -X), "n x d x d")
})
