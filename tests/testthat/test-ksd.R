test_that("KSD tests report their primitives on the documented scale", {
  X <- small_x(6)
  cases <- list(
    list(test = ksd_u_test, stat = ksd_u_statistic, boot = ksd_u_bootstrap,
         seed = 41),
    list(test = ksd_v_test, stat = ksd_v_statistic, boot = ksd_v_bootstrap,
         seed = 42)
  )

  for (cs in cases) {
    K0 <- ksd_uq_matrix(X, normal_score, scaling = 1)
    set.seed(cs$seed)
    boot <- cs$boot(K0, nboot = 5)
    set.seed(cs$seed)
    res <- cs$test(X, normal_score, scaling = 1, nboot = 5,
                   return_raw_boot = TRUE)

    expect_equal(dim(K0), c(6L, 6L))
    expect_finite_numeric(K0)
    expect_length(boot, 5L)
    expect_htest_contract(res)
    # The htest reports exactly the primitive's statistic and draws.
    expect_equal(unname(res$statistic), cs$stat(K0))
    expect_equal(res$bootstrap_samples, boot)
    expect_named(res, c("statistic", "p.value", "method", "data.name",
                        "parameter", "kernel", "bootstrap_samples"))
    expect_equal(unname(res$parameter[c("nboot", "scaling")]), c(5, 1))
    expect_s3_class(res$kernel, "SteinKernel_gaussian_rbf")

    # validate_integer() rounds and as.integer() truncates; the reported count
    # must follow the generated matrix, not the raw argument.
    rounded <- cs$test(X, normal_score, scaling = 1, nboot = 4.999999999,
                       return_raw_boot = TRUE)
    expect_equal(unname(rounded$parameter["nboot"]), 5)
    expect_length(rounded$bootstrap_samples, 5L)
  }

  # The chosen IMQ exponent reaches both the kernel and the reported settings.
  imq <- ksd_v_test(X, normal_score, scaling = 1, nboot = 3, kernel = "imq",
                    imq_beta = -0.75)
  expect_equal(unname(imq$parameter["imq_beta"]), -0.75)
  expect_equal(imq$kernel$beta, -0.75)

  # One matrix builder under two names, sharing the tests' own defaults.
  expect_identical(ksd_uq_matrix, ksd_vq_matrix)
  shared <- c("scaling", "kernel", "imq_beta")
  for (f in list(ksd_u_test, ksd_v_test)) {
    expect_identical(formals(f)[shared], formals(ksd_uq_matrix)[shared])
  }
})

test_that("prepare_ksd_inputs resolves the kernel and its median scale", {
  X <- matrix(c(0, 1, 4, 10), ncol = 1)
  precon <- matrix(4, nrow = 1)
  lazy <- stein_kernel(type = "gaussian_rbf", precon = precon)

  plain <- steinsampling:::.prepare_ksd_inputs(X, normal_score, scaling = NULL)
  expect_named(plain, c("X", "scores", "kernel_obj", "kernel_name", "scaling"))
  expect_equal(plain$kernel_name, "gaussian_rbf")
  expect_gt(plain$scaling, 0)

  # A lazy kernel takes its scale from the preconditioned metric, in place.
  prepped <- steinsampling:::.prepare_ksd_inputs(X, normal_score, kernel = lazy)
  expect_equal(prepped$kernel_obj$precon, precon)
  expect_equal(prepped$kernel_obj$scale2,
               find_median_distance(X %*% t(chol(precon))))
  expect_true(is.na(lazy$scale2))

  # A kernel supplying the whole Stein matrix satisfies both engine paths.
  hess_log_p <- function(Z) array(-1, dim = c(nrow(as.matrix(Z)), 1L, 1L))
  fused <- stein_kernel_imq_score(1, -0.5, hess_log_p)
  expect_htest_contract(
    ksd_v_test(small_x(4), normal_score, kernel = fused, nboot = 3))
  expect_htest_contract(ksd_v_test(
    small_x(4), normal_score, kernel = fused, nboot = 3,
    block_size = 2, block_threshold = 1
  ))
})

test_that("KSD rejects bad samples, scores, and bootstrap controls", {
  X <- small_x(4)
  calls <- 0L
  counting_score <- function(x) {
    calls <<- calls + 1L
    normal_score(x)
  }

  bad_calls <- list(
    list(quote(ksd_u_test(matrix(letters[1:4], ncol = 1), normal_score,
                          nboot = 2)), "numeric"),
    list(quote(ksd_uq_matrix(X, function(x) matrix(0, nrow(as.matrix(x)) - 1L, 1L),
                             scaling = 1)), "shape"),
    list(quote(ksd_u_bootstrap(diag(2), nboot = 0)), "positive"),
    # Markov signs need an explicit change probability strictly inside (0, 1).
    list(quote(ksd_v_test(X, normal_score, nboot = 2, boot_method = "markov",
                          change_prob = 1)), "\\(0, 1\\)"),
    list(quote(ksd_v_bootstrap(diag(2), nboot = 2, boot_method = "markov")),
         "supplied explicitly"),
    list(quote(ksd_u_test(X, normal_score, nboot = 1e20)), "no larger than"),
    list(quote(ksd_v_test(X, normal_score, nboot = 2, block_size = 1.5)),
         "must be a positive integer"),
    list(quote(ksd_u_test(X, normal_score, nboot = 2, block_threshold = 1e20)),
         "no larger than"),
    # These two are refused before `score_function()` ever runs.
    list(quote(ksd_u_test(X, counting_score, nboot = -3)), "positive integer"),
    list(quote(ksd_v_test(X, counting_score, nboot = 2, boot_method = "markov")),
         "supplied explicitly")
  )
  for (cs in bad_calls) expect_error(eval(cs[[1]]), cs[[2]])
  expect_identical(calls, 0L)

  for (bad in list(NA, 1, "yes", c(TRUE, TRUE))) {
    expect_error(
      ksd_u_test(X, normal_score, nboot = 2, return_raw_boot = bad),
      "must be TRUE or FALSE"
    )
  }

  # Shape and finiteness are the whole `W_mat` contract: uncentered,
  # non-sign multipliers are accepted by both bootstraps.
  for (boot_fn in list(ksd_u_bootstrap, ksd_v_bootstrap)) {
    for (bad in list(NA_real_, Inf, NaN)) {
      expect_error(boot_fn(diag(3), W_mat = matrix(c(1, -1, bad), nrow = 3L)),
                   "finite numeric matrix")
    }
    expect_length(boot_fn(diag(3), W_mat = matrix(c(0.5, 2, -3), nrow = 3L)), 1L)
  }
})

test_that("KSD reports degenerate matrices and finite-bootstrap p-values", {
  X <- small_x(8)

  expect_equal(steinsampling:::.bootstrap_pvalue_right_tail(c(0, 1, 2), 1), 3 / 4)

  for (test_fn in list(ksd_u_test, ksd_v_test)) {
    expect_warning(test_fn(X, normal_score, scaling = 1e-8, nboot = 3),
                   "nearly diagonal")
    expect_silent(test_fn(X, normal_score, scaling = 1, nboot = 3))
  }
  # The check is on the assembled k_0p, so it covers IMQ as well as the RBF.
  expect_warning(
    ksd_u_test(X, normal_score, scaling = 1e-24, nboot = 3, kernel = "imq"),
    "nearly diagonal"
  )

  # Every input is finite; the tcrossprod term is what overflows.
  Z <- matrix(c(0, 1e-9, 2e-9), ncol = 1)
  huge_score <- function(x) matrix(rep(1e200, nrow(as.matrix(x))), ncol = 1)
  expect_error(ksd_uq_matrix(Z, huge_score, scaling = 1), "only finite values")
  expect_error(ksd_u_test(Z, huge_score, scaling = 1, nboot = 2),
               "only finite values")
})
