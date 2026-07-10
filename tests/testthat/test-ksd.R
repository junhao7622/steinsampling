test_that("KSD U helpers return matrix, statistic, and bootstrap outputs", {
  set.seed(3)
  X <- small_x(6)
  U <- ksd_uq_matrix(X, normal_score, scaling = 1)
  stat <- ksd_u_statistic(U)
  boot <- ksd_u_bootstrap(U, nboot = 5)
  res <- ksd_u_test(X, normal_score, scaling = 1, nboot = 5)

  expect_equal(dim(U), c(6L, 6L))
  expect_finite_numeric(U)
  expect_length(stat, 1L)
  expect_finite_numeric(stat)
  expect_length(boot, 5L)
  expect_finite_numeric(boot)
  expect_valid_htest(res)
})

test_that("prepare_ksd_u_inputs validates samples, scores, and scaling", {
  X <- small_x(5)
  prep <- steinsampling:::prepare_ksd_u_inputs(
    X, normal_score, scaling = NULL, median_seed = 11
  )
  med <- find_median_distance(X, use_sampling = FALSE)

  expect_named(prep, c("X", "grads", "kernel_obj", "kernel_name", "scaling"))
  expect_equal(dim(prep$X), c(5L, 1L))
  expect_equal(dim(prep$grads), c(5L, 1L))
  expect_equal(prep$kernel_name, "gaussian_rbf")
  expect_gt(prep$scaling, 0)
  expect_length(med, 1L)
  expect_gt(med, 0)
})

test_that("KSD V helpers return matrix, statistic, and bootstrap outputs", {
  set.seed(4)
  X <- small_x(6)
  U <- ksd_vq_matrix(X, normal_score, scaling = 1)
  stat <- ksd_v_statistic(U)
  boot <- ksd_v_bootstrap(U, nboot = 5, boot_method = "rademacher")
  res <- ksd_v_test(X, normal_score, scaling = 1, nboot = 5)

  expect_equal(dim(U), c(6L, 6L))
  expect_finite_numeric(U)
  expect_length(stat, 1L)
  expect_finite_numeric(stat)
  expect_length(boot, 5L)
  expect_finite_numeric(boot)
  expect_valid_htest(res)
})

test_that("KSD tests return htest metadata and raw bootstrap samples", {
  set.seed(101)
  X <- small_x(6)

  ksd_u <- ksd_u_test(
    X, normal_score,
    scaling = 1, nboot = 5,
    return_raw_boot = TRUE
  )
  ksd_v <- ksd_v_test(
    X, normal_score,
    scaling = 1, nboot = 5,
    return_raw_boot = TRUE
  )

  expect_htest_contract(ksd_u)
  expect_named(ksd_u, c(
    "statistic", "p.value", "method", "data.name", "parameter",
    "bootstrap_samples"
  ))
  expect_true(grepl("gaussian_rbf", ksd_u$method, fixed = TRUE))
  expect_equal(unname(ksd_u$parameter["nboot"]), 5)
  expect_equal(unname(ksd_u$parameter["scaling"]), 1)
  expect_length(ksd_u$bootstrap_samples, 5L)
  expect_finite_numeric(ksd_u$bootstrap_samples)

  expect_htest_contract(ksd_v)
  expect_named(ksd_v, c(
    "statistic", "p.value", "method", "data.name", "parameter",
    "bootstrap_samples"
  ))
  expect_true(grepl("gaussian_rbf", ksd_v$method, fixed = TRUE))
  expect_equal(unname(ksd_v$parameter["nboot"]), 5)
  expect_equal(unname(ksd_v$parameter["scaling"]), 1)
  expect_length(ksd_v$bootstrap_samples, 5L)
  expect_finite_numeric(ksd_v$bootstrap_samples)
})

test_that("KSD bootstrap helpers return finite small-sample vectors", {
  set.seed(102)
  U <- matrix(c(1, 0.25, 0.25, 2), nrow = 2)
  u_boot <- ksd_u_bootstrap(U, nboot = 4)
  v_boot <- ksd_v_bootstrap(U, nboot = 4)

  expect_length(u_boot, 4L)
  expect_length(v_boot, 4L)
  expect_finite_numeric(u_boot)
  expect_finite_numeric(v_boot)
})

test_that("KSD rejects nonnumeric samples and bad score shapes", {
  X <- small_x(4)
  bad_score <- function(x) matrix(0, nrow = nrow(as.matrix(x)) - 1L, ncol = 1L)

  expect_error(ksd_u_test(matrix(letters[1:4], ncol = 1), normal_score, nboot = 2), "numeric")
  expect_error(ksd_uq_matrix(X, bad_score, scaling = 1), "shape")
})

test_that("KSD bootstrap inputs reject invalid counts and probabilities", {
  X <- small_x(4)

  expect_error(ksd_u_bootstrap(diag(2), nboot = 0), "positive")
  expect_error(ksd_v_bootstrap(diag(2), nboot = 0), "positive")
  expect_error(
    ksd_v_test(X, normal_score, nboot = 2, boot_method = "markov", change_prob = 2),
    "\\[0, 1\\]"
  )
  expect_error(
    ksd_v_test(X, normal_score, nboot = 2, boot_method = "markov"),
    "supplied explicitly"
  )
  expect_error(
    ksd_v_bootstrap(diag(2), nboot = 2, boot_method = "markov"),
    "supplied explicitly"
  )
  expect_length(
    ksd_v_bootstrap(diag(2), nboot = 2, boot_method = "markov", change_prob = 0.1),
    2L
  )
})

test_that("KSD p-values use the documented finite-bootstrap correction", {
  expect_equal(
    steinsampling:::bootstrap_pvalue_right_tail(c(0, 1, 2), 1),
    3 / 4
  )
})
