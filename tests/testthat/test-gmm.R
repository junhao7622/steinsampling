test_that("gmm() validates its arguments and infers complete model shapes", {
  set.seed(3)
  model <- gmm(d = 3)

  expect_equal(model$d, 3L)
  expect_equal(dim(model$mu), c(3L, 5L))
  expect_equal(dim(model$sigma), c(3L, 3L, 5L))
  expect_equal(dim(rgmm(model, n = 4)), c(4L, 3L))
  expect_equal(gmm(nComp = 2, mu = c(-1, 1), weights = c(2, 3), d = 1)$weights,
               c(0.4, 0.6))

  bad_calls <- list(
    list(quote(gmm(nComp = 2, mu = c(-1, 1), sigma = array(1, c(1, 1, 2)),
                   weights = c(-0.2, 1.2), d = 1)), "weights must be finite"),
    list(quote(gmm(nComp = 1, mu = matrix(c(0, 0), ncol = 1),
                   sigma = matrix(c(1, 2, 2, 1), 2), d = 2)),
         "symmetric positive definite"),
    list(quote(rgmm(list(d = 1))), "returned by"),
    list(quote(rgmm(model, n = 1.5)), "positive integer")
  )
  for (cs in bad_calls) expect_error(eval(cs[[1]]), cs[[2]])
})

test_that("GMM samples, density, and cached score have the required shapes", {
  set.seed(1)
  model <- toy_gmm_1d()
  x <- rgmm(model, n = 8)
  score_eval <- get_score_evaluator(model)
  density <- densitygmm(model, x)
  responsibilities <-
    steinsampling:::.gmm_responsibilities(model, matrix(x, ncol = 1))

  expect_length(x, 8)
  expect_finite_numeric(x)
  expect_equal(dim(responsibilities), c(8L, 2L))
  expect_equal(rowSums(responsibilities), rep(1, 8), tolerance = 1e-8)
  expect_length(density, 8)
  expect_true(all(density > 0))
  expect_equal(dim(score_eval(matrix(x, ncol = 1))), c(8L, 1L))
  expect_length(score_eval(x), 8)
  expect_error(score_eval(1e308), "log densities are -Inf")

  # The row log-sum-exp behind them is exact and -Inf-safe.
  lse <- steinsampling:::.row_logsumexp
  expect_equal(lse(matrix(c(0, 1, 2, 3), nrow = 2)),
               log(c(exp(0) + exp(2), exp(1) + exp(3))))
  expect_equal(lse(matrix(-Inf, nrow = 1, ncol = 2)), -Inf)
})

test_that("GMM scores match the log-density gradient at full precision", {
  precision_cache <- steinsampling:::.build_precision_cache(
    toy_gmm_1d(sigma = array(c(1, 3), c(1, 1, 2)))
  )
  expect_length(precision_cache, 2L)
  # `chol2inv(chol(v))` rounds twice for a 1 x 1 covariance; the scalar path
  # must not.
  expect_identical(as.numeric(precision_cache[[2]]), 1 / 3)

  # A very narrow component must not lose precision in the cache.
  narrow <- gmm(nComp = 1, mu = 0, sigma = 1e-8, d = 1)
  expect_equal(get_score_evaluator(narrow)(matrix(1e-4, nrow = 1)),
               matrix(-1e4, nrow = 1))

  # A single observation may arrive as a bare vector.
  spherical <- gmm(nComp = 1, mu = matrix(c(0, 0), ncol = 1), sigma = diag(2),
                   d = 2)
  expect_equal(get_score_evaluator(spherical)(c(1, -2)), c(-1, 2))

  model <- gmm(
    nComp = 2, mu = matrix(c(-1, 0, 1, 0.5), nrow = 2),
    sigma = array(c(1, 0.2, 0.2, 2, 0.5, 0.1, 0.1, 1), c(2, 2, 2)),
    weights = c(0.4, 0.6), d = 2
  )
  x <- c(0.2, -0.3)
  eps <- 1e-6
  numeric_score <- vapply(seq_along(x), function(j) {
    xp <- xm <- x
    xp[j] <- xp[j] + eps
    xm[j] <- xm[j] - eps
    (log(densitygmm(model, xp)) - log(densitygmm(model, xm))) / (2 * eps)
  }, numeric(1L))
  expect_equal(as.numeric(get_score_evaluator(model)(x)), numeric_score,
               tolerance = 1e-6)
})

test_that("summary() lays out one-dimensional mixtures with several components", {
  s <- summary(toy_gmm_1d(sigma = array(c(1, 4), c(1, 1, 2))))

  expect_equal(s$components, matrix(
    c(0.4, 0.6, -1, 1, 1, 2), nrow = 2,
    dimnames = list(c("comp1", "comp2"), c("weight", "mean_x1", "sd_x1"))
  ))
  expect_named(s, c("nComp", "d", "components"))
})
