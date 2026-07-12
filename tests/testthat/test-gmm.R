test_that("gmm() stores one-dimensional mixture metadata", {
  set.seed(1)
  model <- gmm(
    nComp = 2,
    mu = c(-1, 1),
    sigma = array(1, c(1, 1, 2)),
    weights = c(0.4, 0.6),
    d = 1
  )

  expect_equal(model$nComp, 2)
  expect_equal(model$d, 1)
  expect_equal(sum(model$weights), 1)
})

test_that("gmm() validates mixture weights and covariance matrices", {
  expect_error(
    gmm(
      nComp = 2,
      mu = c(-1, 1),
      sigma = array(1, c(1, 1, 2)),
      weights = c(-0.2, 1.2),
      d = 1
    ),
    "weights must be finite"
  )
  expect_error(
    gmm(nComp = 2, mu = c(-1, 1), sigma = c(1, 1), weights = c(0, 0), d = 1),
    "positive sum"
  )
  expect_error(
    gmm(nComp = 1, mu = matrix(c(0, 0), ncol = 1),
        sigma = matrix(c(1, 2, 2, 1), 2), d = 2),
    "symmetric positive definite"
  )
})

test_that("gmm() respects dimension and infers complete model shapes", {
  set.seed(3)
  model <- gmm(d = 3)

  expect_equal(model$d, 3L)
  expect_equal(dim(model$mu), c(3L, 5L))
  expect_equal(dim(model$sigma), c(3L, 3L, 5L))
  expect_equal(dim(rgmm(model, n = 4)), c(4L, 3L))
  expect_error(rgmm(model, n = 1.5), "positive integer")
})

test_that("rgmm() and perturbgmm() return finite mixture samples", {
  set.seed(1)
  model <- gmm(
    nComp = 2,
    mu = c(-1, 1),
    sigma = array(1, c(1, 1, 2)),
    weights = c(0.4, 0.6),
    d = 1
  )

  x <- rgmm(model, n = 8)
  perturbed <- perturbgmm(model)

  expect_length(x, 8)
  expect_finite_numeric(x)
  expect_equal(perturbed$nComp, model$nComp)
})

test_that("GMM evaluators return posterior, likelihood, and score shapes", {
  set.seed(1)
  model <- gmm(
    nComp = 2,
    mu = c(-1, 1),
    sigma = array(1, c(1, 1, 2)),
    weights = c(0.4, 0.6),
    d = 1
  )
  x <- rgmm(model, n = 8)

  posterior <- posteriorgmm(model, x)
  likelihood <- likelihoodgmm(model = model, X = x)
  score <- scorefunctiongmm(model = model, X = x)
  score_eval <- get_score_evaluator(model)

  expect_equal(dim(posterior), c(8L, 2L))
  expect_equal(rowSums(posterior), rep(1, 8), tolerance = 1e-8)
  expect_length(likelihood, 8)
  expect_finite_numeric(likelihood)
  expect_true(all(likelihood > 0))
  expect_equal(dim(score), c(8L, 1L))
  expect_finite_numeric(score)
  expect_equal(dim(score_eval(matrix(x, ncol = 1))), c(8L, 1L))
  expect_length(score_eval(x), 8)
  expect_error(posteriorgmm(model, 1e308), "log densities are -Inf")
})

test_that("component means and precision cache match a toy mixture", {
  model <- gmm(
    nComp = 2,
    mu = c(-1, 1),
    sigma = array(c(1, 4), c(1, 1, 2)),
    weights = c(0.4, 0.6),
    d = 1
  )

  expect_equal(steinsampling:::get_component_mean(model, 1), -1)
  expect_equal(steinsampling:::get_component_mean(model, 2), 1)

  precision_cache <- steinsampling:::build_precision_cache(model)
  expect_length(precision_cache, 2L)
  expect_equal(dim(precision_cache[[1]]), c(1L, 1L))
  expect_equal(dim(precision_cache[[2]]), c(1L, 1L))
  expect_equal(as.numeric(precision_cache[[1]]), 1)
  expect_equal(as.numeric(precision_cache[[2]]), 0.25)

  narrow <- gmm(nComp = 1, mu = 0, sigma = 1e-8, d = 1)
  expect_equal(scorefunctiongmm(narrow, 1e-4), matrix(-1e4, nrow = 1))
})

test_that("row_logsumexp matches rowwise log-sums", {
  x <- matrix(c(0, 1, 2, 3), nrow = 2)
  expected <- c(log(exp(0) + exp(2)), log(exp(1) + exp(3)))
  expect_equal(steinsampling:::row_logsumexp(x), expected)
  expect_equal(steinsampling:::row_logsumexp(matrix(c(-Inf, -Inf), nrow = 1)), -Inf)
})

test_that("multivariate GMM score accepts one observation as a vector", {
  model <- gmm(
    nComp = 1, mu = matrix(c(0, 0), ncol = 1),
    sigma = diag(2), d = 2
  )

  expect_equal(scorefunctiongmm(model, c(1, -2)), matrix(c(-1, 2), nrow = 1))
  expect_equal(get_score_evaluator(model)(c(1, -2)), c(-1, 2))
})

test_that("multivariate GMM score matches the log-density gradient", {
  model <- gmm(
    nComp = 2,
    mu = matrix(c(-1, 0, 1, 0.5), nrow = 2),
    sigma = array(c(1, 0.2, 0.2, 2, 0.5, 0.1, 0.1, 1), c(2, 2, 2)),
    weights = c(0.4, 0.6), d = 2
  )
  x <- c(0.2, -0.3)
  eps <- 1e-6
  numeric_score <- vapply(seq_along(x), function(j) {
    xp <- xm <- x
    xp[j] <- xp[j] + eps
    xm[j] <- xm[j] - eps
    (log(likelihoodgmm(model, xp)) - log(likelihoodgmm(model, xm))) / (2 * eps)
  }, numeric(1L))

  expect_equal(as.numeric(scorefunctiongmm(model, x)), numeric_score, tolerance = 1e-6)
})

test_that("plotgmm writes a non-empty PDF", {
  set.seed(2)
  model <- gmm(nComp = 2, mu = c(-1, 1), sigma = array(1, c(1, 1, 2)), d = 1)
  x <- rgmm(model, n = 40)
  path <- tempfile(fileext = ".pdf")

  grDevices::pdf(path)
  closed <- FALSE
  on.exit(if (!closed) grDevices::dev.off(), add = TRUE)
  original_mar <- par("mar")
  expect_silent(plotgmm(data = x, mu = model$mu))
  expect_equal(par("mar"), original_mar)
  grDevices::dev.off()
  closed <- TRUE

  expect_true(file.exists(path))
  expect_gt(file.info(path)$size, 0)
})
