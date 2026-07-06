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

expect_valid_htest <- function(x) {
  expect_s3_class(x, "htest")
  expect_true(is.numeric(x$statistic))
  expect_true(length(x$statistic) >= 1L)
  expect_true(all(is.finite(x$statistic)))
  expect_true(is.numeric(x$p.value))
  expect_length(x$p.value, 1L)
  expect_true(is.finite(x$p.value))
  expect_true(x$p.value >= 0 && x$p.value <= 1)
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
    f_vec = as.numeric((X[, 1] - 0.25)^2),
    D_new = -X
  )
}

toy_objective_2d <- function(X) {
  X <- as.matrix(X)
  center <- matrix(c(0.25, -0.25), nrow(X), 2L, byrow = TRUE)
  list(
    f_vec = rowSums((X - center)^2),
    D_new = -X
  )
}
