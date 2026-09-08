test_that("optimizer factories return the documented result record", {
  set.seed(7)
  cases <- list(
    list(opt = fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE),
         obj = toy_objective_1d, X = matrix(0, ncol = 1)),
    list(opt = fmin_mc(lb = -1, ub = 1, n_mc = 5, mu0 = 0,
                       Sigma0 = matrix(0.1), delay = 2),
         obj = toy_objective_1d, X = matrix(0, ncol = 1)),
    list(opt = fmin_nm(lb = c(-1, -1), ub = c(1, 1), n_res = 1,
                       mu0 = c(0, 0), Sigma0 = diag(0.1, 2), delay = 2,
                       control = list(maxit = 5, reltol = 1e-2)),
         obj = toy_objective_2d, X = matrix(c(0, 0), ncol = 2))
  )
  for (cs in cases) {
    res <- cs$opt(cs$obj, cs$X)
    expect_named(res, c("x_min", "d_min", "f_min", "n_eval"))
    expect_gt(res$n_eval, 0)
  }

  # A growing grid takes its resolution from the iteration it is handed.
  grid <- fmin_grid(lb = -1, ub = 1, n0 = 5, grow = TRUE)
  X_curr <- matrix(c(-0.5, 0.5), ncol = 1)
  expect_equal(grid(toy_objective_1d, X_curr, t = 4)$n_eval, 5 + round(sqrt(4)))
  expect_equal(grid(toy_objective_1d, X_curr, t = 100)$n_eval,
               5 + round(sqrt(100)))
})

test_that("optimizer factories validate their construction controls", {
  expect_error(fmin_grid(-1, 1, n0 = 0), "n0 entry")
  expect_error(fmin_grid(c(-1, -1), c(1, 1), n0 = c(2, 2, 2)), "length-d")
  expect_error(fmin_grid(-1, 1, grow = 1), "grow")

  expect_error(fmin_mc(-1, 1, n_mc = 0), "n_mc")
  expect_error(fmin_mc(c(-1, -1), c(1, 1), mu0 = 0), "mu0")
  expect_error(
    fmin_mc(c(-1, -1), c(1, 1), Sigma0 = matrix(1, 1, 1)),
    "Sigma0"
  )
  expect_error(fmin_mc(-1, 1, Sigma0 = matrix(0)), "positive definite")
  expect_error(fmin_mc(-1, 1, sigsq = -1), "sigsq")
  expect_error(fmin_mc(-1, 1, delay = -1), "delay")

  expect_error(fmin_nm(-1, 1, n_res = 0), "n_res")
  expect_error(fmin_nm(-1, 1, control = 1), "control")
})

test_that("stein_thinning selects valid indices for both `pre` forms", {
  set.seed(9)
  X <- small_x(5)
  S <- normal_score(X)
  Z <- cbind(c(-1, 0, 1), c(0, 1, 0))
  M <- diag(c(2, 0.5))

  # A non-default kernel parameter now travels in a stein_kernel() object.
  by_rule <- suppressWarnings(stein_thinning(
    X, S = S, m = 2, kernel = stein_kernel("gaussian_rbf", h = 1),
    pre_subsample = 5
  ))
  expect_length(by_rule, 2L)
  expect_true(all(by_rule >= 1 & by_rule <= nrow(X)))
  expect_length(stein_thinning(Z, S = -Z, m = 2, pre = M), 2L)
  # A matrix `pre` is used as the preconditioner unchanged.
  expect_equal(
    steinsampling:::.build_thinning_precon(Z, m = 2, pre = M, pre_subsample = 1),
    M
  )

  bad_calls <- list(
    list(quote(stein_thinning(X, S = S, m = 0)), "`m` must be a positive integer"),
    list(quote(stein_thinning(X, S = S, m = 1, pre = "sclmed")), "m > 1"),
    # No `...` to swallow it, so R itself rejects a mistyped kernel parameter.
    list(quote(stein_thinning(X, S = S, m = 2, betta = -0.9)), "unused argument"),
    list(quote(stein_thinning(matrix(c(0, Inf), ncol = 1), S = S, m = 2)),
         "`X` must contain only finite numeric values"),
    list(quote(stein_thinning(X, S = matrix(c(0, NA_real_), ncol = 1), m = 2)),
         "`S` must contain only finite numeric values"),
    list(quote(stein_thinning(Z, S = -Z, m = 2, pre = "med",
                              pre_subsample_method = "unused")), "should be one of")
  )
  for (cs in bad_calls) expect_error(eval(cs[[1]]), cs[[2]])

  # Subsampling controls are checked only when they are used.
  for (pre in list(M, "smpcov")) {
    expect_no_error(stein_thinning(
      Z, S = -Z, m = 2, pre = pre, pre_subsample_method = "unused"
    ))
  }
  expect_no_error(stein_thinning(
    Z, S = -Z, m = 2, pre = "med", pre_subsample = c(1, 3),
    pre_subsample_method = "unused"
  ))

  # No kernel row is needed after the final selection.
  local_mocked_bindings(
    .kP_row_vector = function(...) stop("unexpected kernel-row evaluation"),
    .package = "steinsampling"
  )
  expect_length(stein_thinning(X, S = S, m = 1, pre = diag(1)), 1L)
})

test_that("Stein thinning median preconditioners use the documented scale", {
  repeated <- matrix(0, nrow = 4, ncol = 1)
  build <- function(...) steinsampling:::.build_thinning_precon(...)

  expect_warning(med <- build(repeated, m = 2, pre = "med", pre_subsample = 4),
                 "Median pairwise distance is zero")
  expect_warning(sclmed <- build(repeated, m = 2, pre = "sclmed",
                                 pre_subsample = 4),
                 "Median pairwise distance is zero")

  # A zero median distance falls back to unit scale; "sclmed" then adds log(m).
  expect_equal(med, matrix(1, nrow = 1))
  expect_equal(sclmed, matrix(log(2), nrow = 1))
  expect_equal(
    build(matrix(c(0, 1, 2), ncol = 1), m = 4, pre = "sclmed",
          pre_subsample = Inf),
    matrix(log(4), nrow = 1)
  )
  expect_error(
    build(matrix(0, nrow = 1, ncol = 1), m = 1, pre = "sclmed",
          pre_subsample = 1),
    "m > 1"
  )
})

test_that("stein_points bills its evaluations and refuses bad candidate sets", {
  score_rows <- 0L
  log_p_rows <- 0L
  counted_score <- function(X) {
    X <- as.matrix(X)
    score_rows <<- score_rows + nrow(X)
    -X
  }
  counted_log_p <- function(X) {
    X <- as.matrix(X)
    log_p_rows <<- log_p_rows + nrow(X)
    -0.5 * rowSums(X^2)
  }
  grid <- fmin_grid(lb = -1, ub = 1, n0 = 3, grow = FALSE)

  points <- stein_points(counted_score, stein_kernel(type = "gaussian_rbf", h = 1),
                         n_points = 2, d = 1, optimizer = grid,
                         log_p = counted_log_p)
  expect_identical(class(points), "stein_points")
  expect_equal(dim(points$X), c(2L, 1L))
  expect_equal(points$n_eval, c(4L, 3L))
  expect_equal(log_p_rows, 3L)
  expect_equal(score_rows, 4L)

  # Truncation that admits nothing, a kernel with a negative diagonal, and a
  # bandwidth the algorithm cannot pin down are all reported.
  expect_error(
    stein_points(normal_score, stein_kernel(type = "imq", c = 1, beta = -0.5),
                 n_points = 2, d = 1, optimizer = grid, x_init = 0,
                 c2 = 2 * log(2) / 0.5, truncation = "upper"),
    "found no candidate"
  )
  negative <- steinsampling:::new_stein_kernel(
    "negative_stub",
    k0_matrix = function(k, X, S_X, Y, S_Y, M) matrix(-1, nrow(X), nrow(Y)),
    k0_diag = function(k, X, S_X, M) rep(-1, nrow(X))
  )
  expect_warning(stein_points(normal_score, negative, 1, 1, x_init = 0),
                 "Accumulated squared KSD is negative")
  expect_error(
    stein_points(normal_score, stein_kernel(type = "gaussian_rbf"), 2, 1,
                 grid, x_init = 0),
    "fixed h"
  )
})

test_that("stein_points recomputes the selected contribution for its KSD", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  misleading_optimizer <- function(objective, X_curr, t) {
    evaluated <- objective(matrix(1, nrow = 1L))
    list(
      x_min = 1,
      d_min = evaluated$scores[1L, ],
      f_min = 999,
      n_eval = 1L
    )
  }

  for (method in c("greedy", "herding")) {
    fit <- stein_points(
      normal_score, kernel, n_points = 2, d = 1,
      optimizer = misleading_optimizer, method = method, x_init = 0
    )
    direct <- sqrt(sum(stein_kernel_matrix(kernel, fit$X, fit$D))) / 2
    expect_equal(fit$ksd[2L], direct)
  }
})

test_that("stein_codescent keeps a non-increasing KSD", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  X0 <- matrix(c(-1, 1), ncol = 1)

  # An out-of-range search cannot improve the set, so the incumbent survives.
  rejected <- stein_codescent(
    X0, normal_score, kernel, n_iter = 1,
    optimizer = fmin_grid(lb = 10, ub = 11, n0 = 2, grow = FALSE)
  )
  # With one point the update reduces to minimizing k0(x, x).
  single <- stein_codescent(
    matrix(0.75, nrow = 1), normal_score, kernel, n_iter = 1,
    optimizer = fmin_grid(lb = -1, ub = 1, n0 = 5, grow = FALSE)
  )

  expect_identical(class(rejected), "stein_codescent")
  expect_equal(rejected$X, X0)
  expect_equal(rejected$D, normal_score(X0))
  expect_equal(rejected$n_eval, 4L)
  expect_equal(single$X, matrix(0, nrow = 1))
  expect_equal(single$n_eval, 6L)
})

test_that("each point-set summary names what it counts and keeps its diagnostic", {
  kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
  grid <- fmin_grid(lb = -3, ub = 3, n0 = 3, grow = FALSE)
  X0 <- matrix(c(-1, 0, 1), ncol = 1)
  shown <- function(o) paste(capture.output(print(o)), collapse = "\n")

  # stein_points counts optimizer evaluations, which mix score and log_p rows.
  pts_fit <- stein_points(normal_score, kernel, n_points = 4, d = 1,
                          optimizer = grid, log_p = normal_log_p, seed = 1)
  pts <- summary(pts_fit)
  expect_identical(pts$evaluation_label, "objective evaluations")
  expect_false(is.null(pts$ksd_last))
  expect_null(pts$objective_last)
  expect_match(shown(pts_fit), "Stein Points (greedy)", fixed = TRUE)

  # Coordinate objectives stay in the raw result; the summary reports updates.
  cd_fit <- stein_codescent(X0, normal_score, kernel, n_iter = 3,
                            optimizer = grid, seed = 1)
  cd <- summary(cd_fit)
  expect_identical(cd$evaluation_label, "objective evaluations")
  expect_identical(cd$n_iter, 3L)
  expect_null(cd$objective_last)
  expect_null(cd$objective_first)
  expect_null(cd$ksd_last)
  expect_match(shown(cd), "updates: 3")
  expect_match(shown(cd_fit), "updates=3")

  # svgd counts score rows, and only those spent on the update loop.
  sv_fit <- svgd(X0, normal_score, n_iter = 2, step_size = 0.05)
  sv <- summary(sv_fit)
  expect_identical(sv$evaluation_label, "score evaluations (update loop)")
  expect_identical(sv$n_eval_total, 2L * nrow(X0))
  expect_null(sv$ksd_last)
  expect_null(sv$objective_last)
  # Its print carries the settings that define the run, and nothing else.
  expect_match(shown(sv_fit), "SVGD  n=3  d=1  iterations=2  step=0.05")
  expect_false(grepl("n_eval", shown(sv_fit)))

  for (s in list(pts, cd, sv)) {
    expect_null(s$coordinates)
    expect_null(s$ksd_first)
  }
})
