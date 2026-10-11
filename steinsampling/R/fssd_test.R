# The three FSSD tests come first, then compute_tau, fssd_statistic and
# fssd_null_pvalue, then the FSSD-opt optimizer and its gradient.

#' Finite Set Stein Discrepancy goodness-of-fit test
#'
#' Tests independent and identically distributed observations against a target
#' distribution specified by its score function, using Stein features at a
#' finite set of test locations.
#' `variant = "opt"` selects those locations on training rows and tests on
#' held-out rows ([fssd_opt_test()]); `variant = "rand"` draws them and tests
#' on the same rows, so its p-value is heuristic ([fssd_rand_test()]).
#'
#' @details
#' Once the test locations and kernel have been selected, [compute_tau()]
#' evaluates the Stein features at those locations. [fssd_statistic()] combines
#' the feature rows into the scaled test statistic, and [fssd_null_pvalue()]
#' simulates its null distribution and computes the p-value.
#'
#' @param X Numeric vector or matrix containing `n` independent and identically
#'   distributed observations. Rows are observations and columns are
#'   coordinates; a vector is treated as an
#'   \eqn{n\times 1} matrix.
#' @param score_function Function taking an \eqn{n\times d} matrix and
#'   returning the corresponding \eqn{n\times d} matrix of target scores.
#' @param variant `"opt"` (default) or `"rand"`.
#' @param J Number of test locations.
#' @param n_simulations Number of null draws.
#' @param kernel `"gaussian_rbf"`, `"imq"`, or a `SteinKernel` object.
#' @param scaling Positive squared kernel scale: \eqn{h^2} for Gaussian RBF,
#'   \eqn{c^2} for IMQ; a custom kernel may expose its own. It is final for
#'   FSSD-rand and an initial value for FSSD-opt. `NULL` uses a median-based
#'   value when the kernel needs a squared scale. A finite squared scale in a
#'   supplied `SteinKernel` takes precedence; `scaling` is ignored when the
#'   kernel exposes no squared scale.
#' @param train_ratio Requested FSSD-opt training fraction in \eqn{(0,1)};
#'   the random split keeps at least two rows in each part.
#' @param gamma Positive regularizer in the FSSD-opt criterion.
#' @param maxit Maximum L-BFGS-B iterations for FSSD-opt.
#' @param locs_bounds_frac Number of training-set standard deviations added
#'   below each coordinate minimum and above each coordinate maximum.
#' @param scale_lower,scale_upper Bounds applied to the squared kernel scale
#'   when that scale is optimized. `NULL` uses 0.01 and 100 times the median
#'   squared distance of the training rows, kept within 1e-3 and 1e5.
#' @param imq_beta Finite IMQ exponent \eqn{\beta<0}.
#'
#' @return An `htest` object. `statistic` is
#'   \eqn{S_{\tilde n}=\tilde n\widehat{FSSD}^2}, where \eqn{\tilde n} is the
#'   held-out size for FSSD-opt and \eqn{n} for FSSD-rand; `p.value` is its
#'   simulated right-tail p-value; `info$V` is the test-location matrix, and
#'   `info` contains variant-specific diagnostics.
#' Use `print(fit)` to display the test statistic and p-value.
#' @references
#' Jitkrittum, W., Xu, W., Szabó, Z., Fukumizu, K., and Gretton, A. (2017).
#' A Linear-Time Kernel Goodness-of-Fit Test.
#' Advances in Neural Information Processing Systems, 30, 262--271.
#' Curran Associates, Inc.
#' \url{https://papers.nips.cc/paper_files/paper/2017/hash/979d472a84804b9f647bc185a877a8b5-Abstract.html}.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(200, mean = 0.5), ncol = 1)
#' score <- function(X) -X  # Target: N(0, 1)
#' fit <- fssd_test(X, score, J = 3, scaling = 1, train_ratio = 0.5)
#' c(statistic = unname(fit$statistic), p.value = fit$p.value)
#' fit$info$convergence  # 0: successful optimizer termination.
#' @export
fssd_test <- function(X, score_function,
                      variant = c("opt", "rand"),
                      J = 5,
                      n_simulations = 2000,
                      kernel = c("gaussian_rbf", "imq"),
                      scaling = NULL,
                      train_ratio = 0.2,
                      gamma = 1e-4,
                      maxit = 100,
                      locs_bounds_frac = 10,
                      scale_lower = NULL,
                      scale_upper = NULL,
                      imq_beta = -0.5) {
  variant <- match.arg(variant)
  # Record the caller's expression for X here. After forwarding, the variant
  # would only see the name X.
  data_name <- deparse(substitute(X))

  res <- if (variant == "opt") {
    fssd_opt_test(
      X = X, score_function = score_function, J = J,
      n_simulations = n_simulations,
      kernel = kernel, scaling = scaling,
      train_ratio = train_ratio,
      gamma = gamma, maxit = maxit,
      locs_bounds_frac = locs_bounds_frac,
      scale_lower = scale_lower, scale_upper = scale_upper,
      imq_beta = imq_beta
    )
  } else {
    fssd_rand_test(
      X = X, score_function = score_function, J = J,
      n_simulations = n_simulations,
      kernel = kernel, scaling = scaling,
      imq_beta = imq_beta
    )
  }
  res$data.name <- data_name
  res
}


#' FSSD test with random test locations
#'
#' Fits a Gaussian distribution to `X`, draws the test locations
#' \eqn{L = \{v_1,\ldots,v_J\}}, and uses all rows of `X` to compute the FSSD
#' statistic and its plug-in null approximation.
#'
#' @details
#' Let \eqn{\bar x} and \eqn{\widehat\Sigma_X} be the mean and fitted
#' covariance of `X`. The locations are drawn independently as
#' \deqn{
#' v_j \sim N_d(\bar x,\widehat\Sigma_X),
#' \qquad j=1,\ldots,J.
#' }
#' The same rows are used for the statistic and null simulation. The locations,
#' and any data-derived kernel scale, use the tested rows, so the reported
#' p-value is heuristic. See [fssd_null_pvalue()] for the independence condition.
#' For the common interface and choice of variant, see [fssd_test()].
#' @inheritParams fssd_test
#' @param scaling Final positive squared kernel scale (\eqn{h^2} for Gaussian
#'   RBF, \eqn{c^2} for IMQ). `NULL` uses the square of the median of all pairwise
#'   distances in `X` when the kernel needs a squared scale. A Gaussian RBF
#'   kernel carrying `precon` uses its preconditioned distance; otherwise the
#'   distance is Euclidean. A finite squared scale in a supplied `SteinKernel`
#'   takes precedence; `scaling` is ignored when the kernel exposes no squared
#'   scale.
#' @return
#' An object of class `htest`. `statistic` is
#' \eqn{S_n=n\widehat{FSSD}^2}; `p.value` is the same-sample plug-in p-value;
#' and `info` contains the sampled \eqn{J\times d} matrix `V` representing `L`.
#' Use `print(fit)` to display the test statistic and p-value.
#' @references
#' Jitkrittum, W., Xu, W., Szabó, Z., Fukumizu, K., and Gretton, A. (2017).
#' A Linear-Time Kernel Goodness-of-Fit Test.
#' Advances in Neural Information Processing Systems, 30, 262--271.
#' Curran Associates, Inc.
#' \url{https://papers.nips.cc/paper_files/paper/2017/hash/979d472a84804b9f647bc185a877a8b5-Abstract.html}.
#' @examples
#' # Target: N(0, 1).
#' set.seed(1)
#' X <- matrix(rnorm(200, mean = 0.5), ncol = 1)
#' fit <- fssd_rand_test(X, function(X) -X, J = 3, scaling = 1)
#' fit$info$V
#' # Heuristic p-value
#' c(statistic = unname(fit$statistic), p.value = fit$p.value)
#' @export
fssd_rand_test <- function(X, score_function,
                           J = 5,
                           n_simulations = 2000,
                           kernel = c("gaussian_rbf", "imq"),
                           scaling = NULL,
                           imq_beta = -0.5) {
  data_name <- deparse(substitute(X))
  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }

  J <- validate_integer(J, "J")
  n_simulations <- validate_integer(n_simulations, "n_simulations")

  x_mat <- .as_rows(X, "X", 2L)
  scores <- .as_score_matrix(score_function(x_mat), x_mat)
  prep <- .prepare_fssd_kernel(x_mat, kernel, scaling, imq_beta)

  # FSSD-rand draws its locations from the rows it tests, so its p-value is
  # approximate by design.
  V <- .fssd_init_locations(x_mat, J)
  engine_res <- .compute_fssd_test(
    X = x_mat,
    scores = scores,
    V = V,
    kernel_obj = prep$kernel_obj,
    n_simulations = n_simulations
  )

  structure(list(
    statistic = c(fssd = engine_res$statistic),
    p.value = engine_res$p_value,
    method = sprintf("Finite Set Stein Discrepancy (rand) - %s", prep$kernel_name),
    data.name = data_name,
    parameter = c(
      n_simulations = n_simulations,
      J = J,
      scaling = prep$scaling
    ),
    kernel = prep$kernel_obj,
    info = list(variant = "rand", V = V, kernel = prep$kernel_name)
  ), class = "htest")
}


#' FSSD test with optimized test locations
#'
#' Selects the test locations \eqn{L = \{v_1,\ldots,v_J\}} and, when the
#' kernel exposes one, the squared kernel scale \eqn{\rho} on training rows.
#' The FSSD statistic and null calibration use disjoint held-out rows.
#'
#' @details
#' The training criterion to maximize is
#' \deqn{
#' C(L,\rho) =
#' \frac{\widehat{FSSD}_{\mathrm{train}}^2}
#'      {\sqrt{\widehat{\mathrm{Var}}_{H_1}} + \gamma},
#' }
#' where
#' \deqn{
#' \widehat{\mathrm{Var}}_{H_1}
#'   = 4\bar\tau^T\widehat\Sigma_\tau\bar\tau,
#' \qquad
#' \widehat\Sigma_\tau
#'   = \frac{1}{n_{\mathrm{train}}}\sum_{i=1}^{n_{\mathrm{train}}}
#'     (\tau_i-\bar\tau)(\tau_i-\bar\tau)^T.
#' }
#' Here \eqn{\bar\tau} is the mean feature vector on the training rows, and
#' \eqn{\gamma>0} keeps the denominator away from zero.
#'
#' Starting locations are drawn from a Gaussian fitted to the training rows.
#' Bounded L-BFGS-B refines `L` and any optimizable squared scale jointly;
#' the `scaling` argument specifies how its starting value is chosen.
#' [fssd_grad_kernel()] supplies the gradient, one test location at a time.
#'
#' The selected locations and kernel are fixed for the held-out test.
#' For i.i.d. observations, choices based only on the training rows are
#' independent of the held-out rows, as required by the null calibration in
#' [fssd_null_pvalue()].
#' For the common interface and choice of variant, see [fssd_test()].
#' @inheritParams fssd_test
#' @param X Numeric vector or matrix containing `n` independent and identically
#'   distributed observations. Rows are observations and columns are
#'   coordinates; a vector is treated as an \eqn{n\times 1} matrix.
#'   At least four observations are required.
#' @param scaling Starting positive squared kernel scale (\eqn{h^2} for
#'   Gaussian RBF, \eqn{c^2} for IMQ). `NULL` uses a five-value grid around the
#'   training-row median squared scale when the kernel needs one. Supplying a
#'   value gives the starting value instead; an optimizable squared scale is
#'   still optimized from there.
#'   A finite squared scale in a
#'   supplied `SteinKernel` is the starting value and takes precedence;
#'   `scaling` is ignored when the kernel exposes no squared scale.
#' @return
#' An object of class `htest`. `statistic` is
#' \eqn{S_{\tilde n}=\tilde n\widehat{FSSD}^2}, computed on the \eqn{\tilde n}
#' held-out rows;
#' `p.value` is obtained from the held-out plug-in null calibration. `info`
#' contains `V` (test locations), `n_train` and `n_test` (split sizes),
#' `scale2_opt` (optimized squared scale), `objective_opt` (training criterion),
#' `function_evaluations` (optimizer function evaluations), and `convergence`
#' (0 for successful optimizer termination; otherwise see [stats::optim()]).
#'
#' `scale_grid` and `scale_grid_objectives` record the optional initial scale
#' search.
#' Use `print(fit)` to display the test statistic and p-value.
#' @references
#' Jitkrittum, W., Xu, W., Szabó, Z., Fukumizu, K., and Gretton, A. (2017).
#' A Linear-Time Kernel Goodness-of-Fit Test.
#' Advances in Neural Information Processing Systems, 30, 262--271.
#' Curran Associates, Inc.
#' \url{https://papers.nips.cc/paper_files/paper/2017/hash/979d472a84804b9f647bc185a877a8b5-Abstract.html}.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(200, mean = 0.5), ncol = 1)
#' score <- function(X) -X  # Target: N(0, 1)
#' fit <- fssd_opt_test(X, score, J = 3, scaling = 1, train_ratio = 0.5)
#' c(statistic = unname(fit$statistic), p.value = fit$p.value)
#' fit$info$convergence  # 0: successful optimizer termination.
#' @export
fssd_opt_test <- function(X, score_function,
                          J = 5,
                          n_simulations = 2000,
                          kernel = c("gaussian_rbf", "imq"),
                          scaling = NULL,
                          train_ratio = 0.2,
                          gamma = 1e-4,
                          maxit = 100,
                          locs_bounds_frac = 10,
                          scale_lower = NULL,
                          scale_upper = NULL,
                          imq_beta = -0.5) {
  data_name <- deparse(substitute(X))
  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }

  J <- validate_integer(J, "J")
  n_simulations <- validate_integer(n_simulations, "n_simulations")
  maxit <- validate_integer(maxit, "maxit")
  gamma <- validate_positive_scalar(gamma, "gamma")
  locs_bounds_frac <- validate_positive_scalar(locs_bounds_frac, "locs_bounds_frac")
  if (!is.null(scale_lower)) {
    scale_lower <- validate_positive_scalar(scale_lower, "scale_lower")
  }
  if (!is.null(scale_upper)) {
    scale_upper <- validate_positive_scalar(scale_upper, "scale_upper")
  }
  if (!is.null(scale_lower) && !is.null(scale_upper) &&
      scale_lower >= scale_upper) {
    stop("scale_lower must be smaller than scale_upper", call. = FALSE)
  }
  train_ratio <- .validate_fssd_train_ratio(train_ratio)

  x_mat <- .as_rows(X, "X", 4L)
  scores <- .as_score_matrix(score_function(x_mat), x_mat)
  n <- nrow(x_mat)

  # Each part needs at least two rows, because the U statistic and the
  # covariance divide by n minus 1.
  n_train <- max(2L, min(floor(train_ratio * n), n - 2L))
  idx_train <- sample.int(n, size = n_train, replace = FALSE)
  idx_test <- setdiff(seq_len(n), idx_train)

  x_train <- x_mat[idx_train, , drop = FALSE]
  x_test <- x_mat[idx_test, , drop = FALSE]
  scores_train <- scores[idx_train, , drop = FALSE]
  scores_test <- scores[idx_test, , drop = FALSE]

  prep <- .prepare_fssd_kernel(
    X = x_train,
    kernel = kernel,
    scaling = scaling,
    imq_beta = imq_beta
  )

  opt <- .optimize_fssd(
    X_train = x_train,
    scores_train = scores_train,
    J = J,
    kernel_obj = prep$kernel_obj,
    gamma = gamma,
    maxit = maxit,
    locs_bounds_frac = locs_bounds_frac,
    scale_lower = scale_lower,
    scale_upper = scale_upper,
    use_scale_grid = prep$used_default_scale
  )

  # Only the held-out rows enter the test. Tuning used the training rows alone,
  # which keeps the null calibration valid.
  engine_res <- .compute_fssd_test(
    X = x_test,
    scores = scores_test,
    V = opt$V_opt,
    kernel_obj = opt$kernel_obj_opt,
    n_simulations = n_simulations
  )

  structure(list(
    statistic = c(fssd = engine_res$statistic),
    p.value = engine_res$p_value,
    method = sprintf("Finite Set Stein Discrepancy (opt) - %s", prep$kernel_name),
    data.name = data_name,
    parameter = c(
      n_simulations = n_simulations,
      J = J,
      train_ratio = train_ratio,
      gamma = gamma,
      scaling = opt$scale2_opt
    ),
    kernel = opt$kernel_obj_opt,
    info = list(
      variant = "opt",
      V = opt$V_opt,
      kernel = prep$kernel_name,
      n_train = nrow(x_train),
      n_test = nrow(x_test),
      scale2_opt = opt$scale2_opt,
      objective_opt = opt$objective_opt,
      function_evaluations = opt$function_evaluations,
      convergence = opt$convergence,
      scale_grid = opt$scale_grid,
      scale_grid_objectives = opt$scale_grid_objectives
    )
  ), class = "htest")
}


#' Compute the FSSD feature matrix
#'
#' Builds the row-feature matrix used by [fssd_statistic()] and
#' [fssd_null_pvalue()]. The rows of `V` represent the test locations
#' \eqn{L = \{v_1,\ldots,v_J\}}.
#' For the complete goodness-of-fit test, see [fssd_test()].
#'
#' @details
#' For observation \eqn{x_i}, target score \eqn{s_p(x_i)}, and test location
#' \eqn{v_j},
#' define
#' \deqn{
#' \xi_p(x_i,v_j)
#'   = s_p(x_i)k(x_i,v_j) + \nabla_x k(x_i,v_j).
#' }
#' With `d = ncol(X)` and `J = nrow(V)`, the returned row is
#' \deqn{
#' \tau_i = \tau(x_i)
#'   = \frac{1}{\sqrt{dJ}}
#'     \operatorname{vec}\{\xi_p(x_i,v_1),\ldots,\xi_p(x_i,v_J)\}
#'   \in \mathbb{R}^{dJ}.
#' }
#' @param X Numeric \eqn{\tilde n\times d} sample matrix.
#' @param scores Numeric \eqn{\tilde n\times d} score matrix for `X`.
#' @param V Numeric \eqn{J\times d} matrix representing `L`, one test
#'   location per row.
#' @param kernel_obj Stein kernel object. A Gaussian RBF kernel must have a
#'   fixed positive bandwidth.
#'
#' @return
#' Numeric \eqn{\tilde n\times dJ} matrix. Row `i` is `tau(x_i)`; columns are
#' ordered by test location and then coordinate.
#' @seealso [fssd_statistic()], [fssd_null_pvalue()]
#' @references
#' Jitkrittum, W., Xu, W., Szabó, Z., Fukumizu, K., and Gretton, A. (2017).
#' A Linear-Time Kernel Goodness-of-Fit Test.
#' Advances in Neural Information Processing Systems, 30, 262--271.
#' Curran Associates, Inc.
#' \url{https://papers.nips.cc/paper_files/paper/2017/hash/979d472a84804b9f647bc185a877a8b5-Abstract.html}.
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' scores <- -X
#' V <- matrix(0, ncol = 1)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' compute_tau(X, scores, V, kernel)
#' @export
compute_tau <- function(X, scores, V, kernel_obj) {
  x_mat <- as.matrix(X)
  scores_mat <- as.matrix(scores)
  v_mat <- as.matrix(V)
  if (!is.numeric(x_mat) || !is.numeric(scores_mat) || !is.numeric(v_mat) ||
      any(!is.finite(x_mat)) || any(!is.finite(scores_mat)) ||
      any(!is.finite(v_mat))) {
    stop("X, scores, and V must be finite numeric matrices", call. = FALSE)
  }
  if (nrow(x_mat) != nrow(scores_mat) || ncol(x_mat) != ncol(scores_mat)) {
    stop("X and scores must have the same dimensions", call. = FALSE)
  }
  if (ncol(v_mat) != ncol(x_mat) || nrow(v_mat) < 1) {
    stop("V must be a numeric matrix with ncol(V) == ncol(X)", call. = FALSE)
  }
  require_fixed_gaussian_rbf(kernel_obj, "compute_tau")

  .compute_fssd_tau(x_mat, scores_mat, v_mat, kernel_obj)
}

.compute_fssd_tau <- function(X, scores, V, kernel_obj) {
  n <- nrow(X)
  d <- ncol(X)
  J <- nrow(V)
  scale_fac <- 1 / sqrt(d * J)
  tau <- matrix(0, nrow = n, ncol = d * J)

  for (j in seq_len(J)) {
    v_one <- matrix(V[j, ], nrow = 1, ncol = d)
    k_xv <- eval_kernel(kernel_obj, X, v_one)
    grad_arr <- grad_x_kernel(kernel_obj, X, v_one)

    grad_k <- matrix(grad_arr[, 1, ], nrow = n, ncol = d)
    xi <- (scores * as.numeric(k_xv[, 1]) + grad_k) * scale_fac
    # Each location occupies one d-column block of tau.
    tau[, ((j - 1L) * d + 1L):(j * d)] <- xi
  }

  tau
}


#' FSSD statistic and its null calibration
#'
#' Computes the scaled off-diagonal FSSD U-statistic from a feature matrix
#' returned by [compute_tau()], and performs the plug-in null calibration for
#' that statistic.
#' For the complete goodness-of-fit test, see [fssd_test()].
#'
#' @rdname fssd_components
#' @details
#' If `tau_i` is row `i` of `tau_matrix` and \eqn{\tilde n} is its number of
#' rows, the returned value is
#' \deqn{
#' S_{\tilde n}
#'   = \tilde n\widehat{FSSD}^2
#'   = \frac{1}{\tilde n - 1}\sum_{i\ne j}\tau_i^T\tau_j.
#' }
#' Diagonal terms are excluded because population FSSD uses two independent
#' draws. Although population `FSSD^2` is nonnegative, its unbiased
#' off-diagonal estimator and the returned \eqn{S_{\tilde n}} can be negative
#' in finite samples.
#'
#' The fixed-location null calibration performed by [fssd_null_pvalue()]
#' requires the test locations `L` and the
#' kernel, including its scale, to be fixed independently of the observations
#' represented by `tau_matrix`, or selected using separate training data. That
#' function cannot check that condition or correct same-sample selection.
#'
#' Let \eqn{\tilde n} be the number of rows of `tau_matrix`, and define
#' \deqn{
#' \widehat\Sigma_\tau
#'   = \frac{1}{\tilde n}\sum_{i=1}^{\tilde n}
#'     (\tau_i-\bar\tau)(\tau_i-\bar\tau)^T,
#' }
#' and let \eqn{\widehat\lambda_1,\ldots,\widehat\lambda_{dJ}} be its
#' eigenvalues. Numerically negative eigenvalues are replaced by zero. For
#' \eqn{b=1,\ldots,B}, the function simulates
#' \deqn{
#' S_{\tilde n}^{*(b)} = \sum_{q=1}^{dJ}
#'   \widehat\lambda_q\{(Z_q^{(b)})^2-1\},
#' \qquad Z_q^{(b)} \sim N(0,1),
#' }
#' and returns
#' \deqn{
#' \widehat p = \frac{1}{B}\sum_{b=1}^B
#'   \mathbf{1}\{S_{\tilde n}^{*(b)} \ge S_{\tilde n}\}.
#' }
#' The returned proportion does not use an add-one correction. The simulated
#' law is the large-\eqn{\tilde n} limit of \eqn{S_{\tilde n}}, so the test holds its
#' nominal level only asymptotically.
#'
#' @param tau_matrix Numeric \eqn{\tilde n\times dJ} feature matrix returned by
#'   [compute_tau()].
#'
#' @return
#' [fssd_statistic()] returns one numeric value:
#' \eqn{S_{\tilde n}=\tilde n\widehat{FSSD}^2}.
#' [fssd_null_pvalue()] returns a list with four entries:
#' * `p_value`: uncorrected simulated right-tail proportion.
#' * `statistic`: the supplied \eqn{S_{\tilde n}}.
#' * `null_samples`: simulated draws on exactly the same scale as `statistic`.
#' * `eigenvalues`: nonnegative eigenvalues of
#'   \eqn{\widehat\Sigma_\tau} used in the null draws.
#' @references
#' Jitkrittum, W., Xu, W., Szabó, Z., Fukumizu, K., and Gretton, A. (2017).
#' A Linear-Time Kernel Goodness-of-Fit Test.
#' Advances in Neural Information Processing Systems, 30, 262--271.
#' Curran Associates, Inc.
#' \url{https://papers.nips.cc/paper_files/paper/2017/hash/979d472a84804b9f647bc185a877a8b5-Abstract.html}.
#' @examples
#' # Target: N(0, 1).
#' set.seed(1)
#' X <- matrix(rnorm(200, mean = 0.5), ncol = 1)
#' V <- matrix(c(-1, 0, 1), ncol = 1)  # Prespecified test locations
#' kernel <- stein_kernel("gaussian_rbf", h = 1)
#' tau <- compute_tau(X, -X, V, kernel)
#' statistic <- fssd_statistic(tau)
#' result <- fssd_null_pvalue(tau, statistic)
#' c(statistic = result$statistic, p.value = result$p_value)
#' @export
fssd_statistic <- function(tau_matrix) {
  tau <- .as_tau_matrix(tau_matrix)
  nrow(tau) * .fssd_unscaled_u_statistic(tau)
}

.as_tau_matrix <- function(tau_matrix) {
  tau <- as.matrix(tau_matrix)
  if (!is.numeric(tau) || nrow(tau) < 2L || ncol(tau) < 1L ||
      any(!is.finite(tau))) {
    stop("tau_matrix must be a finite numeric matrix with at least two rows", call. = FALSE)
  }
  tau
}

.fssd_unscaled_u_statistic <- function(tau) {
  n <- nrow(tau)
  sum_tau <- colSums(tau)
  # The sum over pairs i not equal to j equals the squared column sum minus the
  # diagonal terms, so no n by n matrix is needed.
  as.numeric((sum(sum_tau * sum_tau) - sum(rowSums(tau * tau))) / (n * (n - 1)))
}


#' @rdname fssd_components
#' @param statistic Observed statistic
#'   \eqn{S_{\tilde n}=\tilde n\widehat{FSSD}^2}.
#' @param n_simulations Number of null draws to simulate.
#'
#' @export
fssd_null_pvalue <- function(tau_matrix, statistic, n_simulations = 2000) {
  tau <- .as_tau_matrix(tau_matrix)
  if (!is.numeric(statistic) || length(statistic) != 1 ||
      !is.finite(statistic)) {
    stop("statistic must be a finite scalar", call. = FALSE)
  }
  n_simulations <- validate_integer(n_simulations, "n_simulations")

  .fssd_null_pvalue_core(tau, statistic, n_simulations)
}

.fssd_null_pvalue_core <- function(tau, statistic, n_simulations) {
  sigma_tau_hat <- .fssd_tau_covariance(tau)
  eigvals <- eigen(sigma_tau_hat, symmetric = TRUE, only.values = TRUE)$values
  # Discard small negative eigenvalues caused by roundoff.
  eigvals[eigvals < 0] <- 0

  z <- matrix(stats::rnorm(n_simulations * length(eigvals)),
              nrow = n_simulations, ncol = length(eigvals))
  # The asymptotic null law of n FSSD^2 is a sum of z^2 - 1 terms weighted by
  # the eigenvalues.
  chi_terms <- z^2 - 1
  s_null <- as.numeric(chi_terms %*% eigvals)

  list(
    p_value = mean(s_null >= statistic),
    statistic = statistic,
    null_samples = s_null,
    eigenvalues = eigvals
  )
}


.compute_fssd_test <- function(X, scores, V, kernel_obj, n_simulations) {
  tau <- .compute_fssd_tau(X, scores, V, kernel_obj)
  statistic <- nrow(tau) * .fssd_unscaled_u_statistic(tau)
  .fssd_null_pvalue_core(tau, statistic, n_simulations)
}


.optimize_fssd <- function(X_train, scores_train, J, kernel_obj,
                           gamma = 1e-4,
                           maxit = 100,
                           locs_bounds_frac = 10,
                           scale_lower = NULL,
                           scale_upper = NULL,
                           use_scale_grid = TRUE) {
  n <- nrow(X_train)
  d <- ncol(X_train)
  has_kernel_param <- !is.null(scale2_kernel(kernel_obj))
  n_params <- d * J + if (has_kernel_param) 1L else 0L
  if (n < n_params) {
    warning(sprintf(
      "FSSD-opt has more parameters (%s) than training rows; increase `train_ratio` or reduce `J`",
      if (has_kernel_param) "d*J+1" else "d*J"
    ), call. = FALSE)
  }

  v_init <- .fssd_init_locations(X_train, J, cov_reg = 1e-6)
  scale2_init <- if (has_kernel_param) scale2_kernel(kernel_obj) else NA_real_
  scale_grid <- scale_grid_objectives <- NULL

  if (has_kernel_param && use_scale_grid) {
    # The criterion is not concave in the scale, so start from the best of five
    # grid values instead of the median alone.
    scale_grid <- .fssd_scale_grid(scale2_init)
    grid_res <- .fssd_grid_search_scale(
      X = X_train,
      scores = scores_train,
      V = v_init,
      kernel_obj_base = kernel_obj,
      scale_grid = scale_grid,
      gamma = gamma
    )
    scale2_init <- grid_res$scale
    scale_grid_objectives <- grid_res$objectives
  }

  if (has_kernel_param && (is.null(scale_lower) || is.null(scale_upper))) {
    M <- kernel_obj$precon
    rho <- find_median_distance(if (is.null(M)) X_train else X_train %*% t(chol(M)))
    if (is.null(scale_lower)) scale_lower <- max(0.01 * rho, 1e-3)
    if (is.null(scale_upper)) scale_upper <- min(100 * rho, 1e5)
  }
  if (has_kernel_param && scale_lower >= scale_upper) {
    stop(sprintf("scale bounds are empty: scale_lower = %g, scale_upper = %g",
                 scale_lower, scale_upper), call. = FALSE)
  }

  scale_idx <- if (has_kernel_param) J * d + 1L else NA_integer_

  x_std <- apply(X_train, 2, stats::sd)
  x_std[!is.finite(x_std) | x_std <= 0] <- 1e-8

  # The search runs on the square root of scale2, which is better conditioned
  # than scale2 itself.
  lower <- c(rep(apply(X_train, 2, min) - locs_bounds_frac * x_std, times = J),
             if (has_kernel_param) sqrt(scale_lower))
  upper <- c(rep(apply(X_train, 2, max) + locs_bounds_frac * x_std, times = J),
             if (has_kernel_param) sqrt(scale_upper))
  par0 <- c(as.numeric(t(v_init)), if (has_kernel_param) sqrt(scale2_init))
  par0 <- pmin(pmax(par0, lower), upper)

  cache <- new.env(parent = emptyenv())
  objective_at <- function(par) {
    # optim calls fn and gr separately at the same par, so the last result is
    # reused instead of computed twice.
    if (!is.null(cache$par) && length(cache$par) == length(par) && all(cache$par == par)) {
      return(cache$objective_grad)
    }
    par_obj <- par
    if (has_kernel_param) par_obj[scale_idx] <- par[scale_idx]^2
    objective_grad <- .fssd_objective_and_grad(
      par = par_obj,
      X = X_train,
      scores = scores_train,
      J = J,
      kernel_obj_base = kernel_obj,
      has_kernel_param = has_kernel_param,
      gamma = gamma
    )
    if (has_kernel_param) {
      # The search variable is par and scale2 equals par^2, so the chain rule
      # multiplies by 2 par.
      objective_grad$grad[scale_idx] <-
        objective_grad$grad[scale_idx] * (2 * par[scale_idx])
    }
    cache$par <- par
    cache$objective_grad <- objective_grad
    objective_grad
  }

  opt_res <- stats::optim(
    par = par0,
    fn = function(par) -objective_at(par)$value,
    gr = function(par) -objective_at(par)$grad,
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    control = list(
      maxit = maxit,
      pgtol = 1e-7,
      factr = 1e-8 / .Machine$double.eps
    )
  )

  best_par <- opt_res$par
  best_value <- objective_at(best_par)$value

  V_opt <- matrix(best_par[seq_len(J * d)], nrow = J, ncol = d, byrow = TRUE)
  scale2_opt <- if (has_kernel_param) best_par[J * d + 1L]^2 else NA_real_

  list(
    V_opt = V_opt,
    scale2_opt = scale2_opt,
    kernel_obj_opt = if (has_kernel_param) {
      scale2_kernel(kernel_obj, scale2_opt)
    } else {
      kernel_obj
    },
    objective_opt = best_value,
    function_evaluations = unname(opt_res$counts[["function"]]),
    convergence = opt_res$convergence,
    scale_grid = scale_grid,
    scale_grid_objectives = scale_grid_objectives
  )
}


.fssd_objective_and_grad <- function(par, X, scores, J, kernel_obj_base,
                                     has_kernel_param, gamma) {
  d <- ncol(X)
  n <- nrow(X)

  V <- matrix(par[seq_len(J * d)], nrow = J, ncol = d, byrow = TRUE)
  kernel_obj_current <- if (has_kernel_param) {
    scale2_kernel(kernel_obj_base, par[J * d + 1L])
  } else {
    kernel_obj_base
  }
  tau <- .compute_fssd_tau(X, scores, V, kernel_obj_current)
  n_features <- ncol(tau)

  mu_tau <- colMeans(tau)
  sum_tau <- colSums(tau)
  fssd2 <- .fssd_unscaled_u_statistic(tau)
  # The derivative of the U statistic with respect to row i is 2 times the sum
  # of the other rows, divided by n times n minus 1.
  grad_fssd2_tau <- (2 / (n * (n - 1L))) *
    (matrix(sum_tau, n, n_features, byrow = TRUE) - tau)

  centered_tau <- sweep(tau, 2, mu_tau, "-")
  sigma_tau <- crossprod(centered_tau) / n
  sigma_h1_sq <- as.numeric(4 * t(mu_tau) %*% sigma_tau %*% mu_tau)
  sigma_h1 <- sqrt(max(sigma_h1_sq, 0))
  denom <- sigma_h1 + gamma

  # Derivative of sigma_h1^2 with respect to each feature row, used for the
  # denominator of the criterion.
  sigma_mu <- as.numeric(sigma_tau %*% mu_tau)
  c_mu <- as.numeric(centered_tau %*% mu_tau)
  grad_sigma_sq <-
    (8 / n) * matrix(sigma_mu, n, n_features, byrow = TRUE) +
    (8 / n) * (c_mu %o% mu_tau)

  d_denom <- if (sigma_h1 > 1e-12) {
    grad_sigma_sq / (2 * sigma_h1)
  } else {
    matrix(0, n, n_features)
  }
  grad_obj_tau <-
    grad_fssd2_tau / denom - (fssd2 / (denom^2)) * d_denom

  scale_fac <- 1 / sqrt(d * J)
  grad_v <- matrix(0, nrow = J, ncol = d)
  grad_param <- 0

  for (j in seq_len(J)) {
    # fssd_grad_kernel carries the gradient with respect to tau back to location
    # j and the kernel scale.
    g_block <- grad_obj_tau[, ((j - 1L) * d + 1L):(j * d), drop = FALSE]
    fssd_grads <- fssd_grad_kernel(
      obj = kernel_obj_current,
      X = X,
      vj = V[j, ],
      grads_X = scores,
      g_block = g_block * scale_fac
    )
    grad_v[j, ] <- fssd_grads$grad_vj
    grad_param <- grad_param + fssd_grads$grad_param
  }

  grad_par <- as.numeric(t(grad_v))
  if (has_kernel_param) grad_par <- c(grad_par, grad_param)

  list(value = fssd2 / denom, grad = grad_par)
}

.prepare_fssd_kernel <- function(X,
                                 kernel = c("gaussian_rbf", "imq"),
                                 scaling = NULL,
                                 imq_beta = -0.5) {
  x_mat <- X
  used_default_scale <- FALSE

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_obj$type
    scale2 <- scale2_kernel(kernel_obj)

    if (!is.null(scale2) && !is.finite(scale2)) {
      if (is.null(scaling)) {
        scaling <- if (identical(kernel_obj$type, "gaussian_rbf")) {
          resolve_gaussian_rbf_scale2(kernel_obj, x_mat)
        } else {
          find_median_distance(x_mat)
        }
        used_default_scale <- TRUE
      }
      scaling <- validate_positive_scalar(scaling, "scaling")
      kernel_obj <- scale2_kernel(kernel_obj, scaling)
    } else {
      if (!is.null(scaling)) {
        warning("`scaling` is ignored because the kernel object sets its own scale",
                call. = FALSE)
      }
      scaling <- if (is.null(scale2)) NA_real_ else scale2
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat)
      used_default_scale <- TRUE
    }
    scaling <- validate_positive_scalar(scaling, "scaling")
    kernel_obj <- instantiate_kernel(kernel_name, scaling, imq_beta)
  }

  list(
    kernel_obj = kernel_obj,
    kernel_name = kernel_name,
    scaling = scaling,
    used_default_scale = used_default_scale
  )
}

.validate_fssd_train_ratio <- function(train_ratio) {
  if (!is.numeric(train_ratio) || length(train_ratio) != 1 ||
      !is.finite(train_ratio) || train_ratio <= 0 || train_ratio >= 1) {
    stop("train_ratio must be a scalar in (0, 1)", call. = FALSE)
  }
  as.numeric(train_ratio)
}


.fssd_scale_grid <- function(center) {
  center * 2^seq(-3, 3, length.out = 5)
}

.fssd_grid_search_scale <- function(X, scores, V, kernel_obj_base,
                                    scale_grid, gamma) {
  J <- nrow(V)
  par_v <- as.numeric(t(V))
  objectives <- vapply(scale_grid, function(scale2) {
    .fssd_objective_and_grad(
      par = c(par_v, scale2),
      X = X,
      scores = scores,
      J = J,
      kernel_obj_base = kernel_obj_base,
      has_kernel_param = TRUE,
      gamma = gamma
    )$value
  }, numeric(1))

  # A failed grid value counts as -Inf. If every value fails, which.max returns
  # the first one.
  finite_objectives <- objectives
  finite_objectives[!is.finite(finite_objectives)] <- -Inf
  list(scale = scale_grid[which.max(finite_objectives)], objectives = objectives)
}


.fssd_symmetric_covariance <- function(s) {
  if (any(!is.finite(s))) {
    stop("the feature covariance is not finite; rescale the sample", call. = FALSE)
  }
  s / 2 + t(s) / 2
}

.fssd_tau_covariance <- function(tau) {
  centered <- sweep(tau, 2, colMeans(tau), "-")
  .fssd_symmetric_covariance(crossprod(centered) / nrow(tau))
}

.fssd_init_locations <- function(X, J, cov_reg = 0) {
  d <- ncol(X)
  # Use an eigen root with a small floor instead of chol, so a singular sample
  # covariance still gives a valid root.
  sigma <- .fssd_symmetric_covariance(stats::cov(X))
  diag(sigma) <- diag(sigma) + cov_reg
  eig <- eigen(sigma, symmetric = TRUE)
  root <- eig$vectors %*% diag(sqrt(pmax(eig$values, 1e-10)), nrow = d)
  z <- matrix(stats::rnorm(J * d), nrow = J, ncol = d)
  sweep(z %*% t(root), 2, colMeans(X), "+")
}
