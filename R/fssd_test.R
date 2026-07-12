#' Finite Set Stein Discrepancy goodness-of-fit test
#'
#' Runs the FSSD test of Jitkrittum et al. The test compares a sample with a
#' target score by evaluating the Stein witness function at a small finite set
#' of test locations. Use `variant = "rand"` for random locations and
#' `variant = "opt"` to learn locations, and for built-in kernels the kernel
#' scale, on a training split.
#'
#' @details
#' FSSD chooses `J` locations \eqn{v_1,\ldots,v_J} and checks whether the Stein
#' witness function vanishes there. Write the target density as `p` with score
#' \eqn{s_p(x) = \nabla_x \log p(x)}. For each sample point `x_i` and location
#' `v_j`, the Stein feature is
#' \deqn{\xi_p(x_i, v_j) = s_p(x_i) k(x_i, v_j) + \nabla_x k(x_i, v_j).}
#' Here `k` is the base kernel, and the gradient is with respect to sample
#' argument `x_i`, not location argument `v_j`; each location contributes a
#' length-`d` vector.
#'
#' Let \eqn{\Xi(x)} be the `d x J` matrix with column
#' \eqn{\xi_p(x,v_j)/\sqrt{dJ}} and let
#' \eqn{\tau(x)=vec(\Xi(x))} stack its columns into length `d * J`, the row
#' representation returned by [compute_tau()]. Then
#' \deqn{\Delta(x, y) = \tau(x)^T \tau(y)
#'      = \frac{1}{dJ}\sum_{j=1}^J
#'        \xi_p(x, v_j)^T \xi_p(y, v_j)}
#' is the FSSD U-statistic kernel; feature normalization by \eqn{\sqrt{dJ}}
#' produces the paper's \eqn{1/(dJ)} inner-product factor.
#'
#' Averaging over distinct sample pairs gives
#' \deqn{
#' \widehat{FSSD}^2 =
#' \frac{1}{n(n - 1)} \sum_{i \ne l} \tau(x_i)^T \tau(x_l).
#' }
#' [compute_tau()] returns the `n x (d * J)` matrix with row `i` equal to
#' `tau(x_i)`; the unbiased statistic is computed without storing every
#' pairwise inner product.
#'
#' The p-value uses the paper's asymptotic null distribution. Estimate
#' \eqn{\hat\Sigma = n^{-1}\sum_i(\tau(x_i)-\bar\tau)
#' (\tau(x_i)-\bar\tau)^T}, compute its eigenvalues, and simulate null draws
#' \eqn{\sum_j\lambda_j(Z_j^2-1)}. Large values indicate disagreement between
#' sample and target score. The `"opt"` variant first uses part of the data to
#' choose useful locations.
#'
#' Unlike [ksd_u_test()] and [ksd_v_test()], which build all pairwise
#' `k0(x_i,x_j)` values through [stein_kernel_matrix()], FSSD uses a finite
#' location matrix `V` and [compute_tau()]. It retains the same `X`, target-score
#' convention, and kernel layer, while adding `J` and, for `"opt"`, a training
#' stage that selects `V` and the built-in kernel scale.
#'
#' @param X Numeric vector or matrix of samples (`n x d`).
#' @param score_function Function returning the target score for each row of
#'   `X`; the output should have shape `n x d`.
#' @param variant `"rand"` draws test locations from a Gaussian fitted to the
#'   data. `"opt"` tunes test locations and the kernel scale on a training
#'   split.
#' @param J Number of test locations.
#' @param nboot Number of simulated null draws used for the p-value.
#' @param kernel `"gaussian_rbf"`, `"imq"`, or a `SteinKernel` object.
#' @param scaling Optional positive squared kernel scale. `NULL` uses the square
#'   of the median distance between all sample-row pairs. This exact calculation
#'   has quadratic time and memory cost; large-sample callers should provide
#'   `scaling` explicitly.
#' @param train_ratio Requested fraction of samples used to tune FSSD-opt. The
#'   split is rounded down and adjusted so both training and test sets contain
#'   at least two rows.
#' @param gamma Small positive regularizer used by the FSSD-opt objective.
#' @param maxit Maximum number of L-BFGS-B optimizer iterations for FSSD-opt.
#' @param locs_bounds_frac Width of the box used to bound optimized test
#'   locations, measured in fitted standard deviations.
#' @param scale_lower,scale_upper Bounds for the optimized squared kernel
#'   scale.
#' @param seed Optional RNG seed for the data split, random locations, and
#'   simulated null draws.
#' @param imq_beta IMQ exponent.
#'
#' @return
#' An object of class `htest`. For both variants the list contains:
#' * `statistic`: one number labeled `fssd`; this is the unbiased estimate of
#'   squared FSSD for the final test locations.
#' * `p.value`: right-tail p-value from simulated asymptotic null draws.
#' * `method`: text naming the FSSD variant and kernel.
#' * `data.name`: expression used for `X`.
#' * `parameter`: numeric vector with `nboot`, `J`, and the final squared
#'   kernel `scaling`; the optimized variant also records `train_ratio` and
#'   `gamma`.
#' * `info`: list with the variant label, final location matrix `V`, kernel
#'   name, and for the optimized variant, optimization diagnostics.
#'
#' The `htest` fields provide standard hypothesis-test printing; `info` retains
#' `V` because a location-free p-value is difficult to reproduce or interpret.
#' @examples
#' X <- matrix(rnorm(8), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' fssd_test(X, score_function, variant = "rand", J = 1, nboot = 10)
#' @export
fssd_test <- function(X, score_function,
                      variant = c("opt", "rand"),
                      J = 5,
                      nboot = 2000,
                      kernel = c("gaussian_rbf", "imq"),
                      scaling = NULL,
                      train_ratio = 0.2,
                      gamma = 1e-4,
                      maxit = 100,
                      locs_bounds_frac = 10,
                      scale_lower = 1e-1,
                      scale_upper = 1e4,
                      seed = NULL,
                      imq_beta = -0.5) {
  variant <- match.arg(variant)

  if (variant == "opt") {
    fssd_opt_test(
      X = X, score_function = score_function, J = J, nboot = nboot,
      kernel = kernel, scaling = scaling,
      train_ratio = train_ratio,
      gamma = gamma, maxit = maxit,
      locs_bounds_frac = locs_bounds_frac,
      scale_lower = scale_lower, scale_upper = scale_upper,
      seed = seed,
      imq_beta = imq_beta
    )
  } else {
    fssd_rand_test(
      X = X, score_function = score_function, J = J, nboot = nboot,
      kernel = kernel, scaling = scaling, seed = seed,
      imq_beta = imq_beta
    )
  }
}


#' FSSD test with random test locations
#'
#' Implements the FSSD-rand variant: draw the finite set of test locations from
#' a Gaussian fitted to the sample, then compute the FSSD statistic and p-value
#' on all rows.
#'
#' @details
#' Having a density for the location distribution is only one condition in the
#' paper's Theorem 1. Its almost-sure identification result also assumes that the
#' state space is connected and open, the base kernel is `C0`-universal and real
#' analytic, the required Stein-kernel and score-difference moments are finite,
#' and the tail boundary condition for the Stein witness holds. The paper states
#' that all of these conditions are assumed throughout and then specializes to
#' the Gaussian kernel. This package's IMQ and custom-kernel options are
#' extensions; for them, a location distribution with a density alone does not
#' establish Theorem 1, and the caller is responsible for verifying the remaining
#' kernel, moment, and boundary assumptions.
#'
#' This implementation draws locations from a practical Gaussian proposal with
#' mean and covariance estimated from `X`. The resulting location matrix `V`
#' has `J` rows and `ncol(X)` columns, and it is stored in `result$info$V` for
#' inspection or reuse. This variant does not split the sample and does not
#' optimize the locations.
#'
#' After the locations are drawn, [compute_tau()] builds the feature matrix,
#' [compute_fssd_unbiased_stat()] computes the U-statistic estimate, and
#' [compute_fssd_null_pvalue()] simulates the null distribution. This gives a
#' compact baseline FSSD test: the randomness is only in the location draw and
#' in the Monte Carlo approximation of the null distribution.
#' @inheritParams fssd_test
#' @return
#' An object of class `htest`. The main fields are `statistic` (one number
#' labeled `fssd`), `p.value`, `method`, `data.name`, and `parameter`. The
#' `info` field is a
#' list with `variant = "rand"`, the sampled `J x d` location matrix `V`, and
#' the kernel name. The statistic is computed on all rows of `X`.
#'
#' The sampled locations are returned because they define the finite set in
#' "Finite Set Stein Discrepancy"; two runs with different `V` are two different
#' finite-feature tests even if the sample and target score are unchanged.
#' @examples
#' X <- matrix(rnorm(8), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' fssd_rand_test(X, score_function, J = 1, nboot = 10)
#' @export
fssd_rand_test <- function(X, score_function,
                           J = 5,
                           nboot = 2000,
                           kernel = c("gaussian_rbf", "imq"),
                           scaling = NULL,
                           seed = NULL,
                           imq_beta = -0.5) {
  data_name <- deparse(substitute(X))

  run_rand <- function() {
    prep <- prepare_fssd_inputs(
      X = X,
      score_function = score_function,
      min_rows = 2,
      kernel = kernel,
      scaling = scaling,
      imq_beta = imq_beta
    )
    J <- validate_positive_integer(J, "J")
    nboot <- validate_positive_integer(nboot, "nboot")

    V <- fssd_init_locations_from_fit(prep$X, J)
    engine_res <- internal_fssd_compute_engine(
      X = prep$X,
      grads = prep$grads,
      V = V,
      kernel_obj = prep$kernel_obj,
      n_simulations = nboot
    )

    structure(list(
      statistic = c(fssd = engine_res$fssd_stat),
      p.value = engine_res$p_value,
      method = sprintf("Finite Set Stein Discrepancy (rand) - %s", prep$kernel_name),
      data.name = data_name,
      parameter = c(
        nboot = nboot,
        J = J,
        scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling
      ),
      info = list(variant = "rand", V = V, kernel = prep$kernel_name)
    ), class = "htest")
  }

  if (is.null(seed)) run_rand() else with_local_seed(seed, run_rand())
}


#' FSSD test with optimized test locations
#'
#' Implements the FSSD-opt variant: use a training split to find test locations
#' that make the FSSD signal large relative to its estimated variability, then
#' run the final goodness-of-fit test on disjoint held-out data.
#'
#' @details
#' The FSSD paper chooses locations, and for Gaussian kernels also the
#' bandwidth, by approximately maximizing test power under the alternative.
#' This package uses the same idea with a small stabilizer:
#' \deqn{
#' \frac{\widehat{FSSD}^2}
#'      {\sqrt{\widehat{var}_{H1}} + \gamma},
#' }
#' where `gamma` keeps the denominator away from zero. The numerator is the
#' training-sample FSSD estimate and the denominator is an empirical standard
#' deviation proxy under the alternative.
#'
#' The algorithm first splits `X` into disjoint training and test rows. Starting
#' locations are drawn from a Gaussian fitted to the training rows. When
#' `scaling = NULL`, built-in kernels try a short grid centered on the exact
#' all-training-row median scale before bounded L-BFGS-B refinement. An explicit
#' `scaling` value is used directly as the optimizer's starting scale and skips
#' that median grid. The locations and squared kernel scale are then refined
#' jointly when applicable.
#' The final statistic and null simulation are always evaluated on the held-out
#' test rows. This disjointness is part of the paper's procedure: locations and
#' bandwidth are learned on the training set, then the goodness-of-fit test is
#' performed on the test set to avoid feature-selection overfitting.
#' With `n = nrow(X)`, the actual training size is
#' `max(2, min(floor(train_ratio * n), n - 2))`; the remaining rows form the
#' test set.
#'
#' The optimized stage and the testing stage use the same feature definition
#' but play different roles. During training, [compute_tau()] is evaluated
#' repeatedly while `V` and the squared scale are changed, and
#' [grad_theta_v_kernel()] supplies derivatives of the finite-location feature
#' with respect to each test location. After optimization, `V` and the kernel
#' scale are fixed; the final evaluation then follows the same lower-level path
#' as [fssd_rand_test()]: [compute_tau()], [compute_fssd_unbiased_stat()], and
#' [compute_fssd_null_pvalue()].
#' @inheritParams fssd_test
#' @return
#' An object of class `htest`. The main fields are `statistic` (one number
#' labeled `fssd`), `p.value`, `method`, `data.name`, and `parameter`. The
#' `info` field is a
#' list with `variant = "opt"`, optimized locations `V`, the kernel name,
#' actual split sizes `n_train` and `n_test`, optimized squared scale
#' `sigma2_opt` when applicable, optimized objective value `objective_opt`,
#' objective-function count `function_evaluations`, and L-BFGS-B `convergence`
#' code. The statistic is computed only on the held-out test rows.
#'
#' The optimization diagnostics are returned because FSSD-opt is a two-stage
#' method: first choose features, then test. If the optimizer stops early or
#' chooses a boundary scale, those diagnostics matter for interpreting the final
#' p-value.
#' @examples
#' X <- matrix(rnorm(12), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' fssd_opt_test(X, score_function, J = 1, nboot = 10, maxit = 1,
#'               train_ratio = 0.5)
#' @export
fssd_opt_test <- function(X, score_function,
                          J = 5,
                          nboot = 2000,
                          kernel = c("gaussian_rbf", "imq"),
                          scaling = NULL,
                          train_ratio = 0.2,
                          gamma = 1e-4,
                          maxit = 100,
                          locs_bounds_frac = 10,
                          scale_lower = 1e-1,
                          scale_upper = 1e4,
                          seed = NULL,
                          imq_beta = -0.5) {
  data_name <- deparse(substitute(X))

  run_opt <- function() {
    x_mat <- validate_samples(X, min_rows = 4)
    grads <- validate_scores(score_function, x_mat)
    n <- nrow(x_mat)

    J <- validate_positive_integer(J, "J")
    nboot <- validate_positive_integer(nboot, "nboot")
    maxit <- validate_positive_integer(maxit, "maxit")
    gamma <- validate_fssd_positive_scalar(gamma, "gamma")
    locs_bounds_frac <- validate_fssd_positive_scalar(locs_bounds_frac, "locs_bounds_frac")
    scale_lower <- validate_fssd_positive_scalar(scale_lower, "scale_lower")
    scale_upper <- validate_fssd_positive_scalar(scale_upper, "scale_upper")
    if (scale_lower >= scale_upper) stop("scale_lower must be smaller than scale_upper")
    train_ratio <- validate_fssd_train_ratio(train_ratio)

    n_train <- max(2L, min(floor(train_ratio * n), n - 2L))
    idx_train <- sample.int(n, size = n_train, replace = FALSE)
    idx_test <- setdiff(seq_len(n), idx_train)

    x_train <- x_mat[idx_train, , drop = FALSE]
    x_test <- x_mat[idx_test, , drop = FALSE]
    grads_train <- grads[idx_train, , drop = FALSE]
    grads_test <- grads[idx_test, , drop = FALSE]

    prep <- prepare_fssd_kernel(
      X = x_train,
      kernel = kernel,
      scaling = scaling,
      imq_beta = imq_beta
    )

    opt <- internal_fssd_optimize_engine(
      X_train = x_train,
      grads_train = grads_train,
      J = J,
      kernel_obj = prep$kernel_obj,
      gamma = gamma,
      maxit = maxit,
      locs_bounds_frac = locs_bounds_frac,
      scale_lower = scale_lower,
      scale_upper = scale_upper,
      use_scale_grid = prep$used_default_scale
    )

    engine_res <- internal_fssd_compute_engine(
      X = x_test,
      grads = grads_test,
      V = opt$V_opt,
      kernel_obj = opt$kernel_obj_opt,
      n_simulations = nboot
    )

    scaling_opt <- if (is.finite(opt$sigma2_opt)) opt$sigma2_opt
                   else kernel_scaling_value(opt$kernel_obj_opt)

    structure(list(
      statistic = c(fssd = engine_res$fssd_stat),
      p.value = engine_res$p_value,
      method = sprintf("Finite Set Stein Discrepancy (opt) - %s", prep$kernel_name),
      data.name = data_name,
      parameter = c(
        nboot = nboot,
        J = J,
        train_ratio = train_ratio,
        gamma = gamma,
        scaling = if (is.null(scaling_opt)) NA_real_ else scaling_opt
      ),
      info = list(
        variant = "opt",
        V = opt$V_opt,
        kernel = prep$kernel_name,
        n_train = nrow(x_train),
        n_test = nrow(x_test),
        sigma2_opt = opt$sigma2_opt,
        objective_opt = opt$objective_opt,
        function_evaluations = opt$function_evaluations,
        convergence = opt$convergence,
        scale_grid = opt$scale_grid,
        scale_grid_objectives = opt$scale_grid_objectives
      )
    ), class = "htest")
  }

  if (is.null(seed)) run_opt() else with_local_seed(seed, run_opt())
}


# ---- Public Computation Primitives --------------------------------------

#' Compute the FSSD feature matrix
#'
#' Builds the matrix of `tau(x_i)` features used by the FSSD statistic and its
#' null simulation. This is the feature-construction step of the FSSD algorithm.
#'
#' @details
#' For sample `x` with target score `s_p(x)` and one test location `v`, the
#' paper's feature is
#' \deqn{\xi_p(x, v) = s_p(x) k(x, v) + \nabla_x k(x, v).}
#' The output row stacks these feature vectors for all `J` locations and scales
#' them by `1 / sqrt(d * J)`, where `d` is the sample dimension. The resulting
#' matrix has one row per sample and `d * J` columns. More explicitly, columns
#' `((j - 1) * d + 1):(j * d)` contain
#' \eqn{\xi_p(x_i, v_j) / \sqrt{dJ}} for all rows `i`. Inner products of these
#' rows are the \eqn{\Delta(x, y) = \tau(x)^T\tau(y)} terms used in the FSSD
#' U-statistic.
#'
#' This function is the FSSD analogue of building a Stein-kernel matrix. It
#' keeps the feature representation instead of immediately taking all pairwise
#' inner products. That design lets [compute_fssd_unbiased_stat()] compute the
#' statistic and lets [compute_fssd_null_pvalue()] estimate the null covariance
#' from the same `tau` matrix.
#'
#' Moving sideways from [stein_kernel_matrix()] to `compute_tau()` changes the
#' second argument of the kernel calculation from sample rows to fixed test
#' locations. For each row `x_i` and location `v_j`, the function calls the
#' base kernel and its `x`-gradient through [eval_kernel()] and
#' [grad_x_kernel()]. It does not call [trace_mixed_kernel()] because FSSD does
#' not directly assemble the full pairwise scalar `k0(x_i, x_l)`. Instead it
#' applies one Stein operator at the sample point and stores the finite-location
#' feature vector; [compute_fssd_unbiased_stat()] later forms the paper's
#' one-sample second-order (pairwise) U-statistic by inner products of feature
#' vectors from distinct observations.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param V Matrix of FSSD test locations.
#' @param kernel_obj Stein kernel object.
#'
#' @return
#' Numeric matrix with `nrow(X)` rows and `ncol(X) * nrow(V)` columns. Row `i`
#' is the stacked and scaled feature vector `tau(x_i)`. Columns are ordered by
#' test location first and coordinate second: all `d` coordinates for `V[1, ]`,
#' then all `d` coordinates for `V[2, ]`, and so on. The matrix is intended to
#' be passed to [compute_fssd_unbiased_stat()] and
#' [compute_fssd_null_pvalue()].
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' grads <- -X
#' V <- matrix(0, ncol = 1)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' compute_tau(X, grads, V, kernel)
#' @export
compute_tau <- function(X, grads, V, kernel_obj) {
  x_mat <- as.matrix(X)
  grads_mat <- as.matrix(grads)
  v_mat <- as.matrix(V)
  if (!is.numeric(x_mat) || !is.numeric(grads_mat) || !is.numeric(v_mat)) {
    stop("X, grads, and V must be numeric matrices")
  }
  if (nrow(x_mat) != nrow(grads_mat) || ncol(x_mat) != ncol(grads_mat)) {
    stop("X and grads must have the same dimensions")
  }
  if (ncol(v_mat) != ncol(x_mat) || nrow(v_mat) < 1) {
    stop("V must be a numeric matrix with ncol(V) == ncol(X)")
  }
  if (!inherits(kernel_obj, "SteinKernel")) {
    stop("kernel_obj must be a SteinKernel object")
  }

  n <- nrow(x_mat)
  d <- ncol(x_mat)
  J <- nrow(v_mat)
  scale_fac <- 1 / sqrt(d * J)
  tau <- matrix(0, nrow = n, ncol = d * J)

  for (j in seq_len(J)) {
    v_one <- matrix(v_mat[j, ], nrow = 1, ncol = d)
    k_xv <- eval_kernel(kernel_obj, x_mat, v_one)
    grad_arr <- grad_x_kernel(kernel_obj, x_mat, v_one)

    grad_k <- matrix(0, nrow = n, ncol = d)
    for (l in seq_len(d)) grad_k[, l] <- grad_arr[, 1, l]

    xi <- (grads_mat * as.numeric(k_xv[, 1]) + grad_k) * scale_fac
    tau[, ((j - 1L) * d + 1L):(j * d)] <- xi
  }

  tau
}


#' Estimate squared FSSD from a feature matrix
#'
#' Computes the unbiased U-statistic estimate of squared FSSD from the feature
#' matrix returned by [compute_tau()].
#'
#' @details
#' If `tau_i` is row `i` of `tau_matrix`, the returned value is
#' \deqn{
#' \frac{1}{n(n - 1)} \sum_{i \ne j} \tau_i^T \tau_j.
#' }
#' This is the sample version of the paper's `FSSD^2` for the selected test
#' locations. The implementation uses column sums to avoid explicitly forming
#' all pairwise inner products.
#'
#' This function is separated from [compute_tau()] because the same feature
#' matrix is also needed for the null simulation. Keeping the statistic as its
#' own step makes the FSSD workflow explicit: features first, U-statistic
#' second, null simulation third.
#'
#' The diagonal terms `tau_i^T tau_i` are excluded for the same reason that a
#' U-statistic excludes self-pairs: the target quantity is an expectation over
#' two independent draws from the sample distribution. The implementation uses
#' the identity
#' \deqn{
#' \sum_{i \ne j} \tau_i^T \tau_j =
#' ||\sum_i \tau_i||^2 - \sum_i ||\tau_i||^2
#' }
#' so it can compute the statistic from column sums and row norms rather than
#' forming the full `n x n` matrix of feature inner products.
#'
#' @param tau_matrix Numeric FSSD feature matrix.
#'
#' @return
#' One numeric value: the unbiased estimate of squared FSSD for the supplied
#' feature matrix. This is the value stored as `statistic` by [fssd_rand_test()]
#' and [fssd_opt_test()] before it is scaled by `n` for the null comparison. It
#' is returned as a plain number because the location matrix `V`, kernel choice,
#' and null-simulation settings are carried by the higher-level test object.
#' @examples
#' tau <- matrix(c(1, 2, 3), ncol = 1)
#' compute_fssd_unbiased_stat(tau)
#' @export
compute_fssd_unbiased_stat <- function(tau_matrix) {
  tau <- as.matrix(tau_matrix)
  n <- nrow(tau)
  if (n < 2) stop("Need at least two rows to compute the unbiased FSSD statistic")

  sum_tau <- colSums(tau)
  as.numeric((sum(sum_tau * sum_tau) - sum(rowSums(tau * tau))) / (n * (n - 1)))
}


#' Simulate the FSSD null distribution
#'
#' Simulates the asymptotic null distribution used to turn an observed FSSD
#' statistic into a p-value.
#'
#' @details
#' Let `lambda_1, ..., lambda_m` be the eigenvalues of the centered empirical
#' feature covariance matrix
#' \deqn{\hat\Sigma = n^{-1}\sum_i(\tau_i-\bar\tau)(\tau_i-\bar\tau)^T.}
#' The null draws have the form
#' \deqn{\sum_j \lambda_j (Z_j^2 - 1),}
#' with independent standard normal `Z_j`. The observed value compared against
#' those draws is `n * fssd_stat`. This matches the FSSD paper's null result for
#' the scaled statistic `n * FSSDhat^2`, using the empirical plug-in covariance
#' of `tau(X)` in place of the population covariance.
#'
#' This is the final lower-level step used by [fssd_rand_test()] and
#' [fssd_opt_test()]. It is exported so users can change the number of null
#' simulations, inspect the eigenvalues, or compare the same observed statistic
#' under different Monte Carlo seeds without recomputing `tau`.
#'
#' @param tau_matrix Numeric FSSD feature matrix.
#' @param fssd_stat Observed FSSD statistic.
#' @param n_simulations Number of null draws to simulate.
#'
#' @return
#' A list with three entries:
#' * `p_value`: fraction of simulated null draws at least as large as
#'   `test_score`.
#' * `test_score`: the observed statistic multiplied by `nrow(tau_matrix)`,
#'   matching the paper's asymptotic null scale.
#' * `eigenvalues`: nonnegative eigenvalues of the empirical covariance matrix
#'   used as weights in the simulated null distribution.
#'
#' The three entries are returned together because they answer different
#' diagnostic questions. `p_value` is the value used by [fssd_rand_test()] and
#' [fssd_opt_test()] to make the hypothesis-test object. `test_score` records
#' the exact scaled number compared with the null draws, which avoids ambiguity
#' about whether the unscaled or `n`-scaled statistic was used. `eigenvalues`
#' expose the estimated null distribution; if most eigenvalues are numerically
#' zero, the finite feature set or sample size may be giving a nearly
#' degenerate null simulation.
#' @examples
#' tau <- matrix(c(1, 2, 3), ncol = 1)
#' compute_fssd_null_pvalue(tau, fssd_stat = 0.1, n_simulations = 10)
#' @export
compute_fssd_null_pvalue <- function(tau_matrix, fssd_stat, n_simulations = 2000) {
  tau <- as.matrix(tau_matrix)
  if (!is.numeric(tau) || nrow(tau) < 2 || ncol(tau) < 1) {
    stop("tau_matrix must be a numeric matrix with at least two rows")
  }
  if (!is.numeric(fssd_stat) || length(fssd_stat) != 1 || !is.finite(fssd_stat)) {
    stop("fssd_stat must be a finite scalar")
  }
  n_simulations <- validate_positive_integer(n_simulations, "n_simulations")

  sigma_tau_hat <- fssd_tau_covariance(tau)
  eigvals <- eigen(sigma_tau_hat, symmetric = TRUE, only.values = TRUE)$values
  eigvals[eigvals < 0] <- 0

  z <- matrix(stats::rnorm(n_simulations * length(eigvals)),
              nrow = n_simulations, ncol = length(eigvals))
  chi_terms <- z^2 - 1
  s_null <- as.numeric(chi_terms %*% eigvals)
  test_score <- nrow(tau) * fssd_stat

  list(
    p_value = mean(s_null >= test_score),
    test_score = test_score,
    eigenvalues = eigvals
  )
}


# ---- Computation Engines -------------------------------------------------

#' Compute the FSSD statistic and p-value
#'
#' Builds the feature matrix, computes the unbiased FSSD statistic, and runs the
#' null simulation.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param V Matrix of FSSD test locations.
#' @param kernel_obj Stein kernel object.
#' @param n_simulations Number of null draws to simulate.
#'
#' @return A list with `fssd_stat` and `p_value`.
#' @noRd
internal_fssd_compute_engine <- function(X, grads, V, kernel_obj, n_simulations) {
  tau <- compute_tau(X = X, grads = grads, V = V, kernel_obj = kernel_obj)
  fssd_stat <- compute_fssd_unbiased_stat(tau)
  null_res <- compute_fssd_null_pvalue(
    tau_matrix = tau,
    fssd_stat = fssd_stat,
    n_simulations = n_simulations
  )

  list(fssd_stat = fssd_stat, p_value = null_res$p_value)
}


#' Optimize the FSSD training criterion
#'
#' Initializes test locations from a Gaussian fit, searches over kernel scales,
#' and then refines the location and scale parameters with bounded optimization.
#' The final FSSD statistic and null simulation are computed separately.
#'
#' @details
#' The optimizer first draws starting locations from a Gaussian fitted to the
#' training data. When `use_scale_grid = TRUE`, it tries a short grid centered
#' on the exact median squared scale and uses the best value as the starting
#' scale. Otherwise the scale already stored in `kernel_obj` is used directly.
#' The final step is bounded optimization of the packed location coordinates
#' and the square root of the squared scale.
#'
#' @param X_train Numeric training sample matrix.
#' @param grads_train Score matrix for `X_train`.
#' @param J Number of FSSD test locations.
#' @param kernel_obj Starting Stein kernel object.
#' @param gamma Positive stabilizer in the training objective.
#' @param maxit Maximum number of optimizer iterations.
#' @param locs_bounds_frac Width of the location box in sample standard
#'   deviations.
#' @param scale_lower,scale_upper Bounds for the optimized squared scale.
#' @param use_scale_grid Logical; whether to initialize the scale from an exact
#'   median-centered grid.
#'
#' @return A list with optimized locations `V_opt`, optimized squared scale
#'   `sigma2_opt`, optimized kernel `kernel_obj_opt`, final objective value,
#'   optimizer diagnostics, and the scale-grid values tried before optimization.
#' @noRd
internal_fssd_optimize_engine <- function(X_train, grads_train, J, kernel_obj,
                                          gamma = 1e-4,
                                          maxit = 100,
                                          locs_bounds_frac = 10,
                                          scale_lower = 1e-1,
                                          scale_upper = 1e4,
                                          use_scale_grid = TRUE) {
  x_train <- validate_samples(X_train, min_rows = 2)
  grads <- as.matrix(grads_train)
  if (!is.numeric(grads) || nrow(grads) != nrow(x_train) || ncol(grads) != ncol(x_train)) {
    stop("grads_train must be a numeric matrix with the same shape as X_train")
  }
  if (!inherits(kernel_obj, "SteinKernel")) stop("kernel_obj must be a SteinKernel object")

  J <- validate_positive_integer(J, "J")
  maxit <- validate_positive_integer(maxit, "maxit")
  gamma <- validate_fssd_positive_scalar(gamma, "gamma")
  locs_bounds_frac <- validate_fssd_positive_scalar(locs_bounds_frac, "locs_bounds_frac")
  scale_lower <- validate_fssd_positive_scalar(scale_lower, "scale_lower")
  scale_upper <- validate_fssd_positive_scalar(scale_upper, "scale_upper")
  if (scale_lower >= scale_upper) stop("scale_lower must be smaller than scale_upper")

  n <- nrow(x_train)
  d <- ncol(x_train)
  has_kernel_param <- inherits(kernel_obj, "SteinKernel_gaussian_rbf") ||
    inherits(kernel_obj, "SteinKernel_imq")
  n_params <- d * J + if (has_kernel_param) 1L else 0L
  if (n < n_params) {
    warning(sprintf(
      "Number of optimization parameters (%s) exceeds the training sample size; FSSD-opt is ill-posed.",
      if (has_kernel_param) "d*J+1" else "d*J"
    ), call. = FALSE)
  }

  v_init <- fssd_init_locations_from_fit(x_train, J, cov_reg = 1e-6)
  scale2_init <- if (has_kernel_param) kernel_scaling_value(kernel_obj) else NA_real_
  scale_grid <- scale_grid_objectives <- NULL

  if (has_kernel_param && use_scale_grid) {
    scale_grid <- fssd_scale_grid(scale2_init)
    grid_res <- fssd_grid_search_scale(
      X = x_train,
      grads = grads,
      V = v_init,
      kernel_obj_base = kernel_obj,
      scale_grid = scale_grid,
      gamma = gamma
    )
    scale2_init <- grid_res$scale
    scale_grid_objectives <- grid_res$objectives
  }

  bounds <- fssd_optimization_bounds(
    X = x_train,
    J = J,
    has_kernel_param = has_kernel_param,
    locs_bounds_frac = locs_bounds_frac,
    scale_lower = scale_lower,
    scale_upper = scale_upper
  )

  opt_bounds <- bounds
  if (has_kernel_param) {
    scale_idx <- J * d + 1L
    opt_bounds$lower[scale_idx] <- sqrt(opt_bounds$lower[scale_idx])
    opt_bounds$upper[scale_idx] <- sqrt(opt_bounds$upper[scale_idx])
  }

  # L-BFGS-B sees sqrt(scale2) for better conditioning. The kernel itself
  # still uses the squared scale.
  par0 <- if (has_kernel_param) c(as.numeric(t(v_init)), sqrt(scale2_init))
          else as.numeric(t(v_init))
  par0 <- fssd_project_parameters(par0, opt_bounds$lower, opt_bounds$upper)

  cache <- new.env(parent = emptyenv())
  objective_at <- function(par) {
    if (!is.null(cache$par) && length(cache$par) == length(par) && all(cache$par == par)) {
      return(cache$og)
    }
    par_obj <- par
    if (has_kernel_param) par_obj[scale_idx] <- par[scale_idx]^2
    og <- fssd_objective_and_grad(
      par = par_obj,
      X = x_train,
      grads = grads,
      J = J,
      kernel_obj_base = kernel_obj,
      has_kernel_param = has_kernel_param,
      gamma = gamma
    )
    if (has_kernel_param) og$grad[scale_idx] <- og$grad[scale_idx] * (2 * par[scale_idx])
    cache$par <- par
    cache$og <- og
    og
  }

  opt_res <- stats::optim(
    par = par0,
    fn = function(par) -objective_at(par)$value,
    gr = function(par) -objective_at(par)$grad,
    method = "L-BFGS-B",
    lower = opt_bounds$lower,
    upper = opt_bounds$upper,
    control = list(
      maxit = maxit,
      pgtol = 1e-7,
      factr = 1e-8 / .Machine$double.eps
    )
  )

  best_par <- opt_res$par
  best_og <- objective_at(best_par)
  best_value <- best_og$value

  if (has_kernel_param) {
    V_opt <- matrix(best_par[seq_len(J * d)], nrow = J, ncol = d, byrow = TRUE)
    sigma2_opt <- best_par[J * d + 1L]^2
  } else {
    sigma2_opt <- NA_real_
    V_opt <- matrix(best_par, nrow = J, ncol = d, byrow = TRUE)
  }

  list(
    V_opt = V_opt,
    sigma2_opt = sigma2_opt,
    kernel_obj_opt = fssd_kernel_with_scale(kernel_obj, sigma2_opt, has_kernel_param),
    objective_opt = best_value,
    function_evaluations = unname(opt_res$counts[["function"]]),
    convergence = opt_res$convergence,
    scale_grid = scale_grid,
    scale_grid_objectives = scale_grid_objectives
  )
}


#' Objective and gradient for FSSD-opt training
#'
#' `par` packs `vec(V)` and, for built-in kernels, the squared kernel scale
#' directly: `h^2` for Gaussian RBF and `c^2` for IMQ. The optimizer uses
#' `FSSD^2 / (sqrt(var_H1) + gamma)`, with fixed `gamma`.
#'
#' @details
#' This function evaluates the training objective used by [fssd_opt_test()] and
#' returns its derivative with respect to the packed parameter vector. The
#' derivative is used by `stats::optim(method = "L-BFGS-B")`.
#'
#' @param par Packed parameter vector containing test locations and optionally
#'   the squared kernel scale.
#' @param X Numeric training sample matrix.
#' @param grads Score matrix for `X`.
#' @param J Number of FSSD test locations.
#' @param kernel_obj_base Starting Stein kernel object.
#' @param has_kernel_param Logical; whether the last parameter is a squared
#'   kernel scale.
#' @param gamma Positive stabilizer in the denominator.
#'
#' @return A list with `value` and `grad`.
#'
#' @noRd
fssd_objective_and_grad <- function(par, X, grads, J, kernel_obj_base,
                                    has_kernel_param, gamma) {
  gamma <- validate_fssd_positive_scalar(gamma, "gamma")
  d <- ncol(X)
  n_local <- nrow(X)

  V <- matrix(par[seq_len(J * d)], nrow = J, ncol = d, byrow = TRUE)
  scale2_k <- if (has_kernel_param) par[J * d + 1L] else NA_real_

  kernel_obj_current <- fssd_kernel_with_scale(kernel_obj_base, scale2_k, has_kernel_param)
  tau <- compute_tau(X, grads, V, kernel_obj_current)
  p_local <- ncol(tau)

  mu_tau <- colMeans(tau)
  sum_tau <- colSums(tau)
  fssd2 <- compute_fssd_unbiased_stat(tau)
  grad_f <- (2 / (n_local * (n_local - 1))) *
    (matrix(sum_tau, n_local, p_local, byrow = TRUE) - tau)

  centered_tau <- sweep(tau, 2, mu_tau, "-")
  sigma_tau <- crossprod(centered_tau) / n_local
  sigma_h1_sq <- as.numeric(4 * t(mu_tau) %*% sigma_tau %*% mu_tau)
  sigma_h1 <- sqrt(max(sigma_h1_sq, 0))
  denom <- sigma_h1 + gamma

  sigma_mu <- as.numeric(sigma_tau %*% mu_tau)
  c_mu <- as.numeric(centered_tau %*% mu_tau)
  grad_sigma_sq <-
    (8 / n_local) * matrix(sigma_mu, n_local, p_local, byrow = TRUE) +
    (8 / n_local) * (c_mu %o% mu_tau)

  d_denom <- if (sigma_h1 > 1e-12) grad_sigma_sq / (2 * sigma_h1)
             else matrix(0, n_local, p_local)
  grad_obj_tau <- grad_f / denom - (fssd2 / (denom^2)) * d_denom

  scale_fac <- 1 / sqrt(d * J)
  grad_v <- matrix(0, nrow = J, ncol = d)
  grad_param <- 0

  for (j in seq_len(J)) {
    g_block <- grad_obj_tau[, ((j - 1L) * d + 1L):(j * d), drop = FALSE]
    fssd_grads <- grad_theta_v_kernel(
      obj = kernel_obj_current,
      X = X,
      vj = V[j, ],
      grads_X = grads,
      g_block = g_block * scale_fac
    )
    grad_v[j, ] <- fssd_grads$grad_vj
    grad_param <- grad_param + fssd_grads$grad_param
  }

  grad_par <- as.numeric(t(grad_v))
  if (has_kernel_param) grad_par <- c(grad_par, grad_param)

  list(value = fssd2 / denom, grad = grad_par)
}


# ---- Helpers -------------------------------------------------------------

#' Prepare samples, scores, and kernel for FSSD
#'
#' Checks sample and score shapes, then prepares the Stein kernel and kernel
#' scale.
#'
#' @param X Numeric vector or matrix of samples.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param min_rows Minimum number of rows allowed in `X`.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object.
#' @param scaling Optional positive squared kernel scale.
#' @param imq_beta IMQ exponent.
#'
#' @return A list with checked samples, scores, kernel object, kernel name, and
#'   selected scale.
#' @noRd
prepare_fssd_inputs <- function(X, score_function, min_rows,
                                kernel = c("gaussian_rbf", "imq"),
                                scaling = NULL,
                                imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = min_rows)
  grads <- validate_scores(score_function, x_mat)
  kernel_prep <- prepare_fssd_kernel(
    X = x_mat,
    kernel = kernel,
    scaling = scaling,
    imq_beta = imq_beta
  )

  c(list(X = x_mat, grads = grads), kernel_prep)
}


#' Prepare a Stein kernel for FSSD
#'
#' Resolves a kernel name or object into a `SteinKernel` object with a usable
#' scale.
#'
#' @details
#' If `scaling` is missing, the square of the median distance between all
#' sample-row pairs is used. For Gaussian RBF kernels this sets `h^2`; for IMQ
#' kernels this sets `c^2`. When a kernel object is supplied, assigning a
#' missing scale preserves its other settings, including its preconditioner.
#'
#' @param X Numeric sample matrix used to choose a default scale.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object.
#' @param scaling Optional positive squared kernel scale.
#' @param imq_beta IMQ exponent.
#'
#' @return A list with `kernel_obj`, `kernel_name`, `scaling`, and a logical
#'   `used_default_scale` indicating whether the exact median supplied the scale.
#' @noRd
prepare_fssd_kernel <- function(X,
                                kernel = c("gaussian_rbf", "imq"),
                                scaling = NULL,
                                imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = 2)
  used_default_scale <- FALSE

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_type_name(kernel_obj)
    kernel_scale <- kernel_scaling_value(kernel_obj)

    if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- find_median_distance(x_mat)
        used_default_scale <- TRUE
      }
      kernel_obj <- fssd_kernel_with_scale(kernel_obj, scaling, has_kernel_param = TRUE)
    } else if (inherits(kernel_obj, "SteinKernel_imq") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- find_median_distance(x_mat)
        used_default_scale <- TRUE
      }
      kernel_obj <- instantiate_kernel("imq", max(scaling, 1e-8), imq_beta)
    } else {
      scaling <- kernel_scale
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat)
      used_default_scale <- TRUE
    }
    if (kernel_name == "imq") scaling <- max(scaling, 1e-8)
    kernel_obj <- instantiate_kernel(kernel_name, scaling, imq_beta)
  }

  list(
    kernel_obj = kernel_obj,
    kernel_name = kernel_name,
    scaling = scaling,
    used_default_scale = used_default_scale
  )
}


#' Build a local grid of FSSD kernel scales
#'
#' Creates five squared scales around a supplied center:
#' `center * 2^seq(-3, 3, length.out = 5)`. FSSD-opt passes the exact median
#' scale already computed while preparing its kernel, avoiding a second
#' quadratic distance calculation.
#'
#' @param center Positive squared scale at the center of the grid.
#'
#' @return Numeric vector of candidate squared scales.
#' @noRd
fssd_scale_grid <- function(center) {
  center <- validate_fssd_positive_scalar(center, "center")
  center * 2^seq(-3, 3, length.out = 5)
}


#' Search FSSD kernel scales with fixed test locations
#'
#' Evaluates the FSSD-opt training objective over a supplied grid of squared
#' kernel scales and keeps the best one.
#'
#' @param X Numeric training sample matrix.
#' @param grads Score matrix for `X`.
#' @param V Matrix of fixed FSSD test locations.
#' @param kernel_obj_base Starting Stein kernel object.
#' @param scale_grid Numeric vector of positive squared scales.
#' @param gamma Positive stabilizer in the training objective.
#'
#' @return A list with the selected `scale`, all `objectives`, and
#'   `best_index`.
#' @noRd
fssd_grid_search_scale <- function(X, grads, V, kernel_obj_base, scale_grid, gamma) {
  scale_grid <- as.numeric(scale_grid)
  scale_grid <- scale_grid[is.finite(scale_grid) & scale_grid > 0]
  if (length(scale_grid) < 1) stop("scale_grid must contain at least one positive finite value")

  J <- nrow(V)
  par_v <- as.numeric(t(V))
  objectives <- vapply(scale_grid, function(scale2) {
    fssd_objective_and_grad(
      par = c(par_v, scale2),
      X = X,
      grads = grads,
      J = J,
      kernel_obj_base = kernel_obj_base,
      has_kernel_param = TRUE,
      gamma = gamma
    )$value
  }, numeric(1))

  finite_objectives <- objectives
  finite_objectives[!is.finite(finite_objectives)] <- -Inf
  if (all(is.infinite(finite_objectives))) {
    return(list(scale = scale_grid[1L], objectives = objectives, best_index = 1L))
  }

  best_index <- which.max(finite_objectives)
  list(scale = scale_grid[best_index], objectives = objectives, best_index = best_index)
}


#' Draw initial FSSD test locations from a fitted Gaussian
#'
#' Fits a Gaussian distribution to `X` using the sample mean and covariance, then
#' draws `J` rows from that Gaussian.
#'
#' @param X Numeric sample matrix.
#' @param J Number of locations to draw.
#' @param cov_reg Nonnegative value added to the covariance diagonal.
#'
#' @return Numeric matrix with `J` rows and `ncol(X)` columns.
#' @noRd
fssd_init_locations_from_fit <- function(X, J, cov_reg = 0) {
  X <- as.matrix(X)
  cov_reg <- if (is.null(cov_reg)) 0 else as.numeric(cov_reg)
  if (length(cov_reg) != 1 || !is.finite(cov_reg) || cov_reg < 0) {
    stop("cov_reg must be a non-negative scalar")
  }
  sigma <- safe_covariance(X)
  if (cov_reg > 0) diag(sigma) <- diag(sigma) + cov_reg
  sample_gaussian_rows(J, colMeans(X), sigma)
}


#' Update a kernel's squared scale
#'
#' Updates a copy of a built-in kernel after the FSSD optimizer changes the
#' squared scale. All non-scale settings, including a preconditioner, are
#' preserved.
#'
#' @param base_kernel Stein kernel object used as the template.
#' @param scale2_k Positive squared scale.
#' @param has_kernel_param Logical; whether the kernel has an optimized scale.
#'
#' @return A Stein kernel object.
#' @noRd
fssd_kernel_with_scale <- function(base_kernel, scale2_k, has_kernel_param) {
  if (!has_kernel_param) return(base_kernel)
  scale2_k <- validate_fssd_positive_scalar(scale2_k, "scale2_k")

  if (inherits(base_kernel, "SteinKernel_gaussian_rbf")) {
    out <- base_kernel
    out$h2 <- scale2_k
    return(out)
  }
  if (inherits(base_kernel, "SteinKernel_imq")) {
    out <- base_kernel
    out$c <- sqrt(scale2_k)
    return(out)
  }

  stop("Kernel parameter optimization is not implemented for this kernel class")
}


#' Build box bounds for FSSD optimization
#'
#' Constructs lower and upper bounds for all test-location coordinates and, when
#' needed, the squared kernel scale.
#'
#' @param X Numeric training sample matrix.
#' @param J Number of FSSD test locations.
#' @param has_kernel_param Logical; whether a squared kernel scale is optimized.
#' @param locs_bounds_frac Width of the location box in sample standard
#'   deviations.
#' @param scale_lower,scale_upper Bounds for the squared kernel scale.
#'
#' @return A list with numeric vectors `lower` and `upper`.
#' @noRd
fssd_optimization_bounds <- function(X, J, has_kernel_param, locs_bounds_frac,
                                     scale_lower, scale_upper) {
  x_mat <- as.matrix(X)
  x_std <- apply(x_mat, 2, stats::sd)
  x_std[!is.finite(x_std) | x_std <= 0] <- 1e-8

  v_lb <- rep(apply(x_mat, 2, min) - locs_bounds_frac * x_std, times = J)
  v_ub <- rep(apply(x_mat, 2, max) + locs_bounds_frac * x_std, times = J)

  if (!has_kernel_param) {
    return(list(lower = v_lb, upper = v_ub))
  }

  list(
    lower = c(v_lb, scale_lower),
    upper = c(v_ub, scale_upper)
  )
}


#' Project FSSD parameters into their bounds
#'
#' Clips each parameter to the interval specified by the matching entries of
#' `lower` and `upper`.
#'
#' @param par Numeric parameter vector.
#' @param lower,upper Numeric vectors of lower and upper bounds.
#'
#' @return Numeric vector inside the requested bounds.
#' @noRd
fssd_project_parameters <- function(par, lower, upper) {
  pmin(pmax(par, lower), upper)
}


#' Estimate the covariance used in the FSSD null simulation
#'
#' Centers the feature matrix by columns and returns the empirical covariance
#' matrix used to simulate the null distribution.
#'
#' @param tau Numeric FSSD feature matrix.
#'
#' @return Symmetric numeric covariance matrix.
#' @noRd
fssd_tau_covariance <- function(tau) {
  x_mat <- as.matrix(tau)
  if (!is.numeric(x_mat) || nrow(x_mat) < 1 || ncol(x_mat) < 1) {
    stop("tau must be a non-empty numeric matrix")
  }
  centered <- sweep(x_mat, 2, colMeans(x_mat), "-")
  cov_hat <- crossprod(centered) / nrow(x_mat)
  cov_hat[!is.finite(cov_hat)] <- 0
  (cov_hat + t(cov_hat)) / 2
}


#' Check a positive scalar for FSSD helpers
#'
#' @param x Value to check.
#' @param arg_name Name used in the error message.
#'
#' @return The checked value as a number.
#' @noRd
validate_fssd_positive_scalar <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) || x <= 0) {
    stop(sprintf("%s must be a positive scalar", arg_name))
  }
  as.numeric(x)
}


#' Check the FSSD training fraction
#'
#' @param train_ratio Requested fraction of rows used for FSSD-opt training.
#'
#' @return The checked value as a number in `(0, 1)`.
#' @noRd
validate_fssd_train_ratio <- function(train_ratio) {
  if (!is.numeric(train_ratio) || length(train_ratio) != 1 ||
      !is.finite(train_ratio) || train_ratio <= 0 || train_ratio >= 1) {
    stop("train_ratio must be a scalar in (0, 1)")
  }
  as.numeric(train_ratio)
}


#' Compute a stable sample covariance matrix
#'
#' Handles the one-dimensional case explicitly and replaces non-finite
#' covariance entries by zero.
#'
#' @param x Numeric sample matrix.
#'
#' @return Symmetric covariance matrix.
#' @noRd
safe_covariance <- function(x) {
  x_mat <- as.matrix(x)
  if (ncol(x_mat) == 1) {
    v <- stats::var(x_mat[, 1])
    if (!is.finite(v) || v < 0) v <- 0
    return(matrix(v, 1, 1))
  }

  s <- stats::cov(x_mat)
  s[!is.finite(s)] <- 0
  (s + t(s)) / 2
}


#' Draw rows from a Gaussian with a stabilized covariance root.
#'
#' Uses an eigenvalue decomposition of `sigma`. Very small or negative
#' eigenvalues are lifted to a small positive value before drawing samples.
#'
#' @param n Number of rows to draw.
#' @param mu Mean vector.
#' @param sigma Covariance matrix.
#'
#' @return Numeric matrix with `n` rows.
#' @noRd
sample_gaussian_rows <- function(n, mu, sigma) {
  d <- length(mu)
  sigma <- as.matrix(sigma)
  if (!all(dim(sigma) == c(d, d))) stop("sigma must be d x d")

  sigma <- (sigma + t(sigma)) / 2
  eig <- eigen(sigma, symmetric = TRUE)
  vals <- pmax(eig$values, 1e-10)
  root <- eig$vectors %*% diag(sqrt(vals), nrow = d)
  z <- matrix(stats::rnorm(n * d), nrow = n, ncol = d)
  sweep(z %*% t(root), 2, mu, "+")
}
